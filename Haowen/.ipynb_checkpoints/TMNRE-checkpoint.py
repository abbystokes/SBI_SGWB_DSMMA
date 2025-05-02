import scipy.stats as stats
import pandas as pd
import numpy as np
import swyft
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch
from scipy.integrate import quad
from scipy import interpolate
import scipy
import jax
from jax import jit
import jax.numpy as jnp
from jax import device_get

import torch.nn as nn
import torch.nn.functional as F
np.random.seed(3407)
jax.config.update("jax_enable_x64", True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define simulator parameters
f_low = 10
f_high = 200
Delta_f = 1./32
N_bins = int((f_high-f_low)/Delta_f) + 1
freq = np.linspace(f_low, f_high+Delta_f, N_bins)
pi = np.pi
sqrt = np.sqrt
H100 = 3.241e-18
h    = 0.679
H0   = h * H100

# Define Simulator
class Simulator(swyft.Simulator):
    def __init__(self, fref, psd, gamma, T_obs, Nbins=len(freq), bounds=None):
        super().__init__()
        self.fref      = fref
        self.psd       = psd
        self.gamma     = gamma
        self.T_obs     = T_obs
        
        self.transform_samples = swyft.to_numpy32
        self.Nbins = Nbins
        self.freq = np.linspace(f_low, f_high, N_bins)
        self.sample_z = swyft.RectBoundSampler([stats.uniform(-12, 4), #omega stats.uniform(-12,5)
                                                stats.uniform(0,4)], #alpha
                                                bounds = bounds) #bounds changes range of the prior

    def psd_interp(self):
        return scipy.interpolate.interp1d(self.psd[:,0], self.psd[:,1])(self.freq)
    
    def gamma_interp(self):
        return scipy.interpolate.interp1d(self.gamma[:, 0], self.gamma[:, 1])(self.freq)
    
    def sigma(self):
        numerator = (20*pi**2*self.freq**3)**2 * self.psd_interp()**2
        denomenator = (3*H0**2)**2 * 8*self.gamma_interp()**2
        #denomenator = (3*H0**2)**2 * 8
        T = 1/(self.freq[1]-self.freq[0])
        N = 2*self.T_obs // T - 1
        return np.sqrt(numerator/denomenator/N)
    
    def C_groundtruth(self, z):
        Omega_ref = 10**z[0]
        alpha     = z[1]
        C_hat_ij = Omega_ref * (self.freq/self.fref)**alpha
        return C_hat_ij
    
    def build(self, graph):
        z = graph.node('z', self.sample_z)
        m = graph.node('m', self.C_groundtruth, z)
        x = graph.node("x", lambda m: m + np.random.normal(0, self.sigma()), m)
        #sigma = graph.node('sigma',self.sigma)
        
# Define swyft network


class AE(swyft.SwyftModule,swyft.AdamW):
    def __init__(self, lr = 1e-5, gamma=0.98, weight_decay=1e-5):
        super().__init__()
        
        marginals = ((0,1), )
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        # AE-Summarizer
        self.summarizer =  nn.Sequential(
            nn.Linear(6081, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.GELU(),
            # nn.Linear(8, 4),
            # nn.GELU(),
        )

        
        self.logratios_1D = swyft.LogRatioEstimator_1dim(
            num_features=4,  # Number of features in compressed x (simulated observation)
            num_params=2,    # Two parameters to infer: alpha and omega
            varnames='z', 
            dropout = 0# Names of the parameters
            # Number of neural network blocks
        )
            
        self.logratios_2D = swyft.LogRatioEstimator_Ndim(num_features = 4, 
                                                       marginals = marginals, varnames = 'z', num_blocks = 4,  dropout = 0)
        
            
    def forward(self, A, B):
        s = self.summarizer(A['x'])

        return self.logratios_1D(s, B['z']),  s, self.logratios_2D(s, B['z'])


def main():
    n_samples = 50_000
    fref = 25
    CE_PSD = np.genfromtxt("../Abby/data/cosmic_explorer_40km_for_paper.txt")
    CE_PSD[0, 0]  = 4.99999999999999999
    CE_PSD[:, 1] = CE_PSD[:, 1]**2
    gamma_HL = np.genfromtxt("../Abby/data/gamma_HL.txt")
    T_obs = 365 * 24 * 3600
    
    sim = Simulator(fref, CE_PSD, gamma_HL, T_obs, bounds=None)
    sigma = sim.sigma()
    sims = sim.sample(N = n_samples)
    sims['x'] = np.log10(np.abs(sims['x'])/sigma)
    
    
    dm = swyft.SwyftDataModule(sims, val_fraction = 0.2, batch_size = 1024)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta = 0., patience=3, verbose=False, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='/scratch/haowen.zhong/SBI/logs/', filename='rings_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
    logger = pl_loggers.TensorBoardLogger(save_dir='/scratch/haowen.zhong/SBI/logs/', name='AE_logs', version=None)
    
    trainer = swyft.SwyftTrainer(accelerator = 'gpu', max_epochs = 300, devices=1,   auto_lr_find=True, logger=logger, callbacks=[lr_monitor, early_stopping_callback, checkpoint_callback], precision=64,log_every_n_steps=5)
    network = AE(lr=8e-6, weight_decay=1e-5, gamma=0.98)#AE(lr=2e-5)#UNet(lr=2e-5, weight_decay=0.01)
    trainer.fit(network, dm)
# Sample from Simulator
if __name__ == '__main__':
    main()
