import numpy as np
import torch
import torch.nn.functional as F
import os

class rand():

    def __init__(self):
        pass
    
    def __str__(self):
         return str(self.__class__)
    
    def encode(self, x):
        return np.random.rand(2)

class none():

    def __init__(self):
        pass
    
    def __str__(self):
         return str(self.__class__)
    
    def encode(self, x):
        return x

class convautoencoder():

    def __init__(self):
        import model
        # self.enc = model.Autoencoder()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.enc = torch.load(dir_path + '/model.pt', map_location='cpu')
        self.enc.eval()

    def __str__(self):
         return str(self.__class__)

    def encode(self, x):
        import torch
        x = torch.from_numpy(x)
        emb = self.enc.encode(x[None, None, :])[0, :, 0].detach().numpy()
        return emb

    def decode(self, x):
        import torch
        x = torch.from_numpy(x)
        emb = self.enc.decode(x[None,:,None]).detach().numpy()
        return emb


class pca():
    
    def __init__(self, data, pca_dim=100):
        from sklearn.decomposition import PCA
        self.pca_dim = pca_dim
        self.pca = PCA(n_components=pca_dim)
        self.pca.fit(data)
        
    def __str__(self):
         return str(self.__class__) + " " + str({"pca_dim":self.pca_dim})
    
    def encode(self, x):
        return self.pca.transform([x])[0]


class fft():
    
    def __init__(self):
        pass
    
    def __str__(self):
         return str(self.__class__)
    
    def encode(self, x):
        import scipy.fftpack
        N = len(x)
        yf = scipy.fftpack.fft(x)
        return np.abs(yf[:N//2])


class biosppy_mean_beat():
    # BioSPPy
    # https://biosppy.readthedocs.io/en/stable/biosppy.html
    # Carreiras, Carlos, et al. BioSPPy: Biosignal Processing in {Python}. 2015, https://github.com/PIA-Group/BioSPPy/.
    
    def __init__(self, sampling_rate=100.):
        self.sampling_rate=sampling_rate
    
    def __str__(self):
         return str(self.__class__) + " " + str({"sampling_rate":self.sampling_rate})
    
    def encode(self, x):
        from biosppy.signals import ecg
        
        try:
            out = ecg.ecg(signal=x, sampling_rate=self.sampling_rate, show=False)
            out = np.concatenate([out["templates"].T.mean(1), out["templates"].T.std(1)], axis=0)
        except ValueError as e:
            print(" Error:",e," Writing zeros instead.")
            out = np.zeros(self.emb_length)
        self.emb_length = len(out)
        return out






