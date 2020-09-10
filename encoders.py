import numpy as np
import torch
import torch.nn.functional as F
import os
import pickle
dir_path = os.path.dirname(os.path.realpath(__file__))

class rand():

    def __init__(self, data=None):
        pass
    
    def __str__(self):
         return "Random"
    
    def encode(self, x):
        return np.random.rand(2)

class none():

    def __init__(self, data=None):
        pass
    
    def __str__(self):
         return "Raw"
    
    def encode(self, x):
        return x

class convautoencoder():

    def __init__(self, model_name="model.pt", data=None):
        if model_name is None:
            import model
            self.enc = model.Autoencoder()
        else:
            self.enc = torch.load(os.path.join(dir_path,model_name), map_location='cpu')
        self.enc.eval()

    def __str__(self):
         return "ConvAE"

    def encode(self, x):
        import torch
        x = torch.from_numpy(x).float()
        emb = self.enc.encode(x[None, None, :])[0, :, 0].detach().numpy()
        return emb

    def decode(self, x):
        import torch
        x = torch.from_numpy(x)
        emb = self.enc.decode(x[None,:,None]).detach().numpy()
        return emb[0][0]

class convautoencoder_random(convautoencoder):
    def __init__(self, model_name=None, data=None):
        convautoencoder.__init__(self, model_name=None)
    def __str__(self):
         return "ConvAE (Random init)"

class pca():
    def __init__(self, data=None, pca_dim=100, pca_path="pca.pkl.gz"):
        from sklearn.decomposition import PCA
        self.pca_dim = pca_dim
        self.pca = PCA(n_components=pca_dim)
        # load precomputed components from training data
        mean,components = pickle.load(open(os.path.join(dir_path,pca_path),"rb"))
        if len(components) < self.pca_dim:
            raise Exception("File only has {} components".format(len(components)))
        self.pca.components_ = components[:self.pca_dim]
        self.pca.mean_ = mean
    def __str__(self):
         return "PCA (dim:{})".format(self.pca_dim)
    def encode(self, x):
        return self.pca.transform([x])[0]
    def decode(self, x):
        return self.pca.inverse_transform([x])[0]

class pca_10(pca):
    def __init__(self):
        pca.__init__(self, pca_dim=10, pca_path="pca_10.pkl.gz")

class pca_50(pca):
    def __init__(self):
        pca.__init__(self, pca_dim=50, pca_path="pca_50.pkl.gz")

class pca_100(pca):
    def __init__(self):
        pca.__init__(self, pca_dim=100, pca_path="pca_100.pkl.gz")


class fft():
    
    def __init__(self, data=None):
        pass
    
    def __str__(self):
         return "FFT"
    
    def encode(self, x):
        import scipy.fftpack
        N = len(x)
        yf = scipy.fftpack.fft(x)
        return np.abs(yf[:N//2])
    
class periodogram():
    # https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.signal.periodogram.html
    def __init__(self, data=None):
        pass
    
    def __str__(self):
         return "Periodogram"
    
    def encode(self, x):
        from scipy import signal
        f, Pxx_den = signal.periodogram(x)
        return Pxx_den


class biosppy_mean_beat():
    # BioSPPy
    # https://biosppy.readthedocs.io/en/stable/biosppy.html
    # Carreiras, Carlos, et al. BioSPPy: Biosignal Processing in {Python}. 2015, https://github.com/PIA-Group/BioSPPy/.
    
    def __init__(self, data=None, sampling_rate=100., default_emb_length_not_inferable=120):
        self.sampling_rate=sampling_rate
        self.default_emb_length_not_inferable = default_emb_length_not_inferable
    
    def __str__(self):
         return "BioSPPy (sample_rate:{})".format(self.sampling_rate)
    
    def encode(self, x):
        from biosppy.signals import ecg
        
        try:
            out = ecg.ecg(signal=x, sampling_rate=self.sampling_rate, show=False)
            out = np.concatenate([out["templates"].T.mean(1), out["templates"].T.std(1)], axis=0)
        except ValueError as e:
            print(" Error:",e," Writing zeros instead.")
            try:
                out = np.zeros(self.emb_length)
            except:
                #some better approach should be found for this
                out = np.zeros(self.default_emb_length_not_inferable)
        self.emb_length = len(out)
        return out






