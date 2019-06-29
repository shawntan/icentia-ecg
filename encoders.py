import numpy as np

class rand():

    def __init__(self):
        pass
    
    def encode(self, x):
        return np.random.rand(2)


class convautoencoder():
    
    def __init__(self):
        import model
        self.enc = model.Autoencoder()
        self.enc.eval()
    
    def encode(self, x):
        import torch
        x = torch.from_numpy(x)
        emb = self.enc.autoencode_1.encode(x[None, None, :])[0, :, 0].detach().numpy()
        return emb


class pca():
    
    def __init__(self, data, pca_dim=100):
        from sklearn.decomposition import PCA
        self.pca_dim = pca_dim
        self.pca = PCA(n_components=pca_dim)
        self.pca.fit(data)
    
    def encode(self, x):
        return self.pca.transform([x])[0]


class fft():
    
    def __init__(self):
        pass
    
    def encode(self, x):
        import scipy.fftpack
        N = len(x)
        yf = scipy.fftpack.fft(x)
        return np.abs(yf[:N//2])


class biosppy():
    
    def __init__(self):
        pass
    
    def encode(self, x):
        from biosppy.signals import ecg
        out = ecg.ecg(signal=x, sampling_rate=500., show=False)
        return np.concatenate([out["templates"].T.mean(1), out["templates"].T.std(1)], axis=0)






