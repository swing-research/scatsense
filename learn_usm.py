import numpy as np
import glob
import os

from core.signal import load_audio_norm,stft
from algorithms.nmf import nmf


outpath = 'data'
div_b = ['IS', 'KL', 'EUC']
b = 0 #divergence: 0 for IS, 1 for KL, 2 for Euc
f0 = 192 #index for first frequency bin to use

prefix = 'usm_%s_sub%s'%(div_b[b], f0)

p = 1 #2 power or 1 magnitude

K = 10 #number of NMF components per speaker

F2 = 513 #freq for fft
F = F2-f0 #final number of frequency bins

outfile = os.path.join(outpath, prefix)

#list of folders, one per speaker (from TIMIT) each containing wavefiles 
path ='data/universal/'
folders = glob.glob(path+'*') #list of speakers for validation
folders = folders[::2]
L = len(folders) #number of speakers

W = np.zeros((F, L*K)) #universal speech model

count = 0
for sp in folders:
    wavefile = glob.glob(sp+'/*.WAV') #get list of WAV files
    Sf_all = np.zeros((F, 0),dtype=np.complex128)
    for wi,w in enumerate(wavefile):
        st0,Sf0 = load_audio_norm(w, F2)
        Sf_all = np.hstack((Sf_all, Sf0[f0:,:])) #concatenate all STFT spectra
    
    Wt,_,_ = nmf(np.abs(Sf_all)**p, b=b, K=K) #factorize power spectrogram: NMF with Kullback-Leibler divergence
    W[:, count*K:(count+1)*K] = Wt #add to usm
    count += 1

np.save(outfile, W)
