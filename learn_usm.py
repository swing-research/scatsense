import numpy as np
import glob
import os

outpath = 'results'
prefix = 'usm_is'
b = 0 #euc, kl, is
p = 1 #power or magnitude

outfile = os.path.join(outpath,prefix) #save true and corresponding estimated directions

#list of folders, one per speaker (from TIMIT) each containing wavefiles 
path ='data/universal/'
folders = glob.glob(path+'*') #list of speakers for validation
folders = folders[::2]
L = len(folders) #number of speakers

from core.signal import load_audio_norm,stft
from algorithms.nmf import nmf

K = 10 #number of NMF components per speaker

F2 = 513 #freq for fft
F = 385 #final #frequenies used
f0 = F2-F #start index

W = np.zeros((F,L*K)) #universal speech model

count = 0
for sp in folders:
    wavefile = glob.glob(sp+'/*.WAV') #get list of WAV files
    Sf_all = np.zeros((F,0),dtype=np.complex128)
    for wi,w in enumerate(wavefile):
        st0,Sf0 = load_audio_norm(w,F2)
        Sf_all = np.hstack((Sf_all,Sf0[f0:,:])) #concatenate all STFT spectra
    
    Wt,_,_ = nmf(np.abs(Sf_all)**p,b=b,K=K) #factorize power spectrogram: NMF with Kullback-Leibler divergence
    W[:,count*K:(count+1)*K] = Wt #add to usm
    count += 1

np.save(outfile,W)
