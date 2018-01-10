import numpy as np
import glob
from core.signal import load_audio_norm

path ='data/test4/*/'
path = 'data/test4_validation/'
folders = glob.glob(path+'*.WAV') #list of speakers for validation

L = len(folders) #number of speakers

S = dict()
St = dict()
count = 0
F2 = 513
for sp in folders:
    st0,Sf0 = load_audio_norm(sp,F2)
    np.save(sp,st0)
    count += 1
