import numpy as np
from numpy.fft.fftpack import rfft,irfft
from math import pi
from scipy.io.wavfile import read


def stft(x,F):
    '''Compute STFT of 1D real signal x
    F: number of frequencies
    '''
    
    wn = 2*(F-1) #window size in samples
    nhop = wn/2 #hop size, 50% overlap
    
    L = len(x) #length of input signal
    zero_pad = nhop #number of zeros to add before and after the signal
    rem = L%nhop #number of samples leftover
    if rem>0: # adjust zero padding to have an integer number of windows
        zero_pad = wn-rem#nhop-rem#wn-rem
    x = np.hstack((np.zeros((nhop,)),x,np.zeros((zero_pad,)))) #zero padding at the beginning to avoid boundary effects
    L = len(x) #new length of signal
    
    N = (L-wn)/nhop + 1 #total number of frames
    X = np.zeros((F,N),dtype=np.complex128) #output STFT matrix
     
   
    w = 0.5- 0.5*np.cos(2*np.pi*np.arange(wn)/(wn)) #Hann window
    for i in range(N): #compute windowed fft for every frame
        xf = np.fft.rfft(x[i*nhop:i*nhop+wn]*w)
        X[:,i] = xf

    return X 

def istft(X,fs,T):
    '''Compute inverse STFT of X
    fs: sample rate in Hz, 
    T: duration in seconds
    '''
    
    Lout = int(fs*T) #actual length of time domain signal
    
    [F,N] = X.shape
    wn = 2*(F-1) #window size
    nhop = (wn)/2

    L = (N-1)*nhop+wn #length of output
    x = np.zeros((L,)) #output
   
    for i in range(N): #compute inverse fft for every frame
        xt = np.fft.irfft(X[:,i])
        x[i*nhop:i*nhop+wn] += xt #overlap and add

    return x[nhop:Lout+nhop] #truncate the zero padding

def load_audio(filename,F,T=-1,wn=None):
    '''Load T seconds of an audio wave file
    Returns the time domain signal and spectrogram'''
    fs,st = read(filename)
    st = np.reshape(st, (len(st),))
    if T>-1:
        N = np.rint(T*fs) #number of samples to read
        if N>len(st):
            raise RuntimeError('Cannot read %d samples. Signal has %d samples.'%(N,len(st)))
        st = st[0:N]
    
    Sf = stft(st,F)
    return st,Sf

def load_audio_norm(filename,F,T=-1,wn=None):
    '''Load T seconds of an audio wave file, normalize amplitude to 1
    Returns the time domain signal and spectrogram'''
    fs,st = read(filename)
    st = st/np.max(np.abs(st))
    st = np.reshape(st, (len(st),))
    if T>-1:
        N = np.rint(T*fs) #number of samples to read
        if N>len(st):
            raise RuntimeError('Cannot read %d samples. Signal has %d samples.'%(N,len(st)))
        st = st[0:N]
    
    Sf = stft(st,F)
    return st,Sf



