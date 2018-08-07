# Compare NMF initializations
# Speech localization with a universal speech model
# Usage: python exp_nmf_init.py config.txt

import logging
from proto import info_pb2
from google.protobuf import text_format
import datetime

from algorithms.nmf import log_l1_nmf
from core.processing import calculate_angle_error
from core.signal import stft

import numpy as np
import os
import sys
import glob

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("whitegrid")
sns.set(font_scale=1.5) 
cmap = "PuBu"


if len(sys.argv) != 2:
  print "Usage:", sys.argv[0], "EXPERIMENT_CONFIG_FILE"
  sys.exit(-1)

# Read the config file.
f = open(sys.argv[1], "rb")
exp_info = info_pb2.DataInfo()
text_format.Merge(f.read(), exp_info)
f.close()

# Create an output directory
outpath = os.path.join(exp_info.output_folder, '{:%Y%m%d_%H%M}'.format(datetime.datetime.now()))
if not os.path.exists(outpath):
    os.makedirs(outpath)

# Get experiment parameters
D = exp_info.D #number of directions
step = np.int32(360./D) #discretization 
Fn = np.int32(exp_info.nfft /2. +1) #number of frequencies in the spectrogram
J = exp_info.J #number of sources
SNR = exp_info.SNR #desired SNR
b = exp_info.beta #type of divergence

runs = exp_info.runs #number of times to run the experiment

# Get input signals
num_folders = len(exp_info.input_folder)
S = dict() #frequency domain signals
St = dict() #time domain signals
L = dict() #number of speakers per folder
for folder_i in range(num_folders):
    path = exp_info.input_folder[folder_i]
    if not path:
        sys.exit("Error: Please provide the path to the input test folder.")
    folders = glob.glob(path+'*.npy') #list of speakers for validation
    folders.sort() #the retrieved order is otherwise different on different machines

    L[folder_i] = len(folders) #number of speakers
    if L[folder_i] == 0:
        sys.exit("Error: Test folder %s is empty."%exp_info.input_folder[folder_i])
    S[folder_i] = dict() #source spectrograms
    St[folder_i] = dict() #source time domain signals (they have different durations)
    count = 0
    for sp in folders:
        St[folder_i][count] = np.load(sp) #load time domain samples
        S[folder_i][count] = stft(St[folder_i][count], Fn) #compute spectrogram
        count += 1

# Get device parameters
device = exp_info.device
dev = device.name
m = device.mic
lam = device.lam #sparsity parameter, determined previously by grid search
gam = device.gam #sparsity parameter, determined previously by grid search
f0 = np.int32(device.f0*exp_info.nfft*1./exp_info.fs) #starting frequency index

# Load the source model
W = np.load(exp_info.source_model)
k = W.shape[1] #size of the dictionary

logfile = os.path.join(outpath,"log.log")

logging.basicConfig(filename=logfile, format='%(asctime)s %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Load transfer functions %s'%dev)
H_theta0 = np.load(device.freq_response) #load transfer functions (includes 4 microphones)
Df = H_theta0.shape[1] #number of directions for a fine discretization

H_theta = H_theta0[f0:,::step,m] #coarse discretization, model
[F,D] = H_theta.shape #F: number of frequencies, D: number of directions

# Load corresponding impulse responses
H_theta0_t = np.load(device.impulse_response)
H_thetaf = H_theta0_t[:,:,m] #fine discretization
H_theta_t = H_theta0_t[:,::step,m] #coarse discretization

anglesf = np.arange(Df, dtype=np.int64)*360./Df # list of angles in degrees

p = k*np.ones((D,), dtype=np.int16) #group sizes

logger.info('Create mixing matrix A')
A = np.zeros((F,k*D)) #model
for d in range(D):
    A[:,d*k:(d+1)*k] = np.abs(H_theta[:,d,np.newaxis])*W
A = A/np.linalg.norm(A, axis=0)

np.random.seed(seed=0)

err_per_source = np.zeros((runs,J))
err_per_source2 = np.zeros((runs,J))

logger.info('Number of sources %s'%(J))
count_diff = 0 #counter for number of times initialization leads to different results
for rns in range(runs):
    theta = np.random.choice(range(Df), J, replace=False) #choose the directions randomly on the fine grid   

    if num_folders==1:
        src_ind = np.random.choice(L[0], J, replace=False) #pick speaker(s)
        folder_ind = np.zeros((J,))
    else: #choose one speaker from each folder
        src_ind = np.zeros((J,))
        folder_ind = np.zeros((J,))
        for j in range(J):
            src_ind[j] = np.random.choice(L[j], 1, replace=False)
            folder_ind[j] = j

    l_min = np.inf #length of shortest signal in the mix
    for si,src_j in enumerate(src_ind):
        l = St[folder_ind[si]][src_j].shape[0] #length of signal
        if l<l_min:
            l_min = l
    obs_len = l_min + H_theta_t.shape[0] - 1 #length of the convolution
                
    yt = np.zeros((obs_len,)) #recorded time domain signal
          
    for j in range(J): #generate spatial images and truncate appropriately to have signals of the same length
        yt += np.convolve(St[folder_ind[j]][src_ind[j]], H_thetaf[:,theta[j]])[:obs_len] #source signal convolved with corresponding directional response
     
    # Generate noise at required SNR    
    sig_norm = np.linalg.norm(yt)
    noise_t = np.random.randn(obs_len,) #additive gaussian noise
    noise_norm = sig_norm/(10.**(SNR/20.))
    noise_t = noise_norm*noise_t/np.linalg.norm(noise_t)

    yt += noise_t #noisy signal

    y = stft(yt, Fn)[f0:,:] #spectrogram of recorded signal
    N = y.shape[1] #number of frames
    
    # Run NMF twice with and without random initialization
    
    # Localization using NMF with non-random initializer X=A^T Y
    xs,_ = log_l1_nmf(np.abs(y), A, p, b=b, lam=lam, gam=gam)
    scores = np.log10(np.linalg.norm(np.reshape(xs.T,(k*N,D), order='F'), axis=0, ord=1) + 1e-80)
    best_dir = np.argpartition(scores, -J)[-J:] #indices of highest norms
    theta_hat = step*best_dir #map coarse index to fine index
    min_err, best_perm = calculate_angle_error(theta, theta_hat, anglesf) #calculate error between estimated and true directions
    for src_j in range(J): #error per source
        err_per_source[rns, src_j] = np.sum(np.absolute(((best_perm[src_j]-theta[src_j]+180) % 360)-180));
    
    # Localization using NMF with random initializer
    xs,_ = log_l1_nmf(np.abs(y), A, p, b=b, lam=lam, gam=gam, rand=True)
    scores = np.log10(np.linalg.norm(np.reshape(xs.T,(k*N,D), order='F'), axis=0, ord=1) + 1e-80)
    best_dir = np.argpartition(scores, -J)[-J:] #indices of highest norms
    theta_hat2 = step*best_dir #map coarse index to fine index
    min_err2, best_perm2 = calculate_angle_error(theta, theta_hat2, anglesf) #calculate error between estimated and true directions
    for src_j in range(J): #error per source
        err_per_source2[rns, src_j] = np.sum(np.absolute(((best_perm2[src_j]-theta[src_j]+180) % 360)-180));
    
    if len(np.intersect1d(best_perm,best_perm2))<J: #if results are different
        count_diff += 1
    
    logger.info('Test %s NMF1, theta: %s, theta_hat: %s, err: %s'%(rns, anglesf[theta], anglesf[best_perm], min_err))
    logger.info('Test %s NMF2, theta: %s, theta_hat: %s, err: %s'%(rns, anglesf[theta], anglesf[best_perm2], min_err2))

# Result statistics
exp_results = exp_info.summary
inds_below_10 = np.all(err_per_source <= exp_info.accuracy_bin, axis=1) #intersection
inds_below_102 = np.all(err_per_source2 <= exp_info.accuracy_bin, axis=1) #intersection
logger.info('NMF1 Accuracy %s'%(np.sum(inds_below_10)*1./len(err_per_source[:,0])*100.))
logger.info('NMF2 Accuracy %s'%(np.sum(inds_below_102)*1./len(err_per_source2[:,0])*100.))
exp_results.accuracy  = (1-count_diff*1./runs)*100. #percentage of times both NMFs give the same answer (not localization accuracy!)

# Write out the results
f = open(os.path.join(outpath, 'config_and_results.txt'), 'w')
f.write(text_format.MessageToString(exp_info))
f.close()

