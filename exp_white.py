# White noise localization
# Usage: python exp_white.py config.txt

import logging
from proto import info_pb2
from google.protobuf import text_format
import datetime

from core.processing import calculate_angle_error
from core.signal import stft

from itertools import combinations
import numpy as np
import os
import sys

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("whitegrid")
sns.set(font_scale=1.5) 
cmap = "PuBu"

if len(sys.argv) != 2:
  print "Usage:",  sys.argv[0],  "EXPERIMENT_CONFIG_FILE"
  sys.exit(-1)

# Read the config file.
f = open(sys.argv[1],  "rb")
exp_info = info_pb2.DataInfo()
text_format.Merge(f.read(), exp_info)
f.close()

# Create an output directory
outpath = os.path.join(exp_info.output_folder,  '{:%Y%m%d_%H%M}'.format(datetime.datetime.now()))
if not os.path.exists(outpath):
    os.makedirs(outpath)

# Read experiment parameters
step = np.int32(360./exp_info.D) #discretization 
Fn = np.int32(exp_info.nfft/2. +1) #number of frequencies in spectrogram
n_samples = int(exp_info.duration*exp_info.fs) #length of signal in samples
J = exp_info.J #number of sources
SNR = exp_info.SNR # desired signal-to-noise ratio
runs = exp_info.runs #number of times to run the experiment

# Read device parameters
device = exp_info.device
dev = device.name
m = device.mic #index of used microphone
f0 = np.int32(device.f0*exp_info.nfft*1./exp_info.fs) #starting frequency index

# Logging
logfile = os.path.join(outpath, "log.log")

logging.basicConfig(filename=logfile, format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Load transfer functions %s'%dev)
H_theta0 = np.load(device.freq_response) #load transfer functions (includes 4 microphones)
Df = H_theta0.shape[1] #number of directions for a fine discretization

H_theta = H_theta0[f0:, ::step, m] #coarse discretization,  model
[F, D] = H_theta.shape #F: number of frequencies,  D: number of directions

# Load corresponding impulse responses
H_theta0_t = np.load(device.impulse_response)[:, :, m]
H_theta_t = H_theta0_t[:, ::step] #coarse discretization

anglesf = np.arange(Df, dtype=np.int64)*360./Df # list of angles in degrees

obs_len = n_samples + H_theta_t.shape[0] - 1 #length of the convolution

np.random.seed(seed=0)

conf_matrix = np.zeros((360, 360)) #confusion  matrix
err_per_source = np.zeros((runs, J))

logger.info('Number of sources %s'%(J))

for rns in range(runs):
    St_all = np.zeros((n_samples, J)) #list of source signals
    for j in range(J):
        St_all[:, j] = np.random.randn(n_samples) #source in time: random gaussian signal 
                   
    theta = np.random.choice(range(Df), J, replace=False) #choose the directions randomly on the fine grid   
                
    yt = np.zeros((obs_len, )) #recorded time domain signal
          
    for j in range(J):
        yt += np.convolve(St_all[:, j], H_theta0_t[:, theta[j]]) #source signal convolved with corresponding directional response
                 
    # Generate noise at required SNR    
    sig_norm = np.linalg.norm(yt)
    noise_t = np.random.randn(obs_len, ) #additive gaussian noise
    noise_norm = sig_norm/(10.**(SNR/20.))
    noise_t = noise_norm*noise_t/np.linalg.norm(noise_t)

    yt += noise_t #noisy signal

    y = stft(yt, Fn)[f0:, :] #spectrogram of recorded signal
    N = y.shape[1] #number of frames
    
    y_mean = np.mean(np.abs(y)**2, axis=1) #mean power frame
    y_mean = y_mean/np.linalg.norm(y_mean) #normalize the observation

    # Exhaustive search algorithm

    # Initialize variables
    best_ind = np.inf #index corresponding to best direction tuple
    smallest_norm = np.inf #smallest projection error
    best_dir = theta #best direction tuple

    # Search all combinations
    pairs2 = combinations(range(D), J)
    for q2, d2 in enumerate(pairs2): 
        Bj = np.abs(H_theta[:, d2])**2 #vectors in current guess
        Pj = Bj.dot(np.linalg.pinv(Bj)) #projection matrix
        proj_error = np.linalg.norm((np.eye(F) - Pj).dot(y_mean)) #projection error

        if proj_error <= smallest_norm:
            smallest_norm = proj_error
            best_ind = q2
            best_dir = d2
    theta_hat = step*np.array(best_dir) #map coarse index to fine index
    min_err, best_perm = calculate_angle_error(theta, theta_hat, anglesf) #calculate error between chosen and true directions
    conf_matrix[theta, best_perm] += 1

    for src_j in range(J): #error per source
        err_per_source[rns, src_j] = np.sum(np.absolute(((best_perm[src_j]-theta[src_j]+180) % 360)-180));
   
    logger.info('Test %s, theta: %s, theta_hat: %s, err: %s'%(rns, anglesf[theta], anglesf[best_perm], min_err))

exp_results = exp_info.summary
inds_below_10 = np.all(err_per_source <= exp_info.accuracy_bin, axis=1) #intersection
exp_results.accuracy  = np.sum(inds_below_10)*1./len(err_per_source[:, 0])*100.
exp_results.robust_mean = np.mean(err_per_source[inds_below_10, :])

per_src = np.zeros((J, ))
for src_j in range(J):
    inds_below_10 = np.any(err_per_source[:, src_j, np.newaxis] <= exp_info.accuracy_bin,  axis=1)
    per_src[src_j] = np.sum(inds_below_10)*1./len(err_per_source[:, src_j])*100.

exp_results.accuracy_per_source = np.mean(per_src)

# Write out the results
f = open(os.path.join(outpath, 'config_and_results.txt'),  'w')
f.write(text_format.MessageToString(exp_info))
f.close()

# Plot confusion matrix.
hm = sns.heatmap(20*np.log10(conf_matrix+1e-80), cmap=cmap, xticklabels=False,  yticklabels=False)
plt.xlabel('Estimate')
plt.ylabel('True')
plt.savefig(os.path.join(outpath, 'conf_matrix.png'))
