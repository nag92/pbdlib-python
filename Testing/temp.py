import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat # loading data from matlab
import os
import pbdlib as pbd
import pbdlib.plot
from pbdlib.utils.jupyter_utils import *
np.set_printoptions(precision=2)
nb_states = 5  # choose the number of states in HMM or clusters in GMM

letter_in = 'X' # INPUT LETTER: choose a letter in the alphabet
letter_out = 'C' # OUTPUT LETTER: choose a letter in the alphabet

datapath = os.path.dirname(pbd.__file__) + '/data/2Dletters/'

data_in = loadmat(datapath + '%s.mat' % letter_in)
data_out = loadmat(datapath + '%s.mat' % letter_out)

demos_in = [d['pos'][0][0].T for d in data_in['demos'][0]] # cleaning matlab data
demos_out = [d['pos'][0][0].T for d in data_out['demos'][0]] # cleaning matlab data

demos = [np.concatenate([d_in, d_out], axis=1)
         for d_in, d_out in zip(demos_in, demos_out)]

gmm = pbd.GMM(nb_states=nb_states)

# initializing model by splitting the demonstrations in k bins
[model.init_hmm_kbins(demos) for model in [gmm]]

# EM to train model
gmm.em(np.concatenate(demos), reg=1e-8)
