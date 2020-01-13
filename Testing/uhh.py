import os
import numpy as np
import matplotlib.pyplot as plt
import pbdlib as pbd


from scipy.io import loadmat # loading data from matlab

datapath = os.path.dirname(pbd.__file__) + '/data/gui/'
data = np.load(datapath + 'test_001.npy', allow_pickle=True)[()]

demos_x = data['x']  #Position data
demos_dx = data['dx'] # Velocity data
demos_xdx = [np.hstack([_x, _dx]) for _x ,_dx in zip(demos_x, demos_dx)] # Position-velocity


demos_x, demos_dx, demos_xdx = pbd.utils.align_trajectories(demos_x, [demos_dx, demos_xdx])

t = np.linspace(0, 100, demos_x[0].shape[0])

demos = [np.hstack([t[:,None], d]) for d in demos_xdx]
data = np.vstack([d for d in demos])

model = pbd.GMM(nb_states=4, nb_dim=5)

model.init_hmm_kbins(demos) # initializing model

# EM to train model
model.em(data, reg=[0.1, 1., 1., 1., 1.])


# plotting
fig, ax = plt.subplots(nrows=4)
fig.set_size_inches(12,7.5)
