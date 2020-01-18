import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat # loading data from matlab
import os
import pbdlib as pbd
import numpy.matlib
import pbdlib.plot
#from pbdlib.utils.jupyter_utils import *

def getTraj(demos, samples):

    nbData = 200  # Length of each trajectory
    dt = 0.01
    kp = 50.0
    kv = (2.0 * kp) ** 0.5
    alpha = 1.0
    x = []
    dx = []
    ddx = []
    sIn = []
    tau = []

    sIn.append(1.0)  # Initialization of decay term
    for t in xrange(1, nbData):
        sIn.append(sIn[t - 1] - alpha * sIn[t - 1] * dt)  # Update of decay term (ds/dt=-alpha s) )

    goal = demos[0][-1]

    for n in xrange(samples):
        demo = demos[n]
        size = demo.shape[0]
        x = pbd.functions.spline(np.arange(1, size + 1), demo, np.linspace(1, size, nbData))
        dx = np.divide(np.diff(x, 1), np.power(dt, 1.0))
        dx = np.vstack((np.append([0.0], dx[0]), np.append([0.0], dx[1])))
        ddx = np.divide(np.diff(x, 2), np.power(dt, 2))
        ddx = np.vstack((np.append([0.0, 0.0], ddx[0]), np.append([0.0, 0.0], ddx[1])))
        goals = np.matlib.repmat(goal, nbData, 1)
        tau_ = ddx - (kp * (goals.transpose() - x)) / sIn + (kv * dx) / sIn
        demos_ = [[s, x_, y_ ] for s, x_, y_ in zip(sIn, tau_[0].tolist(), tau_[1].tolist() )]
        tau.append(np.array(demos_))

    return tau



if __name__ == "__main__":
    np.set_printoptions(precision=3)
    nb_states = 5  # choose the number of states in HMM or clusters in GMM

    datapath = os.path.dirname(pbd.__file__) + '/data/2Dletters/'
    data_in = loadmat(datapath + '%s.mat' % "G")

    demos = [d['pos'][0][0].T for d in data_in['demos'][0]] # cleaning matlab data
    tau = getTraj(demos, nb_states)
    gmm = pbd.GMM(nb_states=nb_states)

    nsamples, nx, ny = np.array(tau).shape
    d2_train_dataset = np.array(tau).reshape((nsamples, nx * ny))
    gmm.init_hmm_kbins(tau)
    gmm.em(np.array(tau), reg=1e-8)
    # plotting demos
    # fig, ax = plt.subplots(ncols=3)
    # fig.set_size_inches(7.5, 2.8)
    # plt.tight_layout()
    # # use dim for selecting dimensions of GMM to plot
    # for p_in in demos:
    #     ax[0].plot(p_in[:, 0], p_in[:, 1])
    #
    # pbd.plot_gmm(gmm.mu, gmm.sigma, ax=ax[0], dim=[0, 1]);
    # n = 0
    #
    # resp_gmm = gmm.compute_resp(demos[0], marginal=slice(0, 2))
    #
    # fig, ax = plt.subplots(nrows=3)
    # fig.set_size_inches(7.5, 3.6)
    #
    # ax[0].plot(resp_gmm.T, lw=1);
    #
    #
    # [ax[i].set_ylim([-0.2, 1.2]) for i in range(3)]
    # plt.xlabel('timestep');
    # plt.tight_layout()
    # plt.show()