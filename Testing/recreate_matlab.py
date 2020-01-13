
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat # loading data from matlab
import os
import pbdlib as pbd
import numpy.matlib
import pbdlib.plot
#from pbdlib.utils.jupyter_utils import *



def init_kmeans(Data, nbStates):

    params_diagRegFact = 1e-4
    Mu, idList, idTmp = kmeansClustering(tau, nbStates)

    for i in xrange(nbStates):
        idtmp = np.where(idList==i)
        Priors = len(idtmp)
        mat = np.asarray([Data[:,idtmp], Data[:,idtmp]])

        sigma = np.cov(mat)
        #Optional regularization term to avoid numerical instability
        #Sigma[i] = Sigma[:,:,i] + np.eye(nbVar) * params_diagRegFact;

#    model.Priors = model.Priors / sum(model.Priors);






def kmeansClustering(Data, nbStates):


    #Criterion to stop the EM iterative update
    cumdist_threshold = 1e-10
    maxIter = 100

    #Initialization of the parameters
    nbData = Data.shape[1]
    cumdist_old = -float("inf")
    nbStep = 0

    idTmp = np.random.permutation(nbData)
    #Mu = Data[:, idTmp[:nbStates]]
    Mu = Data[:, :5]
    searching = True
    distTmp = np.zeros((nbStates, len(Data[0])))
    idList = []

    while searching:

        # E-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
        for i in xrange(0, nbStates):
            # Compute distances
            thing = np.matlib.repmat(Mu[:, i].reshape((3,1)), 1, nbData)
            temp = np.power(Data - thing, 2.0)
            temp2 = np.sum(temp, 0)
            distTmp[i,:] = temp2

        distTmpTrans = distTmp.transpose()
        vTmp = np.min(distTmpTrans, 1)
        cumdist = sum(vTmp)
        idList = []

        for row, min_num in zip(distTmpTrans, vTmp):
            index = np.where(row==min_num)[0]
            idList.append(index[0])

        idList = np.array(idList)
        # M-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i in xrange(nbStates):
            # Update the centers
            id = np.where(idList == i)
            temp = np.mean(Data[:,id], 2)
            Mu[:,i] = np.mean(Data[:,id], 2).reshape((1,3))

        # Stopping criterion %%%%%%%%%%%%%%%%%%%%
        if abs(cumdist-cumdist_old) < cumdist_threshold:
            searching = False

        cumdist_old = cumdist
        nbStep = nbStep + 1

        if nbStep>maxIter:
            print 'Maximum number of kmeans iterations, ' + str(maxIter) + ' is reached'
            searching = False
        print "maxitter ", nbStep

    return Mu, idList, idTmp


def getTraj(demos):

    nbData = 200  # Length of each trajectory
    dt = 0.01
    kp = 50.0
    kv = (2.0 * kp) ** 0.5
    alpha = 1.0
    samples = 4
    x = []
    dx = []
    ddx = []
    sIn = []
    tau = []

    sIn.append(1.0)  # Initialization of decay term
    for t in xrange(1, nbData):
        sIn.append(sIn[t - 1] - alpha * sIn[t - 1] * dt)  # Update of decay term (ds/dt=-alpha s) )
    taux = []
    tauy = []
    goal = demos[0][-1]

    for n in xrange(samples):
        demo = demos[n]
        size = demo.shape[0]
        x = pbd.functions.spline(np.arange(1, size + 1), demo, np.linspace(1, size, nbData))
        dx = np.divide(np.diff(x, 1), np.power(dt, 1.0))
        grad = np.diff(x, 1)
        dx = np.vstack((np.append([0.0], dx[0]), np.append([0.0], dx[1])))
        ddx = np.divide(np.diff(x, 2), np.power(dt, 2))
        ddx = np.vstack((np.append([0.0, 0.0], ddx[0]), np.append([0.0, 0.0], ddx[1])))
        goals = np.matlib.repmat(goal, nbData, 1)
        tau = ddx - (kp * (goals.transpose() - x)) / sIn + (kv * dx) / sIn
        taux = taux + tau[0].tolist()
        tauy = tauy + tau[1].tolist()

    tau = np.vstack((sIn * samples, taux, tauy))
    return tau


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    nb_states = 5  # choose the number of states in HMM or clusters in GMM

    datapath = os.path.dirname(pbd.__file__) + '/data/2Dletters/'
    data_in = loadmat(datapath + '%s.mat' % "G")

    demos = [d['pos'][0][0].T for d in data_in['demos'][0]] # cleaning matlab data
    tau = getTraj(demos)
    gmm = pbd.GMM(nb_states=nb_states)
    gmm.em(np.concatenate(demos), reg=1e-8)

    n = 0

    resp_gmm = gmm.compute_resp(demos[n], marginal=slice(0, 2))

    fig, ax = plt.subplots(nrows=3)
    fig.set_size_inches(7.5, 3.6)

    ax[0].plot(resp_gmm.T, lw=1);


    [ax[i].set_ylim([-0.2, 1.2]) for i in range(3)]
    plt.xlabel('timestep');
    #init_kmeans(tau,5)

#gmm = pbd.GMM(nb_states=5)

# EM to train modeldemos
#gmm.em(tau, kmeans_init=True, reg=1e-8)

#plt.show()
# demos = [np.concatenate([d_in, d_out], axis=1)
#          for d_in, d_out in zip(demos_in, demos_out)]
#
# fig, ax = plt.subplots(ncols=2)
# fig.set_size_inches(5., 2.5)
#
# [ax[i].set_title(s) for i, s in enumerate(['input', 'output'])]
#
# for p_in, p_out in zip(demos_in, demos_out):
#     ax[0].plot(p_in[:, 0], p_in[:, 1])
#     ax[1].plot(p_out[:, 0], p_out[:, 1])



