
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat # loading data from matlab
import os
import pbdlib as pbd
import numpy.matlib
import pbdlib.plot
#from pbdlib.utils.jupyter_utils import *



def kmeansClustering(Data, nbStates):


    #Criterion to stop the EM iterative update
    cumdist_threshold = 1e-10
    maxIter = 100

    #Initialization of the parameters
    nbData = Data.shape[1]
    cumdist_old = -float("inf")
    nbStep = 0

    idTmp = np.random.permutation(nbData)
    Mu = Data[:, idTmp[:nbStates]]
    temp = Data - np.power(np.matlib.repmat(Mu[:, 1].reshape((3, 1)), 1, nbData), 2)
    print "sakdfjlksdfjslkd ", temp.sum(1).reshape((3,1))
    searching = True
    distTmp = np.zeros((5, 800 ))
    while searching:
        # E-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
        for i in xrange(0, nbStates):
            # Compute distances
            thing = np.matlib.repmat(Mu[:, i].reshape((3,1)), 1, nbData)
            temp = np.power(Data - thing, 2)
            temp = np.sum(temp, 0)
            #print sum((Data - np.power(np.matlib.repmat(Mu[:, i], 1, nbData)), 2))
            distTmp[i,:] = temp
        distTmp = distTmp.transpose()
        vTmp = np.min(distTmp, 1)
        cumdist = sum(vTmp)
        idList = []

        for row, min_num in zip(distTmp, vTmp):
            index = np.where(row==min_num)[0]
            idList.append(index[0])

        idList = np.array(idList)
        # M-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i in xrange(0,nbStates):
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

np.set_printoptions(precision=2)
nb_states = 5  # choose the number of states in HMM or clusters in GMM
nbData = 200 #Length of each trajectory
dt = 0.01
kp = 50
kv = (2.0*kp)**0.5
alpha = 1.0
samples = 4
datapath = os.path.dirname(pbd.__file__) + '/data/2Dletters/'
data_in = loadmat(datapath + '%s.mat' % "G")

demos = [d['pos'][0][0].T for d in data_in['demos'][0]] # cleaning matlab data

x = []
dx = []
ddx = []
sIn = []
tau = []

sIn.append(1.0) # Initialization of decay term
for t in xrange(1,nbData):
    sIn.append( sIn[t-1] - alpha * sIn[t-1] * dt) # Update of decay term (ds/dt=-alpha s) )
taux = []
tauy = []
goal = demos[0][samples]

for n in xrange(samples):
    demo = demos[n]
    size = demo.shape[0]
    x = pbd.functions.spline(np.arange(1, size+1), demo, np.linspace(1, size, nbData))
    dx = np.divide(np.diff(x, 1), np.power(dt, 1))
    dx = np.vstack((np.append([0.0], dx[0]), np.append( [0.0], dx[1])))
    ddx = np.divide(np.diff(x, 2), np.power(dt, 2))
    ddx = np.vstack((np.append([0.0, 0.0], ddx[0]), np.append( [0.0, 0.0], ddx[1])))
    goals = np.matlib.repmat(goal, nbData, 1)
    tau = ddx - kp*(goals.transpose() - x) + kv*dx / np.vstack((sIn, sIn))
    taux = taux + tau[0].tolist()
    tauy = tauy + tau[1].tolist()




tau = np.vstack((sIn*samples, taux, tauy))
print len
kmeansClustering(tau, 5)
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



