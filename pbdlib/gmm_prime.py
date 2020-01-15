import numpy as np
from .model import *
from .functions import multi_variate_normal
from scipy.linalg import block_diag

from termcolor import colored
from .mvn import MVN
import gmm

class GMM_Prime(gmm.GMM):

    def __init__(self, nb_states=1, nb_dim=3, init_zeros=False, mu=None, lmbda=None, sigma=None, priors=None):

        # if mu is not None:
        #     nb_states = mu.shape[0]
        #     nb_dim = mu.shape[-1]
        nb_s = nb_states
        nb_d = nb_dim

        gmm.GMM.__init__(self, nb_states=nb_states, nb_dim= nb_dim)
        # flag to indicate that publishing was not init
        self.publish_init = False
        self._mu = mu
        self._lmbda = lmbda
        self._sigma = sigma
        self._priors = priors

        if init_zeros:
            self.init_zeros()

    def kmeansclustering(self, data, reg=1e-8):

        self.reg = reg

        # Criterion to stop the EM iterative update
        cumdist_threshold = 1e-10
        maxIter = 100

        # Initialization of the parameters
        cumdist_old = -float("inf")
        nbStep = 0
        self.nbData = data.shape[1]
        idTmp = np.random.permutation(self.nbData)

        # Mu = Data[:, idTmp[:nbStates]]
        Mu = data[:, idTmp[:self.nb_states]]
        searching = True
        distTmp = np.zeros((self.nb_states, len(data[0])))
        idList = []

        while searching:

            # E-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
            for i in xrange(0, self.nb_states):
                # Compute distances
                thing = np.matlib.repmat(Mu[:, i].reshape((3, 1)), 1, self.nbData)
                temp = np.power(data - thing, 2.0)
                temp2 = np.sum(temp, 0)
                distTmp[i, :] = temp2

            distTmpTrans = distTmp.transpose()
            vTmp = np.min(distTmpTrans, 1)
            cumdist = sum(vTmp)
            idList = []

            for row, min_num in zip(distTmpTrans, vTmp):
                index = np.where(row == min_num)[0]
                idList.append(index[0])

            idList = np.array(idList)
            # M-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for i in xrange(self.nb_states):
                # Update the centers
                id = np.where(idList == i)
                temp = np.mean(data[:, id], 2)
                Mu[:, i] = np.mean(data[:, id], 2).reshape((1, 3))

            # Stopping criterion %%%%%%%%%%%%%%%%%%%%
            if abs(cumdist - cumdist_old) < cumdist_threshold:
                searching = False

            cumdist_old = cumdist
            nbStep = nbStep + 1

            if nbStep > maxIter:
                print 'Maximum number of kmeans iterations, ' + str(maxIter) + ' is reached'
                searching = False
            print "maxitter ", nbStep

        self.mu = Mu

        return idList


    def init_params_kmeans(self, data):

        idList = self.kmeansclustering(data)
        self.priors = np.ones(self.nb_states) / self.nb_states
        self.sigma = np.array([np.eye(self.nb_dim) for i in range(self.nb_states)])
        self.Trans = np.ones((self.nb_states, self.nb_states)) * 0.01

        for i in xrange(self.nb_states):

            idtmp = np.where(idList==i)
            mat = np.vstack((data[:, idtmp][0][0], data[:, idtmp][1][0], data[:, idtmp][2][0]))
            self.priors[i] = len(idtmp)
            self.sigma[i] = np.cov(mat).transpose() + np.eye(self.nb_dim)*self.reg

        self.priors = self.priors / np.sum(self.priors)

    def em(self, data, reg=1e-8, maxiter=100, minstepsize=1e-5, diag=False, reg_finish=False,
           kmeans_init=False, random_init=False, dep_mask=None, verbose=False, only_scikit=False,
           no_init=True):
        """

        :param data:	 		[np.array([nb_timesteps, nb_dim])]
        :param reg:				[list([nb_dim]) or float]
            Regulariazation for EM
        :param maxiter:
        :param minstepsize:
        :param diag:			[bool]
            Use diagonal covariance matrices
        :param reg_finish:		[np.array([nb_dim]) or float]
            Regulariazation for finish step
        :param kmeans_init:		[bool]
            Init components with k-means.
        :param random_init:		[bool]
            Init components randomely.
        :param dep_mask: 		[np.array([nb_dim, nb_dim])]
            Composed of 0 and 1. Mask given the dependencies in the covariance matrices
        :return:
        """

        self.reg = reg

        nb_min_steps = 5  # min num iterations
        nb_max_steps = maxiter  # max iterations
        max_diff_ll = minstepsize  # max log-likelihood increase

        nb_samples = data.shape[1]

        if not no_init:
            if random_init and not only_scikit:
                self.init_params_random(data)
            elif kmeans_init and not only_scikit:
                self.init_params_kmeans(data)
            else:
                if diag:
                    self.init_params_scikit(data, 'diag')
                else:
                    self.init_params_scikit(data, 'full')

        if only_scikit: return
        data = data.T

        LL = np.zeros(nb_max_steps)
        for it in range(nb_max_steps):

            # E - step
            L = np.zeros((self.nb_states, nb_samples))

            for i in range(self.nb_states):
                L [i, :] = self.priors[i] * multi_variate_normal_old(data.T, self.mu[:, i], self.sigma[i])

            GAMMA = L / np.sum(L, axis=0)
            GAMMA2 = GAMMA / np.sum(GAMMA, axis=1)[:, np.newaxis]

            # M-step


            for i in xrange(self.nb_states):
                # update priors
                self.priors[i] = np.sum(GAMMA[i,:]) / self.nbData
                self.mu[:, i] = data.T.dot(GAMMA2[i,:].reshape((-1,1))).T
                mu = np.matlib.repmat(self.mu[:, i].reshape((-1, 1)), 1, self.nbData)
                diff = (data.T - mu)
                self.sigma[i] = data.T.dot(np.diag(GAMMA2[i,:])).dot(data) + np.eye(self.nb_dim) * self.reg;


            # self.mu = np.einsum('ac,ic->ai', GAMMA2, data)  # a states, c sample, i dim
            #
            # dx = data[None, :] - self.mu[:, :, None]  # nb_dim, nb_states, nb_samples
            #
            # self.sigma = np.einsum('acj,aic->aij', np.einsum('aic,ac->aci', dx, GAMMA2),
            #                        dx)  # a states, c sample, i-j dim

            # #self.sigma += self.reg
            #
            # if diag:
            # 	self.sigma *= np.eye(self.nb_dim)
            #
            # if dep_mask is not None:
            # 	self.sigma *= dep_mask

            # print self.Sigma[:,u :, i]

            # Update initial state probablility vector
            self.priors = np.mean(GAMMA, axis=1)

            LL[it] = np.mean(np.log(np.sum(L, axis=0)))
            # Check for convergence
            if it > nb_min_steps:
                if LL[it] - LL[it - 1] < max_diff_ll:
                    if reg_finish is not False:
                        self.sigma = np.einsum(
                            'acj,aic->aij', np.einsum('aic,ac->aci', dx, GAMMA2), dx) + reg_finish

                    if verbose:
                        print(colored('Converged after %d iterations: %.3e' % (it, LL[it]), 'red', 'on_white'))
                    return GAMMA
        if verbose:
            print(
                "GMM did not converge before reaching max iteration. Consider augmenting the number of max iterations.")

        return GAMMA


    def init_hmm_kbins(self, demos, dep=None, reg=1e-8, dep_mask=None):
        """
		Init HMM by splitting each demos in K bins along time. Each K states of the HMM will
		be initialized with one of the bin. It corresponds to a left-to-right HMM.

		:param demos:	[list of np.array([nb_timestep, nb_dim])]
		:param dep:
		:param reg:		[float]
		:return:
		"""

        # delimit the cluster bins for first demonstration
        self.nb_dim = demos[0].shape[1]

        self.init_zeros()

        t_sep = []

        for demo in demos:
            t_sep += [map(int, np.round(np.linspace(0, demo.shape[0], self.nb_states + 1)))]

        # print t_sep
        for i in range(self.nb_states):
            data_tmp = np.empty((0, self.nb_dim))
            inds = []
            states_nb_data = 0  # number of datapoints assigned to state i

            # Get bins indices for each demonstration
            for n, demo in enumerate(demos):
                inds = range(t_sep[n][i], t_sep[n][i + 1])

                data_tmp = np.concatenate([data_tmp, demo[inds]], axis=0)
                states_nb_data += t_sep[n][i + 1] - t_sep[n][i]

            self.priors[i] = states_nb_data
            self.mu[i] = np.mean(data_tmp, axis=0)

            if dep_mask is not None:
                self.sigma *= dep_mask

            if dep is None:
                self.sigma[i] = np.cov(data_tmp.T) + np.eye(self.nb_dim) * reg
            else:
                for d in dep:
                    dGrid = np.ix_([i], d, d)
                    self.sigma[dGrid] = (np.cov(data_tmp[:, d].T) + np.eye(
                        len(d)) * reg)[:, :, np.newaxis]
            # print self.Sigma[:,:,i]

        # normalize priors
        self.priors = self.priors / np.sum(self.priors)

        # Hmm specific init
        self.Trans = np.ones((self.nb_states, self.nb_states)) * 0.01

        nb_data = np.mean([d.shape[0] for d in demos])

        for i in range(self.nb_states - 1):
            self.Trans[i, i] = 1.0 - float(self.nb_states) / nb_data
            self.Trans[i, i + 1] = float(self.nb_states) / nb_data

        self.Trans[-1, -1] = 1.0
        self.init_priors = np.ones(self.nb_states) * 1. / self.nb_states

    def add_trash_component(self, data, scale=2.):
        if isinstance(data, list):
            data = np.concatenate(data, axis=0)

        mu_new = np.mean(data, axis=0)
        sigma_new = scale ** 2 * np.cov(data.T)

        self.priors = np.concatenate([self.priors, 0.01 * np.ones(1)])
        self.priors /= np.sum(self.priors)
        self.mu = np.concatenate([self.mu, mu_new[None]], axis=0)
        self.sigma = np.concatenate([self.sigma, sigma_new[None]], axis=0)

    def mvn_pdf(self, x, reg=None):
        """

		:param x: 			np.array([nb_samples, nb_dim])
			samples
		:param mu: 			np.array([nb_states, nb_dim])
			mean vector
		:param sigma_chol: 	np.array([nb_states, nb_dim, nb_dim])
			cholesky decomposition of covariance matrices
		:param lmbda: 		np.array([nb_states, nb_dim, nb_dim])
			precision matrices
		:return: 			np.array([nb_states, nb_samples])
			log mvn
		"""
        # if len(x.shape) > 1:  # TODO implement mvn for multiple xs
        # 	raise NotImplementedError
        mu, lmbda_, sigma_chol_ = self.mu, self.lmbda, self.sigma_chol

        if x.ndim > 1:
            dx = mu[None] - x[:, None]  # nb_timesteps, nb_states, nb_dim
        else:
            dx = mu - x

        eins_idx = ('baj,baj->ba', 'ajk,baj->bak') if x.ndim > 1 else (
            'aj,aj->a', 'ajk,aj->ak')

        return -0.5 * np.einsum(eins_idx[0], dx, np.einsum(eins_idx[1], lmbda_, dx)) \
               - mu.shape[1] / 2. * np.log(2 * np.pi) - np.sum(
            np.log(sigma_chol_.diagonal(axis1=1, axis2=2)), axis=1)

    def mu(self, value):
        self.nb_dim = value.shape[0]
        self.nb_states = value.shape[1]
        self._mu = value
