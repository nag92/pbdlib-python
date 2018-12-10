import numpy as np
from .gmm import GMM, MVN
from functions import multi_variate_normal, multi_variate_t
from utils import gaussian_moment_matching

class MTMM(GMM):
	"""
	Multivariate t-distribution mixture
	"""

	def __init__(self, *args, **kwargs):
		GMM.__init__(self, *args, **kwargs)

		self._nu = None
		self._k = None

	def __add__(self, other):
		if isinstance(other, MVN):
			gmm = MTMM(nb_dim=self.nb_dim, nb_states=self.nb_states)

			gmm.nu = self.nu
			gmm.k = self.k
			gmm.priors = self.priors
			gmm.mu = self.mu + other.mu[None]
			gmm.sigma = self.sigma + other.sigma[None]

			return gmm

		else:
			raise NotImplementedError

	@property
	def k(self):
		return self._k

	@k.setter
	def k(self, value):
		self._k = value

	@property
	def nu(self):
		return self._nu

	@nu.setter
	def nu(self, value):
		self._nu = value

	def condition_gmm(self, data_in, dim_in, dim_out):
		sample_size = data_in.shape[0]

		# compute responsabilities
		mu_in, sigma_in = self.get_marginal(dim_in)


		h = np.zeros((self.nb_states, sample_size))
		for i in range(self.nb_states):
			h[i, :] = multi_variate_t(data_in[None], self.nu[i],
									  mu_in[i],
									  sigma_in[i])

		h += np.log(self.priors)[:, None]
		h = np.exp(h).T
		h /= np.sum(h, axis=1, keepdims=True)
		h = h.T

		mu_out, sigma_out = self.get_marginal(dim_out)
		mu_est, sigma_est = ([], [])

		inv_sigma_in_in, inv_sigma_out_in = ([], [])

		_, sigma_in_out = self.get_marginal(dim_in, dim_out)

		for i in range(self.nb_states):
			inv_sigma_in_in += [np.linalg.inv(sigma_in[i])]
			inv_sigma_out_in += [sigma_in_out[i].T.dot(inv_sigma_in_in[-1])]
			dx = data_in[None] - mu_in[i]
			mu_est += [mu_out[i] + np.einsum('ij,aj->ai',
											 inv_sigma_out_in[-1], dx)]

			s = np.sum(np.einsum('ai,ij->aj', dx, inv_sigma_in_in[-1]) * dx, axis=1)
			a = (self.nu[i] + s) / (self.nu[i] + mu_in.shape[1])

			sigma_est += [a[:, None, None] *
						  (sigma_out[i] - inv_sigma_out_in[-1].dot(sigma_in_out[i]))[None]]

		mu_est, sigma_est = (np.asarray(mu_est)[:, 0], np.asarray(sigma_est)[:, 0])


		gmm_out = MTMM(nb_states=self.nb_states, nb_dim=mu_out.shape[1])
		gmm_out.nu = self.nu + gmm_out.nb_dim
		gmm_out.mu = mu_est
		gmm_out.sigma = sigma_est
		gmm_out.priors = h[:, 0]

		return gmm_out


	def condition(self, data_in, dim_in, dim_out, h=None, return_gmm=False):
		"""
		[1] M. Hofert, 'On the Multivariate t Distribution,' R J., vol. 5, pp. 129-136, 2013.

		Conditional probabilities in a Joint Multivariate t Distribution Mixture Model

		:param data_in:		[np.array([nb_data, nb_dim])
				Observed datapoints x_in
		:param dim_in:		[slice] or [list of index]
				Dimension of input space e.g.: slice(0, 3), [0, 2, 3]
		:param dim_out:		[slice] or [list of index]
				Dimension of output space e.g.: slice(3, 6), [1, 4]
		:param h:			optional - [np.array([nb_states, nb_data])]
				Overrides marginal probability of states given input dimensions
		:return:
		"""

		sample_size = data_in.shape[0]

		# compute marginal probabilities of states given observation p(k|x_in)
		mu_in, sigma_in = self.get_marginal(dim_in)

		if h is None:
			h = np.zeros((self.nb_states, sample_size))
			for i in range(self.nb_states):
				h[i, :] = multi_variate_t(data_in, self.nu[i],
											   mu_in[i],
											   sigma_in[i])

			h += np.log(self.priors)[:, None]
			h = np.exp(h).T
			h /= np.sum(h, axis=1, keepdims=True)
			h = h.T

		self._h = h # storing value

		mu_out, sigma_out = self.get_marginal(dim_out)  # get marginal distribution of x_out
		mu_est, sigma_est = ([], [])

		# get conditional distribution of x_out given x_in for each states p(x_out|x_in, k)
		inv_sigma_in_in, inv_sigma_out_in = ([], [])

		_, sigma_in_out = self.get_marginal(dim_in, dim_out)

		for i in range(self.nb_states):
			inv_sigma_in_in += [np.linalg.inv(sigma_in[i])]
			inv_sigma_out_in += [sigma_in_out[i].T.dot(inv_sigma_in_in[-1])]
			dx = data_in - mu_in[i]
			mu_est += [mu_out[i] + np.einsum('ij,aj->ai',
											 inv_sigma_out_in[-1], dx)]

			s = np.sum(np.einsum('ai,ij->aj', dx, inv_sigma_in_in[-1]) * dx, axis=1)
			a = (self.nu[i] + s)/(self.nu[i] + mu_in.shape[1])

			sigma_est += [a[:, None, None] *
						  (sigma_out[i] - inv_sigma_out_in[-1].dot(sigma_in_out[i]))[None]]

		mu_est, sigma_est = (np.asarray(mu_est), np.asarray(sigma_est))

		# the conditional distribution is now a still a mixture

		if return_gmm:
			return mu_est, sigma_est
		else:
			# apply moment matching to get a single MVN for each datapoint
			return gaussian_moment_matching(mu_est, sigma_est, h.T)

class VBayesianGMM(MTMM):
	def __init__(self, sk_parameters, *args, **kwargs):
		MTMM.__init__(self, *args, **kwargs)

		from sklearn import mixture

		self._training_data = None
		self._posterior_predictive = None


		self._sk_model = mixture.BayesianGaussianMixture(**sk_parameters)
		self._posterior_samples = None

	@property
	def posterior_samples(self):
		return self._posterior_samples

	def make_posterior_samples(self, nb_samples=10):
		from scipy.stats import wishart
		from .gmm import GMM
		self._posterior_samples = []

		for i in range(nb_samples):
			_gmm = GMM()
			_gmm.mu = np.array(
				[np.random.multivariate_normal(
					self.mu[i], self.sigma[i]/(self.k[i] * (self.nu[i] - self.nb_dim + 1)))
				for i in range(self.nb_states)])

			_gmm.sigma = np.array(
				[wishart.rvs(self.nu[i], self.sigma[i]/self.nu[i])
				 for i in range(self.nb_states)])
			_gmm.priors = self.priors
			self._posterior_samples += [_gmm]

	def posterior(self, data, dims=slice(0, 7)):

		self.nb_dim = data.shape[1]

		self._sk_model.fit(data)

		states = np.where(self._sk_model.weights_ > 5e-2)[0]

		self.nb_states = states.shape[0]

		# see [1] K. P. Murphy, 'Conjugate Bayesian analysis of the Gaussian distribution,' vol. 0, no. 7, 2007. par 9.4
		# or [1] E. Fox, 'Bayesian nonparametric learning of complex dynamical phenomena,' 2009, p 55
		self.priors = np.copy(self._sk_model.weights_[states])
		self.mu = np.copy(self._sk_model.means_[states])
		self.k = np.copy(self._sk_model.mean_precision_[states])
		self.nu = np.copy(self._sk_model.degrees_of_freedom_[states])# - self.nb_dim + 1
		self.sigma = np.copy(self._sk_model.covariances_[states]) * (
		self.k[:, None, None] + 1) * self.nu[:, None, None] \
					 / (self.k[:, None, None] * (self.nu[:, None, None] - self.nb_dim + 1))

	def condition(self, *args, **kwargs):
		if not kwargs.get('samples', False):
			return MTMM.condition(self, *args, **kwargs)
		kwargs.pop('samples')

		mus, sigmas = [], []

		for _gmm in self.posterior_samples:
			mu, sigma = _gmm.condition(*args, **kwargs)
			mus += [mu]; sigmas += [sigma]

		mus, sigmas = np.array(mus), np.array(sigmas)
		# moment matching
		mu = np.mean(mus, axis=0)
		dmu = mu[None] - mus
		sigma = np.mean(sigmas, axis=0) + \
				np.einsum('aki,akj->kij', dmu, dmu) / len(self.posterior_samples)

		return mu, sigma
