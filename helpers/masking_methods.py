# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

"""
    Following masking methods of https://github.com/Js-Mim/ASP.
"""

# imports
import numpy as np
from scipy.fftpack import fft


class FrequencyMasking:
	"""Class containing various time-frequency masking methods,
	   for processing Time-Frequency representations.
	"""

	def __init__(self, mX, sTarget, nResidual, psTarget=[], pnResidual=[], alpha=1.2, method='Wiener'):
		self._mX = mX
		self._eps = np.finfo(np.float).eps
		self._sTarget = sTarget
		self._nResidual = nResidual
		self._pTarget = psTarget
		self._pY = pnResidual
		self._mask = []
		self._Out = []
		self._alpha = alpha
		self._method = method
		self._iterations = 200
		self._lr = 1.5e-3
		self._hetaplus = 1.1
		self._hetaminus = 0.1
		self._amountiter = 0

	def __call__(self, reverse=False):
		if self._method == 'Phase':
			if not self._pTarget.size or not self._pTarget.size:
				raise ValueError('Phase-sensitive masking cannot be performed without phase information.')
			else:
				FrequencyMasking.phaseSensitive(self)
				if not reverse:
					FrequencyMasking.applyMask(self)
				else:
					FrequencyMasking.applyReverseMask(self)

		elif self._method == 'IRM':
			FrequencyMasking.IRM(self)
			if not reverse:
				FrequencyMasking.applyMask(self)
			else:
				FrequencyMasking.applyReverseMask(self)

		elif self._method == 'IAM':
			FrequencyMasking.IAM(self)
			if not reverse:
				FrequencyMasking.applyMask(self)
			else:
				FrequencyMasking.applyReverseMask(self)

		elif self._method == 'IBM':
			FrequencyMasking.IBM(self)
			if not reverse:
				FrequencyMasking.applyMask(self)
			else:
				FrequencyMasking.applyReverseMask(self)

		elif self._method == 'UBBM':
			FrequencyMasking.UBBM(self)
			if not reverse:
				FrequencyMasking.applyMask(self)
			else:
				FrequencyMasking.applyReverseMask(self)

		elif self._method == 'Wiener':
			FrequencyMasking.Wiener(self)
			if not reverse:
				FrequencyMasking.applyMask(self)
			else:
				FrequencyMasking.applyReverseMask(self)

		elif self._method == 'alphaWiener':
			FrequencyMasking.alphaHarmonizableProcess(self)
			if not reverse:
				FrequencyMasking.applyMask(self)
			else:
				FrequencyMasking.applyReverseMask(self)

		elif self._method == 'expMask':
			FrequencyMasking.ExpM(self)
			if not reverse:
				FrequencyMasking.applyMask(self)
			else:
				FrequencyMasking.applyReverseMask(self)

		elif self._method == 'MWF':
			print('Multichannel Wiener Filtering')
			FrequencyMasking.MWF(self)

		return self._Out

	def IRM(self):
		"""
			Computation of Ideal Amplitude Ratio Mask. As appears in :
			H Erdogan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux,
	   		"Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks,"
	   		in ICASSP 2015, Brisbane, April, 2015.
		Args:
			sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
			nResidual: (2D ndarray) Magnitude Spectrogram of the residual component
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values

		"""
		print('Ideal Amplitude Ratio Mask')
		self._mask = np.divide(self._sTarget, (self._sTarget + self._nResidual + self._eps))

	def IAM(self):
		"""
			Computation of Ideal Amplitude Mask. As appears in :
			H. Erdogan, J. R. Hershey, S. Watanabe, and J. Le Roux,
	   		"Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks,"
	   		in ICASSP 2015, Brisbane, April, 2015.
		Args:
			sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
			nResidual: (2D ndarray) Magnitude Spectrogram of the residual component
									(In this case the observed mixture should be placed)
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values

		"""
		print('Ideal Amplitude Mask')
		self._mask = np.divide(self._sTarget, (self._nResidual + self._eps))

	def ExpM(self):
		"""
			Approximate a signal via element-wise exponentiation. As appears in :
			S.I. Mimilakis, K. Drossos, T. Virtanen, and G. Schuller,
			"Deep Neural Networks for Dynamic Range Compression in Mastering Applications,"
			in proc. of the 140th Audio Engineering Society Convention, Paris, 2016.
		Args:
			sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
			nResidual: (2D ndarray) Magnitude Spectrogram of the residual component
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values

		"""
		print('Exponential mask')
		self._mask = np.divide(np.log(self._sTarget.clip(self._eps, np.inf) ** self._alpha), \
							   np.log(self._nResidual.clip(self._eps, np.inf) ** self._alpha))

	def IBM(self):
		"""
			Computation of Ideal Binary Mask.
		Args:
			sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
			nResidual: (2D ndarray) Magnitude Spectrogram of the residual component
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values

		"""
		print('Ideal Binary Mask')
		theta = 0.5
		mask = np.divide(self._sTarget ** self._alpha, (self._eps + self._nResidual ** self._alpha))
		bg = np.where(mask >= theta)
		sm = np.where(mask < theta)
		mask[bg[0], bg[1]] = 1.
		mask[sm[0], sm[1]] = 0.
		self._mask = mask

	def UBBM(self):
		"""
			Computation of Upper Bound Binary Mask. As appears in :
			- J.J. Burred, "From Sparse Models to Timbre Learning: New Methods for Musical Source Separation", PhD Thesis,
			TU Berlin, 2009.

		Args:
			sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
			nResidual: (2D ndarray) Magnitude Spectrogram of the residual component (Should not contain target source!)
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values
		"""
		print('Upper Bound Binary Mask')
		mask = 20. * np.log(self._eps + np.divide((self._eps + (self._sTarget ** self._alpha)),
												  ((self._eps + (self._nResidual ** self._alpha)))))
		bg = np.where(mask >= 0)
		sm = np.where(mask < 0)
		mask[bg[0], bg[1]] = 1.
		mask[sm[0], sm[1]] = 0.
		self._mask = mask

	def Wiener(self):
		"""
			Computation of Wiener-like Mask. As appears in :
			H Erdogan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux,
	   		"Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks,"
	   		in ICASSP 2015, Brisbane, April, 2015.
		Args:
				sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
				nResidual: (2D ndarray) Magnitude Spectrogram of the residual component
		Returns:
				mask:      (2D ndarray) Array that contains time frequency gain values
		"""
		print('Wiener-like Mask')
		localsTarget = self._sTarget ** 2.
		numElements = len(self._nResidual)
		if numElements > 1:
			localnResidual = self._nResidual[0] ** 2. + localsTarget
			for indx in range(1, numElements):
				localnResidual += self._nResidual[indx] ** 2.
		else:
			localnResidual = self._nResidual[0] ** 2. + localsTarget

		self._mask = np.divide((localsTarget + self._eps), (self._eps + localnResidual))

	def alphaHarmonizableProcess(self):
		"""
			Computation of Wiener like mask using fractional power spectrograms. As appears in :
			A. Liutkus, R. Badeau, "Generalized Wiener filtering with fractional power spectrograms",
    		40th International Conference on Acoustics, Speech and Signal Processing (ICASSP),
    		Apr 2015, Brisbane, Australia.
		Args:
			sTarget:   (2D ndarray) Magnitude Spectrogram of the target component
		    nResidual: (2D ndarray) Magnitude Spectrogram of the residual component or a list
									of 2D ndarrays which will be added together
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values

		"""
		print('Harmonizable Process with alpha:', str(self._alpha))
		localsTarget = self._sTarget ** self._alpha
		numElements = len(self._nResidual)
		if numElements > 1:
			localnResidual = self._nResidual[0] ** self._alpha + localsTarget
			for indx in range(1, numElements):
				localnResidual += self._nResidual[indx] ** self._alpha
		else:
			localnResidual = self._nResidual[0] ** self._alpha + localsTarget

		self._mask = np.divide((localsTarget + self._eps), (self._eps + localnResidual))

	def phaseSensitive(self):
		"""
			Computation of Phase Sensitive Mask. As appears in :
			H Erdogan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux,
	   		"Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks,"
	   		in ICASSP 2015, Brisbane, April, 2015.

		Args:
			mTarget:   (2D ndarray) Magnitude Spectrogram of the target component
			pTarget:   (2D ndarray) Phase Spectrogram of the output component
			mY:        (2D ndarray) Magnitude Spectrogram of the residual component
			pY:        (2D ndarray) Phase Spectrogram of the residual component
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values

		"""
		print('Truncated Phase Sensitive Masking.')
		# Compute Phase Difference
		Theta = (self._pTarget - self._pY)
		Theta = np.clip(np.cos(Theta), a_min=0., a_max=1.)
		self._mask = np.divide(self._sTarget, self._eps + self._nResidual) * Theta

	def optAlpha(self, initloss):
		"""
			A simple gradiend descent method using the RProp algorithm,
			for finding optimum power-spectral density exponents (alpha) for generalized Wiener filtering.
		Args:
			sTarget  : (2D ndarray) Magnitude Spectrogram of the target component
			nResidual: (2D ndarray) Magnitude Spectrogram of the residual component or a list
									of 2D ndarrays which will be added together
			initloss : (float)		Initial loss, for comparisson
		Returns:
			mask:      (2D ndarray) Array that contains time frequency gain values

		"""
		# Initialization of the parameters
		# Put every source spectrogram into an array, given an input list.
		slist = list(self._nResidual)
		slist.insert(0, self._sTarget)
		numElements = len(slist)
		slist = np.asarray(slist)

		alpha = np.array([1.15] * (numElements))  # Initialize an array of alpha values to be found.
		dloss = np.array([0.] * (numElements))  # Initialize an array of loss functions to be used.
		lrs = np.array(
			[self._lr] * (numElements))  # Initialize an array of learning rates to be applied to each source.

		# Begin of otpimization
		isloss = []
		alphalog = []
		for iter in xrange(self._iterations):
			# The actual function of additive power spectrograms
			Xhat = np.sum(np.power(slist, np.reshape(alpha, (numElements, 1, 1))), axis=0)
			for source in xrange(numElements):
				# Derivative with respect to the function of additive power spectrograms
				dX = (slist[source, :, :] ** alpha[source]) * np.log(slist[source, :, :] + self._eps)

				# Chain rule between the above derivative and the IS derivative
				dloss[source] = self._dIS(Xhat) * np.mean(dX)

			alpha -= (lrs * dloss)

			# Make sure the initial alpha are inside reasonable and comparable values
			alpha = np.clip(alpha, a_min=0.5, a_max=2.)
			alpha = np.round(alpha * 100.) / 100.

			# Store the evolution of alphas
			alphalog.append(alpha)

			# Check IS Loss by computing Xhat
			Xhat = 0
			for source in xrange(numElements):
				Xhat += slist[source, :, :] ** alpha[source]

			isloss.append(self._IS(Xhat))

			# Apply RProp
			if (iter > 2):
				if (isloss[-2] - isloss[-1] > 0):
					lrs *= self._hetaplus

				if (isloss[-2] - isloss[-1] < 0):
					lrs *= self._hetaminus

				if (iter > 4):
					if (np.abs(isloss[-2] - isloss[-1]) < 1e-4 and np.abs(isloss[-3] - isloss[-2]) < 1e-4):
						if (isloss[-1] > 3e-1):
							print('Stuck...')
							alpha = alphalog[np.argmin(isloss) - 1]
							isloss[-1] = isloss[np.argmin(isloss)]
						else:
							print('Local Minimum Found')

						print('Final Loss: ' + str(isloss[-1]) + ' with characteristic exponent(s): ' + str(alpha))
						break

			print('Loss: ' + str(isloss[-1]) + ' with characteristic exponent(s): ' + str(alpha))

		# If the operation was terminated by the end of iterations pick the minimum
		if iter == self._iterations:
			self._alpha = alphalog[np.argmin(isloss)]
			self._closs = isloss[np.argmin(isloss)]
		else:
			self._closs = isloss[-1]
			self._alpha = alpha

		# Export the amount of iterations
		self._amountiter = iter

		# Evaluate Xhat for the mask update
		Xhat = 0
		for source in xrange(numElements):
			Xhat += slist[source, :, :] ** alpha[source]

		self._mask = np.divide((slist[0, :, :] ** alpha[0] + self._eps), (Xhat + self._eps))

	def MWF(self):
		""" Multi-channel Wiener filtering as appears in:
		I. Cohen, J. Benesty, and S. Gannot, Speech Processing in Modern
		Communication, Springer, Berlin, Heidelberg, 2010, Chapter 9.
		Args:
			mTarget:   (3D ndarray) Magnitude Spectrogram of the target component
			mY:        (3D ndarray) Magnitude Spectrogram of the output component
								    (M channels x F frequency samples x T time-frames).
		Returns:
			_Out:      (3D ndarray) Array that contains the estimated source.
		"""
		# Parameter for the update
		flambda = 0.99  # Forgetting Factor

		cX = self._sTarget ** self._alpha
		cN = self._nResidual ** self._alpha

		M = self._mX.shape[0]  # Number of channels
		gF = 1. / M  # Gain factor
		eM = cX.shape[0]  # Number of estimated channels
		F = cX.shape[1]  # Number of frequency samples
		T = cX.shape[2]  # Number of time-frames
		fout = np.zeros((M, F, T), dtype=np.float32)  # Initializing output
		I = np.eye(M)  # Identity matrix

		# Initialization of covariance matrices
		Rxx = np.repeat(np.reshape(I, (M, M, 1)), F, axis=-1)
		Rnn = np.repeat(np.reshape(I, (M, M, 1)), F, axis=-1)

		# Recursive updates
		for t in xrange(T):
			for f in xrange(F):
				if eM == 1:
					Rxx[:, :, f] = flambda * Rxx[:, :, f] + (1. - flambda) * (cX[:, f, t])
					Rnn[:, :, f] = flambda * Rnn[:, :, f] + (1. - flambda) * (cN[:, f, t])
				else:
					Rxx[:, :, f] = (np.dot(cX[:, f:f + 1, t], cX[:, f:f + 1, t].T)) / np.sum(cX[:, f, t], axis=0)
					Rnn[:, :, f] = (np.dot(cN[:, f:f + 1, t], cN[:, f:f + 1, t].T)) / np.sum(cN[:, f, t], axis=0)

				inv = np.dot(np.linalg.pinv(Rnn[:, :, f]), (Rnn[:, :, f] + Rxx[:, :, f]))
				if eM == 1:
					Wf = ((inv - I) / ((cN[:, f, t] + cX[:, f, t] + 1e-6) / (cX[:, f, t] + 1e-6) + np.trace(inv) * gF))
				else:
					Wf = ((inv - I) / (gF * np.trace(inv)))

				fout[:, f, t] = np.dot(Wf.T, self._mX[:, f, t])

		self._Out = np.abs(fout)

	def applyMask(self):
		""" Compute the filtered output spectrogram.
		Args:
			mask:   (2D ndarray) Array that contains time frequency gain values
			mX:     (2D ndarray) Input Magnitude Spectrogram
		Returns:
			Y:      (2D ndarray) Filtered version of the Magnitude Spectrogram
		"""
		if self._method == 'expMask':
			self._Out = (self._mX ** self._alpha) ** self._mask
		else:
			self._Out = np.multiply(self._mask, self._mX)

	def applyReverseMask(self):
		""" Compute the filtered output spectrogram, reversing the gain values.
		Args:
			mask:   (2D ndarray) Array that contains time frequency gain values
			mX:     (2D ndarray) Input Magnitude Spectrogram
		Returns:
			Y:      (2D ndarray) Filtered version of the Magnitude Spectrogram
		"""
		if self._method == 'expMask':
			raise ValueError('Cannot compute that using such masking method.')
		else:
			self._Out = np.multiply((1. - self._mask), self._mX)

	def _IS(self, Xhat):
		""" Compute the Itakura-Saito distance between the observed magnitude spectrum
			and the estimated one.
		Args:
			mX    :   	(2D ndarray) Input Magnitude Spectrogram
			Xhat  :     (2D ndarray) Estimated Magnitude Spectrogram
		Returns:
			dis   :     (float) Average Itakura-Saito distance
		"""
		r1 = (np.abs(self._mX) ** self._alpha + self._eps) / (np.abs(Xhat) + self._eps)
		lg = np.log((np.abs(self._mX) ** self._alpha + self._eps)) - np.log((np.abs(Xhat) + self._eps))
		return np.mean(r1 - lg - 1.)

	def _dIS(self, Xhat):
		""" Computation of the first derivative of Itakura-Saito function. As appears in :
			Cedric Fevotte and Jerome Idier, "Algorithms for nonnegative matrix factorization
			with the beta-divergence", in CoRR, vol. abs/1010.1763, 2010.
		Args:
			mX    :   	(2D ndarray) Input Magnitude Spectrogram
			Xhat  :     (2D ndarray) Estimated Magnitude Spectrogram
		Returns:
			dis'  :     (float) Average of first derivative of Itakura-Saito distance.
		"""
		dis = (np.abs(Xhat + self._eps) ** (-2.)) * (np.abs(Xhat) - np.abs(self._mX) ** self._alpha)
		return (np.mean(dis))


if __name__ == "__main__":

	# Small test
	kSin = (0.5 * np.cos(np.arange(4096) * (1000.0 * (3.1415926 * 2.0) / 44100)))
	noise = (np.random.uniform(-0.25,0.25,4096))
	# Noisy observation
	obs = (kSin + noise)

	kSinX = fft(kSin, 4096)
	noisX = fft(noise, 4096)
	obsX  = fft(obs, 4096)

	# Wiener Case
	mask = FrequencyMasking(np.abs(obsX), np.abs(kSinX), [np.abs(noisX)], [], [], alpha = 2., method = 'alphaWiener')
	sinhat = mask()
	noisehat = mask(reverse = True)
	# Access the mask if needed
	ndmask = mask._mask