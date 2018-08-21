# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import numpy as np
from helpers import tf_methods as tf
from scipy.signal import firwin2, freqz, hanning
from scipy.interpolate import InterpolatedUnivariateSpline as uspline


class PsychoacousticModel:
    """ Class that performs a very basic psychoacoustic model.
        - Bark scaling is based on Perceptual-Coding-In-Python, [Online] :
        https://github.com/stephencwelch/Perceptual-Coding-In-Python
        - Perceptual filters and correction based on :
        A. Härmä, and K. Palomäki, ''HUTear – a free Matlab toolbox for modeling of human hearing'',
        in Proceedings of the Matlab DSP Conference, pp 96-99, Espoo, Finland 1999.
    """

    def __init__(self, N=4096, fs=44100, nfilts=24, type='rasta', width=1.0, minfreq=0, maxfreq=22050):

        self.nfft = N
        self.fs = fs
        self.nfilts = nfilts
        self.width = width
        self.min_freq = minfreq
        self.max_freq = maxfreq
        self.max_freq = fs/2
        self.nfreqs = N/2
        self._LTeq = np.zeros(nfilts, dtype=np.float32)

        print('Initializing Psychoacoustic model')

        # Non-linear superposition parameters
        self._alpha = 0.9                                       # Exponent alpha
        self._maxb = 1./self.nfilts                             # Bark-band normalization
        self._fa = 1./(10 ** (14.5/20.) * 10 ** (12.5/20.))     # Tone masking approximation
        self._fb = 1./(10**(7.5/20.))                           # Upper slope of spreading function
        self._fbb = 1./(10**(26./20.))                          # Lower slope of spreading function
        self._fd = 1./self._alpha                               # One over alpha exponent

        # Type of transformation
        self.type = type

        # Computing the matrix for forward Bark transformation
        self.W = self.mX2Bark(type)

        # Computing the inverse matrix for backward Bark transformation
        self.W_inv = self.bark2mX()

    def mX2Bark(self, type):
        """ Method to perform the transofrmation.
        Args :
            type : (str)        String denoting the type of transformation. Can be either
                                'rasta' or 'peaq'.
        Returns  :
            W    : (ndarray)    The transformation matrix.

        """
        if type == 'rasta':
            W = self.fft2bark_rasta()
        elif type == 'peaq':
            W = self.fft2bark_peaq()
        else:
            assert 'Unknown method'

        return W.astype(np.float32)

    def fft2bark_peaq(self):
        """ Method construct the weight matrix.
        Returns  :
            W    : (ndarray)    The transformation matrix, used in PEAQ evaluation.
        """

        nfft = self.nfft
        nfilts = self.nfilts
        fs = self.fs

        # Acquire frequency analysis
        df = float(fs)/nfft

        # Acquire filter responses
        fc, fl, fu = self.CB_filters()

        W = np.zeros((nfilts, nfft))

        for k in range(nfft/2+1):
            for i in range(nfilts):
                temp = (np.amin([fu[i], (k+0.5)*df]) - np.amax([fl[i], (k-0.5)*df])) / df
                W[i, k] = np.amax([0, temp])

        return W

    def fft2bark_rasta(self):
        """ Method construct the weight matrix.
        Returns  :
            W    : (ndarray)    The transformation matrix, used in PEAQ evaluation.
        """
        minfreq = self.min_freq
        maxfreq = self.max_freq
        nfilts = self.nfilts
        nfft = self.nfft
        fs = self.fs

        min_bark = self.hz2bark(minfreq)
        nyqbark = self.hz2bark(maxfreq) - min_bark

        if (nfilts == 0):
            nfilts = np.ceil(nyqbark)+1

        W = np.zeros((nfilts, nfft))

        # Bark per filter
        step_barks = nyqbark/(nfilts-1)

        # Frequency of each FFT bin in Bark
        binbarks = self.hz2bark(np.linspace(0, (nfft/2), (nfft/2)+1)*fs/nfft)

        for i in xrange(nfilts):
            f_bark_mid = min_bark + i * step_barks

            # Compute the absolute threshold
            self._LTeq[i] = 3.64 * (self.bark2hz(f_bark_mid + 1) / 1000.) ** -0.8 - \
                            6.5 * np.exp(-0.6 * (self.bark2hz(f_bark_mid + 1) / 1000. - 3.3) ** 2.) + \
                            1e-3 * ((self.bark2hz(f_bark_mid + 1) / 1000.) ** 4.)

            W[i, 0:(nfft/2)+1] = (np.round(binbarks/step_barks) == i)

        return W

    def bark2mX(self):
        """ Method construct the inverse weight matrix, to map back to FT domain.
        Returns  :
            W    : (ndarray)    The inverse transformation matrix.
        """
        W_inv = np.dot(np.diag((1.0/np.sum(self.W[:, 0:self.nfreqs + 1], 1)) ** 0.5),
                       self.W[:, 0:self.nfreqs + 1]).T

        return W_inv.astype(np.float32)

    def hz2bark(self, f):
        """ Method to compute Bark from Hz.
        Args     :
            f    : (ndarray)    Array containing frequencies in Hz.
        Returns  :
            Brk  : (ndarray)    Array containing Bark scaled values.
        """
        Brk = 6. * np.arcsinh(f/600.)  # Method from RASTA model for computable inverse function.
        return Brk

    def bark2hz(self, Brk):
        """ Method to compute Hz from Bark scale.
        Args     :
            Brk  : (ndarray)    Array containing Bark scaled values.
        Returns  :
            Fhz  : (ndarray)    Array containing frequencies in Hz.
        """
        Fhz = 650. * np.sinh(Brk/7.)

        return Fhz

    def CB_filters(self):
        """ Method to acquire critical band filters for creation of the PEAQ FFT model.
        Returns         :
            fc, fl, fu  : (ndarray)    Arrays containing the values in Hz for the
                                       bandwidth and centre frequencies used in creation
                                       of the transformation matrix.
        """

        fl = np.array([80.000,   103.445,   127.023,   150.762,   174.694,
                       198.849,   223.257,   247.950,   272.959,   298.317,
                       324.055,   350.207,   376.805,   403.884,   431.478,
                       459.622,   488.353,   517.707,   547.721,   578.434,
                       609.885,   642.114,   675.161,   709.071,   743.884,
                       779.647,   816.404,   854.203,   893.091,   933.119,
                       974.336,  1016.797,  1060.555,  1105.666,  1152.187,
                      1200.178,  1249.700,  1300.816,  1353.592,  1408.094,
                      1464.392,  1522.559,  1582.668,  1644.795,  1709.021,
                      1775.427,  1844.098,  1915.121,  1988.587,  2064.590,
                      2143.227,  2224.597,  2308.806,  2395.959,  2486.169,
                      2579.551,  2676.223,  2776.309,  2879.937,  2987.238,
                      3098.350,  3213.415,  3332.579,  3455.993,  3583.817,
                      3716.212,  3853.817,  3995.399,  4142.547,  4294.979,
                      4452.890,  4616.482,  4785.962,  4961.548,  5143.463,
                      5331.939,  5527.217,  5729.545,  5939.183,  6156.396,
                      6381.463,  6614.671,  6856.316,  7106.708,  7366.166,
                      7635.020,  7913.614,  8202.302,  8501.454,  8811.450,
                      9132.688,  9465.574,  9810.536, 10168.013, 10538.460,
                      10922.351, 11320.175, 11732.438, 12159.670, 12602.412,
                      13061.229, 13536.710, 14029.458, 14540.103, 15069.295,
                      15617.710, 16186.049, 16775.035, 17385.420])

        fc = np.array([91.708,   115.216,   138.870,   162.702,   186.742,
                       211.019,   235.566,   260.413,   285.593,   311.136,
                       337.077,   363.448,   390.282,   417.614,   445.479,
                       473.912,   502.950,   532.629,   562.988,   594.065,
                       625.899,   658.533,   692.006,   726.362,   761.644,
                       797.898,   835.170,   873.508,   912.959,   953.576,
                       995.408,  1038.511,  1082.938,  1128.746,  1175.995,
                      1224.744,  1275.055,  1326.992,  1380.623,  1436.014,
                      1493.237,  1552.366,  1613.474,  1676.641,  1741.946,
                      1809.474,  1879.310,  1951.543,  2026.266,  2103.573,
                      2183.564,  2266.340,  2352.008,  2440.675,  2532.456,
                      2627.468,  2725.832,  2827.672,  2933.120,  3042.309,
                      3155.379,  3272.475,  3393.745,  3519.344,  3649.432,
                      3784.176,  3923.748,  4068.324,  4218.090,  4373.237,
                      4533.963,  4700.473,  4872.978,  5051.700,  5236.866,
                      5428.712,  5627.484,  5833.434,  6046.825,  6267.931,
                      6497.031,  6734.420,  6980.399,  7235.284,  7499.397,
                      7773.077,  8056.673,  8350.547,  8655.072,  8970.639,
                      9297.648,  9636.520,  9987.683, 10351.586, 10728.695,
                      11119.490, 11524.470, 11944.149, 12379.066, 12829.775,
                      13294.850, 13780.887, 14282.503, 14802.338, 15341.057,
                      15899.345, 16477.914, 17077.504, 17690.045])

        fu = np.array([103.445,   127.023,   150.762,   174.694,   198.849,
                       223.257,   247.950,   272.959,   298.317,   324.055,
                       350.207,   376.805,   403.884,   431.478,   459.622,
                       488.353,   517.707,   547.721,   578.434,   609.885,
                       642.114,   675.161,   709.071,   743.884,   779.647,
                       816.404,   854.203,   893.091,   933.113,   974.336,
                      1016.797,  1060.555,  1105.666,  1152.187,  1200.178,
                      1249.700,  1300.816,  1353.592,  1408.094,  1464.392,
                      1522.559,  1582.668,  1644.795,  1709.021,  1775.427,
                      1844.098,  1915.121,  1988.587,  2064.590,  2143.227,
                      2224.597,  2308.806,  2395.959,  2486.169,  2579.551,
                      2676.223,  2776.309,  2879.937,  2987.238,  3098.350,
                      3213.415,  3332.579,  3455.993,  3583.817,  3716.212,
                      3853.348,  3995.399,  4142.547,  4294.979,  4452.890,
                      4643.482,  4785.962,  4961.548,  5143.463,  5331.939,
                      5527.217,  5729.545,  5939.183,  6156.396,  6381.463,
                      6614.671,  6856.316,  7106.708,  7366.166,  7635.020,
                      7913.614,  8202.302,  8501.454,  8811.450,  9132.688,
                      9465.574,  9810.536, 10168.013, 10538.460, 10922.351,
                      11320.175, 11732.438, 12159.670, 12602.412, 13061.229,
                      13536.710, 14029.458, 14540.103, 15069.295, 15617.710,
                      16186.049, 16775.035, 17385.420, 18000.000])

        return fc, fl, fu

    def forward(self, spc):
        """ Method to transform FT domain to Bark.
        Args         :
            spc      : (ndarray)    2D Array containing the magnitude spectra.
        Returns      :
            Brk_spc  : (ndarray)    2D Array containing the Bark scaled magnitude spectra.
        """
        W_short = self.W[:, 0:self.nfreqs]
        Brk_spc = np.dot(W_short, spc)
        return Brk_spc

    def backward(self, Brk_spc):
        """ Method to reconstruct FT domain from Bark.
        Args         :
            Brk_spc  : (ndarray)    2D Array containing the Bark scaled magnitude spectra.
        Returns      :
            Xhat     : (ndarray)    2D Array containing the reconstructed magnitude spectra.
        """
        Xhat = np.dot(self.W_inv, Brk_spc)
        return Xhat

    def OutMidCorrection(self, correctionType, firOrd, fs):
        """
        Method to "correct" the middle outer ear transfer function.
            As appears in :
            - A. Härmä, and K. Palomäki, ''HUTear – a free Matlab toolbox for modeling of human hearing'',
            in Proceedings of the Matlab DSP Conference, pp 96-99, Espoo, Finland 1999.
        """
        # Lookup tables for correction
        f1 = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100,
                      125, 150, 177, 200, 250, 300, 350, 400, 450, 500, 550,
                      600, 700, 800, 900, 1000, 1500, 2000, 2500, 2828, 3000,
                      3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000,
                      12748, 15000])

        ELC = np.array([31.8, 26.0, 21.7, 18.8, 17.2, 15.4, 14.0, 12.6, 11.6, 10.6,
                        9.2, 8.2, 7.7, 6.7, 5.3, 4.6, 3.9, 2.9, 2.7, 2.3,
                        2.2, 2.3, 2.5, 2.7, 2.9, 3.4, 3.9, 3.9, 3.9, 2.7,
                        0.9, -1.3, -2.5, -3.2, -4.4, -4.1, -2.5, -0.5, 2.0, 5.0,
                        10.2, 15.0, 17.0, 15.5, 11.0, 22.0])

        MAF = np.array([73.4, 65.2, 57.9, 52.7, 48.0, 45.0, 41.9, 39.3, 36.8, 33.0,
                        29.7, 27.1, 25.0, 22.0, 18.2, 16.0, 14.0, 11.4, 9.2, 8.0,
                        6.9,  6.2,  5.7,  5.1,  5.0,  5.0,  4.4,  4.3, 3.9, 2.7,
                        0.9, -1.3, -2.5, -3.2, -4.4, -4.1, -2.5, -0.5, 2.0, 5.0,
                        10.2, 15.0, 17.0, 15.5, 11.0, 22.0])

        f2 = np.array([125,  250,  500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 14000, 16000])

        MAP = np.array([30.0, 19.0, 12.0,  9.0, 11.0, 16.0, 16.0, 14.0, 14.0,  9.9, 24.7, 32.7, 44.1, 63.7])

        if correctionType == 'ELC':
            freqTable = f1
            CorrectionTable = ELC
        elif correctionType == 'MAF':
            freqTable = f1
            CorrectionTable = MAF
        elif correctionType == 'MAP':
            freqTable = f2
            CorrectionTable = MAP
        else:
            print('Unrecongised operation: ELC will be used instead...')
            freqTable = f1
            CorrectionTable = ELC

        freqN = np.arange(0, firOrd) * fs/2. / (firOrd-1)
        spline = uspline(freqTable, CorrectionTable)
        crc = spline(freqN)
        crclin = 10. ** (-crc / 10.)
        return crclin, freqN, crc

    def MOEar(self, correctionType='ELC'):
        """ Method to approximate middle-outer ear transfer function for linearly scaled
            frequency representations, using an FIR approximation of order 600 taps.
            As appears in :
            - A. Härmä, and K. Palomäki, ''HUTear – a free Matlab toolbox for modeling of human hearing'',
            in Proceedings of the Matlab DSP Conference, pp 96-99, Espoo, Finland 1999.
        Arguments          :
            correctionType : (string)     String which specifies the type of correction :
                                          'ELC' - Equal Loudness Curves at 60 dB (default)
                                          'MAP' - Minimum Audible Pressure at ear canal
                                          'MAF' - Minimum Audible Field
        Returns            :
            LTq            : (ndarray)    1D Array containing the transfer function, without the DC sub-band.
        """
        # Parameters
        firOrd = self.nfft
        Cr, fr, Crdb = self.OutMidCorrection(correctionType, firOrd, self.fs)
        Cr[self.nfft - 1] = 0.

        # FIR Design
        A = firwin2(firOrd, fr, Cr, nyq=self.fs/2)
        B = 1
        _, LTq = freqz(A, B, firOrd, self.fs)

        LTq = 20. * np.log10(np.abs(LTq) + 1e-6)
        LTq -= max(LTq)
        return LTq[:self.nfft/2 + 1].astype(np.float32)

    def maskingThreshold(self, mX):
        """ Method to compute the masking threshold by non-linear superposition.
        As used in :
        - G. Schuller, B. Yu, D. Huang and B. Edler, "Perceptual Audio Coding Using Adaptive
        Pre and Post-filters and Lossless Compression", in IEEE Transactions on Speech and Audio Processing,
        vol. 10, n. 6, pp. 379-390, September, 2002.
        As appears in :
        - F. Baumgarte, C. Ferekidis, H Fuchs,  "A Nonlinear Psychoacoustic Model Applied to ISO/MPEG Layer 3 Coder",
        in Proceedings of the 99th Audio Engineering Society Convention, October, 1995.

        Args         :
            mX       : (ndarray)    2D Array containing the magnitude spectra (1 time frame x frequency subbands)
        Returns      :
            mT       : (ndarray)    2D Array containing the masking threshold.

        Authors      : Gerald Schuller('shl'), S.I. Mimilakis ('mis')
        """
        # Bark Scaling with the initialized, from the class, matrix W.
        if mX.shape[1] % 2 == 0:
            nyq = False
            mX = np.dot(np.abs(mX), self.W[:, :self.nfreqs].T)
        else:
            nyq = True
            mX = np.dot(np.abs(mX), self.W[:, :self.nfreqs + 1].T)

        # Parameters
        Numsubbands = mX.shape[1]
        timeFrames = mX.shape[0]
        fc = self._maxb * Numsubbands

        # Initialization of the matrix containing the masking threshold
        maskingThreshold = np.zeros((timeFrames, Numsubbands), dtype=np.float32)

        for frameindx in xrange(mX.shape[0]):
            mT = np.zeros(Numsubbands)
            for n in xrange(Numsubbands):
                for m in xrange(0, n):
                    mT[n] += (mX[frameindx, m] * self._fa * (self._fb ** ((n - m) * fc))) ** self._alpha

                for m in xrange(n+1, Numsubbands):
                    mT[n] += (mX[frameindx, m] * self._fa * (self._fbb ** ((m - n) * fc))) ** self._alpha

                mT[n] = mT[n] ** self._fd

            maskingThreshold[frameindx, :] = mT

        # Inverse the bark scaling with the initialized, from the class, matrix W_inv.
        if nyq == False:
            maskingThreshold = np.dot(maskingThreshold, self.W_inv[:-1, :self.nfreqs].T)
        else:
            maskingThreshold = np.dot(maskingThreshold, self.W_inv[:, :self.nfreqs].T)

        return maskingThreshold

    def NMREval(self, xn, xnhat):
        """ Method to perform NMR-based evaluation of audio quality between two signals.
        Args        :
            xn      :   (ndarray) 1D Array containing the true time domain signal.
            xnhat   :   (ndarray) 1D Array containing the estimated time domain signal.
        Returns     :
            NMR     :   (float)   A measurement providing an indication of how close
                                  the errors are to the masking threshold. The average through
                                  time is computed. Values less than 0 indicate imperceptiple
                                  differences.
        Based on :
        - K. Brandenburg and T. Sporer,  “NMR and Masking Flag: Evaluation of Quality Using Perceptual Criteria,” in
        Proceedings of the AES 11th International Conference on Test and Measurement, Portland, USA, May 1992, pp. 169–179.
        """

        mX, _ = tf.TimeFrequencyDecomposition.STFT(xn, hanning(self.nfft/2 + 1), self.nfft, self.nfft/4)
        mXhat, _ = tf.TimeFrequencyDecomposition.STFT(xnhat, hanning(self.nfft/2 + 1), self.nfft, self.nfft/4)

        # Remove zero-padded frames
        mX = mX[4:-4, :]
        mXhat = mXhat[4:-4, :]

        # Compute Error
        Err = np.abs(mX - mXhat) ** 2.

        # Acquire Masking Threshold
        mT = self.maskingThreshold(mX)

        # Inverse the filter of masking threshold
        imT = (1./(mT + 1e-12)) ** 2.

        # NMR computation
        NMR = (10. * np.log10(np.mean(np.mean(imT * Err, axis=-1) + 1e-12)))

        print(NMR)
        return NMR

# EOF
