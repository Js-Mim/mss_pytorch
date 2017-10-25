# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import math, sys, os
import numpy as np
from scipy.fftpack import fft, ifft, dct, dst
from scipy.signal import firwin2, freqz, cosine, hanning, hamming, fftconvolve
from scipy.interpolate import InterpolatedUnivariateSpline as uspline
try:
    from QMF import qmf_realtime_class as qrf
except ImportError:
    print('PQMF class was not found. ')

eps = np.finfo(np.float32).tiny

class TimeFrequencyDecomposition:
    """ A Class that performs time-frequency decompositions by means of a
        Discrete Fourier Transform, using Fast Fourier Transform algorithm
        by SciPy, MDCT with modified type IV bases, PQMF,
        and Fractional Fast Fourier Transform.
    """
    @staticmethod
    def DFT(x, w, N):
        """ Discrete Fourier Transformation(Analysis) of a given real input signal
        via an FFT implementation from scipy. Single channel is being supported.
        Args:
            x       : (array) Real time domain input signal
            w       : (array) Desired windowing function
            N       : (int)   FFT size
        Returns:
            magX    : (2D ndarray) Magnitude Spectrum
            phsX    : (2D ndarray) Phase Spectrum
        """

        # Half spectrum size containing DC component
        hlfN = (N/2)+1

        # Half window size. Two parameters to perform zero-phase windowing technique
        hw1 = int(math.floor((w.size+1)/2))
        hw2 = int(math.floor(w.size/2))

        # Window the input signal
        winx = x*w

        # Initialize FFT buffer with zeros and perform zero-phase windowing
        fftbuffer = np.zeros(N)
        fftbuffer[:hw1] = winx[hw2:]
        fftbuffer[-hw2:] = winx[:hw2]

        # Compute DFT via scipy's FFT implementation
        X = fft(fftbuffer)

        # Acquire magnitude and phase spectrum
        magX = (np.abs(X[:hlfN]))
        phsX = (np.angle(X[:hlfN]))

        return magX, phsX

    @staticmethod
    def iDFT(magX, phsX, wsz):
        """ Discrete Fourier Transformation(Synthesis) of a given spectral analysis
        via an inverse FFT implementation from scipy.
        Args:
            magX    : (2D ndarray) Magnitude Spectrum
            phsX    : (2D ndarray) Phase Spectrum
            wsz     :  (int)   Synthesis window size
        Returns:
            y       : (array) Real time domain output signal
        """

        # Get FFT Size
        hlfN = magX.size;
        N = (hlfN-1)*2

        # Half of window size parameters
        hw1 = int(math.floor((wsz+1)/2))
        hw2 = int(math.floor(wsz/2))

        # Initialise synthesis buffer with zeros
        fftbuffer = np.zeros(N)
        # Initialise output spectrum with zeros
        Y = np.zeros(N, dtype = complex)
        # Initialise output array with zeros
        y = np.zeros(wsz)

        # Compute complex spectrum(both sides) in two steps
        Y[0:hlfN] = magX * np.exp(1j*phsX)
        Y[hlfN:] = magX[-2:0:-1] * np.exp(-1j*phsX[-2:0:-1])

        # Perform the iDFT
        fftbuffer = np.real(ifft(Y))

        # Roll-back the zero-phase windowing technique
        y[:hw2] = fftbuffer[-hw2:]
        y[hw2:] = fftbuffer[:hw1]

        return y

    @staticmethod
    def STFT(x, w, N, hop):
        """ Short Time Fourier Transform analysis of a given real input signal,
        via the above DFT method.
        Args:
            x   : 	(array)  Time-domain signal
            w   :   (array)  Desired windowing function
            N   :   (int)    FFT size
            hop :   (int)    Hop size
        Returns:
            sMx :   (2D ndarray) Stacked arrays of magnitude spectra
            sPx :   (2D ndarray) Stacked arrays of phase spectra
        """

        # Analysis Parameters
        wsz = w.size

        # Add some zeros at the start and end of the signal to avoid window smearing
        x = np.append(np.zeros(3*hop),x)
        x = np.append(x, np.zeros(3*hop))

        # Initialize sound pointers
        pin = 0
        pend = x.size - wsz
        indx = 0

        # Normalise windowing function
        if np.sum(w)!= 0. :
            w = w / np.sqrt(N)

        # Initialize storing matrix
        xmX = np.zeros((len(x)/hop, N/2 + 1), dtype = np.float32)
        xpX = np.zeros((len(x)/hop, N/2 + 1), dtype = np.float32)

        # Analysis Loop
        while pin <= pend:
            # Acquire Segment
            xSeg = x[pin:pin+wsz]

            # Perform DFT on segment
            mcX, pcX = TimeFrequencyDecomposition.DFT(xSeg, w, N)

            xmX[indx, :] = mcX
            xpX[indx, :] = pcX

            # Update pointers and indices
            pin += hop
            indx += 1

        return xmX, xpX

    @staticmethod
    def GLA(wsz, hop, N = 4096):
        """ LSEE-MSTFT algorithm for computing the synthesis window used in
        inverse STFT method below.
        Args:
            wsz :   (int)    Synthesis window size
            hop :   (int)    Hop size
            N   :   (int)    DFT Size
        Returns :
            symw:   (array)  Synthesised windowing function

        References :
            [1] Daniel W. Griffin and Jae S. Lim, ``Signal estimation from modified short-time
            Fourier transform,'' IEEE Transactions on Acoustics, Speech and Signal Processing,
            vol. 32, no. 2, pp. 236-243, Apr 1984.
        """
        synw = hamming(wsz)/np.sqrt(N)
        synwProd = synw ** 2.
        synwProd.shape = (wsz, 1)
        redundancy = wsz/hop
        env = np.zeros((wsz, 1))
        for k in xrange(-redundancy, redundancy + 1):
            envInd = (hop*k)
            winInd = np.arange(1, wsz+1)
            envInd += winInd

            valid = np.where((envInd > 0) & (envInd <= wsz))
            envInd = envInd[valid] - 1
            winInd = winInd[valid] - 1
            env[envInd] += synwProd[winInd]

        synw = synw/env[:, 0]
        return synw

    @staticmethod
    def iSTFT(xmX, xpX, wsz, hop, smt = False) :
        """ Short Time Fourier Transform synthesis of given magnitude and phase spectra,
        via the above iDFT method.
        Args:
            xmX :   (2D ndarray)  Magnitude spectrum
            xpX :   (2D ndarray)  Phase spectrum
            wsz :   (int)         Synthesis window size
            hop :   (int)         Hop size
            smt :   (bool)        Whether or not use a post-processing step in time domain
                                  signal recovery, using synthesis windows.
        Returns :
            y   :   (array)       Synthesised time-domain real signal.
        """

        # GL-Algorithm or simple OLA
        if smt == True:
            rs = TimeFrequencyDecomposition.GLA(wsz, hop, (wsz - 1)*2.)
        else:
            rs = np.sqrt(2. * (wsz-1))

        # Acquire half window sizes
        hw1 = int(math.floor((wsz+1)/2))
        hw2 = int(math.floor(wsz/2))

        # Acquire the number of STFT frames
        numFr = xmX.shape[0]

        # Initialise output array with zeros
        y = np.zeros(numFr * hop + hw1 + hw2)

        # Initialise sound pointer
        pin = 0

        # Main Synthesis Loop
        for indx in range(numFr):
            # Inverse Discrete Fourier Transform
            ybuffer = TimeFrequencyDecomposition.iDFT(xmX[indx, :], xpX[indx, :], wsz)

            # Overlap and Add
            y[pin:pin+wsz] += ybuffer*rs

            # Advance pointer
            pin += hop

        # Delete the extra zeros that the analysis had placed
        y = np.delete(y, range(3*hop))
        y = np.delete(y, range(y.size-(3*hop + 1), y.size))

        return y

    @staticmethod
    def MCSTFT(x, w, N, hop):
        """ Short Time Fourier Transform analysis of a given real input signal,
        over multiple channels.
        Args:
            x   : 	(2D array)  Multichannel time-domain signal (nsamples x nchannels)
            w   :   (array)     Desired windowing function
            N   :   (int)       FFT size
            hop :   (int)       Hop size
        Returns:
            sMx :   (3D ndarray) Stacked arrays of magnitude spectra
            sPx :   (3D ndarray) Stacked arrays of phase spectra
                                 Of the shape (Channels x Frequency-samples x Time-frames)
        """
        M = x.shape[1]      # Number of channels

        # Analyse the first incoming channel to acquire the dimensions
        mX, pX = TimeFrequencyDecomposition.STFT(x[:, 0], w, N, hop)
        smX = np.zeros((M, mX.shape[1], mX.shape[0]), dtype = np.float32)
        spX = np.zeros((M, pX.shape[1], pX.shape[0]), dtype = np.float32)
        # Storing it to the actual return and free up some memory
        smX[0, :, :] = mX.T
        spX[0, :, :] = pX.T
        del mX, pX

        for channel in xrange(1, M):
            mX, pX = TimeFrequencyDecomposition.STFT(x[:, channel], w, N, hop)
            smX[channel, :, :] = mX.T
            spX[channel, :, :] = pX.T

        del mX, pX

        return smX, spX

    @staticmethod
    def MCiSTFT(xmX, xpX, wsz, hop, smt = False):
        """ Short Time Fourier Transform synthesis of given magnitude and phase spectra
        over multiple channels.
        Args:
            xMx :   (3D ndarray) Stacked arrays of magnitude spectra
            xPx :   (3D ndarray) Stacked arrays of phase spectra
                                 Of the shape (Channels x Frequency samples x Time-frames)
            wsz :   (int)        Synthesis Window size
            hop :   (int)        Hop size
            smt :   (bool)       Whether or not use a post-processing step in time domain
                                 signal recovery, using synthesis windows
        Returns :
            y   :   (2D array)   Synthesised time-domain real signal of the shape (nsamples x nchannels)
        """
        M = xmX.shape[0]      # Number of channels
        F = xmX.shape[1]      # Number of frequency samples
        T = xmX.shape[2]      # Number of time-frames

        # Synthesize the first incoming channel to acquire the dimensions
        y = TimeFrequencyDecomposition.iSTFT(xmX[0, :, :].T, xpX[0, :, :].T, wsz, hop, smt)
        yout = np.zeros((len(y), M), dtype = np.float32)
        # Storing it to the actual return and free up some memory
        yout[:, 0] = y
        del y

        for channel in xrange(1, M):
            y = TimeFrequencyDecomposition.iSTFT(xmX[channel, :, :].T, xpX[channel, :, :].T, wsz, hop, smt)
            yout[:, channel] = y

        del y

        return yout

    @staticmethod
    def nuttall4b(M, sym=False):
        """
        Returns a minimum 4-term Blackman-Harris window according to Nuttall.
        The typical Blackman window famlity define via "alpha" is continuous
        with continuous derivative at the edge. This will cause some errors
        to short time analysis, using odd length windows.

        Args    :
            M   :   (int)   Number of points in the output window.
            sym :   (array) Synthesised time-domain real signal.

        Returns :
            w   :   (ndarray) The windowing function

        References :
            [1] Heinzel, G.; Rüdiger, A.; Schilling, R. (2002). Spectrum and spectral density
               estimation by the Discrete Fourier transform (DFT), including a comprehensive
               list of window functions and some new flat-top windows (Technical report).
               Max Planck Institute (MPI) für Gravitationsphysik / Laser Interferometry &
               Gravitational Wave Astronomy, 395068.0

            [2] Nuttall A.H. (1981). Some windows with very good sidelobe behaviour. IEEE
               Transactions on Acoustics, Speech and Signal Processing, Vol. ASSP-29(1):
               84-91.
        """

        if M < 1:
            return np.array([])
        if M == 1:
            return np.ones(1, 'd')
        if not sym :
            M = M + 1

        a = [0.355768, 0.487396, 0.144232, 0.012604]
        n = np.arange(0, M)
        fac = n * 2 * np.pi / (M - 1.0)

        w = (a[0] - a[1] * np.cos(fac) +
             a[2] * np.cos(2 * fac) - a[3] * np.cos(3 * fac))

        if not sym:
            w = w[:-1]

        return w

    @staticmethod
    def pqmf_analysis(x):
        """
            Method to analyse an input time-domain signal using PQMF.
            See QMF class for more information.

            Arguments   :
                x       : (1D Array) Input signal

            Returns     :
                ms      : (2D Array) Analysed time-frequency representation by means of PQMF analysis.
        """
        # Parameters
        N = 1024
        nTimeSlots = len(x) / N


        # Initialization
        ms = np.zeros((nTimeSlots, N), dtype=np.float32)
        qrf.reset_rt()
        # Perform Analysis
        for m in xrange(nTimeSlots):
            ms[m, :] = qrf.PQMFAnalysis.analysisqmf_realtime(x[m*N:(m+1)*N], N)

        return ms

    @staticmethod
    def pqmf_synthesis(ms):
        """
            Method to synthesise a time-domain signal using PQMF.
            See QMF class for more information.

            Arguments   :
                ms      : (2D Array) Analysed time-frequency representation by means of PQMF analysis.

            Returns     :
                xrec    : (1D Array) Reconstructed signal
        """
        # Parameters
        N = 1024
        nTimeSlots = ms.shape[0]


        # Initialization
        xrec = np.zeros((nTimeSlots * N), dtype=np.float32)
        qrf.reset_rt()
        # Perform Analysis
        for m in xrange(nTimeSlots):
            xrec[m * N: (m + 1) * N] = qrf.PQMFSynthesis.synthesisqmf_realtime(ms[m, :], N)

        return xrec

    @staticmethod
    def coreModulation(win, N, type = 'MDCT'):
        """
            Method to produce Analysis and Synthesis matrices for the offline
            PQMF class, using polyphase matrices.

            Arguments  :
                win    :  (1D Array) Windowing function
                N      :  (int) Number of subbands
                type   :  (str) Selection between 'MDCT' or 'PQMF' basis functions

            Returns  :
                Cos   :   (2D Array) Cosine Modulated Polyphase Matrix
                Sin   :   (2D Array) Sine Modulated Polyphase Matrix

        """
        global Cos

        lfb = len(win)
        # Initialize Storing Variables
        Cos = np.zeros((N,lfb), dtype = np.float32)
        #Sin = np.zeros((N,lfb), dtype = np.float32)

        # Generate Matrices        
        if type == 'MDCT' :
	        print('MDCT')
	        for k in xrange(0, N):
	            for n in xrange(0, lfb):
	                Cos[k, n] = win[n] * np.cos(np.pi/N * (k + 0.5) * (n + 0.5 + N/2)) * np.sqrt(2. / N)
	                #Sin[k, n] = win[n] * np.sin(np.pi/N * (k + 0.5) * (n + 0.5 + N/2)) * np.sqrt(2. / N)

        elif type == 'PQMF-polyphase' :
	        print('PQMF-polyphase')
	        for k in xrange(0, N):
	            for n in xrange(0, lfb):
	                Cos[k, n] = win[n] * np.cos(np.pi/N * (k + 0.5) * (n + 0.5)) * np.sqrt(2. / N)
	                #Sin[k, n] = win[n] * np.sin(np.pi/N * (k + 0.5) * (n + 0.5)) * np.sqrt(2. / N)
        
        else :
            assert('Unknown type')

        return Cos

    @staticmethod
    def real_analysis(x, N = 1024):
        """
            Method to compute the subband samples from time domain signal x.
            A real valued output matrix will be computed using DCT.

            Arguments   :
                x       : (1D Array) Input signal
                N       : (int)      Number of sub-bands

            Returns     :
                y       : (2D Array) Real valued output of the analysis

        """
        # Parameters and windowing function design
        win = cosine(2*N, True)
        lfb = len(win)
        nTimeSlots = len(x)/N - 2

        # Initialization
        ycos = np.zeros((len(x)/N, N), dtype = np.float32)
        ysin = np.zeros((len(x)/N, N), dtype = np.float32)

        # Check global variables in order to avoid
        # computing over and over again the transformation matrices.
        glvars = globals()

        if 'Cos' in glvars and ((glvars['Cos'].T).shape[1] == N):
            print('... using pre-computed transformation matrices')
            global Cos
            # Perform Analysis
            for m in xrange(0, nTimeSlots):
                ycos[m, :] = np.dot(x[m * N: m * N + lfb], Cos.T)
        else :
            print('... computing transformation matrices')
            # Analysis Matrix
            Cos = TimeFrequencyDecomposition.coreModulation(win, N)
            # Perform Analysis
            for m in xrange(0, nTimeSlots):
                ycos[m, :] = np.dot(x[m * N: m * N + lfb], Cos.T)

        return ycos

    @staticmethod
    def real_synthesis(y):
        """
            Method to compute the resynthesis of the MDCT.
            A real valued input matrix is asummed as input, derived from DCT typeIV.

            Arguments   :
                y       : (2D Array) Real Representation (time frames x frequency sub-bands (N))

            Returns     :
                xrec    : (1D Array) Time domain reconstruction

        """
        # Parameters and windowing function design
        N = y.shape[1]
        win = cosine(2*N, True)
        lfb = len(win)
        nTimeSlots = y.shape[0]
        SignalLength = nTimeSlots * N + 2 * N

        # Check global variables in order to avoid
        # computing over and over again the transformation matrices.
        glvars = globals()

        if 'Cos' in glvars and ((glvars['Cos'].T).shape[1] == N):
            print('... using pre-computed transformation matrix')
            global Cos
            # Initialization
            zcos = np.zeros((1, SignalLength), dtype=np.float32)

            # Perform Synthesis
            for m in xrange(0, nTimeSlots):
                zcos[0, m * N: m * N + lfb] += np.dot((y[m, :]).T, Cos)

        else:
            print('... computing transformation matrix')
            # Synthesis marix
            Cos = TimeFrequencyDecomposition.coreModulation(win, N)

            # Initialization
            zcos = np.zeros((1, SignalLength), dtype=np.float32)

            # Perform Synthesis
            for m in xrange(0, nTimeSlots):
                zcos[0, m * N: m * N + lfb] += np.dot((y[m, :]).T, Cos)

        return zcos.T

    @staticmethod
    def frft(f, a):
        """
        Fractional Fourier transform. As appears in :
        -https://nalag.cs.kuleuven.be/research/software/FRFT/
        -https://github.com/audiolabs/frft/
        Args:
            f       : (array) Input data
            a       : (float) Alpha factor
        Returns:
            ret    : (array) Complex valued analysed data

        """
        ret = np.zeros_like(f, dtype=np.complex)
        f = f.copy().astype(np.complex)
        N = len(f)
        shft = np.fmod(np.arange(N) + np.fix(N / 2), N).astype(int)
        sN = np.sqrt(N)
        a = np.remainder(a, 4.0)

        # Special cases
        if a == 0.0:
            return f
        if a == 2.0:
            return np.flipud(f)
        if a == 1.0:
            ret[shft] = np.fft.fft(f[shft]) / sN
            return ret
        if a == 3.0:
            ret[shft] = np.fft.ifft(f[shft]) * sN
            return ret

        # reduce to interval 0.5 < a < 1.5
        if a > 2.0:
            a = a - 2.0
            f = np.flipud(f)
        if a > 1.5:
            a = a - 1
            f[shft] = np.fft.fft(f[shft]) / sN
        if a < 0.5:
            a = a + 1
            f[shft] = np.fft.ifft(f[shft]) * sN

        # the general case for 0.5 < a < 1.5
        alpha = a * np.pi / 2
        tana2 = np.tan(alpha / 2)
        sina = np.sin(alpha)
        f = np.hstack((np.zeros(N - 1), TimeFrequencyDecomposition.sincinterp(f), np.zeros(N - 1))).T

        # chirp premultiplication
        chrp = np.exp(-1j * np.pi / N * tana2 / 4 * np.arange(-2 * N + 2, 2 * N - 1).T ** 2)
        f = chrp * f

        # chirp convolution
        c = np.pi / N / sina / 4
        ret = fftconvolve(np.exp(1j * c * np.arange(-(4 * N - 4), 4 * N - 3).T ** 2), f)
        ret = ret[4 * N - 4:8 * N - 7] * np.sqrt(c / np.pi)

        # chirp post multiplication
        ret = chrp * ret

        # normalizing constant
        ret = np.exp(-1j * (1 - a) * np.pi / 4) * ret[N - 1:-N + 1:2]

        return ret

    @staticmethod
    def ifrft(f, a):
        """
        Inverse fractional Fourier transform. As appears in :
        -https://nalag.cs.kuleuven.be/research/software/FRFT/
        -https://github.com/audiolabs/frft/
        ----------
        Args:
            f       : (array) Complex valued input array
            a       : (float) Alpha factor
        Returns:
            ret     : (array) Real valued synthesised data
        """
        return TimeFrequencyDecomposition.frft(f, -a)

    @staticmethod
    def sincinterp(x):
        """
        Sinc interpolation for computation of fractional transformations.
        As appears in :
        -https://github.com/audiolabs/frft/
        ----------
        Args:
            f       : (array) Complex valued input array
            a       : (float) Alpha factor
        Returns:
            ret     : (array) Real valued synthesised data
        """
        N = len(x)
        y = np.zeros(2 * N - 1, dtype=x.dtype)
        y[:2 * N:2] = x
        xint = fftconvolve( y[:2 * N], np.sinc(np.arange(-(2 * N - 3), (2 * N - 2)).T / 2),)
        return xint[2 * N - 3: -2 * N + 3]

    @staticmethod
    def stfrft(x, w, hop, a):
        """ Short Time Fractional Fourier Transform analysis of a given real input signal,
        via the above DFT method.
        Args:
            x   : 	(array)  Magnitude Spectrum
            w   :   (array)  Desired windowing function determining the analysis size
            hop :   (int)    Hop size
            a   :   (float)  Alpha factor
        Returns:
            sMx :   (2D ndarray) Stacked arrays of magnitude spectra
            sPx :   (2D ndarray) Stacked arrays of phase spectra
        """

        # Analysis Parameters
        wsz = w.size

        # Add some zeros at the start and end of the signal to avoid window smearing
        x = np.append(np.zeros(3*hop),x)
        x = np.append(x, np.zeros(3*hop))

        # Initialize sound pointers
        pin = 0
        pend = x.size - wsz
        indx = 0

        # Initialize storing matrix
        xmX = np.zeros((len(x)/hop, wsz), dtype = np.float32)
        xpX = np.zeros((len(x)/hop, wsz), dtype = np.float32)

        # Analysis Loop
        while pin <= pend:
            # Acquire Segment
            xSeg = x[pin:pin+wsz] * w

            # Perform frFT on segment
            cX = TimeFrequencyDecomposition.frft(xSeg, a)

            xmX[indx, :] = np.abs(cX)
            xpX[indx, :] = np.angle(cX)

            # Update pointers and indices
            pin += hop
            indx += 1

        return xmX, xpX

    @staticmethod
    def istfrft(xmX, xpX, hop, a):
        """ Inverse Short Time Fractional Fourier Transform synthesis of given magnitude and phase spectra.
        Args:
            xmX :   (2D ndarray)  Magnitude Spectrum
            xpX :   (2D ndarray)  Phase Spectrum
            hop :   (int)         Hop Size
            a   :   (float)       Alpha factor
        Returns :
            y   :   (array) Synthesised time-domain real signal.
        """
        # Acquire the number of frames
        numFr = xmX.shape[0]
        # Amount of samples
        wsz = xmX.shape[1]


        # Initialise output array with zeros
        y = np.zeros(numFr * hop + wsz)

        # Initialise sound pointer
        pin = 0

        # Main Synthesis Loop
        for indx in range(numFr):
            # Inverse FrFT
            cX = xmX[indx, :] * np.exp(1j*xpX[indx, :])
            ybuffer = TimeFrequencyDecomposition.ifrft(cX, a)

            # Overlap and Add
            y[pin:pin+wsz] += np.real(ybuffer)

            # Advance pointer
            pin += hop

        # Delete the extra zeros that the analysis had placed
        y = np.delete(y, range(3*hop))
        y = np.delete(y, range(y.size-(3*hop + 1), y.size))

        return y