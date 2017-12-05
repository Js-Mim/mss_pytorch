# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import math
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import hamming

# definition
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
        hlfN = magX.size
        N = (hlfN-1)*2

        # Half of window size parameters
        hw1 = int(math.floor((wsz+1)/2))
        hw2 = int(math.floor(wsz/2))

        # Initialise output spectrum with zeros
        Y = np.zeros(N, dtype=complex)
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
        if np.sum(w) != 0.:
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
    def GLA(wsz, hop, N=4096):
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
    def iSTFT(xmX, xpX, wsz, hop, smt=False) :
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
    def MCiSTFT(xmX, xpX, wsz, hop, smt=False):
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