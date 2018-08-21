# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
from helpers.io_methods import AudioIO as Io
from helpers.masking_methods import FrequencyMasking as Fm
from numpy.lib import stride_tricks
from helpers import tf_methods as tf
from helpers import psychoacoustic_model as pm
import torch
import numpy as np
import os

# definitions
dataset_path = '/mnt/IDMT-WORKSPACE/DATA-STORE/mis/Datasets/musdb18/'
test_dataset_path = '/home/avdata/audio/public/musdb18/'
keywords = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav', 'mixture.wav']
foldersList = ['train', 'test']
save_path = 'results/'

pm = pm.PsychoacousticModel(N=4096, fs=44100, nfilts=24)
mm = pm.MOEar()
mm = 10**(mm/20.)

__all__ = [
    'prepare_overlap_sequences',
    'get_data',
    'test_eval',
    'test_nnet'
]


def prepare_overlap_sequences(ms, vs, bk, l_size, o_lap, bsize):
    """
        Method to prepare overlapping sequences of the given magnitude spectra.
        Args:
            ms               : (2D Array)  Mixture magnitude spectra (Time frames times Frequency sub-bands).
            vs               : (2D Array)  Singing voice magnitude spectra (Time frames times Frequency sub-bands).
            bk               : (2D Array)  Background magnitude spectra (Time frames times Frequency sub-bands).
            l_size           : (int)       Length of the time-sequence.
            o_lap            : (int)       Overlap between spectrogram time-sequences
                                           (to recover the missing information from the context information).
            bsize            : (int)       Batch size.

        Returns:
            ms               : (3D Array)  Mixture magnitude spectra training data
                                           reshaped into overlapping sequences.
            vs               : (3D Array)  Singing voice magnitude spectra training data
                                           reshaped into overlapping sequences.
            bk               : (3D Array)  Background magnitude spectra training data
                                           reshaped into overlapping sequences.

    """
    trim_frame = ms.shape[0] % (l_size - o_lap)
    trim_frame -= (l_size - o_lap)
    trim_frame = np.abs(trim_frame)
    # Zero-padding
    if trim_frame != 0:
        ms = np.pad(ms, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))
        vs = np.pad(vs, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))
        bk = np.pad(bk, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))

    # Reshaping with overlap
    ms = stride_tricks.as_strided(ms, shape=(ms.shape[0] / (l_size - o_lap), l_size, ms.shape[1]),
                                  strides=(ms.strides[0] * (l_size - o_lap), ms.strides[0], ms.strides[1]))
    ms = ms[:-1, :, :]

    vs = stride_tricks.as_strided(vs, shape=(vs.shape[0] / (l_size - o_lap), l_size, vs.shape[1]),
                                  strides=(vs.strides[0] * (l_size - o_lap), vs.strides[0], vs.strides[1]))
    vs = vs[:-1, :, :]

    bk = stride_tricks.as_strided(bk, shape=(bk.shape[0] / (l_size - o_lap), l_size, bk.shape[1]),
                                  strides=(bk.strides[0] * (l_size - o_lap), bk.strides[0], bk.strides[1]))
    bk = bk[:-1, :, :]

    b_trim_frame = (ms.shape[0] % bsize)
    if b_trim_frame != 0:
        ms = ms[:-b_trim_frame, :, :]
        vs = vs[:-b_trim_frame, :, :]
        bk = bk[:-b_trim_frame, :, :]

    return ms, vs, bk


def get_data(current_set, set_size, wsz=2049, N=4096, hop=384, T=100, L=20, B=16):
    """
        Method to acquire training data. The STFT analysis is included.
        Args:
            current_set      : (int)       An integer denoting the current training set.
            set_size         : (int)       The amount of files a set has.
            wsz              : (int)       Window size in samples.
            N                : (int)       The FFT size.
            hop              : (int)       Hop size in samples.
            T                : (int)       Length of the time-sequence.
            L                : (int)       Number of context frames from the time-sequence.
            B                : (int)       Batch size.

        Returns:
            ms_train        :  (3D Array)  Mixture magnitude training data, for the current set.
            vs_train        :  (3D Array)  Singing voice magnitude training data, for the current set.

    """

    # Generate full paths for dev and test
    dev_list = sorted(os.listdir(dataset_path + foldersList[0]))
    dev_list = [dataset_path + foldersList[0] + '/' + i for i in dev_list]

    # Current lists for training
    c_train_mlist = dev_list[(current_set - 1) * set_size: current_set * set_size]

    for index in range(len(c_train_mlist)):
        # Reading
        print(c_train_mlist[index])
        vox, _ = Io.wavRead(os.path.join(c_train_mlist[index], keywords[3]), mono=False)
        mix, _ = Io.wavRead(os.path.join(c_train_mlist[index], keywords[4]), mono=False)
        bkg = mix - vox

        # STFT Analysing
        ms_seg, _ = tf.TimeFrequencyDecomposition.STFT(np.sum(mix, axis=-1)*0.5, tf.hamming(wsz, True), N, hop)
        vs_seg, _ = tf.TimeFrequencyDecomposition.STFT(np.sum(vox, axis=-1)*0.5, tf.hamming(wsz, True), N, hop)
        bk_seg, _ = tf.TimeFrequencyDecomposition.STFT(np.sum(bkg, axis=-1)*0.5, tf.hamming(wsz, True), N, hop)

        # Remove null frames
        ms_seg = ms_seg[3:-3, :]
        vs_seg = vs_seg[3:-3, :]
        bk_seg = bk_seg[3:-3, :]

        # Stack some spectrograms and fit
        if index == 0:
            ms_train = ms_seg
            vs_train = vs_seg
            bk_train = bk_seg
        else:
            ms_train = np.vstack((ms_train, ms_seg))
            vs_train = np.vstack((vs_train, vs_seg))
            bk_train = np.vstack((bk_train, bk_seg))

    # Data preprocessing
    # Freeing up some memory
    ms_seg = None
    vs_seg = None

    # Learning the filtering process
    mask = Fm(ms_train, vs_train, bk_train, [], [], alpha=2., method='IRM')
    vs_train = mask()

    # Yet another memory free-up
    mask = None
    ms_train, vs_train, _ = prepare_overlap_sequences(ms_train.clip(max=1.),
                                                      vs_train.clip(max=1.),
                                                      ms_train.clip(max=1.), T, L*2, B)

    return ms_train, vs_train


def get_data_for_pm(current_set, set_size, wsz=2049, N=4096, hop=384, T=100, L=20, B=16):
    """
        Method to acquire training data. The STFT analysis is included.
        Args:
            current_set      : (int)       An integer denoting the current training set.
            set_size         : (int)       The amount of files a set has.
            wsz              : (int)       Window size in samples.
            N                : (int)       The FFT size.
            hop              : (int)       Hop size in samples.
            T                : (int)       Length of the time-sequence.
            L                : (int)       Number of context frames from the time-sequence.
            B                : (int)       Batch size.

        Returns:
            None             : Just saves the masking threshold tensors.

    """

    # Generate full paths for dev and test
    dev_list = sorted(os.listdir(dataset_path + foldersList[0]))
    dev_list = [dataset_path + foldersList[0] + '/' + i for i in dev_list]

    # Current lists for training
    c_train_mlist = dev_list[(current_set - 1) * set_size: current_set * set_size]

    for index in range(len(c_train_mlist)):
        # Reading
        print(c_train_mlist[index])
        vox, _ = Io.wavRead(os.path.join(c_train_mlist[index], keywords[3]), mono=False)
        mix, _ = Io.wavRead(os.path.join(c_train_mlist[index], keywords[4]), mono=False)
        bkg = mix - vox

        # STFT Analysing
        ms_seg, _ = tf.TimeFrequencyDecomposition.STFT(np.sum(mix, axis=-1) * 0.5, tf.hamming(wsz, True), N, hop)
        vs_seg, _ = tf.TimeFrequencyDecomposition.STFT(np.sum(vox, axis=-1) * 0.5, tf.hamming(wsz, True), N, hop)
        bk_seg, _ = tf.TimeFrequencyDecomposition.STFT(np.sum(bkg, axis=-1) * 0.5, tf.hamming(wsz, True), N, hop)

        # Remove null frames
        ms_seg = ms_seg[3:-3, :]
        vs_seg = vs_seg[3:-3, :]
        bk_seg = bk_seg[3:-3, :]

        # Stack some spectrograms and fit
        if index == 0:
            ms_train = ms_seg
            vs_train = vs_seg
            bk_train = bk_seg
        else:
            ms_train = np.vstack((ms_train, ms_seg))
            vs_train = np.vstack((vs_train, vs_seg))
            bk_train = np.vstack((bk_train, bk_seg))

            # Data preprocessing
            # Freeing up some memory
    ms_seg = None
    vs_seg = None

    # Learning the filtering process
    mask = Fm(ms_train, vs_train, bk_train, [], [], alpha=2., method='IRM')
    vs_train = mask()

    # Yet another memory free-up
    mask = None

    # Acquire Masking Threshold
    masking_threshold = pm.maskingThreshold(vs_train)

    # Inverse the filter of masking threshold
    masking_threshold = mm/(masking_threshold + 1e-4)

    # Prepare overlapping sequences
    ms_train, vs_train, masking_threshold = prepare_overlap_sequences(ms_train.clip(max=1.),
                                                                      (vs_train + masking_threshold).clip(max=1.),
                                                                      masking_threshold, T, L*2, B)

    return ms_train, vs_train, masking_threshold


def test_nnet(nnet, seqlen=100, olap=40, wsz=2049, N=4096, hop=384, B=16):
    """
        Method to test the model on some data. Writes the outcomes in ".wav" format and.
        stores them under the defined results path.
        Args:
            nnet             : (List)      A list containing the Pytorch modules of the skip-filtering model.
            seqlen           : (int)       Length of the time-sequence.
            olap             : (int)       Overlap between spectrogram time-sequences
                                           (to recover the missing information from the context information).
            wsz              : (int)       Window size in samples.
            N                : (int)       The FFT size.
            hop              : (int)       Hop size in samples.
            B                : (int)       Batch size.
    """
    nnet[0].eval()
    nnet[1].eval()
    nnet[2].eval()
    nnet[3].eval()
    L = olap/2
    w = tf.hamming(wsz, True)
    #x, fs = Io.wavRead('/home/mis/Documents/Python/Projects/SourceSeparation/testFiles/supreme_test.wav', mono=True)
    #bb_set = '/mnt/IDMT-WORKSPACE/DATA-STORE/mis/Datasets/BeachBoysTestSet/'
    bb_set = '/home/mis/Documents/Python/Projects/SourceSeparation/testFiles/'

    import matplotlib.pyplot as plt
    for item in os.listdir(bb_set):

        if item.endswith('.wav'):
            x, fs = Io.wavRead(os.path.join(bb_set, item), mono=True)
            #x = x[:30*fs]

            mx, px = tf.TimeFrequencyDecomposition.STFT(x, w, N, hop)
            mx, px, _ = prepare_overlap_sequences(mx, px, mx, seqlen, olap, B)
            vs_out = np.zeros((mx.shape[0], seqlen-olap, wsz), dtype=np.float32)
            mx_in = np.zeros((mx.shape[0], seqlen - olap, wsz), dtype=np.float32)
            H_enc_out = np.zeros((mx.shape[0], seqlen - olap, 2050), dtype=np.float32)
            H_dec_out = np.zeros((mx.shape[0], seqlen - olap, 2050), dtype=np.float32)

            for batch in xrange(mx.shape[0]/B):
                # Mixture to Singing voice
                H_enc = nnet[0](mx[batch * B: (batch+1)*B, :, :])
                H_j_dec = nnet[1](H_enc)
                vs_hat, mask = nnet[2](H_j_dec, mx[batch * B: (batch+1)*B, :, :])
                y_out = nnet[3](vs_hat)
                mx_in[batch * B: (batch+1)*B, :, :] = mx[batch * B: (batch+1)*B, L:-L, :]

                H_enc_out[batch * B: (batch + 1) * B, :, :] = torch.tanh(H_enc).data.cpu().numpy()

                H_dec_out[batch * B: (batch + 1) * B, :, :] = np.clip(H_j_dec.data.cpu().numpy(), 0., 1.)
                vs_out[batch * B: (batch + 1) * B, :, :] = y_out.data.cpu().numpy()

            vs_out.shape = (vs_out.shape[0]*vs_out.shape[1], wsz)
            H_enc_out.shape = (H_enc_out.shape[0] * H_enc_out.shape[1], 2050)
            H_dec_out.shape = (H_dec_out.shape[0] * H_dec_out.shape[1], 2050)
            mx_in.shape = (mx_in.shape[0] * mx_in.shape[1], wsz)


            """
            # ploting
            plt.figure(1)
            plt.imshow(mx_in.T, aspect='auto', origin='lower')
            plt.figure(2)
            plt.imshow(H_enc_out.T, aspect='auto', origin='lower')
            plt.figure(3)
            plt.imshow(H_dec_out.T, aspect='auto', origin='lower')
            plt.figure(4)
            plt.imshow(vs_out.T, aspect='auto', origin='lower')
            plt.show(block=False)
            print('end of printing')
            """

            if olap == 1:
                mx = np.ascontiguousarray(mx, dtype=np.float32)
                px = np.ascontiguousarray(px, dtype=np.float32)
            else:
                mx = np.ascontiguousarray(mx[:, olap/2:-olap/2, :], dtype=np.float32)
                px = np.ascontiguousarray(px[:, olap/2:-olap/2, :], dtype=np.float32)

            mx.shape = (mx.shape[0]*mx.shape[1], wsz)
            px.shape = (px.shape[0]*px.shape[1], wsz)

            # Approximated sources
            # Iterative G-L algorithm
            for GLiter in range(1):
                y_recb = tf.TimeFrequencyDecomposition.iSTFT(vs_out, px, wsz, hop, True)
                _, px = tf.TimeFrequencyDecomposition.STFT(y_recb, tf.hamming(wsz, True), N, hop)

            x = x[olap/2 * hop:]

            Io.audioWrite(y_recb, 44100, 16, item[:-4]+'_sv_est.mp3', 'mp3')
            Io.audioWrite(x[:len(y_recb)], 44100, 16, item[:-4]+'_mix.mp3', 'mp3')

    return None


# EOF
