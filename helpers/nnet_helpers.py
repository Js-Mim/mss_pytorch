# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
from helpers.io_methods import AudioIO as Io
from helpers.masking_methods import FrequencyMasking as Fm
from mir_eval import separation as bss_eval
from numpy.lib import stride_tricks
from helpers import iterative_inference as it_infer
from helpers import tf_methods as tf
import pickle as pickle
import numpy as np
import os

# definitions
mixtures_path = 'DSD100/Mixtures/'
sources_path = 'DSD100/Sources/'
keywords = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav', 'mixture.wav']
foldersList = ['Dev', 'Test']
save_path = 'results/GRU_sskip_filt/inference_m3_i10plus/'

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
    dev_mixtures_list = sorted(os.listdir(mixtures_path + foldersList[0]))
    dev_mixtures_list = [mixtures_path + foldersList[0] + '/' + i for i in dev_mixtures_list]
    dev_sources_list = sorted(os.listdir(sources_path + foldersList[0]))
    dev_sources_list = [sources_path + foldersList[0] + '/' + i for i in dev_sources_list]

    # Current lists for training
    c_train_slist = dev_sources_list[(current_set - 1) * set_size: current_set * set_size]
    c_train_mlist = dev_mixtures_list[(current_set - 1) * set_size: current_set * set_size]

    for index in range(len(c_train_mlist)):

        # print('Reading:' + c_train_mlist[index])

        # Reading
        vox, _ = Io.wavRead(os.path.join(c_train_slist[index], keywords[3]), mono=False)
        mix, _ = Io.wavRead(os.path.join(c_train_mlist[index], keywords[4]), mono=False)

        # STFT Analysing
        ms_seg, _ = tf.TimeFrequencyDecomposition.STFT(0.5*np.sum(mix, axis=-1), tf.hamming(wsz, True), N, hop)
        vs_seg, _ = tf.TimeFrequencyDecomposition.STFT(0.5*np.sum(vox, axis=-1), tf.hamming(wsz, True), N, hop)

        # Remove null frames
        ms_seg = ms_seg[3:-3, :]
        vs_seg = vs_seg[3:-3, :]

        # Stack some spectrograms and fit
        if index == 0:
            ms_train = ms_seg
            vs_train = vs_seg
        else:
            ms_train = np.vstack((ms_train, ms_seg))
            vs_train = np.vstack((vs_train, vs_seg))

    # Data preprocessing
    # Freeing up some memory
    ms_seg = None
    vs_seg = None

    # Learning the filtering process
    mask = Fm(ms_train, vs_train, ms_train, [], [], alpha=1., method='IRM')
    vs_train = mask()
    vs_train *= 2.
    vs_train = np.clip(vs_train, a_min=0., a_max=1.)
    ms_train = np.clip(ms_train, a_min=0., a_max=1.)
    mask = None
    ms_train, vs_train, _ = prepare_overlap_sequences(ms_train, vs_train, ms_train, T, L*2, B)

    return ms_train, vs_train


def test_eval(nnet, B, T, N, L, wsz, hop):
    """
        Method to test the model on the test data. Writes the outcomes in ".wav" format and.
        stores them under the defined results path. Optionally, it performs BSS-Eval using
        MIREval python toolbox (Used only for comparison to BSSEval Matlab implementation).
        The evaluation results are stored under the defined save path.
        Args:
            nnet             : (List)      A list containing the Pytorch modules of the skip-filtering model.
            B                : (int)       Batch size.
            T                : (int)       Length of the time-sequence.
            N                : (int)       The FFT size.
            L                : (int)       Number of context frames from the time-sequence.
            wsz              : (int)       Window size in samples.
            hop              : (int)       Hop size in samples.
    """
    nnet[0].eval()
    nnet[1].eval()
    nnet[2].eval()
    nnet[3].eval()

    def my_res(mx, vx, L, wsz):
        """
            A helper function to reshape data according
            to the context frame.
        """
        mx = np.ascontiguousarray(mx[:, L:-L, :], dtype=np.float32)
        mx.shape = (mx.shape[0]*mx.shape[1], wsz)
        vx = np.ascontiguousarray(vx[:, L:-L, :], dtype=np.float32)
        vx.shape = (vx.shape[0]*vx.shape[1], wsz)

        return mx, vx

    # Paths for loading and storing the test-set
    # Generate full paths for test
    test_sources_list = sorted(os.listdir(sources_path + foldersList[1]))
    test_sources_list = [sources_path + foldersList[1] + '/' + i for i in test_sources_list]

    # Initializing the containers of the metrics
    sdr = []
    sir = []
    sar = []

    for indx in xrange(len(test_sources_list)):
        print('Reading:' + test_sources_list[indx])
        # Reading
        bass, _ = Io.wavRead(os.path.join(test_sources_list[indx], keywords[0]), mono=False)
        drums, _ = Io.wavRead(os.path.join(test_sources_list[indx], keywords[1]), mono=False)
        oth, _ = Io.wavRead(os.path.join(test_sources_list[indx], keywords[2]), mono=False)
        vox, _ = Io.wavRead(os.path.join(test_sources_list[indx], keywords[3]), mono=False)

        bk_true = np.sum(bass + drums + oth, axis=-1) * 0.5
        mix = np.sum(bass + drums + oth + vox, axis=-1) * 0.5
        sv_true = np.sum(vox, axis=-1) * 0.5

        # STFT Analysing
        mx, px = tf.TimeFrequencyDecomposition.STFT(mix, tf.hamming(wsz, True), N, hop)

        # Data reshaping (magnitude and phase)
        mx, px, _ = prepare_overlap_sequences(mx, px, px, T, 2*L, B)

        # The actual "denoising" part
        vx_hat = np.zeros((mx.shape[0], T-L*2, wsz), dtype=np.float32)

        for batch in xrange(mx.shape[0]/B):
            H_enc = nnet[0](mx[batch * B: (batch+1)*B, :, :])

            H_j_dec = it_infer.iterative_recurrent_inference(nnet[1], H_enc,
                                                             criterion=None, tol=1e-3, max_iter=10)

            vs_hat, mask = nnet[2](H_j_dec, mx[batch * B: (batch+1)*B, :, :])
            y_out = nnet[3](vs_hat)
            vx_hat[batch * B: (batch+1)*B, :, :] = y_out.data.cpu().numpy()

        # Final reshaping
        vx_hat.shape = (vx_hat.shape[0]*vx_hat.shape[1], wsz)
        mx, px = my_res(mx, px, L, wsz)

        # Time-domain recovery
        # Iterative G-L algorithm
        for GLiter in range(10):
            sv_hat = tf.TimeFrequencyDecomposition.iSTFT(vx_hat, px, wsz, hop, True)
            _, px = tf.TimeFrequencyDecomposition.STFT(sv_hat, tf.hamming(wsz, True), N, hop)

        # Removing the samples that no estimation exists
        mix = mix[L*hop:]
        sv_true = sv_true[L*hop:]
        bk_true = bk_true[L*hop:]

        # Background music estimation
        if len(sv_true) > len(sv_hat):
            bk_hat = mix[:len(sv_hat)] - sv_hat
        else:
            bk_hat = mix - sv_hat[:len(mix)]

        # Disk writing for external BSS_eval using DSD100-tools (used in our paper)
        Io.wavWrite(sv_true, 44100, 16, os.path.join(save_path, 'tf_true_sv_' + str(indx) + '.wav'))
        Io.wavWrite(bk_true, 44100, 16, os.path.join(save_path, 'tf_true_bk_' + str(indx) + '.wav'))
        Io.wavWrite(sv_hat, 44100, 16, os.path.join(save_path, 'tf_hat_sv_' + str(indx) + '.wav'))
        Io.wavWrite(bk_hat, 44100, 16, os.path.join(save_path, 'tf_hat_bk_' + str(indx) + '.wav'))
        Io.wavWrite(mix, 44100, 16, os.path.join(save_path, 'tf_mix_' + str(indx) + '.wav'))

        # Internal BSSEval using librosa (just for comparison)
        if len(sv_true) > len(sv_hat):
            c_sdr, _, c_sir, c_sar, _ = bss_eval.bss_eval_images_framewise([sv_true[:len(sv_hat)], bk_true[:len(sv_hat)]],
                                                                           [sv_hat, bk_hat])
        else:
            c_sdr, _, c_sir, c_sar, _ = bss_eval.bss_eval_images_framewise([sv_true, bk_true],
                                                                           [sv_hat[:len(sv_true)], bk_hat[:len(sv_true)]])

        sdr.append(c_sdr)
        sir.append(c_sir)
        sar.append(c_sar)

        # Storing the results iteratively
        pickle.dump(sdr, open(os.path.join(save_path, 'SDR.p'), 'wb'))
        pickle.dump(sir, open(os.path.join(save_path, 'SIR.p'), 'wb'))
        pickle.dump(sar, open(os.path.join(save_path, 'SAR.p'), 'wb'))

    return None


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
    x, fs = Io.wavRead('results/test_files/test.wav', mono=True)

    mx, px = tf.TimeFrequencyDecomposition.STFT(x, w, N, hop)
    mx, px, _ = prepare_overlap_sequences(mx, px, mx, seqlen, olap, B)
    vs_out = np.zeros((mx.shape[0], seqlen-olap, wsz), dtype=np.float32)

    for batch in xrange(mx.shape[0]/B):
        # Mixture to Singing voice
        H_enc = nnet[0](mx[batch * B: (batch+1)*B, :, :])
        H_j_dec = it_infer.iterative_recurrent_inference(nnet[1], H_enc,
                                                         criterion=None, tol=1e-3, max_iter=10)

        vs_hat, mask = nnet[2](H_j_dec, mx[batch * B: (batch+1)*B, :, :])
        y_out = nnet[3](vs_hat)
        vs_out[batch * B: (batch+1)*B, :, :] = y_out.data.cpu().numpy()

    vs_out.shape = (vs_out.shape[0]*vs_out.shape[1], wsz)

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
    for GLiter in range(10):
        y_recb = tf.TimeFrequencyDecomposition.iSTFT(vs_out, px, wsz, hop, True)
        _, px = tf.TimeFrequencyDecomposition.STFT(y_recb, tf.hamming(wsz, True), N, hop)

    x = x[olap/2 * hop:]

    Io.wavWrite(y_recb, 44100, 16, 'results/test_files/test_sv.wav')
    Io.wavWrite(x[:len(y_recb)], 44100, 16, 'results/test_files/test_mix.wav')

    return None

# EOF
