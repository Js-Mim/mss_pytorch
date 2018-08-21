# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'
import musdb
import museval
import torch
import numpy as np
import os
from helpers.io_methods import AudioIO as Io
from helpers.nnet_helpers import prepare_overlap_sequences
from helpers import tf_methods as tf
from modules import cls_sparse_skip_filt as s_s_net

output_dir = '/mnt/IDMT-WORKSPACE/DATA-STORE/mis/TrainingFruitsBackup/skip_filt_MaD_regularizations_costs/KL/'
#output_dir = '/mnt/IDMT-WORKSPACE/DATA-STORE/mis/TrainingFruitsBackup/skip_filt_MaD_regularizations_costs/nmr/'
dataset_dir = '/mnt/IDMT-WORKSPACE/DATA-STORE/mis/Datasets/musdb18/'
method = 'KL'

# Analysis
wsz = 2049  # Window-size
Ns = 4096   # FFT size
hop = 384   # Hop size
fs = 44100  # Sampling frequency

# Parameters
B = 16    # Batch-size
T = 60    # Length of the sequence
N = 2049  # Frequency sub-bands to be processed
F = 1025  # Frequency sub-bands for encoding
L = 10    # Context parameter (2*L frames will be removed)


print('------------   Building model   ------------')
encoder = s_s_net.BiGRUEncoder(B, T, N, F, L)
decoder = s_s_net.Decoder(B, T, N, F, L, infr=True)
sp_decoder = s_s_net.SparseDecoder(B, T, N, F, L)
source_enhancement = s_s_net.SourceEnhancement(B, T, N, F, L)

if torch.has_cudnn:
    print('------------   CUDA Enabled   --------------')
    encoder.cuda()
    decoder.cuda()
    sp_decoder.cuda()
    source_enhancement.cuda()

print('-------  Loading pre-trained model   -------')
print('-------  Loading inference weights  -------')
encoder.load_state_dict(torch.load(os.path.join(output_dir, 'torch_sps_encoder_'+method+'50.pytorch'),
                                   map_location={'cuda:1': 'cuda:0'}))
decoder.load_state_dict(torch.load(os.path.join(output_dir, 'torch_sps_decoder_'+method+'50.pytorch'),
                                   map_location={'cuda:1': 'cuda:0'}))
sp_decoder.load_state_dict(torch.load(os.path.join(output_dir, 'torch_sps_sp_decoder_'+method+'50.pytorch'),
                                      map_location={'cuda:1': 'cuda:0'}))
source_enhancement.load_state_dict(torch.load(os.path.join(output_dir, 'torch_sps_se_'+method+'50.pytorch'),
                                   map_location={'cuda:1': 'cuda:0'}))
print('-------------      Done        -------------')


def test_eval_stereo(track):
    mix = track.audio
    encoder.eval()
    decoder.eval()
    sp_decoder.eval()
    source_enhancement.eval()

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

    # STFT Analysing
    mx_mc, px_mc = tf.TimeFrequencyDecomposition.MCSTFT(mix, tf.hamming(wsz, True), Ns, hop)

    for m_channel in range(mx_mc.shape[0]):
        mx = mx_mc[m_channel, :, :].T
        px = px_mc[m_channel, :, :].T

        # Data reshaping (magnitude and phase)
        mx, px, _ = prepare_overlap_sequences(mx, px, px, T, 2*L, B)

        # The actual "denoising" part
        vx_hat = np.zeros((mx.shape[0], T-L*2, wsz), dtype=np.float32)
        bx_hat = np.zeros((mx.shape[0], T-L*2, wsz), dtype=np.float32)

        for batch in xrange(mx.shape[0]/B):
            H_enc = encoder.forward(mx[batch * B: (batch+1)*B, :, :])

            H_j_dec = decoder.forward(H_enc)

            vs_hat, _ = sp_decoder.forward(H_j_dec, mx[batch * B: (batch+1)*B, :, :])
            y_out = source_enhancement.forward(vs_hat)
            vx_hat[batch * B: (batch + 1) * B, :, :] = y_out.data.cpu().numpy()
            bx_hat[batch * B: (batch + 1) * B, :, :] = np.clip(mx[batch * B: (batch+1)*B, L:-L, :] -
                                                               vx_hat[batch * B: (batch + 1) * B, :, :], 0., 16.)

        # Final reshaping
        vx_hat.shape = (vx_hat.shape[0] * vx_hat.shape[1], wsz)
        bx_hat.shape = (bx_hat.shape[0] * bx_hat.shape[1], wsz)
        mx, px = my_res(mx, px, L, wsz)

        # Time-domain recovery
        sv_hat = tf.TimeFrequencyDecomposition.iSTFT(vx_hat, px, wsz, hop, True)
        bk_hat = tf.TimeFrequencyDecomposition.iSTFT(bx_hat, px, wsz, hop, True)

        if m_channel == 0:
            out_sv_hat = sv_hat
            out_bk_hat = bk_hat
        else:
            out_sv_hat = np.vstack((out_sv_hat, sv_hat)).T
            out_bk_hat = np.vstack((out_bk_hat, bk_hat)).T

    # Removing the samples that no estimation exists
    out_sv_hat = np.pad(out_sv_hat, [(L * hop, 0), (0, 0)], mode='constant')
    out_bk_hat = np.pad(out_bk_hat, [(L * hop, 0), (0, 0)], mode='constant')

    # Background music estimation
    if len(mix[:, 0]) > len(out_sv_hat[:, 0]):
        zp = len(mix[:, 0]) - len(out_sv_hat[:, 0])
        out_sv_hat = np.pad(out_sv_hat, [(0, zp), (0, 0)], mode='constant')
        out_bk_hat = np.pad(out_bk_hat, [(0, zp), (0, 0)], mode='constant')
        #out_bk_hat = mix - out_sv_hat
    else:
        out_sv_hat = out_sv_hat[:len(mix[:, 0]), :]
        out_bk_hat = out_bk_hat[:len(mix[:, 0]), :]
        #out_bk_hat = mix - out_sv_hat

    estimates = {
        'vocals': out_sv_hat,
        'accompaniment': out_bk_hat
    }


    # Writing
    os.makedirs(os.path.join(output_dir, method, track.name))
    Io.wavWrite(estimates['vocals'], 44100, 16, os.path.join(output_dir, method, track.name,
                                                             'vocals.wav'))
    Io.wavWrite(estimates['accompaniment'], 44100, 16, os.path.join(output_dir, method, track.name,
                                                                    'accompaniment.wav'))

    return estimates


def estimate_and_evaluate(track):
    print(track.name)

    # generate your estimates
    estimates = test_eval_stereo(track)

    # Evaluate using museval
    scores = museval.eval_mus_track(
        track, estimates, output_dir=os.path.join(output_dir, method)
    )

    # print nicely formatted mean scores
    print(scores)

    # return estimates as usual
    return estimates


if __name__ == '__main__':
    mus = musdb.DB(root_dir=dataset_dir, is_wav=True)
    tracks = mus.load_mus_tracks(subsets='test')
    mus.run(estimate_and_evaluate, cpus=7, tracks=tracks)


# EOF
