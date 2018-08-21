# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from helpers import visualize, nnet_helpers
from torch.autograd import Variable
from modules import cls_sparse_skip_filt as s_s_net
from losses import loss_functions
from torch.optim.lr_scheduler import ReduceLROnPlateau as RedLR

# path definition
output_dir = '/mnt/IDMT-WORKSPACE/DATA-STORE/mis/TrainingFruitsBackup/skip_filt_MaD_regularizations_costs/NMR/'
mt_path = '/mnt/IDMT-WORKSPACE/DATA-STORE/mis/Datasets/musdb18/mt_train/'


def main(training):
    """
        The main function to train and test.
    """
    # Reproducible results
    np.random.seed(218)
    torch.manual_seed(218)
    torch.cuda.manual_seed(218)
    # Torch model
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Analysis
    wsz = 2049                  # Window-size
    Ns = 4096                   # FFT size
    hop = 384                   # Hop size
    fs = 44100                  # Sampling frequency

    # Parameters
    B = 16                      # Batch-size
    T = 60                      # Length of the sequence
    N = 2049                    # Frequency sub-bands to be processed
    F = 1025                    # Frequency sub-bands for encoding
    L = 10                      # Context parameter (2*L frames will be removed)
    epochs = 100                # Epochs
    init_lr = 1e-4              # Initial learning rate
    mnorm = 1.5	                # L2-based norm clipping

    # Data (Pre-defined)
    totTrainFiles = 100
    numFilesPerTr = 4

    print('------------   Building model   ------------')
    encoder = s_s_net.BiGRUEncoder(B, T, N, F, L)
    decoder = s_s_net.Decoder(B, T, N, F, L, infr=True)
    sp_decoder = s_s_net.SparseDecoder(B, T, N, F, L)
    source_enhancement = s_s_net.SourceEnhancement(B, T, N, F, L)

    encoder.train(mode=True)
    decoder.train(mode=True)
    sp_decoder.train(mode=True)
    source_enhancement.train(mode=True)

    if torch.has_cudnn:
        print('------------   CUDA Enabled   --------------')
        encoder.cuda()
        decoder.cuda()
        sp_decoder.cuda()
        source_enhancement.cuda()

    # Defining objectives
    rec_criterion = loss_functions.nmr  # Reconstruction criterion

    optimizer = optim.Adam(list(encoder.parameters()) +
                           list(decoder.parameters()) +
                           list(sp_decoder.parameters()) +
                           list(source_enhancement.parameters()),
                           lr=init_lr
                           )

    scheduler = RedLR(optimizer, 'min', factor=0.8, patience=3, verbose=True)
    if training:
        win_viz, winb_viz = visualize.init_visdom()
        batch_loss = []
        # Over epochs
        batch_index = 0
        for epoch in range(epochs):
            print('Epoch: ' + str(epoch + 1))
            epoch_loss = []
            # Over the set of files
            for index in range(totTrainFiles / numFilesPerTr):
                # Get Data
                ms, vs = nnet_helpers.get_data(index + 1, numFilesPerTr, wsz, Ns, hop, T, L, B)
                mt = np.load(os.path.join(mt_path, 'masking_threshold_pt_' + str(index) + '.npy'))

                # Shuffle data
                shf_indices = np.random.permutation(ms.shape[0])
                ms = ms[shf_indices]
                vs = vs[shf_indices]
                mt = mt[shf_indices]

                # Over batches
                for batch in tqdm(range(ms.shape[0] / B)):
                    # Mixture to Singing voice
                    H_enc = encoder.forward(ms[batch * B: (batch + 1) * B, :, :])
                    H_j_dec = decoder.forward(H_enc)
                    vs_hat_b = sp_decoder.forward(H_j_dec, ms[batch * B: (batch + 1) * B, :, :])[0]
                    vs_hat_b_filt = source_enhancement.forward(vs_hat_b)

                    # Loss
                    loss_den = rec_criterion(Variable(
                        torch.from_numpy(vs[batch * B: (batch + 1) * B, L:-L, :]).cuda()),
                        vs_hat_b_filt,
                        Variable(torch.from_numpy(mt[batch * B: (batch + 1) * B, L:-L, :]).cuda())
                    )

                    loss_mask = rec_criterion(
                        Variable(torch.from_numpy(vs[batch * B: (batch + 1) * B, L:-L, :]).cuda()),
                        vs_hat_b,
                        Variable(torch.from_numpy(mt[batch * B: (batch + 1) * B, L:-L, :]).cuda())
                        )

                    # Accumulated reconstruction loss
                    loss = loss_den + loss_mask

                    # Store loss for display and scheduler
                    batch_loss += [loss.data[0]]
                    epoch_loss += [loss.data[0]]

                    # Update graphs
                    win_viz = visualize.viz.line(X=np.arange(batch_index, batch_index + 1),
                                                 Y=np.reshape(batch_loss[batch_index], (1,)),
                                                 win=win_viz, update='append')

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(list(encoder.parameters()) +
                                                  list(decoder.parameters()) +
                                                  list(sp_decoder.parameters()) +
                                                  list(source_enhancement.parameters()),
                                                  max_norm=mnorm, norm_type=2)
                    optimizer.step()

                    batch_index += 1

            if epoch + 1 >= 10:
                scheduler.step((torch.from_numpy(np.asarray(np.mean(epoch_loss)))))

            if (epoch + 1) % 50 == 0:
                print('------------   Saving model   ------------')
                torch.save(encoder.state_dict(), 'results/torch_sps_encoder_NMR' + str(epoch + 1) + '.pytorch')
                torch.save(decoder.state_dict(), 'results/torch_sps_decoder_NMR' + str(epoch + 1) + '.pytorch')
                torch.save(sp_decoder.state_dict(), 'results/torch_sps_sp_decoder_NMR' + str(epoch + 1) + '.pytorch')
                torch.save(source_enhancement.state_dict(), 'results/torch_sps_se_NMR' + str(epoch + 1) + '.pytorch')
                print('------------       Done       ------------')

    else:
        print('-------  Loading pre-trained model   -------')
        print('-------  Loading inference weights  -------')
        encoder.load_state_dict(
            torch.load(os.path.join(output_dir, 'torch_sps_encoder_NMR50.pytorch'),
                       map_location={'cuda:1': 'cuda:0'}))
        decoder.load_state_dict(
            torch.load(os.path.join(output_dir, 'torch_sps_decoder_NMR50.pytorch'),
                       map_location={'cuda:1': 'cuda:0'}))
        sp_decoder.load_state_dict(
            torch.load(os.path.join(output_dir, 'torch_sps_sp_decoder_NMR50.pytorch'),
                       map_location={'cuda:1': 'cuda:0'}))
        source_enhancement.load_state_dict(
            torch.load(os.path.join(output_dir, 'torch_sps_se_NMR50.pytorch'),
                       map_location={'cuda:1': 'cuda:0'}))
        print('-------------      Done        -------------')

    return encoder, decoder, sp_decoder, source_enhancement


if __name__ == '__main__':

    with torch.cuda.device(0):
        do_training = True              # Whether to train or test the trained model (requires the optimized parameters)
        sfiltnet = main(do_training)
        #print('-------------     DNN-Test     -------------')
        #nnet_helpers.test_nnet(sfiltnet, 60, 10*2, 2049, 4096, 384, 16)
        #print('-------------       Done       -------------')

# EOF
