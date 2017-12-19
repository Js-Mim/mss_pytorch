# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch
import torch.nn as nn
from torch.autograd import Variable


class BiGRUEncoder(nn.Module):

    """ Class that builds skip-filtering
        connections neural network.
        Encoder part.
    """

    def __init__(self, B, T, N, F, L):
        """
        Constructing blocks of the model based
        on the sparse skip-filtering connections.
        Args :
            B      : (int) Batch size
            T      : (int) Length of the time-sequence.
            N      : (int) Original dimensionallity of the input.
            F      : (int) Dimensionallity of the input
                               (Amount of frequency sub-bands).
            L      : (int) Length of the half context time-sequence.
        """
        super(BiGRUEncoder, self).__init__()
        self._B = B
        self._T = T
        self._N = N
        self._F = F
        self._L = L
        self._alpha = 1.

        # Bi-GRU Encoder
        self.gruEncF = nn.GRUCell(self._F, self._F)
        self.gruEncB = nn.GRUCell(self._F, self._F)

        # Initialize the weights
        self.initialize_encoder()

    def initialize_encoder(self):
        """
            Manual weight/bias initialization.
        """
        nn.init.orthogonal(self.gruEncF.weight_hh)
        nn.init.xavier_normal(self.gruEncF.weight_ih)
        self.gruEncF.bias_hh.data.zero_()
        self.gruEncF.bias_ih.data.zero_()

        nn.init.orthogonal(self.gruEncB.weight_hh)
        nn.init.xavier_normal(self.gruEncB.weight_ih)
        self.gruEncB.bias_hh.data.zero_()
        self.gruEncB.bias_ih.data.zero_()
        print('Initialization of the encoder done...')

        return None

    def forward(self, input_x):

        if torch.has_cudnn:
            # Initialization of the hidden states
            h_t_fr = Variable(torch.zeros(self._B, self._F).cuda(), requires_grad=False)
            h_t_bk = Variable(torch.zeros(self._B, self._F).cuda(), requires_grad=False)
            H_enc = Variable(torch.zeros(self._B, self._T - (2 * self._L), 2 * self._F).cuda(), requires_grad=False)

            # Input is of the shape : (B (batches), T (time-sequence), N(frequency sub-bands))
            # Cropping some "un-necessary" frequency sub-bands
            cxin = Variable(torch.pow(torch.from_numpy(input_x[:, :, :self._F]).cuda(), self._alpha))

        else:
            # Initialization of the hidden states
            h_t_fr = Variable(torch.zeros(self._B, self._F), requires_grad=False)
            h_t_bk = Variable(torch.zeros(self._B, self._F), requires_grad=False)
            H_enc = Variable(torch.zeros(self._B, self._T - (2 * self._L), 2 * self._F), requires_grad=False)

            # Input is of the shape : (B (batches), T (time-sequence), N(frequency sub-bands))
            # Cropping some "un-necessary" frequency sub-bands
            cxin = Variable(torch.pow(torch.from_numpy(input_x[:, :, :self._F]), self._alpha))

        for t in range(self._T):
            # Bi-GRU Encoding
            h_t_fr = self.gruEncF((cxin[:, t, :]), h_t_fr)
            h_t_bk = self.gruEncB((cxin[:, self._T - t - 1, :]), h_t_bk)
            # Residual connections
            h_t_fr += cxin[:, t, :]
            h_t_bk += cxin[:, self._T - t - 1, :]

            # Remove context and concatenate
            if (t >= self._L) and (t < self._T - self._L):
                h_t = torch.cat((h_t_fr, h_t_bk), dim=1)
                H_enc[:, t - self._L, :] = h_t

        return H_enc


class Decoder(nn.Module):

    """ Class that builds skip-filtering
        connections neural network.
        Decoder part.
    """

    def __init__(self, B, T, N, F, L, infr):
        """
        Constructing blocks of the model based
        on the sparse skip-filtering connections.
        Args :
            B      : (int) Batch size
            T      : (int) Length of the time-sequence.
            N      : (int) Original dimensionallity of the input.
            F      : (int) Dimensionallity of the input
                           (Amount of frequency sub-bands).
            L      : (int) Length of the half context time-sequence.
            infr   : (bool)If the decoder uses recurrent inference or not.
        """
        super(Decoder, self).__init__()
        self._B = B
        self._T = T
        self._N = N
        self._F = F
        self._L = L
        if infr:
            self._gruout = 2*self._F
        else:
            self._gruout = self._F

        # GRU Decoder
        self.gruDec = nn.GRUCell(2*self._F, self._gruout)

        # Initialize the weights
        self.initialize_decoder()

    def initialize_decoder(self):
        """
            Manual weight/bias initialization.
        """
        nn.init.orthogonal(self.gruDec.weight_hh)
        nn.init.xavier_normal(self.gruDec.weight_ih)
        self.gruDec.bias_hh.data.zero_()
        self.gruDec.bias_ih.data.zero_()

        print('Initialization of the decoder done...')
        return None

    def forward(self, H_enc):
        if torch.has_cudnn:
            # Initialization of the hidden states
            h_t_dec = Variable(torch.zeros(self._B, self._gruout).cuda(), requires_grad=False)

            # Initialization of the decoder output
            H_j_dec = Variable(torch.zeros(self._B, self._T - (self._L * 2), self._gruout).cuda(), requires_grad=False)

        else:
            # Initialization of the hidden states
            h_t_dec = Variable(torch.zeros(self._B, self._gruout), requires_grad=False)

            # Initialization of the decoder output
            H_j_dec = Variable(torch.zeros(self._B, self._T - (self._L * 2), self._gruout), requires_grad=False)

        for ts in range(self._T - (self._L * 2)):
            # GRU Decoding
            h_t_dec = self.gruDec(H_enc[:, ts, :], h_t_dec)
            H_j_dec[:, ts, :] = h_t_dec

        return H_j_dec


class SparseDecoder(nn.Module):

    """ Class that builds skip-filtering
        connections neural network.
        Decoder part.
    """

    def __init__(self, B, T, N, F, L, ifnr=True):
        """
        Constructing blocks of the model based
        on the sparse skip-filtering connections.
        Args :
            B      : (int) Batch size
            T      : (int) Length of the time-sequence.
            N      : (int) Original dimensionallity of the input.
            F      : (int) Dimensionallity of the input
                           (Amount of frequency sub-bands).
            L      : (int) Length of the half context time-sequence.
            infr   : (bool)If the GRU decoder used recurrent inference or not.
        """
        super(SparseDecoder, self).__init__()
        self._B = B
        self._T = T
        self._N = N
        self._F = F
        self._L = L

        # FF Sparse Decoder
        if ifnr:
            self.ffDec = nn.Linear(2*self._F, self._N)
        else:
            self.ffDec = nn.Linear(self._F, self._N)

        # Initialize the weights
        self.initialize_decoder()

        # Additional functions
        self.relu = nn.ReLU()

    def initialize_decoder(self):
        """
            Manual weight/bias initialization.
        """
        nn.init.xavier_normal(self.ffDec.weight)
        self.ffDec.bias.data.zero_()

        print('Initialization of the sparse decoder done...')
        return None

    def forward(self, H_j_dec, input_x):
        if torch.has_cudnn:
            # Input is of the shape : (B, T, N)
            input_x = Variable(torch.from_numpy(input_x[:, self._L:-self._L, :]).cuda(), requires_grad=True)

        else:
            # Input is of the shape : (B, T, N)
            # Cropping some "un-necessary" frequency sub-bands
            input_x = Variable(torch.from_numpy(input_x[:, self._L:-self._L, :]), requires_grad=True)

        # Decode/Sparsify mask
        mask_t1 = self.relu(self.ffDec(H_j_dec))
        # Apply skip-filtering connections
        Y_j = torch.mul(mask_t1, input_x)

        return Y_j, mask_t1


class SourceEnhancement(nn.Module):

    """ Class that builds the source enhancement
        module of the skip-filtering connections
        neural network. This could be used for
        recursive inference.
    """

    def __init__(self, B, T, N, F, L):
        """
        Constructing blocks of the model based
        on the sparse skip-filtering connections.
        Args :
            B            : (int) Batch size
            T            : (int) Length of the time-sequence.
            N            : (int) Original dimensionallity of the input.
            F            : (int) Dimensionallity of the input
                                 (Amount of frequency sub-bands).
            L            : (int) Length of the half context time-sequence.
        """
        super(SourceEnhancement, self).__init__()
        self._B = B
        self._T = T
        self._N = N
        self._F = F
        self._L = L

        # FF Source Enhancement Layer
        self.ffSe_enc = nn.Linear(self._N, self._N/2)
        self.ffSe_dec = nn.Linear(self._N/2, self._N)

        # Initialize the weights
        self.initialize_module()

        # Additional functions
        self.relu = nn.ReLU()

    def initialize_module(self):
        """
            Manual weight/bias initialization.
        """
        nn.init.xavier_normal(self.ffSe_dec.weight)
        self.ffSe_dec.bias.data.zero_()
        nn.init.xavier_normal(self.ffSe_enc.weight)
        self.ffSe_enc.bias.data.zero_()
        print('Initialization of the source enhancement module done...')

        return None

    def forward(self, Y_hat):
        # Enhance Source
        mask_enc_hl = self.relu(self.ffSe_enc(Y_hat))
        mask_t2 = self.relu(self.ffSe_dec(mask_enc_hl))
        # Apply skip-filtering connections
        Y_hat_filt = torch.mul(mask_t2, Y_hat)

        return Y_hat_filt


# EOF
