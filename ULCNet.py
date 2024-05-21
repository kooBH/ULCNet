import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
Channelwise Feature Orientation

Shetu, Shrishti Saha, et al. "Ultra Low Complexity Deep Learning Based Noise Suppression." arXiv preprint arXiv:2312.08132 (2023).
<-
Liu, Haohe, et al. "Channel-wise subband input for better voice and accompaniment separation on high resolution music." arXiv preprint arXiv:2008.05216 (2020).
"""
# ChannelwiseReorientation
class CR(nn.Module) : 
    def __init__(self, n_band, overlap=1/3, **kwargs):
        super(CR, self).__init__()
        self.n_band = n_band
        self.overlap = overlap
        """
        if type_window == "None" :
            self.window = torch.tensor(1.0)
        elif type_window == "Rectengular" : 
            self.window = torch.kaiser_window(window_length ,beta = 0.0)
        elif type_window == "Hanning":
            self.window = torch.hann_window(window_length)
        else :
            raise NotImplementedError
        """

    def forward(self,x):
        idx = 0

        B,C,T,F = x.shape
        n_freq = x.shape[3]
        sz_band = n_freq/(self.n_band*(1-self.overlap))
        sz_band = int(np.ceil(sz_band))
        y = torch.zeros(B,self.n_band*C,T,sz_band).to(x.device)
        
        for i in range(self.n_band):
            if idx+sz_band > F :
                sz_band = F - idx
            y[:,i*C:(i+1)*C,:,:sz_band] = x[:,:,:,idx:idx+sz_band]
            n_idx = idx + int(sz_band*(1-self.overlap))
            idx = n_idx
        return y

class Encoders(nn.Module):
    def __init__(self, in_channels=8):
        super(Encoders,self).__init__()

        self.conv11 = nn.Conv2d(in_channels,in_channels,(1,3),groups = in_channels)
        self.bn11 = nn.BatchNorm2d(in_channels)
        self.conv12 = nn.Conv2d(in_channels,32,(1,1))
        self.bn12 = nn.BatchNorm2d(32)

        self.conv21 = nn.Conv2d(32,32,(1,3),padding=(0,1),groups=32)
        self.bn21 = nn.BatchNorm2d(32)
        self.conv22 = nn.Conv2d(32,64,(1,1))
        self.bn22 = nn.BatchNorm2d(64)

        self.conv31 = nn.Conv2d(64,64,(1,3),padding=(0,1),groups=64)
        self.bn31 = nn.BatchNorm2d(64)
        self.conv32 = nn.Conv2d(64,96,(1,1))
        self.bn32 = nn.BatchNorm2d(96)

        self.conv41 = nn.Conv2d(96,96,(1,3),padding=(0,1),groups=96)
        self.bn41 = nn.BatchNorm2d(96)
        self.conv42 = nn.Conv2d(96,128,(1,1))
        self.bn42 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d((1,2))

    def forward(self,x):
        # Except for thr first Conv layers downsampling is achieved through max-pooing at a factor of
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))

        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = self.maxpool(x)

        return x

class FC(nn.Module):
    def __init__(self,in_dim = 256):
        super(FC,self).__init__()
        self.FC1 = nn.Linear(256,257)
        self.FC2 = nn.Linear(257,257)

    def forward(self,x):
        x = F.relu(self.FC1(x))
        x = F.sigmoid(self.FC2(x))

        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(2,32,(1,3),padding=(0,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,32,(1,3),padding=(0,1))
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


"""
NOTE : No mention about the activation function and the normalization method.
=> default : ReLU and Batch Normalization, sigmoid on mask module
"""
class ULCNet(nn.Module):
    def __init__(self) :
        super(ULCNet,self).__init__()
        """
        CR : 
            1.5kHz resolution(48 ferq bin)
            overlap 0.33
            => 8 band
        """
        self.CR = CR(8,overlap=0.33)
        """
        Conv :
            4 separable conv layers
                (1 x 3) kernel
                channels { 32, 64, 96, 128} 
                except for thr first Conv layers
                downsampling is achieved through max-pooing at a factor of 2.
        """
        self.encoders = Encoders()

        """
        FGRU :
            bidirectional
            64 units
        """
        self.FGRU = nn.GRU(128,64,bidirectional=True,batch_first=True)

        """
        Pointwise Conv : 
            64 filters
        """
        self.pointwise1 = nn.Conv2d(128,64,(1,1))

        """
        Subband GRU : 
        2-subband TGRU blocks with 2 layers with 128 units
        """
        self.TGRU1 = nn.GRU(160,128,num_layers=2,bidirectional=False,batch_first=True)
        self.TGRU2 = nn.GRU(160,128,num_layers=2,bidirectional=False,batch_first=True)

        """
        FC : 
            2-layers with 257 neurons
        """
        self.FC = FC()

        """
        2nd Stage

        CNN
            two 2D Convs with 32 filter
            (1,3) kernel
        """
        self.CNN = CNN()

        """
        Pointwise 
            2 output channels
        """
        self.pointwise2 = nn.Conv2d(32,2,(1,1))

    """
    x : TF domain after STFT.
    window length : 32 ms
    16 ms hop size
    n_fft  = 512
    """
    def forward(self,x):
        # x : [B,C,T,F,2]
        x_r = x[:,:,:,:,0]
        x_i = x[:,:,:,:,1]
        x_m = torch.sqrt(x_r**2 + x_i**2)
        x_p = torch.atan2(x_i,x_r)

        ### 1st Stage ###

        # CR
        z = self.CR(x_m)

        # Conv
        z = self.encoders(z)

        # FGRU
        # [B,C,T,F] -> [B*T,F,C]
        nB, nC, nT, nF = z.shape
        z = z.permute(0,2,3,1).contiguous()
        z = torch.reshape(z,(-1,nF,nC)).contiguous()
        z, _ = self.FGRU(z)
        # [B*T,F,C] -> [B,C,T,F]
        z = torch.reshape(z,(nB,nT,nF,nC)).contiguous()
        z = z.permute(0,3,1,2).contiguous()

        # Pointwise COnv
        z = self.pointwise1(z)

        # Subband Splitting
        # Flattening
        nB, nC, nT, nF = z.shape
        z = z.permute(0,2,1,3).contiguous()
        z = torch.reshape(z,(nB,nT,nC*nF)).contiguous()
        # Subband GRU blocks
        z1, h1 = self.TGRU1(z[:,:,:160])
        z2, h2 = self.TGRU2(z[:,:,160:])

        # Featrue Concatenation
        z = F.relu(torch.cat((z1,z2),dim=2))

        # FC Layers
        M_m = self.FC(z)

        # Intermediate Feature Computation
        Y_hat_r = M_m * torch.cos(x_p[:,0])
        Y_hat_i = M_m * torch.sin(x_p[:,0])
        Y_hat = torch.stack((Y_hat_r,Y_hat_i),dim=1)

        ### 2nd Stage ###

        # CNN
        z = self.CNN(Y_hat)

        # Pointwise Conv
        M = F.sigmoid(self.pointwise2(z))

        # Clean Speach Estimation
        s_r = M[:,0] * x_r[:,0]
        s_i = M[:,1] * x_i[:,0]

        z = torch.stack((s_r,s_i),dim=-1)

        # (Optional) Power Law Decomposition

        return z

class ULCNet_helper(nn.Module):
    """
        x : TF domain after STFT.
        window length : 32 ms
        16 ms hop size
        n_fft  = 512
    """
    def __init__(self,
        n_fft = 512,
        n_hop = 256
        ) :
        super(ULCNet_helper,self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.window = torch.hann_window(self.n_fft)

        self.model = ULCNet()

    def forward(self,x):
        B,L = x.shape

        X = torch.stft(x,n_fft = self.n_fft, hop_length=self.n_hop, window =self.window.to(x.device), return_complex=False)

        # [B,C,T,F,2]
        X = torch.permute(X,(0,2,1,3))
        X = torch.unsqueeze(X,1)

        Y = self.model(X)

        Y = Y[...,0] + Y[...,0]*1j
        # [B,C,T,F,2]
        Y = torch.permute(Y,(0,2,1))

        y = torch.istft(Y,self.n_fft, hop_length=self.n_hop, window=self.window.to(x.device),length=L)

        return y


if __name__ == "__main__":
    # B, C, F, T
    #x = torch.rand(2,1,150,257,2)
    x = torch.rand(2,16000)

    #m =  ULCNet()
    m = ULCNet_helper()


    y = m(x)

    print("========")
    print(x.shape)
    print(y.shape)
