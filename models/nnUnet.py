import torch
import torch.nn as nn


class nnUnet(nn.Module):
    def __init__(self):
        super(nnUnet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(x):
        x = self.encoder(x)
        return self.decoder(x)
                
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = [32, 64, 128, 256, 320]
        self.layer1 = nn.Sequential( 
        CNA(in_channels=c[0], out_channels=c[0], stride = (1,1,1), kernel_size=(1,3,3)),
        CNA(in_channels=c[0], out_channels=c[0], stride = (1,1,1), kernel_size=(1,3,3)),
        DownScale(in_channels = c[0],out_channels = c[1], stride = (1,1,1), kernel_size=(1,3,3))
        )
        self.layer2 = nn.Sequential( 
        CNA(in_channels=c[1], out_channels=c[1], stride = (1,2,2), kernel_size=(3,3,3)),
        CNA(in_channels=c[0], out_channels=c[0], stride = (1,1,1), kernel_size=(3,3,3)),
        DownScale(in_channels = c[1], out_channels = c[2], stride = (1,1,1), kernel_size=(3,3,3))
        )
        self.layer3 = nn.Sequential( 
        CNA(in_channels=c[2], out_channels=c[2], stride = (2,2,2), kernel_size=(3,3,3)),
        CNA(in_channels=c[2], out_channels=c[2], stride = (1,1,1), kernel_size=(3,3,3)),
        DownScale(in_channels = c[2], out_channels = c[3], stride = (1,1,1), kernel_size=(3,3,3))
        )
        self.layer4 = nn.Sequential( 
        CNA(in_channels=c[3], out_channels=c[3], stride = (2,2,2), kernel_size=(3,3,3)),
        CNA(in_channels=c[3], out_channels=c[3], stride = (1,1,1), kernel_size=(3,3,3)),
        DownScale(in_channels = c[3], out_channels = c[4], stride = (1,1,1), kernel_size=(3,3,3))
        )
        self.layer5 = nn.Sequential( 
        CNA(in_channels=c[4], out_channels=c[4], stride = (1,2,2), kernel_size=(3,3,3)),
        CNA(in_channels=c[4], out_channels=c[4], stride = (1,1,1), kernel_size=(3,3,3)),
        DownScale(in_channels = c[4], out_channels = c[4], stride = (1,1,1), kernel_size=(3,3,3))
        )
    def forward(x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = [320, 256, 128, 64, 32]
        self.layer1 = nn.Sequential( 
        TCNA(in_channels=c[0], out_channels=c[0], stride = (1,1,1), kernel_size=(3,3,3)),
        TCNA(in_channels=c[0], out_channels=c[0], stride = (1,1,1), kernel_size=(3,3,3)),
        UpScale(in_channels = c[0],out_channels = c[1] , stride = (1,1,1), kernel_size=(1,3,3))
        )
        self.layer2 = nn.Sequential( 
        TCNA(in_channels=c[1], out_channels=c[1], stride = (1,2,2), kernel_size=(3,3,3)),
        TCNA(in_channels=c[0], out_channels=c[0], stride = (1,1,1), kernel_size=(3,3,3)),
        UpScale(in_channels = c[1], out_channels = c[2], stride = (1,1,1), kernel_size=(3,3,3))
        )
        self.layer3 = nn.Sequential( 
        TCNA(in_channels=c[2], out_channels=c[2], stride = (2,2,2), kernel_size=(3,3,3)),
        TCNA(in_channels=c[2], out_channels=c[2], stride = (1,1,1), kernel_size=(3,3,3)),
        UpScale(in_channels = c[2], out_channels = c[3], stride = (1,1,1), kernel_size=(3,3,3))
        )
        self.layer4 = nn.Sequential( 
        TCNA(in_channels=c[3], out_channels=c[3], stride = (2,2,2), kernel_size=(3,3,3)),
        TCNA(in_channels=c[3], out_channels=c[3], stride = (1,1,1), kernel_size=(3,3,3)),
        UpScale(in_channels = c[3], out_channels = c[4], stride = (1,1,1), kernel_size=(3,3,3))
        )
        self.layer5 = nn.Sequential( 
        TCNA(in_channels=c[4], out_channels=c[4], stride = (1,2,2), kernel_size=(1,3,3)),
        TCNA(in_channels=c[4], out_channels=c[4], stride = (1,1,1), kernel_size=(1,3,3)),
        UpScale(in_channels = c[4], out_channels = c[4], stride = (1,1,1), kernel_size=(1,3,3))
        )
    def forward(x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    

        
"""  Convolution -> Normalization -> Activiation = CNA  """        
class CNA(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super(CNA, self).__init__()
        self.conv = conv3x3(in_channels, out_channels, kernel_size, stride)
        self.norm = group_norm(32, in_channels)
        self.act = lrelu()
    
    def forward(x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)
        
"""  Transposed Convolution -> Normalization -> Activiation = TCNA  """               
class TCNA(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super(TCNA, self).__init__()
        self.conv = transp_conv3x3(in_channels, out_channels, kernel_size, stride)
        self.norm = group_norm(32, in_channels)
        self.act = lrelu()
    
    def forward(x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)
        
class DownScale(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super(DownScale, self).__init__()
        self.conv = conv3x3(in_channels, out_channels, kernel_size, stride)
        
    def forward(x):
        return self.conv(x)
    
class UpScale(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super(UpScale, self).__init__()
        self.conv = transp_conv3x3(in_channels, out_channels, kernel_size, stride)
        
    def forward(x):
        return self.conv(x)
        
        
def conv3x3(in_channels, out_channels, kernel_size=1, stride=2, bias = False):
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride, bias)

def transp_conv3x3(in_channels, out_channels, kernel_size=(3,3,3), stride=(1,1,1), bias = False):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, bias)
    
def group_norm(num_groups = 32, num_channels = 32):
    return nn.GroupNorm(num_groups, num_channels)

def lrelu():
    return nn.LeakyReLU()

def out_act():
    return nn.Sigmoid()
