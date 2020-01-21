import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(inplane,outplane, stride=1,padding=0):
	return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding)
	
def crop_tensor(x):
    x = x.narrow(2,1,x.shape[2]-3).narrow(3,1,x.shape[3]-3).narrow(4,1,x.shape[4]-3).contiguous()
    return x

class BasicBlock(nn.Module):
    def __init__(self,inplane,outplane,stride = 1):
        super(BasicBlock, self).__init__()
        #self.drop=nn.dropout(0.1)
        self.conv1 = conv3x3(inplane,outplane,padding=0,stride=stride)
        self.bn1 = nn.BatchNorm3d(outplane)
        self.relu = nn.LeakyReLU(inplace=True)
        self.pad1=nn.ConstantPad3d((1,1,1,1,1,1), 0)

    def forward(self,x):
        #x = periodic_padding_3d(x,pad=(1,1,1,1,1,1))
        x=self.pad1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.drop(out)
        out = self.relu(out)
        return out

class VanillaUnet(nn.Module):
    def __init__(self, block):
        super(VanillaUnet,self).__init__()
        self.layerm1 = self._make_layer(block, 1, 16, blocks=1,stride=1)
        self.layerm2 = self._make_layer(block, 1, 16, blocks=1,stride=1)
        self.layer01 = self._make_layer(block, 16, 32, blocks=1,stride=2)
        self.layer02 = self._make_layer(block, 16, 32, blocks=1,stride=2)
        self.layer1 = self._make_layer(block, 64, 64, blocks=2,stride=1)
        self.layer2 = self._make_layer(block,64,128, blocks=1,stride=2)
        self.layer3 = self._make_layer(block,128,128,blocks=2,stride=1)
        self.layer4 = self._make_layer(block,128,256,blocks=1,stride=2)
        self.layer5 = self._make_layer(block,256,256,blocks=2,stride=1)
        self.deconv1 = nn.ConvTranspose3d(256,128,3,stride=2,padding=0)
        self.deconv_batchnorm1 = nn.BatchNorm3d(num_features = 128,momentum=0.1)
        self.layer6 = self._make_layer(block,256,128,blocks=2,stride=1)
        self.deconv2 = nn.ConvTranspose3d(128,64,3,stride=2,padding=0)
        self.deconv_batchnorm2 = nn.BatchNorm3d(num_features = 64,momentum=0.1)
        self.layer7 = self._make_layer(block,128,64,blocks=1,stride=1)
        self.layer8 = self._make_layer(block,64,32,blocks=1,stride=1)
        self.deconv4 = nn.ConvTranspose3d(32,1,1,stride=1,padding=0)
        self.pad=nn.ConstantPad3d((0,1,0,1,0,1), 0)

    def _make_layer(self,block,inplanes,outplanes,blocks,stride=1):
        layers = []
        for i in range(0,blocks):
            layers.append(block(inplanes,outplanes,stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def forward(self,x,y):
        x  = self.layerm1(x)
        x  = self.layer01(x)
        y  = self.layerm2(y)
        y  = self.layer02(y)
        x  = torch.cat((x,y),dim=1)
        x1 = self.layer1(x)
        x  = self.layer2(x1)
        x2 = self.layer3(x)
        x  = self.layer4(x2)
        x  = self.layer5(x)
        x  = self.pad(x)
        x  = F.leaky_relu(self.deconv_batchnorm1(crop_tensor(self.deconv1(x))),inplace=True)
        x  = torch.cat((x,x2),dim=1)
        x  = self.layer6(x)
        x  = self.pad(x)
        x  = F.leaky_relu(self.deconv_batchnorm2(crop_tensor(self.deconv2(x))),inplace=True)
        x  = torch.cat((x,x1),dim=1)
        x  = self.layer7(x)
        x  = self.layer8(x)
        x  = F.relu(self.deconv4(x),inplace=True)

        return x
        
class two_phase_conv(nn.Module):
    def __init__(self,first_pmodel,second_pmodel):
        super(two_phase_conv,self).__init__()
        self.fp = first_pmodel
        for param in self.fp.parameters():
            param.requires_grad = False
        self.sp = second_pmodel
        self.thres = 0.5
    
    def forward(self,X):
        output = self.fp(X)
        outputs = F.softmax(output, dim=1)[:,1,:,:,:]
        mask_value = (outputs > self.thres).float()
        result = mask_value * self.sp(X)
        return result

class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        return F.leaky_relu(x, inplace=True)
        
                
class InceptionE(nn.Module):

    def __init__(self, in_channels, conv1_out, conv3_out, conv5_out, pool_out):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv3d(in_channels, conv1_out, kernel_size=1)
        self.branch3x3 = BasicConv3d(in_channels, conv3_out, kernel_size=3, padding = 1)
        self.branch5x5 = BasicConv3d(in_channels, conv5_out, kernel_size=5, padding = 2)
        self.branch_pool = BasicConv3d(in_channels, pool_out, kernel_size=1)


    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3(x)
        
        branch5x5 = self.branch5x5(x)
        
        x = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        x = self.branch_pool(x)
        x = [branch1x1, branch3x3, branch5x5, x]
        return torch.cat(x, 1)


class Inception(nn.Module):
    def __init__(self, channels, conv1_out, conv3_out, conv5_out, reg = 0):
        super(Inception, self).__init__()
        self.convm1 = BasicConv3d(channels, 4, kernel_size = 3, padding = 1)
        self.incep1 = InceptionE(4, conv1_out//2, conv3_out//2, conv5_out//2, 2)
        conv_in = conv1_out//2 + conv3_out//2 + conv5_out//2 + 2
        self.conv0 = BasicConv3d(conv_in, conv_in+6, kernel_size = 3, padding = 1, stride=2)
        self.conv1 = BasicConv3d(conv_in+6, 8, kernel_size = 3, padding = 1)
        self.incep2 = InceptionE(8, conv1_out, conv3_out, conv5_out, 3)
        conv_in = conv1_out + conv3_out + conv5_out + 3
        self.conv2 = BasicConv3d(conv_in, conv_in//2, kernel_size = 3, padding = 1)
        self.reg=reg
        if reg:
            dim_out = 1
        else:
            dim_out = 2
        self.conv3 = BasicConv3d(conv_in//2, dim_out, kernel_size = 1)
    def forward(self, x):
        x = self.convm1(x)
        x = self.incep1(x)
        x = self.conv0(x)
        x=self.conv1(x)
        x = self.incep2(x)
        x=self.conv2(x) 
        x=self.conv3(x)
        if self.reg:
            x = x.squeeze(1)
        return x
