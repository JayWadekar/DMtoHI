import torch
import torch.nn as nn
import numpy as np

def conv3x3(inplane,outplane, stride=1,padding=0):
	return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True)
	
def crop_tensor(x):
    x = x.narrow(2,1,x.shape[2]-3).narrow(3,1,x.shape[3]-3).narrow(4,1,x.shape[4]-3).contiguous()
    return x

class BasicBlock(nn.Module):
    def __init__(self,inplane,outplane,stride = 1):
        super(BasicBlock, self).__init__()
        #self.drop=nn.dropout(0.1)
        self.conv1 = conv3x3(inplane,outplane,padding=0,stride=stride)
        self.bn1 = nn.BatchNorm3d(outplane)
        self.relu = nn.ReLU(inplace=True)
        self.pad1=nn.ConstantPad3d((1,1,1,1,1,1), 0)

    def forward(self,x):
        #x = periodic_padding_3d(x,pad=(1,1,1,1,1,1))
        x=self.pad1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.drop(out)
        out = self.relu(out)
        return out

class Lpt2NbodyNet(nn.Module):
    def __init__(self, block):
        super(Lpt2NbodyNet,self).__init__()
        self.layerm1 = self._make_layer(block, 2, 32, blocks=2,stride=1)
        self.layer0 = self._make_layer(block, 32, 64, blocks=1,stride=2)
        self.layer1 = self._make_layer(block, 64, 64, blocks=1,stride=1)
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
        self.layer8 = self._make_layer(block,64,32,blocks=2,stride=1)
        self.deconv4 = nn.ConvTranspose3d(32,1,1,stride=1,padding=0)
        self.pad=nn.ConstantPad3d((0,1,0,1,0,1), 0)

    def _make_layer(self,block,inplanes,outplanes,blocks,stride=1):
        layers = []
        for i in range(0,blocks):
            layers.append(block(inplanes,outplanes,stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def forward(self,x):
        x  = self.layerm1(x)
        #x0 = nn.MaxPool3d(2, stride=2)(x)
        x  = self.layer0(x)
        x1 = self.layer1(x)
        x  = self.layer2(x1)
        x2 = self.layer3(x)
        x  = self.layer4(x2)
        x  = self.layer5(x)
        x  = self.pad(x)
        x  = nn.functional.relu(self.deconv_batchnorm1(crop_tensor(self.deconv1(x))),inplace=True)
        x  = torch.cat((x,x2),dim=1)
        x  = self.layer6(x)
        x  = self.pad(x)
        x  = nn.functional.relu(self.deconv_batchnorm2(crop_tensor(self.deconv2(x))),inplace=True)
        #x  = torch.cat((x,x1),dim=1)
        #x  = torch.cat((x,x0),dim=1)
        #x  = self.layer7(x)
        x  = self.layer8(x)
        x  = self.deconv4(x)
        x  = nn.functional.relu(x,inplace=True)

        return x
