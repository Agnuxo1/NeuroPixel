"""Pixel-only native NCA/retina and near-parameter-matched CNN comparator."""
import torch
from torch import nn
from neuropixel.model import NeuroPixel,Retina
class PixelNCA(nn.Module):
    def __init__(self):
        super().__init__();self.core=NeuroPixel(11,(4,4),c_id=8,c=16,hidden=32,steps=8,fire_rate=1,retina=True);self.core.retina=Retina(8,w=8)
    def forward(self,image):
        canvas=torch.zeros((len(image),8,8),dtype=torch.long,device=image.device);cam=torch.ones_like(canvas,dtype=torch.bool)
        return self.core(canvas,rgb=image,cam=cam)['logits']
class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Sequential(nn.Conv2d(3,18,3,padding=1),nn.ReLU(),nn.Conv2d(18,26,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));self.head=nn.Linear(26,11)
    def forward(self,image):
        logits=self.head(self.features(image).flatten(1));logits[:,0]=-10000;return logits
def make(family,seed):
    torch.manual_seed(seed);model={'nca':PixelNCA,'cnn':PixelCNN}[family]();return model
