import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy



class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        

    def forward(self,x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU6, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)), inplace=True)

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBnReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)
        

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
# get expected value, soft argmin
# return: depth [B, 1, H, W]
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    depth = depth.unsqueeze(1)
    return depth

def depth_regression_1(p, depth_values):
    depth = torch.sum(p * depth_values, 1)
    depth = depth.unsqueeze(1)
    return depth
