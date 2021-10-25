import torch
import torch.nn as nn
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn as e2nn

class DoubleConv(e2nn.EquivariantModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = e2nn.R2Conv(in_channels, mid_channels, kernel_size=5, padding=2)
        self.bn1 = e2nn.InnerBatchNorm(mid_channels)
        self.relu1 = e2nn.ReLU(mid_channels)
        self.do1 = e2nn.PointwiseDropout(mid_channels, p=0.1)
        self.conv2 = e2nn.R2Conv(mid_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = e2nn.InnerBatchNorm(out_channels)
        self.relu2 = e2nn.ReLU(out_channels)
        self.do2 = e2nn.PointwiseDropout(out_channels, p=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.do1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.do2(x)
        return x
    
    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape


class Down(e2nn.EquivariantModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = e2nn.PointwiseMaxPool(in_channels, 2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x
    
    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape


class Up(e2nn.EquivariantModule):
    """Upscaling then double conv"""

    def __init__(self, in_channels_upsamp, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.in_channels_upsamp = in_channels_upsamp
        self.in_channels = in_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = e2nn.R2Upsampling(in_channels_upsamp, scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = e2nn.R2ConvTransposed(in_channels, in_channels_upsamp, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1.tensor, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x1 = e2nn.GeometricTensor(x1, self.in_channels_upsamp)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2.tensor, x1.tensor], dim=1)
        x = e2nn.GeometricTensor(x, self.in_channels)

        return self.conv(x)
    
    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape


class OutConv(e2nn.EquivariantModule):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = e2nn.R2Conv(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape
        
        
class UNet_eq(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_eq, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        self.type_in  = e2nn.FieldType(self.r2_act,  n_channels*[self.r2_act.trivial_repr])
        
        feat_type_in  = e2nn.FieldType(self.r2_act,  n_channels*[self.r2_act.trivial_repr])
        feat_type_down1 = e2nn.FieldType(self.r2_act, 5*[self.r2_act.regular_repr])
        feat_type_down2 = e2nn.FieldType(self.r2_act, 10*[self.r2_act.regular_repr])
        feat_type_down3 = e2nn.FieldType(self.r2_act, 20*[self.r2_act.regular_repr])
        feat_type_down4 = e2nn.FieldType(self.r2_act, 20*[self.r2_act.regular_repr])
        factor = 2 if bilinear else 1
        feat_type_down5 = e2nn.FieldType(self.r2_act, (40//factor)*[self.r2_act.regular_repr])

        feat_type_up1_in = e2nn.FieldType(self.r2_act, (40//factor)*[self.r2_act.regular_repr])
        feat_type_up1 = e2nn.FieldType(self.r2_act, 2*(40//factor)*[self.r2_act.regular_repr])
        feat_type_up12 = e2nn.FieldType(self.r2_act, (40//factor)*[self.r2_act.regular_repr])
        feat_type_up21_in = e2nn.FieldType(self.r2_act, (40//factor)*[self.r2_act.regular_repr])
        feat_type_up21 = e2nn.FieldType(self.r2_act, 2*(40//factor)*[self.r2_act.regular_repr])
        feat_type_up22 = e2nn.FieldType(self.r2_act, (20//factor)*[self.r2_act.regular_repr])
        feat_type_up31_in = e2nn.FieldType(self.r2_act, (20//factor)*[self.r2_act.regular_repr])
        feat_type_up31 = e2nn.FieldType(self.r2_act, 2*(20//factor)*[self.r2_act.regular_repr])
        feat_type_up32 = e2nn.FieldType(self.r2_act, (10//factor)*[self.r2_act.regular_repr])
        feat_type_up41_in = e2nn.FieldType(self.r2_act, (10//factor)*[self.r2_act.regular_repr])
        feat_type_up41 = e2nn.FieldType(self.r2_act, 2*(10//factor)*[self.r2_act.regular_repr])
        feat_type_up42 = e2nn.FieldType(self.r2_act, 5*[self.r2_act.regular_repr])
        feat_type_n_classes = e2nn.FieldType(self.r2_act, n_classes*[self.r2_act.trivial_repr])

        feat_type_downlin = e2nn.FieldType(self.r2_act, 40*[self.r2_act.trivial_repr])        

        self.inc = DoubleConv(feat_type_in, feat_type_down1)
        self.down1 = Down(feat_type_down1, feat_type_down2)
        self.down2 = Down(feat_type_down2, feat_type_down3)
        self.down3 = Down(feat_type_down3, feat_type_down4)
        self.down4 = Down(feat_type_down4, feat_type_down5)
        self.up1 = Up(feat_type_up1_in, feat_type_up1, feat_type_up12, bilinear)
        self.up2 = Up(feat_type_up21_in, feat_type_up21, feat_type_up22, bilinear)
        self.up3 = Up(feat_type_up31_in, feat_type_up31, feat_type_up32, bilinear)
        self.up4 = Up(feat_type_up41_in, feat_type_up41, feat_type_up42, bilinear)
        self.outc = OutConv(feat_type_up42, feat_type_n_classes)
        
        self.outlin = OutConv(feat_type_down5, feat_type_downlin)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(4000, 512)
        self.do1 = torch.nn.Dropout()
        self.lin2 = nn.Linear(512, 256)
        self.do2 = torch.nn.Dropout()
        self.lin3 = nn.Linear(256, 4)

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.type_in)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = logits.tensor
        
        pred = self.outlin(x5)
        pred = self.flatten(pred.tensor)
        pred = F.relu(self.lin1(pred))
        pred = self.do1(pred)
        pred = F.relu(self.lin2(pred))
        pred = self.do2(pred)
        pred = self.lin3(pred)
        return logits, pred