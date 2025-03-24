import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, device=torch.device('cuda')):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = None
        self.device = device

    # def gauss_kernel(self, device=torch.device('cuda'), channels=3):
    #     kernel = torch.tensor([[1., 4., 6., 4., 1],
    #                            [4., 16., 24., 16., 4.],
    #                            [6., 24., 36., 24., 6.],
    #                            [4., 16., 24., 16., 4.],
    #                            [1., 4., 6., 4., 1.]])
    #     kernel /= 256.
    #     kernel = kernel.repeat(channels, 1, 1, 1)
    #     kernel = kernel.to(device)
    #     return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Trans_lower(nn.Module):
    def __init__(self, num_residual_blocks):
        super(Trans_lower, self).__init__()

        model = [nn.Conv2d(3, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        out = torch.tanh(out)
        return out

class Trans_upper(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_upper, self).__init__()

        self.num_high = num_high

        model = [nn.Conv2d(9, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

        for i in range(self.num_high):
            trans_mask_block = nn.Sequential(
                nn.Conv2d(1, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 1, 1))
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        mask = self.model(x)

        for i in range(self.num_high):
            mask = nn.functional.interpolate(mask, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3]))
            self.trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            mask = self.trans_mask_block(mask)
            result_highfreq = torch.mul(pyr_original[-2-i], mask)
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq)

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq)

        pyr_result.append(fake_low)

        return pyr_result
    
class Hypernet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Adjust number of filters and use irregular kernels to reduce the output to (5x5)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # First conv layer (8 filters), irregular kernel
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (608x896 -> 304x448)
            
            nn.Conv2d(8, 16, kernel_size=(1, 3), stride=1, padding=(0, 1)),  # Second conv layer (16 filters)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (304x448 -> 152x224)
            
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # Third conv layer (32 filters)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (152x224 -> 76x112)
            
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),  # Fourth conv layer (64 filters)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),  # (76x112 -> 25x37)
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),  # Fifth conv layer (128 filters), square kernel
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(5, 7), stride=(5, 7)),  # Irregular pooling (25x37 -> 5x5)
            
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, image):
        x = self.conv_layers(image)
        x = x.view(5, 5)      
        x = x.repeat(3, 1, 1, 1)
        return x

class HipyrNet(nn.Module):
    def __init__(self, nrb_low=5, nrb_high=3, num_high=3, device = torch.device('cuda')):
        super(HipyrNet, self).__init__()

        self.device = device
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, self.device)
        self.ltm = Trans_lower(nrb_low).to(self.device)
        self.utm = Trans_upper(nrb_high, num_high=num_high).to(self.device)
        self.hyper_net = Hypernet().to(self.device)

    def forward(self, inp):
        self.lap_pyramid.kernel = self.hyper_net(inp) #dynamic kernel prediction

        inp_pyr = self.lap_pyramid.pyramid_decom(inp) #pyramid decomposition

        trans_pyr_low = self.ltm(inp_pyr[-1])
        pyr_low_up = nn.functional.interpolate(inp_pyr[-1], size=(inp_pyr[-2].shape[2], inp_pyr[-2].shape[3]))
        trans_low_up = nn.functional.interpolate(trans_pyr_low, size=(inp_pyr[-2].shape[2], inp_pyr[-2].shape[3]))
        high_with_low = torch.cat([inp_pyr[-2], pyr_low_up, trans_low_up], 1)
        trans_pyr = self.utm(high_with_low, inp_pyr, trans_pyr_low)

        op = self.lap_pyramid.pyramid_recons(trans_pyr) #pyramid reconstruction
        return op