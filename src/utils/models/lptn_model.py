import importlib
import torch
import cv2
import numpy as np
from collections import OrderedDict
from os import path as osp
import os
import matplotlib.pyplot as plt
from utils.models.base_model import BaseModel
from utils.models.losses import compute_gradient_penalty

from .archs.LPTN import LPTNPaper
from .archs.HipyrNet import HipyrNet
from .archs.discriminator_arch import Discriminator
from .losses.losses import MSELoss, GANLoss

loss_module = importlib.import_module('utils.models.losses')

class LPTNModel(BaseModel):

    def __init__(self, loss_weight, kernel_loss_weight, device, lr, gan_type='standard', nrb_low=3, nrb_high=5, use_hypernet = False):
        super(LPTNModel, self).__init__(loss_weight, kernel_loss_weight, device, lr)

        self.gan_type = gan_type
        self.kernel_loss_weight = kernel_loss_weight
        self.nrb_low = nrb_low
        self.nrb_high = nrb_high

        self.use_hypernet = use_hypernet

        # creating discriminator object
        self.device = torch.device(device)
        disc = Discriminator()
        disc = disc.to(self.device)
        if use_hypernet == True:
            model = HipyrNet(
            nrb_high = self.nrb_high,
            nrb_low = self.nrb_low,
            num_high=2,
            device=self.device,
            )
            print('HyperNet Turned ON')
            
        else:
            model = LPTNPaper(
            nrb_low =self.nrb_low,
            nrb_high =self.nrb_high,
            num_high= 2,
            device=self.device,
            )
    
        # using model as generator
        self.net_g = model.to(self.device)
        self.print_network(self.net_g)

        self.net_d = disc.to(self.device)
        self.print_network(self.net_d)

        self.init_training_settings()
        
        glw = 1
        print("GAN TURNED OFF" if glw==0 else "GAN TURNED ON")

        # initialize losses
        self.KLoss = MSELoss(loss_weight=self.kernel_loss_weight, reduction = 'mean').to(self.device)
        self.MLoss = MSELoss(loss_weight=self.loss_weight, reduction='mean').to(self.device)
        self.GLoss = GANLoss(gan_type=self.gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=glw).to(self.device)
        
        #optimal kernel
        self.opt_kernel = torch.tensor([[1., 4., 6., 4., 1],
                                            [4., 16., 24., 16., 4.],
                                            [6., 24., 36., 24., 6.],
                                            [4., 16., 24., 16., 4.],
                                            [1., 4., 6., 4., 1.]])
        self.opt_kernel /= 256.
        self.opt_kernel = self.opt_kernel.repeat(3, 1, 1, 1)
        self.opt_kernel = self.opt_kernel.to(device)
    
    def load_network(self, load_path,device = 'cuda', strict=True):
        
        load_net = torch.load(load_path, map_location=device)
        self.net_g.load_state_dict(load_net, strict=strict)
        print("Network is loaded")      

    def init_training_settings(self):
        self.net_g.train()
        self.net_d.train()
        self.optimizers = []
        self.gp_weight = 100
        self.net_d_iters = 1
        self.net_d_init_iters = 0

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
    
        self.optimizer_g = torch.optim.Adam(optim_params,
                                                 lr=0.0001, weight_decay=0, betas=[0.9, 0.99])                     
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                                 lr=0.0001, weight_decay=0, betas=[0.9, 0.99])                     

        self.optimizers.append(self.optimizer_d)
        
        if(self.use_hypernet):            
            self.optimizer_h = torch.optim.Adam(self.net_g.hyper_net.parameters(),
                                                lr=0.0001, weight_decay=0, betas=[0.9, 0.99])

            self.optimizers.append(self.optimizer_h)

    def feed_data(self, LLI, HLI):
        """
        Args:
            LLI : Low Light Image
            HLI : High Light Image
        """
        self.LLI = LLI.to(self.device)
        self.HLI = HLI.to(self.device)

    def optimize_parameters(self, current_iter):
        torch.autograd.set_detect_anomaly(True)

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.LLI)

        if(self.use_hypernet):
            self.output_kernel = self.net_g.hyper_net(self.LLI)

        l_g_total = 0
        l_g_ker = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            
            # pixel loss
            l_g_pix = self.MLoss(self.output, self.HLI).to(self.device)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.GLoss(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            
            #kernel loss
            if(self.use_hypernet):
                self.optimizer_h.zero_grad()
                l_g_ker = self.KLoss(self.output_kernel, self.opt_kernel).to(self.device)
                l_g_total += self.kernel_loss_weight*l_g_ker

            l_g_total.backward()
            self.optimizer_g.step()

            if self.use_hypernet:
                self.optimizer_h.step()
            

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        self.output = self.net_g(self.LLI)
        
        # real
        real_d_pred = self.net_d(self.HLI)
        l_d_real = self.GLoss(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())

        # fake
        fake_d_pred = self.net_d(self.output)
        l_d_fake = self.GLoss(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        gradient_penalty = compute_gradient_penalty(self.net_d, self.HLI, self.output, self.device)
        l_d = l_d_real + l_d_fake + self.gp_weight * gradient_penalty

        l_d.backward()
        self.optimizer_d.step()
        
        visuals = self.get_current_visuals()
        input_img = visuals['Low_Limage'] 
        result_img = visuals['result']
        if 'High_Limage' in visuals:
            HLI_img = visuals['High_Limage']
            #del self.HLI
      
        psnr_t,ssim_t = self.calculate_metrics(result_img,HLI_img)

        return l_g_total, l_g_ker, psnr_t, ssim_t
    

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.LLI)
        self.net_g.train()

    def nondist_validation(self, dataloader):
        psnr = 0
        ssim = 0

        for idx,batch in enumerate(dataloader):
            low_limage, high_limage = batch
            self.feed_data(low_limage, high_limage)
            self.test()
            
            visuals = self.get_current_visuals()
            input_img = visuals['Low_Limage'] 
            result_img = visuals['result']
            if 'High_Limage' in visuals:
                HLI_img = visuals['High_Limage']
                del self.HLI

            x, y = self.calculate_metrics(result_img,HLI_img)
            psnr = x + psnr
            ssim = y + ssim

        # print(psnr)
        # print(ssim)
        psnr /= (idx + 1)
        ssim /= (idx + 1)
        
        print(f'Val PSNR {psnr}')
        print(f'Val SSIM {ssim}')
                
        return psnr, ssim
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['Low_Limage'] = self.LLI.detach().to(self.device)
        out_dict['result'] = self.output.detach().to(self.device)
        if hasattr(self, 'HLI'):
            out_dict['High_Limage'] = self.HLI.detach().to(self.device)
        return out_dict

    def save(self, path):
        self.save_network(self.net_g, 'net_g', path+'_g.pth')
        self.save_network(self.net_d, 'net_d', path+'_d.pth')
        
    def visualise(self, save_dir='output_images', iteration=0):
        output = self.net_g(self.LLI)
        input = self.LLI
        label = self.HLI
        
        os.makedirs(save_dir, exist_ok=True)
        
        unique_index = iteration

        label = label.detach().cpu().numpy()
        label = np.transpose(label, (0, 2, 3, 1))  # CHW to HWC
        label = label[0]

        label = (label * 255.).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'label_image_{unique_index}.png'), cv2.cvtColor(label, cv2.COLOR_RGB2BGR))

        output = output.detach().cpu().numpy()
        output = np.transpose(output, (0, 2, 3, 1))  # CHW to HWC
        img = output[0]
        img = (img * 255.).astype(np.uint8)

        mean = [0.41441402, 0.41269127, 0.37940571]
        std = [0.33492465, 0.33443474, 0.33518072]

        input = input.detach().cpu().numpy()
        input = np.transpose(input, (0, 2, 3, 1))  # CHW to HWC
        img_in = input[0]
        img_in = (img_in * 255.).astype(np.uint8)

        print("imaged")

        cv2.imwrite(os.path.join(save_dir, f'output_image_{unique_index}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f'input_image_{unique_index}.png'), cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR))
