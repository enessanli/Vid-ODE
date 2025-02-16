import torch
import torch.nn as nn

from models.base_conv_gru import *
from models.ode_func import ODEFunc, DiffeqSolver
from models.layers import create_convnet
import torchvision.models as models
from diffusers import AutoencoderTiny
import os
import torch
import torch.nn.functional as F
from models.modeling_vae import CVVAEModel
from einops import rearrange
from torchvision.io import write_video
from torch.utils.data import DataLoader
from fire import Fire
from decord import cpu
from models.modeling_vae import CVVAESD3Model
from decord import VideoReader, cpu
import torch
import os
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
from fire import Fire
import time

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # Use the layers up to (and including) the max pool.
        self.features = nn.Sequential(
            resnet.conv1,   # Output: (bs, 64, 64, 64) for 128×128 input (approximately)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # Output: (bs, 64, 32, 32)
        )
        # Adjust channel dimensions from 64 to 128
        self.adjust = nn.Conv2d(64, 128, kernel_size=1)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)       # (bs, 64, 32, 32)
        x = self.adjust(x)         # (bs, 128, 32, 32)
        return x
    
class Adjuster(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.adjust(x)
    



class VidODE(nn.Module):
    
    def __init__(self, opt, device):
        super(VidODE, self).__init__()
        
        self.opt = opt
        self.device = device
        
        # initial function
        self.build_model()
        
        # tracker
        self.tracker = utils.Tracker()
    
    def build_model(self):
        
        # channels for encoder, ODE, init decoder
        init_dim = self.opt.init_dim
        #print("init_dim:", init_dim)
        resize = 2 ** (self.opt.n_downs+1)
        #print("resize:", resize)
        base_dim = init_dim * resize
        input_size = (self.opt.input_size // resize, self.opt.input_size // resize)
        #print("input_size:", input_size)
        ode_dim = base_dim
        print(base_dim)
        #print("opt_n_downs:", self.opt.n_downs)
        #print("opt_n_layers:", self.opt.n_layers)
        print(f"Building models... base_dim:{base_dim}")
        self.adjuster = Adjuster(16, 256).to(self.device)
        self.adjuster2 = Adjuster(256, 16).to(self.device)
        self.adjuster3 = Adjuster(3, 6).to(self.device)
        ##### Conv Encoder
        #self.encoder = ResNet18Encoder().to(self.device)
        #self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(self.device).float()
        #for param in self.vae.parameters():
        #    param.requires_grad = False

        self.vae3d = CVVAESD3Model.from_pretrained(
            "/kuacc/users/hpc-esanli/CV-VAE", subfolder="vae3d_sd3", torch_dtype=torch.float16
        )
        self.vae3d.requires_grad_(False).cuda()
        self.vae3d = self.vae3d.to(torch.float32)


        ##### ODE Encoder
        ode_func_netE = create_convnet(n_inputs=ode_dim,
                                       n_outputs=base_dim,
                                       n_layers=self.opt.n_layers,
                                       n_units=base_dim // 2).to(self.device)
        
        rec_ode_func = ODEFunc(opt=self.opt,
                               input_dim=ode_dim,
                               latent_dim=base_dim,  # channels after encoder, & latent dimension
                               ode_func_net=ode_func_netE,
                               device=self.device).to(self.device)
        
        z0_diffeq_solver = DiffeqSolver(base_dim,
                                        ode_func=rec_ode_func,
                                        method="euler",
                                        latents=base_dim,
                                        odeint_rtol=1e-3,
                                        odeint_atol=1e-4,
                                        device=self.device)
        
        self.encoder_z0 = Encoder_z0_ODE_ConvGRU(input_size=input_size,
                                                 input_dim=base_dim,
                                                 hidden_dim=base_dim,
                                                 kernel_size=(3, 3),
                                                 num_layers=1,
                                                 dtype=torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor,
                                                 batch_first=True,
                                                 bias=True,
                                                 return_all_layers=True,
                                                 z0_diffeq_solver=z0_diffeq_solver,
                                                 run_backwards=self.opt.run_backwards).to(self.device)
        
        ##### ODE Decoder
        ode_func_netD = create_convnet(n_inputs=ode_dim,
                                       n_outputs=base_dim,
                                       n_layers=self.opt.n_layers,
                                       n_units=base_dim // 2).to(self.device)
        
        gen_ode_func = ODEFunc(opt=self.opt,
                               input_dim=ode_dim,
                               latent_dim=base_dim,
                               ode_func_net=ode_func_netD,
                               device=self.device).to(self.device)
        
        self.diffeq_solver = DiffeqSolver(base_dim,
                                          gen_ode_func,
                                          self.opt.dec_diff, base_dim,
                                          odeint_rtol=1e-3,
                                          odeint_atol=1e-4,
                                          device=self.device)
        
        ##### Conv Decoder
        #self.decoder = Decoder(input_dim=base_dim * 2, output_dim=self.opt.input_dim + 3, n_ups=self.opt.n_downs).to(self.device)

    def encode(self, x):
        print("encode giris:", x.shape)
        latents = self.vae3d.encode(x).latent_dist.sample()
        print("latents:", latents.shape)
        latents = latents.permute(0,2, 1, 3, 4)  # (b, 5, 16, 32, 32)
        batch = latents.shape[0]
        print("latents: permute", latents.shape)

        adjusted_latents = self.adjuster(latents.view(-1, 16, 16, 16))
        print("adjusted latents:", adjusted_latents.shape)
        adjusted_latents = adjusted_latents.view(batch, -1, 256, 16, 16)
        return adjusted_latents
    
    def decode(self, x):

        x = self.adjuster2(x)

        x = self.vae3d.decode(x).sample
        print("decoded x:", x.shape)
        #print("decoded x:", x.shape)
        return x.view(-1, 3, 128, 128)
        
    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, mask=None, out_mask=None):
        
        truth = truth.to(self.device)
        truth_time_steps = truth_time_steps.to(self.device)
        mask = mask.to(self.device)
        out_mask = out_mask.to(self.device)
        time_steps_to_predict = time_steps_to_predict.to(self.device)
        
        resize = 2 ** (self.opt.n_downs+1)
        b, t, c, h, w = truth.shape
        print("truth:", truth.shape)
        print("time_steps_to_predict:", time_steps_to_predict.shape)
        pred_t_len = len(time_steps_to_predict)
        
        ##### Skip connection forwarding
        skip_image = truth[:, -1, ...] if self.opt.extrap else truth[:, 0, ...]
        skip_conn_embed = self.encode(skip_image).view(b, -1, h // resize, w // resize)
        print("skip_conn_embed:", skip_conn_embed.shape)

        ##### Conv encoding
        e_truth = self.encode(truth.view(b * t, c, h, w)).view(b, t, -1, h // resize, w // resize)
        print("e_truth:", e_truth.shape)

        ##### ODE encoding
        first_point_mu, first_point_std = self.encoder_z0(input_tensor=e_truth, time_steps=truth_time_steps, mask=mask, tracker=self.tracker)
        
        # Sampling latent features
        first_point_enc = first_point_mu.unsqueeze(0).repeat(1, 1, 1, 1, 1)
        
        # ==================================================================================== #
        
        ##### ODE decoding
        first_point_enc = first_point_enc.squeeze(0)
        sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict)
        self.tracker.write_info(key="sol_y", value=sol_y.clone().cpu())
        
        ##### Conv decoding
        sol_y = sol_y.contiguous().view(b, pred_t_len, -1, h // resize, w // resize)
        # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w

        pred_outputs = self.get_flowmaps(sol_out=sol_y, first_prev_embed=skip_conn_embed, mask=out_mask) # b, t, 6, h, w
        pred_outputs = torch.cat(pred_outputs, dim=1)  # Listeyi tensöre dönüştür
        pred_x = pred_outputs.view(b, -1, c, h, w)
        print("pred_outputs:", pred_outputs.shape)
        
        pred_x = pred_outputs.view(b, -1, c, h, w)
        
        ### extra information
        extra_info = {}
        
        return pred_x, extra_info
    
    def get_mse(self, truth, pred_x, mask=None):
    
        b, _, c, h, w = truth.size()
        
        if mask is None:
            selected_time_len = truth.size(1)
            selected_truth = truth
        else:
            selected_time_len = int(mask[0].sum())
            selected_truth = truth[mask.squeeze(-1).byte()].view(b, selected_time_len, c, h, w)
        loss = torch.sum(torch.abs(pred_x - selected_truth)) / (b * selected_time_len * c * h * w)
        return loss
    
    
    def get_diff(self, data, mask=None):
        
        data_diff = data[:, 1:, ...] - data[:, :-1, ...]
        b, _, c, h, w = data_diff.size()
        selected_time_len = int(mask[0].sum())
        masked_data_diff = data_diff[mask.squeeze(-1).byte()].view(b, selected_time_len, c, h, w)
        
        return masked_data_diff

    
    def export_infos(self):
        infos = self.tracker.export_info()
        self.tracker.clean_info()
        return infos
    
    def get_flowmaps(self, sol_out, first_prev_embed, mask):
        """ Get flowmaps recursively
        Input:
            sol_out - Latents from ODE decoder solver (b, time_steps_to_predict, c, h, w)
            first_prev_embed - Latents of last frame (b, c, h, w)
        
        Output:
            pred_flows - List of predicted flowmaps (b, time_steps_to_predict, c, h, w)
        """
        b, _, c, h, w = sol_out.size()
        pred_time_steps = int(mask[0].sum())
        pred_flows = list()
    
        prev = first_prev_embed.clone()
        time_iter = range(pred_time_steps)
        

        sol_out = sol_out.view(b, pred_time_steps, c, h, w)
        #print("Time iter: ", time_iter)
        for t in time_iter:
            cur_and_prev = sol_out[:, t, ...]
            print("cur_and_prev:", cur_and_prev.shape)
            pred_flow = self.decode(cur_and_prev).unsqueeze(1)
            print("pred_flow decoder out:", pred_flow.shape)
            pred_flows += [pred_flow]
            prev = sol_out[:, t, ...].clone()
    
        return pred_flows
    
    def get_warped_images(self, pred_flows, start_image, grid):
        """ Get warped images recursively
        Input:
            pred_flows - Predicted flowmaps to use (b, time_steps_to_predict, c, h, w)
            start_image- Start image to warp
            grid - pre-defined grid

        Output:
            pred_x - List of warped (b, time_steps_to_predict, c, h, w)
        """
        warped_time_steps = pred_flows.size(1)
        pred_x = list()
        last_frame = start_image
        b, _, c, h, w = pred_flows.shape
        print("pred_flows:", pred_flows.shape)
        
        for t in range(warped_time_steps):
            pred_flow = pred_flows[:, t, ...]           # b, 2, h, w
            pred_flow = torch.cat([pred_flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), pred_flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
            pred_flow = pred_flow.permute(0, 2, 3, 1)   # b, h, w, 2
            print("pred_flow:", pred_flow.shape)
            print("grid:", grid.shape)
            flow_grid = grid.clone() + pred_flow.clone()# b, h, w, 2
            warped_x = nn.functional.grid_sample(last_frame, flow_grid, padding_mode="border")
            pred_x += [warped_x.unsqueeze(1)]           # b, 1, 3, h, w
            last_frame = warped_x.clone()
        
        return pred_x
    
    def compute_all_losses(self, batch_dict):
        
        batch_dict["tp_to_predict"] = batch_dict["tp_to_predict"].to(self.device)
        batch_dict["observed_data"] = batch_dict["observed_data"].to(self.device)
        batch_dict["observed_tp"] = batch_dict["observed_tp"].to(self.device)
        batch_dict["observed_mask"] = batch_dict["observed_mask"].to(self.device)
        batch_dict["data_to_predict"] = batch_dict["data_to_predict"].to(self.device)
        batch_dict["mask_predicted_data"] = batch_dict["mask_predicted_data"].to(self.device)

        pred_x, extra_info = self.get_reconstruction(
            time_steps_to_predict=batch_dict["tp_to_predict"],
            truth=batch_dict["observed_data"],
            truth_time_steps=batch_dict["observed_tp"],
            mask=batch_dict["observed_mask"],
            out_mask=batch_dict["mask_predicted_data"])
        
        # batch-wise mean
        loss = torch.mean(self.get_mse(truth=batch_dict["data_to_predict"],
                                       pred_x=pred_x,
                                       mask=batch_dict["mask_predicted_data"]))

        if not self.opt.extrap:
            init_image = batch_dict["observed_data"][:, 0, ...]
        else:
            init_image = batch_dict["observed_data"][:, -1, ...]

        data = torch.cat([init_image.unsqueeze(1), batch_dict["data_to_predict"]], dim=1)
        data_diff = self.get_diff(data=data, mask=batch_dict["mask_predicted_data"])
        diff_loss = 0.0
        diff_loss = torch.mean(self.get_mse(truth=data_diff, pred_x=data_diff, mask=None))

        loss = loss

        results = {}
        results["loss"] = torch.mean(loss)
        results["pred_y"] = pred_x
        results['diff_loss'] = diff_loss
        return results
