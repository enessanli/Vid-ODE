import torch
import matplotlib.pyplot as plt
import cv2
import os

from templates import *
from dataloader_penn import load_data

def calculate_psnr(mse, max_pixel_value=1.0):
    return 20 * torch.log10(torch.tensor(max_pixel_value)) - 10 * torch.log10(mse)

device = 'cuda:0'
conf = ffhq128_autoenc_130M()
model = LitModel(conf)

# Model ağırlıklarını yükle
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

mse_scores = []
psnr_scores = []

data_train, data_val, dataset_kwargs = load_data(batch_size=1)

# Min ve max değerleri bulmak için başlangıç değerleri
global_min = 0
global_max = 1


# Max pixel value'yi belirle
max_pixel_value = 1.0  # Eğer veriler 0-1 arasında normalize edildiyse, bu doğru değer olur.

for batch_idx, (input_a, input_b) in enumerate(data_train):
    result = torch.cat((input_a, input_b), dim=1)
    
    for i in range(result.shape[1]):
        single_frame = result[0, i].unsqueeze(0)
        cond = model.encode(single_frame.to(device))
        xT = model.encode_stochastic(single_frame.to(device), cond, T=250)
        ori = single_frame
        pred = model.render(xT, cond, T=20)
        pred = pred * 2 - 1
        
        mse_loss = torch.nn.functional.mse_loss(
            ori[0].permute(1, 2, 0).cpu(), 
            pred[0].permute(1, 2, 0).cpu(),
            reduction='mean'
        )
        
        psnr_value = calculate_psnr(mse_loss, max_pixel_value)
        
        mse_scores.append(mse_loss.item())
        psnr_scores.append(psnr_value.item())
        
        print(f"Frame {i}, MSE Loss: {mse_loss.item()}, PSNR: {psnr_value.item()}")
    break

# Ortalama MSE ve PSNR hesaplama
avg_mse = sum(mse_scores) / len(mse_scores)
avg_psnr = sum(psnr_scores) / len(psnr_scores)

print(f"Ortalama MSE: {avg_mse}")
print(f"Ortalama PSNR: {avg_psnr}")
