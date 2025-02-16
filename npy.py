import torch
import matplotlib.pyplot as plt

from templates import *
from dataloader_taichi import load_data
import cv2
import os
import numpy as np

device = 'cuda:0'
conf = ffhq128_autoenc_130M()
model = LitModel(conf)

# Model ağırlıklarını yükle
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

data_train, data_val, dataset_kwargs = load_data(batch_size=1)
xT_values = []
xT_recon_values = []
condition_values = []
condition_recon_values = []
original_images = []
reconstructed_images = []

# DataLoader'daki her video için döngü
for video_idx, (input_a, input_b) in enumerate(data_train):
    # Tüm frame'leri birleştiriyoruz (input_a ve input_b)
    result = torch.cat((input_a, input_b), dim=1)  # Shape: (1, 20, 3, H, W)
    print("result shape:",result.shape)
    print("input_a shape:",input_a.shape)
    print("input_b shape:",input_b.shape)
    
    # xT, Original ve Reconstructed için geçici depolama
    xT_video = []
    xT_recon_video = []
    original_video = []
    reconstructed_video = []
    condition_values_frame = []
    condition_recon_values_frame = []
    print(video_idx)
    # Video içerisindeki her frame için döngü
    for i in range(result.shape[1]):
        print(i)
        single_frame = result[0, i]  # İlk örnek, i. frame (Shape: (3, H, W))

        # Tek frame'i modele vermek için boyut düzeltmesi yapalım: (1, 3, H, W)
        single_frame = single_frame.unsqueeze(0)

        # Modele encode
        cond = model.encode(single_frame.to(device))
        # encode_stochastic ile T=250 adımlık encode
        xT = model.encode_stochastic(single_frame.to(device), cond, T=250)

        # Diffusion ile yeniden (render)
        pred = model.render(xT, cond, T=20)

        # Rekonstrüksiyondan tekrar bir xT elde et
        cond_reconstructed = model.encode(pred.to(device))
        xT_reconstructed = model.encode_stochastic(pred.to(device),
                                                   cond_reconstructed, T=250)

        # CPU'ya alıp numpy'a dönüştür
        xT_video.append(xT[0].cpu().numpy())  
        xT_recon_video.append(xT_reconstructed[0].cpu().numpy())  
        condition_values_frame.append(cond[0].cpu().numpy())  
        condition_recon_values_frame.append(cond_reconstructed[0].cpu().numpy())
        original_video.append(single_frame[0].cpu().numpy())  
        reconstructed_video.append(pred[0].cpu().numpy())  

    # Videoları numpy array olarak sakla
    xT_values.append(np.stack(xT_video))  
    xT_recon_values.append(np.stack(xT_recon_video))  
    condition_values.append(np.stack(condition_values_frame))
    condition_recon_values.append(np.stack(condition_recon_values_frame))
    original_images.append(np.stack(original_video))
    reconstructed_images.append(np.stack(reconstructed_video))

    # İlk 10 video ile sınırlıyoruz
    if video_idx == 1:
        break

# -----------------------------
# Verileri bir araya getirme
# -----------------------------
xT_array = np.stack(xT_values)  # (10, 20, 3, H, W)
xT_recon_array = np.stack(xT_recon_values)  # (10, 20, 3, H, W)
original_array = np.stack(original_images)  # (10, 20, 3, H, W)
reconstructed_array = np.stack(reconstructed_images)  # (10, 20, 3, H, W)

# condition lar da benzer şekle sahip
condition_array = np.stack(condition_values)  # (10, 20, 3, H, W) olduğunu varsayıyoruz
condition_recon_array = np.stack(condition_recon_values)  # (10, 20, 3, H, W)

# Eğer condition gerçekte (C, H, W) değil de (C) boyutlu bir vektörse,
# yukarıdaki kodu (10, 20, C) boyutlu olacak şekilde yorumlayabilirsiniz.
# Örneğin condition encode sonucu (batch, 128) gibi bir boyuta sahipse,
# condition_values_frame.append(cond[0].cpu().numpy())  # (128, ) şekline sahiptir.

# Transpoze (istiyorsanız, xT gibi verilerde son boyutu [H, W, C] şeklinde kullanırız)
# Ama condition vektör ise transpozeye gerek olmayabilir. 
# Örneğin:
# xT_array = np.transpose(xT_array, (0, 1, 3, 4, 2))  # (10, 20, H, W, C) 
# original_array = np.transpose(original_array, (0, 1, 3, 4, 2))  # ...
# reconstructed_array = np.transpose(reconstructed_array, (0, 1, 3, 4, 2))  # ...

# condition_array ve condition_recon_array vektör ise flatten veya doğrudan kaydedebilirsiniz.
# Şu an (10, 20, 3, H, W) ise, vektöre çevirmek isterseniz:
# shape = (10, 20, 3*H*W)
# condition_array = condition_array.reshape(condition_array.shape[0],
#                                          condition_array.shape[1],
#                                          -1)
# condition_recon_array = condition_recon_array.reshape(condition_recon_array.shape[0],
#                                                      condition_recon_array.shape[1],
#                                                      -1)

# Numpy dosyası olarak kaydet
os.makedirs("output", exist_ok=True)
np.save("output/xT_array.npy", xT_array)
np.save("output/xT_recon_array.npy", xT_recon_array)
np.save("output/original_array.npy", original_array)
np.save("output/reconstructed_array.npy", reconstructed_array)
np.save("output/condition_array.npy", condition_array)
np.save("output/condition_recon_array.npy", condition_recon_array)

print("Tüm veriler kaydedildi!")

def images_to_video(image_folder, video_name, fps=10):
    images = [img for img in os.listdir(image_folder) 
              if img.startswith("ori_vs_pred_") and img.endswith(".png")]
    images.sort()
    if not images:
        print(f"'{image_folder}' klasöründe 'ori_vs_pred_' ile başlayan PNG yok.")
        return
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()

def images_to_video2(image_folder, video_name, fps=10):
    images = [img for img in os.listdir(image_folder) 
              if img.startswith("ori_vs_xT_") and img.endswith(".png")]
    images.sort()
    if not images:
        print(f"'{image_folder}' klasöründe 'ori_vs_xT_' ile başlayan PNG yok.")
        return
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()

images_to_video(".", "output_video.mp4", fps=10)
images_to_video2(".", "output_video_xT.mp4", fps=10)
print("Video saved as output_video.mp4 and output_video_xT.mp4")
