import torch
import matplotlib.pyplot as plt

from templates import *

from dataloader_taichi import load_data
import cv2
import os

device = 'cuda:0'
conf = ffhq128_autoenc_130M()
model = LitModel(conf)

# Model ağırlıklarını yükle
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

data_train, data_val, dataset_kwargs = load_data(batch_size=1)
for batch_idx, (input_a, input_b) in enumerate(data_train):
    result = torch.cat((input_a, input_b), dim=1)
    previousXT = None
    for i in range(result.shape[1]):

        single_frame = result[0, i]  # ilk örnek, ilk frame

        # Tek frame'i modele vermek için boyut düzeltmesi yapalım: (1, 3, H, W)
        single_frame = single_frame.unsqueeze(0)

        img_for_save = single_frame[0].permute(1, 2, 0).cpu().numpy()
        plt.imsave(f"original_image_{i}.png", img_for_save)

        # Modele encode
        print("model encode start")
        cond = model.encode(single_frame.to(device))
        print("model encode end")
        # encode_stochastic ile T=250 adımlık encode

        print("model encode stochastic start")
        xT = model.encode_stochastic(single_frame.to(device), cond, T=250)
        print("model encode stochastic end")
        # Ori ve xT karşılaştırması
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        

        ori = single_frame
        ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(xT[0].permute(1, 2, 0).cpu())
        ax[1].set_title("xT Encoded")
        ax[1].axis("off")

        fig.savefig(f"ori_vs_xT_{i}.png", bbox_inches='tight')

        plt.close(fig)  # Artık ekrana göstermiyoruz, kaydetme sonrası kapatıyoruz
        print("xT shape:", xT)
        # Diffusion ile yeniden (render)
        print("model render start")
        pred = model.render(xT, cond, T=20)
        print("model render end")
        pred = pred*2-1
        # Ori ve pred karşılaştırması
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(pred[0].permute(1, 2, 0).cpu())
        ax[1].set_title("Predicted")
        ax[1].axis("off")
        mse_loss = torch.nn.functional.mse_loss(ori[0].permute(1, 2, 0).cpu(), pred[0].permute(1, 2, 0).cpu(),reduction='sum')
        print("MSE Loss:", mse_loss.item())
        """
        if previousXT == None:
            previousXT = xT
        
        pred = model.render((xT+previousXT)/2, cond, T=20)
        pred = pred*2-1
        previousXT = xT

        ax[2].imshow(pred[0].permute(1, 2, 0).cpu())
        ax[2].set_title("Interpolate")
        ax[2].axis("off")
        """
        fig.savefig(f"ori_vs_pred_{i}.png", bbox_inches='tight')
        plt.close(fig)
    break

def images_to_video(image_folder, video_name, fps=10):
    images = [img for img in os.listdir(image_folder) if img.startswith("ori_vs_pred_") and img.endswith(".png")]
    images.sort()
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
    images = [img for img in os.listdir(image_folder) if img.startswith("ori_vs_xT_") and img.endswith(".png")]
    images.sort()
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
print("Video saved as output_video.mp4")