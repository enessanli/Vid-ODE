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
# Veri yükleme fonksiyonları
#from dataloader_taichi import TaiChiDataset, load_data  
from dataloader_penn import PennDataset, load_data  

def mse_metric(original, reconstructed):
    """Mean Squared Error (MSE) hesaplar."""
    return F.mse_loss(original.float(), reconstructed.float()).item()

def psnr_metric(original, reconstructed, max_pixel=255.0):
    """Peak Signal-to-Noise Ratio (PSNR) hesaplar."""
    mse = mse_metric(original, reconstructed)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_pixel)) - 10 * torch.log10(torch.tensor(mse))

def process_videos(vae_path, root_dir, save_dir, image_size=256):
    # VAE modelini yükle
    vae3d = CVVAEModel.from_pretrained(vae_path, subfolder="vae3d", torch_dtype=torch.float16)

    vae3d = CVVAESD3Model.from_pretrained(
        vae_path, subfolder="vae3d_sd3", torch_dtype=torch.float16
    )
    vae3d.requires_grad_(False).cuda()

    # Test splitindeki verileri yükle
    test_loader, _, _ = load_data(root_dir=root_dir, batch_size=1, image_size=image_size, input_length=11, target_length=10, num_workers=4)

    os.makedirs(save_dir, exist_ok=True)

    total_mse = 0
    total_psnr = 0
    video_count = 0
    start_point = time.time()

    for idx, (input_a, input_b) in enumerate(test_loader):
        input_a = torch.cat((input_a, input_b), dim=1)  # (20, 3, 128, 128)
        print("ilk input_a", input_a.shape)

        video = input_a.squeeze(0).cuda()  # (20, 3, 128, 128)

        # VAE encode işlemi
        video = (video * 2) - 1  # Normalize [-1,1]
        video = rearrange(video, 't c h w -> 1 c t h w').half()
        latent = vae3d.encode(video).latent_dist.sample()
        print("latent shape:", latent.shape)
        print("video", video.shape)
        # VAE decode işlemi
        results = vae3d.decode(latent).sample
        results = rearrange(results.squeeze(0), 'c t h w -> t h w c')
        results = ((torch.clamp(results, -1.0, 1.0) + 1.0) * 127.5).to('cpu', dtype=torch.uint8)

        # MSE ve PSNR hesapla
        print("results", results.shape)
        original_video = (input_a.squeeze(0).permute(0, 2, 3, 1) * 255).to(dtype=torch.uint8)  # (T, H, W, C)
        print("original_video", original_video.shape)
        print("results", results.shape)
        # İlk frame farkına bak
        asd
        mse_score = mse_metric(original_video[0:17], results)
        psnr_score = psnr_metric(original_video[0:17], results)

        total_mse += mse_score
        total_psnr += psnr_score
        video_count += 1

        print(f"Video {idx}: MSE = {mse_score:.4f}, PSNR = {psnr_score:.2f} dB")

        # Video kaydet
        save_path = os.path.join(save_dir, f"processed_video_{idx}.mp4")
        write_video(save_path, results, fps=10, options={'crf': '10'})
    end_point = time.time()
    print("Process time:",start_point - end_point)
    # Test seti için ortalama MSE ve PSNR
    avg_mse = total_mse / video_count
    avg_psnr = total_psnr / video_count
    print(f"\nTest Seti Ortalama Sonuçları:")
    print(f"✅ Ortalama MSE: {avg_mse:.4f}")
    print(f"✅ Ortalama PSNR: {avg_psnr:.2f} dB")

if __name__ == "__main__":
    Fire(process_videos)
