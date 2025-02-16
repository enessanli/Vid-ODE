import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# 1. Veriyi yükleme
data = np.load('output/xT_array.npy')  # Shape: (10, 20, 128, 128, 3)
data_xt = np.load('output/xT_recon_array.npy')  # Shape: (10, 20, 128, 128, 3)

data_original = np.load('output/original_array.npy')  # Shape: (10, 20, 128, 128, 3)
data_reconstructed = np.load('output/reconstructed_array.npy')  # Shape: (10, 20, 128, 128, 3)

# 2. Videoları birleştirme (frame'leri tek bir örnek haline getiriyoruz)
videos = [data[i].reshape(-1) for i in range(data.shape[0])]  # Her video (491520,)
videos_xt = [data_xt[i].reshape(data_xt.shape[1], -1) for i in range(data_xt.shape[0])]

videos_original = [data_original[i].reshape(-1) for i in range(data_original.shape[0])]  # Her video (491520,)
videos_reconstructed = [data_reconstructed[i].reshape(-1) for i in range(data_reconstructed.shape[0])]  # Her video (491520,)

# 3. PCA Analizi
n_components = 2

pca = PCA(n_components=n_components)

# xT için PCA
combined_videos = np.array(videos + videos_xt)  # Shape: (20, 491520)

pca_xT = pca.fit_transform(combined_videos)  # Shape: (10, 2)

# Original ve Reconstructed için PCA
pca_combined = PCA(n_components=n_components)
combined_videos = np.array(videos_original + videos_reconstructed)  # Shape: (20, 491520)
pca_combined_result = pca_combined.fit_transform(combined_videos)  # Shape: (20, 2)

# 4. PCA Sonuçlarını Ayrı Grafiklerde Çizim
output_dir = "pca_video_based_results_lines"
os.makedirs(output_dir, exist_ok=True)

# Renk paleti oluştur
colors = plt.cm.get_cmap("tab10", len(videos))  # 10 video için 10 renk

# 4.1 PCA xT için grafik (Çizgi ile)
plt.figure(figsize=(10, 8))
for i in range(len(pca_xT)):
    plt.plot(
        [pca_xT[i, 0],pca_xT[i+ len(videos), 0]], [pca_xT[i, 1],pca_xT[i+len(videos), 0]],
        color=colors(i), label=f'xT - Video {i + 1}', alpha=0.8, marker='o', linestyle='-'
    )
plt.title('PCA Analysis - xT Only (Line Plot)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.savefig(os.path.join(output_dir, "pca_xT_video_based_line.png"), bbox_inches='tight')
plt.close()

# 4.2 PCA Original ve Reconstructed için grafik (Çizgi ile)
plt.figure(figsize=(10, 8))
for i in range(len(videos)):
    # Original
    plt.plot(
        [pca_combined_result[i, 0], pca_combined_result[i + len(videos), 0]],
        [pca_combined_result[i, 1], pca_combined_result[i + len(videos), 1]],
        color=colors(i), label=f'Video {i + 1}', alpha=0.8, linestyle='-'
    )
plt.title('PCA Analysis - Original and Reconstructed (Line Plot)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.savefig(os.path.join(output_dir, "pca_original_reconstructed_video_based_line.png"), bbox_inches='tight')
plt.close()

print(f"PCA grafikleri '{output_dir}' klasörüne kaydedildi.")
