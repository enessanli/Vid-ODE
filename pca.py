import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# 1. Veriyi yükleme
data = np.load('output/xT_array.npy')                # Shape örneği: (10, 20, 128, 128, 3)
data_xt = np.load('output/xT_recon_array.npy')       # Shape örneği: (10, 20, 128, 128, 3)
data_original = np.load('output/original_array.npy') # Shape örneği: (10, 20, 128, 128, 3)
data_reconstructed = np.load('output/reconstructed_array.npy')  # (10, 20, 128, 128, 3)

# Condition verilerini de yükle
condition_array = np.load("output/condition_array.npy")           # Shape örneği: (10, 20, latent_dim)
condition_recon_array = np.load("output/condition_recon_array.npy") # Aynı shape: (10, 20, latent_dim)

# Örnek olarak ilk 3 video ile sınırlamak isterseniz:
data = data[:3]
data_xt = data_xt[:3]
data_original = data_original[:3]
data_reconstructed = data_reconstructed[:3]
condition_array = condition_array[:3]
condition_recon_array = condition_recon_array[:3]

# 2. Veriyi yeniden şekillendirme (flatten)
#    Her video 20 frame'den oluşuyor ve her frame'i tek bir vektör haline getiriyoruz.
videos = [data[i].reshape(data.shape[1], -1) for i in range(data.shape[0])]               # (20, ?)
videos_xt = [data_xt[i].reshape(data_xt.shape[1], -1) for i in range(data_xt.shape[0])]   # (20, ?)
videos_original = [data_original[i].reshape(data_original.shape[1], -1) for i in range(data_original.shape[0])]
videos_reconstructed = [data_reconstructed[i].reshape(data_reconstructed.shape[1], -1) for i in range(data_reconstructed.shape[0])]

# Condition verileri de benzer şekilde (20, ?) boyutuna dönüştürülüyor
videos_condition = [condition_array[i].reshape(condition_array.shape[1], -1) for i in range(condition_array.shape[0])]
videos_condition_recon = [condition_recon_array[i].reshape(condition_recon_array.shape[1], -1) for i in range(condition_recon_array.shape[0])]

# 3. PCA Analizi
n_components = 2
pca_xT = []
pca_xT2 = []
pca_original = []
pca_reconstructed = []

# Condition için PCA
pca_condition = []
pca_condition_recon = []

# xT
for video in videos:
    pca = PCA(n_components=n_components)
    pca_xT.append(pca.fit_transform(video))  # Shape: (20, 2)

# xT Reconstructed
for video in videos_xt:
    pca = PCA(n_components=n_components)
    pca_xT2.append(pca.fit_transform(video)) # Shape: (20, 2)

# Original
for video in videos_original:
    pca = PCA(n_components=n_components)
    pca_original.append(pca.fit_transform(video))

# Reconstructed
for video in videos_reconstructed:
    pca = PCA(n_components=n_components)
    pca_reconstructed.append(pca.fit_transform(video))

# Condition
for video in videos_condition:
    pca = PCA(n_components=n_components)
    pca_condition.append(pca.fit_transform(video))

# Condition Reconstructed
for video in videos_condition_recon:
    pca = PCA(n_components=n_components)
    pca_condition_recon.append(pca.fit_transform(video))

# 4. PCA Sonuçlarını Ayrı Grafiklerde Çizim
output_dir = "pca_line_results"
os.makedirs(output_dir, exist_ok=True)

# Renk paleti (örnek: tab10)
colors = plt.cm.get_cmap("tab10", len(videos))  # kaç tane video varsa o kadar renk

# 4.1. PCA xT (Line Plot)
plt.figure(figsize=(10, 8))
for i in range(len(videos)):
    plt.plot(
        pca_xT[i][:, 0], pca_xT[i][:, 1],
        color=colors(i), label=f'xT - Video {i+1}', alpha=0.8, marker='o'
    )
    plt.plot(
        pca_xT2[i][:, 0], pca_xT2[i][:, 1],
        color=colors(i), label=f'xT Recon - Video {i+1}', alpha=0.8, marker='s'
    )
plt.title('PCA Analysis - xT / xT Reconstructed (Line Plot)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.savefig(os.path.join(output_dir, "pca_xT_line.png"), bbox_inches='tight')
plt.close()

# 4.2. PCA Original / Reconstructed (Line Plot)
plt.figure(figsize=(10, 8))
for i in range(len(videos)):
    plt.plot(
        pca_original[i][:, 0], pca_original[i][:, 1],
        color=colors(i), label=f'Original - Video {i+1}', alpha=0.8, linestyle='--'
    )
    plt.plot(
        pca_reconstructed[i][:, 0], pca_reconstructed[i][:, 1],
        color=colors(i), label=f'Reconstructed - Video {i+1}', alpha=0.8, linestyle='-'
    )
plt.title('PCA Analysis - Original / Reconstructed (Line Plot)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.savefig(os.path.join(output_dir, "pca_original_reconstructed_line.png"), bbox_inches='tight')
plt.close()

# 4.3. PCA Condition / Condition Reconstructed (Line Plot)
plt.figure(figsize=(10, 8))
for i in range(len(videos_condition)):
    plt.plot(
        pca_condition[i][:, 0], pca_condition[i][:, 1],
        color=colors(i), label=f'Cond - Video {i+1}', alpha=0.8, marker='o'
    )
    plt.plot(
        pca_condition_recon[i][:, 0], pca_condition_recon[i][:, 1],
        color=colors(i), label=f'Cond Recon - Video {i+1}', alpha=0.8, marker='s'
    )
plt.title('PCA Analysis - Condition / Condition Reconstructed (Line Plot)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.savefig(os.path.join(output_dir, "pca_condition_line.png"), bbox_inches='tight')
plt.close()

print(f"PCA grafikleri '{output_dir}' klasörüne kaydedildi.")
