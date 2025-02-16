import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Frame dosyalarını yükleme
frames = []
for i in range(80):
    frame = np.load(f'./latents/frames_h_{i}.npy')  # (1, 512) boyutunda
    print(f"Frame {i} shape: {frame.shape}")

    frames.append(frame[-1,:].flatten())  # (512,) boyutuna dönüştür
frames_original = []
for j in range(2):
    for i in range(20):
        frames_original.append(frames[j*40 + i*2].copy())
    for i in range(20):
        frames_original.append(frames[j*40 + i*2+1].copy())
frames = frames_original
# 2. Veriyi (80, 512) şekline dönüştürme
data = np.array(frames)  # (80, 512)

# 3. PCA ile 2 boyuta indirgeme
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data[:40,:])  # (80, 2)
data_pca2 = pca.fit_transform(data[40:,:])  # (80, 2)
data_pca = np.concatenate((data_pca, data_pca2), axis=0)
# Video parçalarının indeks aralıklarını tanımlayalım
# 0-19: Video1 ilk yarı
# 20-39: Video1 ikinci yarı
# 40-59: Video2 ilk yarı
# 60-79: Video2 ikinci yarı
segments = [
    (0, 20),   # Video1 ilk yarı
    (20, 40),  # Video1 ikinci yarı
    (40, 60),  # Video2 ilk yarı
    (60, 80)   # Video2 ikinci yarı
]

# Renkleri ve marker'ları tanımlayalım
colors = ['blue', 'blue', 'red', 'red']
markers = ['o', 'x', 'o', 'x']
labels = [
    "Video 1 (ilk yarı)",
    "Video 1 (ikinci yarı)",
    "Video 2 (ilk yarı)",
    "Video 2 (ikinci yarı)"
]

plt.figure(figsize=(8, 6))

# Her segmenti plot edelim
for (start_idx, end_idx), color, marker, label in zip(segments, colors, markers, labels):
    plt.plot(data_pca[start_idx:end_idx, 0],
             data_pca[start_idx:end_idx, 1],
             color=color,
             marker=marker,
             linestyle='-',
             label=label)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Analizi ile Video Frame Dağılımı')
plt.grid()
plt.legend()

# Görseli kaydetme
plt.savefig("pca_analysis.png", dpi=300)
plt.close()
