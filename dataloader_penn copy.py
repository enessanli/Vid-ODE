import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class PennDataset(Dataset):
    """Tai-Chi Dataset for Video Prediction Tasks."""

    def __init__(self, root_dir, split='frames', input_length=10, target_length=10,
                 image_size=128, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            split (str): 'train' or 'test' split.
            input_length (int): Number of frames in the first sequence (input_a).
            target_length (int): Number of frames in the second sequence (input_b).
            image_size (int): Desired image size (images will be resized to image_size x image_size).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert split in ['frames', 'frames'], "split must be 'train' or 'test'"
        self.root_dir = os.path.join(root_dir, split)
        self.train = split == 'frames'
        self.input_length = input_length  # Length of input_a
        self.target_length = target_length  # Length of input_b
        self.seq_length = self.input_length + self.target_length  # Total sequence length
        self.image_size = image_size
        self.transform = transform

        self.video_dirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir)
                           if os.path.isdir(os.path.join(self.root_dir, d))]
        self.video_dirs.sort()
        self.samples = self._prepare_dataset()

    def _prepare_dataset(self):
        """Prepares the dataset by listing all possible sequences."""
        samples = []
        i = 0
        for video_dir in self.video_dirs:
            frame_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
                           if f.endswith('.jpg')]
            frame_files.sort()
            num_frames = len(frame_files)
            if i == 50:
                break
            if num_frames >= self.seq_length:
                i += 1
                samples.append({
                    'video_dir': video_dir,
                    'frame_files': frame_files,
                    'num_frames': num_frames
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_files = sample['frame_files']
        num_frames = sample['num_frames']

        # Randomly select a starting index
        max_start_idx = num_frames - self.seq_length
        if max_start_idx > 0:
            start_idx = random.randint(0, max_start_idx)
        else:
            start_idx = 0
        if not self.train:
            start_idx = 0
        start_idx = 0
        
        end_idx = start_idx + self.seq_length 
        selected_frames = frame_files[start_idx:end_idx]

        # Load images and preprocess
        frames = []
        for frame_path in selected_frames:
            image = Image.open(frame_path).convert('RGB')  # Convert to RGB
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
            frame = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
            frame = np.transpose(frame, (2, 0, 1))  # Convert to (C, H, W)
            frames.append(frame)

        frames = np.stack(frames)  # Shape: (seq_length, C, H, W)

        # Convert to torch tensor
        frames = torch.from_numpy(frames)  # Shape: (seq_length, C, H, W)

        # Optionally apply transformations
        if self.transform:
            frames = self.transform(frames)

        # Split into input_a and input_b sequences
        input_a = frames[:self.input_length]  # Shape: (input_length, C, H, W)
        input_b = frames[self.input_length:self.seq_length]  # Shape: (target_length, C, H, W)

        return [input_a, input_b]
    

def load_data(root_dir="/kuacc/users/hpc-esanli/CV-VAE/Penn_Action", batch_size=8, image_size=128,
                              input_length=10, target_length=10,
                              num_workers=4, distributed=False):
    from torch.utils.data import DataLoader

    # Create training dataset
    #train_dataset = TaiChiDataset(root_dir=root_dir, split='train',
    ##                              input_length=input_length, target_length=target_length,
    #                              image_size=image_size)

    # Create testing dataset
    test_dataset = PennDataset(root_dir=root_dir, split='frames',
                                 input_length=input_length, target_length=target_length,
                                 image_size=image_size)

    # Create data samplers


    train_sampler = None
    test_sampler = None

    # Create data loaders
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    #                          num_workers=num_workers, sampler=train_sampler, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, sampler=test_sampler, drop_last=True)
    

    dataset_kwargs = {
        'image_size': 128,
        'num_channels': 3,
        'dt': 0.1,
    }
    
    return test_loader, test_loader, dataset_kwargs
if __name__ == '__main__':
    # Parameters
    root_dir = '/kuacc/users/hpc-esanli/CV-VAE/Penn_Action'
    batch_size = 8
    image_size = 128
    input_length = 10
    target_length = 10
    num_workers = 4

    # Create data loaders
    train_loader, test_loader, _ = load_data(
        root_dir=root_dir,
        batch_size=batch_size,
        image_size=image_size,
        input_length=input_length,
        target_length=target_length,
        num_workers=num_workers
    ) # type: ignore

    # Iterate over the training data loader
    for batch in train_loader:
        input_a, input_b = batch  # Each is of shape (batch_size, sequence_length, C, H, W)
        print(f'Input A shape: {input_a.shape}')      # Expected: (batch_size, input_length, C, H, W)
        print(f'Input B shape: {input_b.shape}')      # Expected: (batch_size, target_length, C, H, W)
        print(max(input_a[0].numpy().flatten()))
        print(min(input_a[0].numpy().flatten()))
        # Your training code here
        break  # Remove this line to iterate over the entire dataset

    # Iterate over the testing data loader
    for batch in test_loader:
        input_a, input_b = batch
        print(f'Test Input A shape: {input_a.shape}')
        print(f'Test Input B shape: {input_b.shape}')
        print(max(input_a[0].numpy().flatten()))
        print(min(input_a[0].numpy().flatten()))
        # Your evaluation code here
        break