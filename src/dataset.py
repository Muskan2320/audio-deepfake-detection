import os
import torch
from torch.utils.data import Dataset
from .preprocess import extract_mfcc

class AudioDataset(Dataset):
    def __init__(self, real_dir, fake_dir):
        self.samples = []
        self.labels = []

        for file in os.listdir(real_dir):
            self.samples.append(os.path.join(real_dir, file))
            self.labels.append(0)  # real

        for file in os.listdir(fake_dir):
            self.samples.append(os.path.join(fake_dir, file))
            self.labels.append(1)  # fake

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mfcc = extract_mfcc(self.samples[idx])
        mfcc = torch.tensor(mfcc).unsqueeze(0)  # (1, 40, T)
        label = torch.tensor(self.labels[idx])

        return mfcc, label
