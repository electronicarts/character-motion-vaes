# Minimal MVAE Dataset for the ACCAD-converted mocap.npz
import numpy as np
from torch.utils.data import Dataset

class MVAEACCADDataset(Dataset):
    """
    Returns pairs (p_{t-1}, p_t) from concatenated buffer X.
    The underlying file is produced by convert_accad_to_mocap.py (keys: X, lengths, offsets, dim, meta).
    """
    def __init__(self, npz_path, seq_hop=1, context=1, normalize=True, subset=None):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z['X'].astype(np.float32)
        self.lengths = z['lengths'].astype(np.int64)
        self.offsets = z['offsets'].astype(np.int64)
        self.dim = int(z['dim'][0])
        self.normalize = normalize
        self.context = context
        # build index of valid pairs (t-1, t)
        pairs = []
        for off, L in zip(self.offsets, self.lengths):
            for t in range(1, L, seq_hop):
                pairs.append((off + t - 1, off + t))
        if subset is not None:
            pairs = pairs[:subset]
        self.pairs = np.array(pairs, dtype=np.int64)
        # simple feature normalization
        if self.normalize and len(self.X) > 0:
            self.mean = self.X.mean(axis=0, keepdims=True)
            self.std = self.X.std(axis=0, keepdims=True) + 1e-6
        else:
            self.mean = np.zeros((1,self.dim), np.float32)
            self.std = np.ones((1,self.dim), np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        prev = (self.X[i:i+1] - self.mean) / self.std
        curr = (self.X[j:j+1] - self.mean) / self.std
        return prev.squeeze(0), curr.squeeze(0)
