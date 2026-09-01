from datasets import load_from_disk
from torch.utils.data import Dataset
import torch

class SpacetimeWikiDataset(Dataset):
    def __init__(self, db_path, space_embeddings=None):
        self.data = load_from_disk(db_path)
        self.space_embeddings = space_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        input_ids = torch.tensor(record['input_ids'], dtype=torch.long)
        time_features = torch.tensor(record['time_features'], dtype=torch.float32)
        if self.space_embeddings is not None:
            space_coords = self.space_embeddings[input_ids]   # [seq, 3]
        else:
            space_coords = torch.zeros(len(input_ids), 3, dtype=torch.float32)
        coords_raw = torch.cat([space_coords, time_features.unsqueeze(-1)], dim=-1)  # [seq, 4]
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'coords_raw': coords_raw,
            'length': len(input_ids)
        }