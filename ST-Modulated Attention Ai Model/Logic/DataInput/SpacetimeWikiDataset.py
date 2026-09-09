from datasets import load_from_disk
from torch.utils.data import Dataset
import torch


class SpacetimeWikiDataset(Dataset):
    def __init__(self, db_path, space_embeddings=None):
        self.data = load_from_disk(db_path)
        print(f"Loaded dataset from {db_path}, type: {type(self.data)}, len: {len(self.data)}")
        # 打印第一条数据的键，确认字段
        if len(self.data) > 0:
            print(f"Sample keys: {self.data[0].keys()}")
        self.space_embeddings = space_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        input_ids = torch.tensor(record['input_ids'], dtype=torch.long)

        # 如果记录中已有 coords_raw，直接使用
        if 'coords_raw' in record:
            coords_raw = torch.tensor(record['coords_raw'], dtype=torch.float32)
        else:
            # 否则从 time_features 构建
            if 'time_features' in record:
                time_features = torch.tensor(record['time_features'], dtype=torch.float32)
            else:
                # 默认线性时间
                time_features = torch.linspace(0, 1, len(input_ids), dtype=torch.float32)

            if self.space_embeddings is not None:
                space_coords = self.space_embeddings[input_ids]
            else:
                space_coords = torch.zeros(len(input_ids), 3, dtype=torch.float32)
            coords_raw = torch.cat([space_coords, time_features.unsqueeze(-1)], dim=-1)

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'coords_raw': coords_raw,
            'length': len(input_ids)
        }