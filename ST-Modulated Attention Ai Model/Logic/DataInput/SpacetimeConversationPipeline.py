import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from typing import Optional, List, Dict  # ⬅️ 必须导入 List, Dict
import torch.nn as nn
from transformers import AutoTokenizer
from datetime import datetime


# ===== 基于数据库的对话数据集 =====
class SpacetimeConversationDataset(Dataset):
    def __init__(self, db_path: str, space_embeddings: Optional[nn.Parameter] = None):
        self.data = load_from_disk(db_path)
        self.space_embeddings = space_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        input_ids = torch.tensor(record['input_ids'], dtype=torch.long)
        time_features = torch.tensor(record['time_features'], dtype=torch.float32)

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


# ===== 基于内存对话列表的数据集（旧版功能） =====
class SpacetimeConversationListDataset(Dataset):
    """从内存中的对话列表构建数据集"""

    def __init__(self,
                 conversations: List[List[Dict]],
                 tokenizer_name: str = 'gpt2',
                 max_length: int = 512,
                 time_mode: str = 'delta_seconds',
                 space_mode: str = 'learned_space',
                 vocab_size: int = 50257,
                 d_model: int = 128):
        self.conversations = conversations
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.time_mode = time_mode
        self.space_mode = space_mode
        if space_mode == 'learned_space':
            self.space_embeddings = nn.Parameter(
                torch.randn(vocab_size, 3) * 0.1
            )
        else:
            self.space_embeddings = None

    def _parse_timestamp(self, ts_str: str) -> datetime:
        try:
            cleaned = ts_str.strip().replace(' ', 'T')
            return datetime.fromisoformat(cleaned)
        except Exception as e:
            print(f"警告：时间戳解析失败 '{ts_str}'，使用当前时间替代。错误：{e}")
            return datetime.now()

    def _compute_time_feature(self, timestamps: List[str]) -> List[float]:
        if self.time_mode == 'delta_seconds':
            dt_list = [self._parse_timestamp(ts) for ts in timestamps]
            if len(dt_list) == 0:
                return []
            base = dt_list[0]
            deltas = [(dt - base).total_seconds() for dt in dt_list]
            max_delta = max(deltas) + 1e-8
            return [d / max_delta for d in deltas]
        elif self.time_mode == 'cumulative_turns':
            return list(range(len(timestamps)))
        else:
            raise ValueError(f"Unknown time_mode: {self.time_mode}")

    def _get_space_coords(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.space_mode == 'learned_space':
            return self.space_embeddings[input_ids]
        else:
            return torch.zeros(len(input_ids), 3)

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]

        # ---- 1. 提取消息内容和时间戳 ----
        texts = [msg['content'] for msg in conv]
        timestamps = [msg.get('timestamp', '2025-01-01 00:00:00') for msg in conv]
        time_values = self._compute_time_feature(timestamps)

        # ---- 2. 逐条消息分词 ----
        all_input_ids = []
        all_timestamps = []

        for msg, t_val in zip(conv, time_values):
            tokens = self.tokenizer(msg['content'], truncation=False, add_special_tokens=False)['input_ids']
            if not tokens:
                tokens = [self.tokenizer.pad_token_id]
            all_input_ids.extend(tokens)
            all_timestamps.extend([t_val] * len(tokens))

        # ---- 3. 截断 ----
        if len(all_input_ids) > self.max_length:
            all_input_ids = all_input_ids[:self.max_length]
            all_timestamps = all_timestamps[:self.max_length]

        # ---- 4. 转为Tensor ----
        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        t_tensor = torch.tensor(all_timestamps, dtype=torch.float32).unsqueeze(-1)

        # ---- 5. 获取空间坐标 ----
        space_tensor = self._get_space_coords(input_ids_tensor)

        # ---- 6. 组合 ----
        coords_raw = torch.cat([space_tensor, t_tensor], dim=-1)

        return {
            'input_ids': input_ids_tensor,
            'attention_mask': torch.ones(len(input_ids_tensor), dtype=torch.long),
            'coords_raw': coords_raw,
            'length': len(input_ids_tensor)
        }