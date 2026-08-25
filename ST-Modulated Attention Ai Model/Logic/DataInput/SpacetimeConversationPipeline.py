import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import torch.nn as nn


class SpacetimeConversationDataset(Dataset):
    """
    将对话数据转换为时空注意力模型所需的格式
    支持两种模式：
    - mode='learned_space': x,y,z 初始化为可学习的随机嵌入（推荐）
    - mode='zero_space': x,y,z 初始化为0（让模型自己学，收敛较慢）
    """

    def __init__(self,
                 conversations: List[List[Dict]],  # 每条对话是消息列表
                 tokenizer_name: str = 'gpt2',
                 max_length: int = 512,
                 time_mode: str = 'delta_seconds',  # 'delta_seconds' 或 'cumulative_turns'
                 space_mode: str = 'learned_space',  # 'learned_space' 或 'zero_space'
                 vocab_size: int = 50257,
                 d_model: int = 128):
        """
        conversations 格式示例：
        [
            [  # 对话1
                {"role": "user", "content": "你好", "timestamp": "2025-01-01 10:00:00"},
                {"role": "assistant", "content": "你好！", "timestamp": "2025-01-01 10:00:05"},
            ],
            [  # 对话2
                ...
            ]
        ]
        """
        self.conversations = conversations
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.time_mode = time_mode
        self.space_mode = space_mode

        # 如果使用可学习空间嵌入，为每个Token位置预创建参数
        # 注意：这里我们为整个词汇表创建可学习空间嵌入（类似Word2Vec+坐标）
        if space_mode == 'learned_space':
            # 每个Token有一个固定的 (x,y,z) 坐标，训练时更新
            self.space_embeddings = nn.Parameter(
                torch.randn(vocab_size, 3) * 0.1  # 初始化为小随机数
            )
        else:
            self.space_embeddings = None

    def _parse_timestamp(self, ts_str: str) -> datetime:
        try:
            # 去除首尾空格，并将空格替换为 T（兼容 ISO 格式）
            cleaned = ts_str.strip().replace(' ', 'T')
            return datetime.fromisoformat(cleaned)
        except Exception as e:
            # 打印警告以便调试（不要静默吞噬错误）
            print(f"警告：时间戳解析失败 '{ts_str}'，使用当前时间替代。错误：{e}")
            return datetime.now()

    def _compute_time_feature(self, timestamps: List[str]) -> List[float]:
        """计算每个消息的时间特征 t"""
        if self.time_mode == 'delta_seconds':
            # 模式1：相对于第一条消息的秒数差
            dt_list = [self._parse_timestamp(ts) for ts in timestamps]
            if len(dt_list) == 0:
                return []
            base = dt_list[0]
            deltas = [(dt - base).total_seconds() for dt in dt_list]
            # 归一化到 [0, 1] 区间（防止数值过大）
            max_delta = max(deltas) + 1e-8
            return [d / max_delta for d in deltas]

        elif self.time_mode == 'cumulative_turns':
            # 模式2：累积轮次（从0开始计数）
            return list(range(len(timestamps)))

        else:
            raise ValueError(f"Unknown time_mode: {self.time_mode}")

    def _get_space_coords(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        获取空间坐标 (x,y,z)
        - 若为 learned_space：从可学习嵌入中查找
        - 若为 zero_space：全零
        """
        if self.space_mode == 'learned_space':
            # input_ids: [Seq] -> 查找每个Token的 (x,y,z)
            return self.space_embeddings[input_ids]  # [Seq, 3]
        else:
            return torch.zeros(len(input_ids), 3)

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]

        # ---- 1. 逐条消息处理 ----
        all_input_ids = []
        all_timestamps = []

        # 先计算所有消息的时间特征（标量列表）
        timestamps = [msg.get('timestamp', '2025-01-01 00:00:00') for msg in conv]
        time_values = self._compute_time_feature(timestamps)  # [t0, t1, t2, ...]

        for msg, t_val in zip(conv, time_values):
            # 逐条分词
            tokens = self.tokenizer(msg['content'], truncation=False, add_special_tokens=False)['input_ids']
            # 如果 tokens 为空（极短文本），跳过或补一个空格
            if not tokens:
                tokens = [self.tokenizer.pad_token_id]  # 占位
            # 为这条消息的所有Token分配相同的时间值
            all_input_ids.extend(tokens)
            all_timestamps.extend([t_val] * len(tokens))

        # ---- 2. 截断（保留前 max_length 个Token） ----
        if len(all_input_ids) > self.max_length:
            all_input_ids = all_input_ids[:self.max_length]
            all_timestamps = all_timestamps[:self.max_length]

        # ---- 3. 转为Tensor ----
        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        t_tensor = torch.tensor(all_timestamps, dtype=torch.float32).unsqueeze(-1)  # [Seq, 1]

        # ---- 4. 获取空间坐标 (x,y,z) ----
        space_tensor = self._get_space_coords(input_ids_tensor)  # [Seq, 3]

        # ---- 5. 组合 ----
        coords_raw = torch.cat([space_tensor, t_tensor], dim=-1)  # [Seq, 4]

        return {
            'input_ids': input_ids_tensor,
            'attention_mask': torch.ones(len(input_ids_tensor), dtype=torch.long),
            'coords_raw': coords_raw,
            'length': len(input_ids_tensor)
        }