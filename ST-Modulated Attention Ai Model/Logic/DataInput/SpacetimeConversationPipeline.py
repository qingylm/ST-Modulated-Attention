import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple


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
        """解析时间戳字符串"""
        try:
            return datetime.fromisoformat(ts_str.replace(' ', 'T'))
        except:
            return datetime.now()  # 降级处理

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

        # ---- 1. 提取文本和时间戳 ----
        texts = [msg['content'] for msg in conv]
        timestamps = [msg.get('timestamp', '2025-01-01 00:00:00') for msg in conv]

        # ---- 2. 分词 ----
        # 将多条消息拼接，用特殊分隔符隔开
        full_text = ' '.join(texts)
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        input_ids = tokenized['input_ids']

        # 如果超过max_length，对应的时间戳也需要截断
        # 简单处理：取前 max_length 个Token
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        # ---- 3. 生成时间特征 t（按Token级别扩展） ----
        # 由于一条消息可能包含多个Token，我们需要为每个Token分配相同的时间特征
        # 简化方法：按消息粒度扩展
        token_time_values = []
        for i, text in enumerate(texts):
            # 计算这条消息对应的Token数量（近似）
            tokens_in_msg = self.tokenizer(text, truncation=False)['input_ids']
            # 如果截断，可能导致消息被切碎；这里用简化策略：只取前max_length个Token
            t_val = self._compute_time_feature(timestamps)[i]
            token_time_values.extend([t_val] * len(tokens_in_msg))

        # 保证与input_ids长度一致
        if len(token_time_values) > len(input_ids):
            token_time_values = token_time_values[:len(input_ids)]
        elif len(token_time_values) < len(input_ids):
            # 补充最后一个时间值（通常不会发生）
            token_time_values.extend([token_time_values[-1]] * (len(input_ids) - len(token_time_values)))

        t_tensor = torch.tensor(token_time_values, dtype=torch.float32).unsqueeze(-1)  # [Seq, 1]

        # ---- 4. 生成空间坐标 (x,y,z) ----
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        space_tensor = self._get_space_coords(input_ids_tensor)  # [Seq, 3]

        # ---- 5. 组合为完整的 coords_raw (x,y,z,t) ----
        coords_raw = torch.cat([space_tensor, t_tensor], dim=-1)  # [Seq, 4]

        # ---- 6. 构建attention_mask（全1，因为未填充） ----
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': attention_mask,
            'coords_raw': coords_raw,
            'length': len(input_ids)
        }