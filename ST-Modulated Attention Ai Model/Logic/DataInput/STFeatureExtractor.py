import torch
import torch.nn as nn

class STExtractor(nn.Module):
    def __init__(self, vocab_size, d_model, st_dim=4):
        super().__init__()
        self.semantic_emb = nn.Embedding(vocab_size, d_model - st_dim) # 语义部分
        self.st_predictor = nn.Sequential(  # 坐标回归器
            nn.Linear(d_model - st_dim, 128),
            nn.ReLU(),
            nn.Linear(128, st_dim)
        )
    def forward(self, x):
        sem = self.semantic_emb(x)
        st = self.st_predictor(sem)
        # 关键：时间t务必加上绝对序列偏移，防止塌缩
        st[:, :, 3] = st[:, :, 3] + torch.linspace(0, 10, x.size(1)).to(x.device)
        return torch.cat([sem, st], dim=-1) # 输出 [Batch, Seq, D_model]