import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# 导入自定义模块
from Logic.DataInput.SpacetimeConversationPipeline import SpacetimeConversationListDataset
from Logic.SpacetimeTransformer import SpacetimeLM
from Logic.CoreAttention import PhysicsRegularizationLoss
from Logic.DataInput.DataAugmentation import SpacetimeDataAugmentor
from Logic.DataInput.SpacetimeWikiDataset import SpacetimeWikiDataset



# ---------- 配置参数 ----------
config = {
    'vocab_size': 50257,  # GPT-2 词汇量
    'd_model': 128,
    'd_space': 32,
    'd_time': 16,
    'num_heads': 4,
    'window_size': 1024,  # 滑动窗口大小
    'num_layers': 2,
    'dropout': 0.1,
    'batch_size': 2,
    'max_length': 64,
    'learning_rate': 1e-3,
    'warmup_steps': 100,
    'total_steps': 1000,
    'weight_decay': 0.01,
    'grad_clip_norm': 1.0,
    'lambda_causal': 0.1,
    'lambda_time': 0.01,
    'lambda_norm': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './checkpoints',
    'log_interval': 10,
    'eval_interval': 500,
}

# ---------- 数据准备 ----------
# 使用之前的示例数据，也可以替换为更大规模的真实对话数据
# 正确示例
sample_conversations = [
    [
        {"role": "user", "content": "今天天气怎么样？", "timestamp": "2025-01-01 10:00:00"},
        {"role": "assistant", "content": "今天阳光明媚，适合出游。", "timestamp": "2025-01-01 10:00:10"},
        {"role": "user", "content": "那我们去公园吧。", "timestamp": "2025-01-01 10:01:00"},
    ],
    [
        {"role": "user", "content": "Hello AI", "timestamp": "2025-01-01 11:00:00"},
        {"role": "assistant", "content": "Hi there!", "timestamp": "2025-01-01 11:00:03"},
    ]
]

# 初始化增强器
augmentor = SpacetimeDataAugmentor(
    time_scale_range=(0.9, 1.1),
    time_shift_range=(-30, 30),
    synonym_prob=0.15,
    window_size=4,
    window_stride=2
)

# 原始数据
raw_conversations = sample_conversations  # 你已有的2条

# ---- 数据增强 ----
augmented_conversations = []
for conv in raw_conversations:
    new_samples = augmentor.augment(conv, count=10)  # 每条原始对话生成10个变体
    augmented_conversations.extend(new_samples)

print(f"原始样本数: {len(raw_conversations)}")
print(f"增强后样本数: {len(augmented_conversations)}")

# 同前，但建议使用更大的数据集


tokenizer = AutoTokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- 使用增强后的数据创建 Dataset ----
# dataset = SpacetimeConversationDataset(
#     conversations=augmented_conversations,  # 替换原始数据
#     tokenizer_name='gpt2',
#     max_length=config['max_length'],
#     time_mode='delta_seconds',
#     space_mode='learned_space',
#     vocab_size=config['vocab_size'],
#     d_model=config['d_model']
# )

# ---- 创建训练数据集（使用增强后的对话数据） ----
train_dataset = SpacetimeConversationListDataset(
    conversations=augmented_conversations,
    tokenizer_name='gpt2',
    max_length=config['max_length'],
    time_mode='delta_seconds',
    space_mode='learned_space',
    vocab_size=config['vocab_size'],
    d_model=config['d_model']
)
# ---- 验证集可以使用维基数据（或单独的对话验证集） ----
current_dir = os.path.dirname(os.path.abspath(__file__))
root_name = "ST-Modulated Attention Ai Model"
while True:
    # 如果当前目录名匹配根名称，则停止
    if os.path.basename(current_dir) == root_name:
        root_path = current_dir
        break
    parent = os.path.dirname(current_dir)
    if parent == current_dir:  # 到达文件系统根目录
        raise FileNotFoundError("未找到项目根目录")
    current_dir = parent
target_path = os.path.join(root_path, "data", "database", "wiki_db")
val_dataset = SpacetimeWikiDataset(db_path=target_path)

#wiki_dataset = SpacetimeWikiDataset(db_path="data/database/wiki_db")
def collate_with_global_tokenizer(batch, tokenizer=tokenizer):
    # 1. 过滤掉 None 元素（防止 __getitem__ 返回 None）
    batch = [item for item in batch if item is not None]

    # 2. 如果过滤后为空，直接返回 None（由训练循环跳过）
    if len(batch) == 0:
        return None

    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    coords_raw = [item['coords_raw'] for item in batch]
    lengths = [item['length'] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    padded_coords = pad_sequence(coords_raw, batch_first=True, padding_value=0.0)

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_masks,
        'coords_raw': padded_coords,
        'lengths': torch.tensor(lengths)
    }


dataloader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collate_with_global_tokenizer
)

# 为验证集，可再创建一个数据集（此处简单用相同数据演示）
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    collate_fn=collate_with_global_tokenizer
)
# ---------- 模型、优化器、损失 ----------
model = SpacetimeLM(
    vocab_size=config['vocab_size'],
    d_model=config['d_model'],
    d_space=config['d_space'],
    d_time=config['d_time'],
    num_heads=config['num_heads'],
    window_size=config['window_size'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
).to(config['device'])

# 物理损失函数
phys_loss_fn = PhysicsRegularizationLoss(
    lambda_causal=config['lambda_causal'],
    lambda_time=config['lambda_time'],
    lambda_norm=config['lambda_norm']
)

# 优化器（使用AdamW，权重衰减）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

# 学习率调度器（线性warmup + 线性衰减）
total_steps = config['total_steps']
warmup_steps = config['warmup_steps']
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 混合精度训练（如果GPU支持）
scaler = GradScaler() if config['device'] == 'cuda' else None

# 检查点目录
os.makedirs(config['save_dir'], exist_ok=True)


# ---------- 辅助函数 ----------
def compute_loss(logits, input_ids, attention_mask):
    """计算交叉熵损失（忽略padding）"""
    # 移除了最后的token预测（因果模型，所有位置都预测下一个token）
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()
    # 计算损失，忽略padding
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
    # 应用mask
    loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
    return loss


# ---------- 训练循环 ----------
def train():
    global_step = 0
    best_loss = float('inf')
    model.train()

    for epoch in range(100):
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
        for batch in progress_bar:
            if batch is None:
                continue

            # 注意：如果 batch_size > 1，重置缓存会清空所有样本的状态，
            # 若模型支持 batch 级缓存则无问题，否则需逐个样本处理。
            model.reset_caches()

            input_ids = batch['input_ids'].to(config['device'])
            attention_mask = batch['attention_mask'].to(config['device'])
            coords_raw = batch['coords_raw'].to(config['device'])

            # 前向传播（混合精度）
            if scaler:
                with autocast():
                    logits, _ = model(input_ids, coords_raw, attention_mask)
                    ce_loss = compute_loss(logits, input_ids, attention_mask)
                    phys_loss, _ = phys_loss_fn(coords_raw, mask=attention_mask)
                    loss = ce_loss + phys_loss

                # 反向传播（混合精度）
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(input_ids, coords_raw, attention_mask)
                ce_loss = compute_loss(logits, input_ids, attention_mask)
                phys_loss, _ = phys_loss_fn(coords_raw, mask=attention_mask)
                loss = ce_loss + phys_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
                optimizer.step()

            scheduler.step()
            global_step += 1

            # 日志
            if global_step % config['log_interval'] == 0:
                progress_bar.set_postfix({
                    'step': global_step,
                    'ce': ce_loss.item(),
                    'phys': phys_loss.item(),
                    'lr': scheduler.get_last_lr()[0]
                })

            # 验证
            if global_step % config['eval_interval'] == 0:
                val_loss = evaluate()
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(global_step, val_loss)

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break


def evaluate():
    model.eval()
    total_loss = 0
    total_steps = 0
    # 限制验证样本数（例如最多 50 个 batch）
    max_batches = 50
    with torch.no_grad():
        val_progress = tqdm(val_dataloader, desc="Evaluating", leave=False)
        for batch_idx, batch in enumerate(val_progress):
            if batch_idx >= max_batches:
                break
            model.reset_caches()
            input_ids = batch['input_ids'].to(config['device'])
            attention_mask = batch['attention_mask'].to(config['device'])
            coords_raw = batch['coords_raw'].to(config['device'])
            logits, _ = model(input_ids, coords_raw, attention_mask)
            ce_loss = compute_loss(logits, input_ids, attention_mask)
            phys_loss, _ = phys_loss_fn(coords_raw, mask=attention_mask)
            loss = ce_loss + phys_loss
            total_loss += loss.item()
            total_steps += 1
    model.train()
    return total_loss / max(1, total_steps)


def save_checkpoint(step, loss):
    path = os.path.join(config['save_dir'], f'checkpoint_step{step}_loss{loss:.4f}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'loss': loss,
        'config': config,
    }, path)
    print(f'Checkpoint saved at {path}')


if __name__ == '__main__':
    print("增强前对话:", sample_conversations[0])
    print("增强后第1条变体:", augmented_conversations[0])
    print("增强后第2条变体:", augmented_conversations[1])
    train()