import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from torch.utils.tensorboard import SummaryWriter
import logging
import gc

# 导入自定义模块
from Logic.DataInput.SpacetimeConversationPipeline import SpacetimeConversationListDataset
from Logic.SpacetimeTransformer import SpacetimeLM
from Logic.CoreAttention import PhysicsRegularizationLoss
from Logic.DataInput.DataAugmentation import SpacetimeDataAugmentor
from Logic.DataInput.SpacetimeWikiDataset import SpacetimeWikiDataset
from Logic.SpacetimeTransformer import EarlyStopping



# ---------- 配置参数 ----------
config = {
    'vocab_size': 50257,
    'd_model': 512,
    'd_space': 32,
    'd_time': 16,
    'num_heads': 4,
    'window_size': 1024,
    'num_layers': 6,
    'dropout': 0.1,
    'batch_size': 8,
    'max_length': 64,
    'learning_rate': 1e-3,
    'warmup_steps': 100,
    'total_steps': 30000,
    'weight_decay': 0.01,
    'grad_clip_norm': 1.0,
    'lambda_causal': 1.0,
    'lambda_time': 0.5,
    'lambda_norm': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './checkpoints',
    'log_interval': 10,
    'eval_interval': 5000,
    'early_stop_patience': 5,  # 连续多少次验证无改善则停止
    'early_stop_min_delta': 0.001,  # 视为改善的最小阈值
}

# ---------- 数据增强（对话数据） ----------
# sample_conversations = [
#     [
#         {"role": "user", "content": "今天天气怎么样？", "timestamp": "2025-01-01 10:00:00"},
#         {"role": "assistant", "content": "今天阳光明媚，适合出游。", "timestamp": "2025-01-01 10:00:10"},
#         {"role": "user", "content": "那我们去公园吧。", "timestamp": "2025-01-01 10:01:00"},
#     ],
#     [
#         {"role": "user", "content": "Hello AI", "timestamp": "2025-01-01 11:00:00"},
#         {"role": "assistant", "content": "Hi there!", "timestamp": "2025-01-01 11:00:03"},
#     ]
# ]
#
# augmentor = SpacetimeDataAugmentor(
#     time_scale_range=(0.9, 1.1),
#     time_shift_range=(-30, 30),
#     synonym_prob=0.15,
#     window_size=4,
#     window_stride=2
# )
#
# augmented_conversations = []
# for conv in sample_conversations:
#     new_samples = augmentor.augment(conv, count=10)
#     augmented_conversations.extend(new_samples)
#
# print(f"原始样本数: {len(sample_conversations)}")
# print(f"增强后样本数: {len(augmented_conversations)}")

tokenizer = AutoTokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================================
# 1. 创建对话数据集
# ============================================================================
# dialog_dataset = SpacetimeConversationListDataset(
#     conversations=augmented_conversations,
#     tokenizer_name='gpt2',
#     max_length=config['max_length'],
#     time_mode='delta_seconds',
#     space_mode='learned_space',
#     vocab_size=config['vocab_size'],
#     d_model=config['d_model']
# )

# ============================================================================
# 1.加载 LCCC 数据集（10 个分片，使用 SpacetimeWikiDataset 包装以保证键一致）
# ============================================================================
print("开始加载LCCC数据集")
train_dir = "I:\\data\\database\\lccc_db\\train"
valid_dir = "I:\\data\\database\\lccc_db\\valid"
lccc_train_dataset = SpacetimeWikiDataset(db_path=train_dir)
lccc_valid_dataset = SpacetimeWikiDataset(db_path=valid_dir)
lccc_datasets = [lccc_train_dataset]   # 列表
lccc_valid_datasets = [lccc_valid_dataset]  # 列表

# ============================================================================
# 2. 加载 CCI 数据集（10 个分片，使用 SpacetimeWikiDataset 包装以保证键一致）
# ============================================================================
print("开始加载CCI_HQ数据集")
cci_base  = "I:\\data\\database\\cci_hq_dp"
cci_datasets = [SpacetimeWikiDataset(db_path=os.path.join(cci_base, f"database_part{i}")) for i in range(1, 11)]


# ============================================================================
# 3. 加载 Wiki 数据集
# ============================================================================
# 自动查找项目根目录下的 Wiki 数据库路径
print("开始加载Wiki数据集")
wiki_db_path = "I:\\data\\database\\wiki_db"
wiki_dataset = SpacetimeWikiDataset(db_path=wiki_db_path)
wiki_datasets = [wiki_dataset]
if wiki_datasets is None:
    print("wiki_dataset数据集加载失败")
# ============================================================================
# 4. 拼接所有数据集（简单拼接）
# ============================================================================
all_datasets = lccc_datasets + cci_datasets + wiki_datasets + lccc_valid_datasets
full_train_dataset = ConcatDataset(all_datasets)

print(f"训练集总样本数: {len(full_train_dataset)}")

# ============================================================================
# 5. DataLoader 与验证集
# ============================================================================
def collate_with_global_tokenizer(batch, tokenizer=tokenizer):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    coords_raw = [item['coords_raw'] for item in batch]
    lengths = [item['length'] for item in batch]

    # 🔧 确保 coords_raw 是 Tensor（若已是 Tensor 则跳过转换）
    coords_tensors = []
    for c in coords_raw:
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float)
        coords_tensors.append(c)

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
    full_train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collate_with_global_tokenizer
)

# 验证集使用 Wiki 数据（可选，也可以使用独立对话验证集）
val_dataset = lccc_valid_dataset  # 注意：此处已正确定义
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

    # 初始化早停
    early_stopping = EarlyStopping(
        patience=config['early_stop_patience'],
        min_delta=config['early_stop_min_delta']
    )

    # ---------- 可选：TensorBoard 日志 ----------
    USE_TENSORBOARD = True  # 设为 False 可关闭
    if USE_TENSORBOARD:
        writer = SummaryWriter(log_dir='runs/spacetime_lm')

    # ---------- 可选：文件日志 ----------
    USE_FILE_LOG = True
    if USE_FILE_LOG:
        logging.basicConfig(
            filename='training.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        logging.info("训练开始")

    for epoch in range(100):
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_phys = 0.0
        epoch_steps = 0

        for batch in progress_bar:
            if batch is None:
                continue

            # 每个 batch 重置缓存（若模型支持 batch 级缓存可优化）
            model.reset_caches()

            input_ids = batch['input_ids'].to(config['device'])
            attention_mask = batch['attention_mask'].to(config['device'])
            coords_raw = batch['coords_raw'].to(config['device'])

            # ---------- 前向传播 ----------
            if scaler:
                with autocast():
                    logits, _ = model(input_ids, coords_raw, attention_mask)
                    ce_loss = compute_loss(logits, input_ids, attention_mask)
                    phys_loss, phys_dict = phys_loss_fn(coords_raw, mask=attention_mask)
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
                phys_loss, phys_dict = phys_loss_fn(coords_raw, mask=attention_mask)
                loss = ce_loss + phys_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
                optimizer.step()

            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()
            epoch_ce += ce_loss.item()
            epoch_phys += phys_loss.item()
            epoch_steps += 1

            # ---------- 每 N 步输出详细日志 ----------
            if global_step % config['log_interval'] == 0:
                # 更新进度条
                progress_bar.set_postfix({
                    'step': global_step,
                    'ce': f'{ce_loss.item():.3f}',
                    'phys': f'{phys_loss.item():.4f}',
                    'causal': f'{phys_dict["loss_causal"]:.4f}',
                    'time': f'{phys_dict["loss_time"]:.4f}',
                    'norm': f'{phys_dict["loss_norm"]:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.4e}'
                })

                # TensorBoard 记录
                if USE_TENSORBOARD:
                    writer.add_scalar('Loss/train_ce', ce_loss.item(), global_step)
                    writer.add_scalar('Loss/train_phys', phys_loss.item(), global_step)
                    writer.add_scalar('Loss/train_total', loss.item(), global_step)
                    writer.add_scalar('Phys/causal', phys_dict['loss_causal'], global_step)
                    writer.add_scalar('Phys/time', phys_dict['loss_time'], global_step)
                    writer.add_scalar('Phys/norm', phys_dict['loss_norm'], global_step)
                    writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)

                # 文件日志
                if USE_FILE_LOG:
                    logging.info(
                        f"Step {global_step}: ce={ce_loss.item():.4f}, phys={phys_loss.item():.4f}, "
                        f"causal={phys_dict['loss_causal']:.4f}, time={phys_dict['loss_time']:.4f}, "
                        f"norm={phys_dict['loss_norm']:.4f}, lr={scheduler.get_last_lr()[0]:.4e}"
                    )

                # GPU 显存监控（可选）
                if config['device'] == 'cuda' and global_step % (config['log_interval'] * 5) == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    tqdm.write(f"GPU Memory: allocated {allocated:.2f} GB, reserved {reserved:.2f} GB")

            # ---------- 验证 ----------
            if global_step % config['eval_interval'] == 0:
                val_loss, val_ce, val_phys = evaluate()
                tqdm.write(
                    f"Step {global_step}: Val Loss: {val_loss:.4f}, "
                    f"Val CE: {val_ce:.4f}, Val Phys: {val_phys:.4f}"
                )
                if USE_TENSORBOARD:
                    writer.add_scalar('Loss/val_total', val_loss, global_step)
                    writer.add_scalar('Loss/val_ce', val_ce, global_step)
                    writer.add_scalar('Loss/val_phys', val_phys, global_step)

                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(global_step, val_loss)
                    if USE_FILE_LOG:
                        logging.info(f"Best model saved at step {global_step} with val loss {val_loss:.4f}")

                # 早停判断
                if early_stopping.step(val_loss):
                    tqdm.write(f"Early stopping triggered at step {global_step}")
                    if USE_FILE_LOG:
                        logging.info(f"Early stopping triggered at step {global_step}")
                    # 如果希望自动加载最佳模型，可在这里加载 best_checkpoint_path
                    # checkpoint = torch.load(best_checkpoint_path)
                    # model.load_state_dict(checkpoint['model_state_dict'])
                    return  # 终止训练

            if global_step >= total_steps:
                break

        # ---------- 每个 Epoch 结束输出平均损失 ----------
        avg_loss = epoch_loss / epoch_steps
        avg_ce = epoch_ce / epoch_steps
        avg_phys = epoch_phys / epoch_steps
        tqdm.write(
            f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, "
            f"Avg CE: {avg_ce:.4f}, Avg Phys: {avg_phys:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.4e}"
        )
        if USE_FILE_LOG:
            logging.info(
                f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, "
                f"Avg CE: {avg_ce:.4f}, Avg Phys: {avg_phys:.4f}"
            )

        if global_step >= total_steps:
            break

    if USE_TENSORBOARD:
        writer.close()
        tqdm.write("TensorBoard writer closed.")


def evaluate(max_batches=2000):  # 限制验证 batch 数，避免过慢
    model.eval()
    total_ce = 0.0
    total_phys = 0.0
    total_steps = 0
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
            total_ce += ce_loss.item()
            total_phys += phys_loss.item()
            total_steps += 1
    model.train()
    avg_ce = total_ce / max(1, total_steps)
    avg_phys = total_phys / max(1, total_steps)
    return avg_ce + avg_phys, avg_ce, avg_phys


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
    train()