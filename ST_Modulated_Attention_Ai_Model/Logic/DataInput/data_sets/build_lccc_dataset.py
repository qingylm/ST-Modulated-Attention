# build_lccc_dataset.py
import os
import json
import jieba
import glob
import numpy as np
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import shutil

# ===== 配置 =====
LCCC_DATA_DIR = "I:\\data\\LCCC\\LCCC-base-split"          # 存放三个 json 的目录
OUTPUT_DB_DIR = "I:\\data\\database\\lccc_db"              # 最终输出目录（里面会有 train/valid/test）
MAX_LEN = 512                    # 每个样本最大 token 数（对话总长）
TOKENIZER_NAME = "gpt2"
CHUNK_SIZE = 150000               # 每 5 万条对话保存一个临时分片

# ===== 初始化 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===== 角色空间坐标映射（可自定义） =====
ROLE_SPACE = {
    "user":      [0.0, 0.0, 0.0],
    "assistant": [1.0, 0.0, 0.0],
}
# 若对话中出现其他角色（如 system），可自行添加

def process_conversation(conv: list) -> dict | None:
    """
    输入：对话列表，每个元素是字符串（一个轮次）
    输出：包含 input_ids, attention_mask, coords_raw 的字典
    coords_raw 是 (seq_len, 4) 的浮点数列表
    """
    all_token_ids = []
    all_coords = []
    role_sequence = []   # 记录每条消息的角色

    # 先分词并获取每个句子的 token ids
    for idx, text in enumerate(conv):
        # 角色按奇偶交替（0->user, 1->assistant, 2->user...）
        role = "user" if idx % 2 == 0 else "assistant"
        # 分词（与 CCI 脚本一致，使用 jieba 分词 + 空格连接）
        seg_list = jieba.cut(text, cut_all=False)
        tokenized_text = ' '.join(seg_list)
        tokens = tokenizer.tokenize(tokenized_text)   # 转成 token 字符串列表
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # 截断单个消息过长（防止整体超长）
        if len(token_ids) > MAX_LEN // 2:
            token_ids = token_ids[:MAX_LEN // 2]
        if not token_ids:
            continue
        all_token_ids.extend(token_ids)
        # 记录每个 token 属于哪个角色
        role_sequence.extend([role] * len(token_ids))

    if not all_token_ids:
        return None

    # 整体截断
    if len(all_token_ids) > MAX_LEN:
        all_token_ids = all_token_ids[:MAX_LEN]
        role_sequence = role_sequence[:MAX_LEN]

    # 构建坐标
    seq_len = len(all_token_ids)
    # 先计算每个 token 的时间归一化值（基于消息级别）
    # 我们需要知道每个 token 对应的消息索引，以便为同一消息内的 token 分配相同时间
    # 由于 role_sequence 已经记录了每个 token 的角色，我们通过遍历构建 msg_idx
    # 简便方法：累计每个消息的长度
    msg_idx = []
    msg_counter = 0
    for i, token in enumerate(all_token_ids):
        # 判断当前 token 是否属于新的消息（即角色改变）
        if i == 0:
            msg_counter = 0
        elif role_sequence[i] != role_sequence[i-1]:
            msg_counter += 1
        msg_idx.append(msg_counter)
    total_msgs = max(msg_idx) + 1 if msg_idx else 1
    # 时间 t = msg_idx / (total_msgs - 1) 归一化到 [0,1]
    time_t = [idx / max(1, total_msgs - 1) for idx in msg_idx]

    coords_raw = []
    for idx, token in enumerate(all_token_ids):
        role = role_sequence[idx]
        space = ROLE_SPACE.get(role, [0.5, 0.5, 0.5])   # 未知角色用中间值
        t = time_t[idx]
        coords_raw.append(space + [t])   # 四维 [x, y, z, t]

    return {
        'input_ids': all_token_ids,
        'attention_mask': [1] * len(all_token_ids),
        'coords_raw': coords_raw,   # 二维列表，每个元素长度为4
        'length': len(all_token_ids)
    }

def process_one_json(input_path, output_dir, split_name):
    """处理单个 JSON 文件，生成一个 dataset 分片并保存"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 处理 {split_name}: {input_path}")

    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return None

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)   # LCCC 是顶层列表

    all_samples = []
    chunk_idx = 0
    temp_paths = []

    for conv in tqdm(data, desc=f"处理 {split_name}"):
        record = process_conversation(conv)
        if record:
            all_samples.append(record)
        if len(all_samples) >= CHUNK_SIZE:
            print(f"⚠️ 保存临时分片 {chunk_idx}...")
            tmp_ds = Dataset.from_list(all_samples)
            tmp_path = os.path.join(output_dir, f"temp_{split_name}_{chunk_idx}")
            tmp_ds.save_to_disk(tmp_path)
            temp_paths.append(tmp_path)
            chunk_idx += 1
            all_samples = []

    if all_samples:
        print(f"⚠️ 保存最后 {len(all_samples)} 个样本...")
        tmp_ds = Dataset.from_list(all_samples)
        tmp_path = os.path.join(output_dir, f"temp_{split_name}_{chunk_idx}")
        tmp_ds.save_to_disk(tmp_path)
        temp_paths.append(tmp_path)

    if not temp_paths:
        print(f"❌ {split_name} 没有生成有效样本")
        return None

    # 合并分片
    print(f"合并 {len(temp_paths)} 个临时分片...")
    combined_ds = None
    for p in temp_paths:
        ds = load_from_disk(p)
        if combined_ds is None:
            combined_ds = ds
        else:
            combined_ds = concatenate_datasets([combined_ds, ds])

    final_path = os.path.join(output_dir, split_name)
    combined_ds.save_to_disk(final_path)
    print(f"✅ {split_name} 完成，共 {len(combined_ds)} 样本，保存在 {final_path}")

    # 清理临时文件
    for p in temp_paths:
        shutil.rmtree(p)

    return final_path


if __name__ == '__main__':
    # 三种 split
    splits = ['valid', 'test']
    for split in splits:
        json_path = os.path.join(LCCC_DATA_DIR, f"LCCC-base_{split}.json")
        if os.path.exists(json_path):
            process_one_json(json_path, OUTPUT_DB_DIR, split)
        else:
            print(f"⚠️ 跳过 {split}，文件不存在: {json_path}")
    print("🎉 全部处理完成！")