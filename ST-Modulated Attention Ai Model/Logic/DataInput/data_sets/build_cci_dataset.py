# build_cci_dataset.py (分块保存版)
import os
import json
import jieba
import glob
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import shutil

# ===== 配置 =====
CCI_DATA_DIR = "I:\data\BAAI_CCI3_HQ\cci3_hq\part5"
OUTPUT_DB_DIR = "I:\data\database\cci_hq_dp\database_part5"
MAX_LEN = 512
TOKENIZER_NAME = "gpt2"
CHUNK_SIZE = 150000  # 每 5 万条保存一个临时分片

# ===== 初始化 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def process_document(line: str) -> dict:
    try:
        data = json.loads(line)
        text = data.get('text', '')
        if not text:
            return None
    except json.JSONDecodeError:
        return None

    seg_list = jieba.cut(text, cut_all=False)
    tokenized_text = ' '.join(seg_list)
    tokens = tokenizer(tokenized_text, truncation=True, max_length=MAX_LEN, add_special_tokens=False)['input_ids']
    if not tokens:
        return None

    time_features = [i / max(1, len(tokens)-1) for i in range(len(tokens))]
    return {
        'input_ids': tokens,
        'time_features': time_features,
    }

# ===== 主程序 =====
if __name__ == '__main__':
    if not os.path.exists(CCI_DATA_DIR):
        print(f"❌ 路径 {CCI_DATA_DIR} 不存在！")
        exit(1)

    if os.path.isfile(CCI_DATA_DIR):
        file_list = [CCI_DATA_DIR]
    else:
        file_list = glob.glob(os.path.join(CCI_DATA_DIR, "*.jsonl"))
        if not file_list:
            file_list = glob.glob(os.path.join(CCI_DATA_DIR, "*.json"))
        print(f"找到 {len(file_list)} 个数据文件")

    if not file_list:
        print("❌ 未找到任何数据文件。")
        exit(1)

    all_samples = []
    chunk_idx = 0
    temp_paths = []

    for file_path in file_list:
        print(f"📂 处理文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=os.path.basename(file_path)):
                record = process_document(line)
                if record:
                    all_samples.append(record)
                # 达到阈值则保存分片
                if len(all_samples) >= CHUNK_SIZE:
                    print(f"⚠️ 样本数达到 {CHUNK_SIZE}，正在保存临时分片 {chunk_idx}...")
                    # 保存临时分片
                    tmp_ds = Dataset.from_list(all_samples)
                    tmp_path = os.path.join(os.path.dirname(OUTPUT_DB_DIR), f"temp_chunk_{chunk_idx}")
                    tmp_ds.save_to_disk(tmp_path)
                    temp_paths.append(tmp_path)
                    chunk_idx += 1
                    all_samples = []  # 清空内存

    # 处理剩余样本
    if all_samples:
        print(f"⚠️ 保存最后的 {len(all_samples)} 个样本...")
        tmp_ds = Dataset.from_list(all_samples)
        tmp_path = os.path.join(os.path.dirname(OUTPUT_DB_DIR), f"temp_chunk_{chunk_idx}")
        tmp_ds.save_to_disk(tmp_path)
        temp_paths.append(tmp_path)

    # 合并所有临时分片
    if temp_paths:
        print(f"合并 {len(temp_paths)} 个临时分片...")
        combined_ds = None
        for p in temp_paths:
            ds = load_from_disk(p)
            if combined_ds is None:
                combined_ds = ds
            else:
                combined_ds = concatenate_datasets([combined_ds, ds])
        # 保存最终数据集
        os.makedirs(os.path.dirname(OUTPUT_DB_DIR), exist_ok=True)
        combined_ds.save_to_disk(OUTPUT_DB_DIR)
        print(f"✅ CCI3-HQ 数据集已建立，共 {len(combined_ds)} 个样本，保存在 {OUTPUT_DB_DIR}")
        # 清理临时文件
        for p in temp_paths:
            shutil.rmtree(p)
    else:
        print("❌ 没有生成任何样本，请检查数据。")