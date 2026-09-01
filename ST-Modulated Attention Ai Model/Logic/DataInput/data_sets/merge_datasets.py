
import os
import json
import jieba
import glob
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import shutil
def merge_datasets(temp_paths, final_path):
    """合并所有临时分片"""
    if not temp_paths:
        return
    print(f"正在合并 {len(temp_paths)} 个临时分片...")
    combined = None
    for p in tqdm(temp_paths, desc="合并"):
        ds = load_from_disk(p)
        if combined is None:
            combined = ds
        else:
            combined = concatenate_datasets([combined, ds])
    combined.save_to_disk(final_path)
    print(f"✅ 最终数据集已保存，共 {len(combined)} 条样本，路径：{final_path}")
    # 可选：删除临时分片（确认合并成功后可手动或自动删除）
    # import shutil
    # for p in temp_paths:
    #     shutil.rmtree(p)
