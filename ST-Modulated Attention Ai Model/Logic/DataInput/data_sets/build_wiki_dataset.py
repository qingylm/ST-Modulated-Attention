import os
import glob
import json
import re
import jieba
from tqdm import tqdm
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer

# ===== 获取项目根目录 =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
print(f"项目根目录: {PROJECT_ROOT}")

# ===== 配置 =====
WIKI_DIR = os.path.join(PROJECT_ROOT, "data", "Wiki", "wiki_zh")
OUTPUT_DB_DIR = os.path.join(PROJECT_ROOT, "data", "database", "wiki_db")
MAX_LEN = 512
USER_DICT_PATH = os.path.join(PROJECT_ROOT, "userdict.txt")
TOKENIZER_NAME = "gpt2"

# ===== 加载自定义词典 =====
if os.path.exists(USER_DICT_PATH):
    jieba.load_userdict(USER_DICT_PATH)

# ===== 初始化 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_segments(text: str, max_len: int = MAX_LEN, overlap: int = 50):
    sentences = re.split(r'[。！？!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    segments = []
    current_seg = []
    current_len = 0
    for sent in sentences:
        sent_tokens = tokenizer.tokenize(sent) if tokenizer else []
        sent_len = len(sent_tokens) if sent_tokens else len(sent) // 2
        if current_len + sent_len > max_len and current_seg:
            segments.append(' '.join(current_seg))
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current_seg):
                s_len = len(tokenizer.tokenize(s)) if tokenizer else len(s)//2
                if overlap_len + s_len <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += s_len
                else:
                    break
            current_seg = overlap_sents
            current_len = overlap_len
        current_seg.append(sent)
        current_len += sent_len
    if current_seg:
        segments.append(' '.join(current_seg))
    return segments

def process_article(article: dict) -> list:
    text = article.get('text', '')
    if not text:
        return []
    text = clean_text(text)
    segments = split_into_segments(text, max_len=MAX_LEN, overlap=50)
    samples = []
    total_segments = len(segments)
    for idx, seg in enumerate(segments):
        seg_list = jieba.cut(seg, cut_all=False)
        tokenized_text = ' '.join(seg_list)
        tokens = tokenizer(tokenized_text, truncation=True, max_length=MAX_LEN, add_special_tokens=False)['input_ids']
        if not tokens:
            continue
        t_val = idx / max(1, total_segments - 1) if total_segments > 1 else 0.0
        time_features = [t_val] * len(tokens)
        samples.append({
            'input_ids': tokens,
            'time_features': time_features,
        })
    return samples

# ===== 主程序 =====
if __name__ == '__main__':
    if not os.path.exists(WIKI_DIR):
        print(f"❌ 路径 {WIKI_DIR} 不存在！")
        exit(1)

    if os.path.isfile(WIKI_DIR):
        file_list = [WIKI_DIR]
    else:
        # 递归搜索所有 .jsonl 文件（包括子目录）
        file_list = glob.glob(os.path.join(WIKI_DIR, "**", "*.jsonl"), recursive=True)
        if not file_list:
            # 尝试 .json 后缀
            file_list = glob.glob(os.path.join(WIKI_DIR, "**", "*.json"), recursive=True)
        if not file_list:
            # 尝试无后缀的 wiki_* 文件
            file_list = glob.glob(os.path.join(WIKI_DIR, "**", "wiki_*"), recursive=True)
        print(f"找到 {len(file_list)} 个文件")

    if not file_list:
        print("❌ 未找到任何数据文件，请检查目录结构。")
        exit(1)

    CHUNK_SIZE = 5000  # 每 5000 个样本保存为一个临时分片
    all_samples = []
    chunk_idx = 0
    temp_paths = []

    for file_path in file_list:
        print(f"📂 处理文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc=f"Processing {os.path.basename(file_path)}")):
                line = line.strip()
                if not line:
                    continue
                try:
                    article = json.loads(line)
                    samples = process_article(article)
                    all_samples.extend(samples)
                    # 达到阈值则保存并清空
                    if len(all_samples) >= CHUNK_SIZE:
                        tmp_ds = Dataset.from_list(all_samples)
                        tmp_path = os.path.join(OUTPUT_DB_DIR, f"temp_chunk_{chunk_idx}")
                        tmp_ds.save_to_disk(tmp_path)
                        temp_paths.append(tmp_path)
                        chunk_idx += 1
                        all_samples = []  # 清空内存
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第 {line_num} 行 JSON 解析失败：{e}")
                except Exception as e:
                    print(f"⚠️ 第 {line_num} 行处理异常：{e}")

    # 处理剩余样本
    if all_samples:
        tmp_ds = Dataset.from_list(all_samples)
        tmp_path = os.path.join(OUTPUT_DB_DIR, f"temp_chunk_{chunk_idx}")
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
        combined_ds.save_to_disk(OUTPUT_DB_DIR)
        print(f"✅ 维基百科数据集已建立，共 {len(combined_ds)} 个样本，保存在 {OUTPUT_DB_DIR}")
        # 清理临时文件
        import shutil

        for p in temp_paths:
            shutil.rmtree(p)
    else:
        print("❌ 没有生成任何样本，请检查数据。")