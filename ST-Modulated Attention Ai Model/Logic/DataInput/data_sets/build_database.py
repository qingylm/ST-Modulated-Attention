# build_database.py
import json
import os
import jieba
from datasets import Dataset
from transformers import AutoTokenizer
from datetime import datetime

# ===== 配置参数（可改为命令行参数） =====
MAX_LEN = 512
INPUT_JSONL = "conversations.jsonl"  # 输入文件路径
OUTPUT_DB_DIR = "data/database/conversations_db"  # 输出数据库目录
USER_DICT_PATH = "../../../userdict.txt"  # 自定义词典路径（与脚本同目录）

# ===== 加载自定义词典（如果存在） =====
if os.path.exists(USER_DICT_PATH):
    jieba.load_userdict(USER_DICT_PATH)
    print(f"✅ 已加载自定义词典：{USER_DICT_PATH}")
else:
    print("ℹ️ 未找到自定义词典，使用默认分词")

# ===== 初始化 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


def parse_timestamp(ts_str: str) -> datetime:
    """健壮的时间戳解析，支持多种格式"""
    try:
        # 尝试 ISO 格式 (YYYY-MM-DDTHH:MM:SS)
        return datetime.fromisoformat(ts_str.replace(' ', 'T'))
    except ValueError:
        try:
            # 尝试其他常见格式，如 YYYY-MM-DD HH:MM:SS
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # 如果都不行，使用当前时间并打印警告
            print(f"⚠️ 警告：无法解析时间戳 '{ts_str}'，使用当前时间")
            return datetime.now()


def process_conversation(conv_dict: dict, max_len: int = MAX_LEN):
    """
    处理单条对话，返回 input_ids 和 time_features
    conv_dict 必须包含 'messages' 键，每条消息有 'content' 和 'timestamp'
    """
    messages = conv_dict['messages']
    texts = [msg['content'] for msg in messages]
    timestamps = [msg['timestamp'] for msg in messages]

    # ---- 1. 计算时间特征（归一化秒差） ----
    try:
        base = parse_timestamp(timestamps[0])
        deltas = [(parse_timestamp(ts) - base).total_seconds() for ts in timestamps]
        max_delta = max(deltas) + 1e-8  # 避免除零
        time_feats = [d / max_delta for d in deltas]  # 每条消息一个标量
    except Exception as e:
        print(f"⚠️ 时间特征计算失败：{e}，全部设为 0")
        time_feats = [0.0] * len(messages)

    # ---- 2. 逐条消息进行 jieba 分词 + tokenization ----
    all_input_ids = []
    all_time_features = []

    for msg, t_val in zip(messages, time_feats):
        # 使用 jieba 精确模式分词
        seg_list = jieba.cut(msg['content'], cut_all=False)
        # 将分词结果用空格连接（保持词语边界）
        tokenized_text = ' '.join(seg_list)
        # 调用 tokenizer 转换为 token IDs（不添加特殊标记）
        tokens = tokenizer(tokenized_text, truncation=False, add_special_tokens=False)['input_ids']

        if not tokens:
            tokens = [tokenizer.pad_token_id]  # 占位

        all_input_ids.extend(tokens)
        # 为这条消息的所有 Token 分配相同的时间值
        all_time_features.extend([t_val] * len(tokens))

    # ---- 3. 截断到 max_len ----
    if len(all_input_ids) > max_len:
        all_input_ids = all_input_ids[:max_len]
        all_time_features = all_time_features[:max_len]

    return {
        'input_ids': all_input_ids,
        'time_features': all_time_features,
    }


# ---- 主程序 ----
if __name__ == "__main__":
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_JSONL):
        print(f"❌ 错误：输入文件 '{INPUT_JSONL}' 不存在！")
        print("请确保 'conversations.jsonl' 位于当前目录，或修改 INPUT_JSONL 变量指向正确路径。")
        exit(1)

    # 读取所有记录
    all_records = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as fp:
        for line_num, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
                record = process_conversation(conv)
                all_records.append(record)
            except json.JSONDecodeError as e:
                print(f"⚠️ 第 {line_num} 行 JSON 解析失败：{e}，已跳过")
            except KeyError as e:
                print(f"⚠️ 第 {line_num} 行缺少必要字段 {e}，已跳过")

    if not all_records:
        print("❌ 没有有效的记录，数据库未建立。")
        exit(1)

    # 保存为 Hugging Face Dataset
    dataset = Dataset.from_list(all_records)
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_DB_DIR), exist_ok=True)
    dataset.save_to_disk(OUTPUT_DB_DIR)
    print(f"✅ 数据库已建立，共 {len(dataset)} 条记录，保存在 {OUTPUT_DB_DIR}")