import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from .SpacetimeConversationPipeline import SpacetimeConversationDataset
from torch.nn.utils.rnn import pad_sequence


# ---- 准备数据 ----
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

# ---- 初始化 ----
tokenizer = AutoTokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SpacetimeConversationDataset(
    conversations=sample_conversations,
    tokenizer_name='gpt2',
    max_length=64,
    time_mode='delta_seconds',
    space_mode='learned_space',
    vocab_size=tokenizer.vocab_size,
    d_model=128
)

# ---- 自定义collate（需绑定tokenizer） ----
def collate_with_global_tokenizer(batch, tokenizer=tokenizer):
    """
    注意：这里 tokenizer=tokenizer 将外部变量捕获为默认值
    """
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
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_with_global_tokenizer
)

# ---- 测试读取 ----
for batch in dataloader:
    print("input_ids shape:", batch['input_ids'].shape)      # [Batch, Seq]
    print("coords_raw shape:", batch['coords_raw'].shape)    # [Batch, Seq, 4]
    print("coords_raw (last dim):", batch['coords_raw'][0, :, 3])  # 时间t
    break