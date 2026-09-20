# from torch.nn.utils.rnn import pad_sequence
# import torch
#
#
# def spacetime_collate_fn(batch):
#     """
#     将多个样本填充到相同长度
#     """
#     input_ids = [item['input_ids'] for item in batch]
#     attention_masks = [item['attention_mask'] for item in batch]
#     coords_raw = [item['coords_raw'] for item in batch]
#     lengths = [item['length'] for item in batch]
#
#     # 填充input_ids (用pad_token_id)
#     padded_input_ids = pad_sequence(
#         input_ids,
#         batch_first=True,
#         padding_value=tokenizer.pad_token_id  # 需要全局访问tokenizer
#     )
#
#     # 填充attention_mask (用0)
#     padded_masks = pad_sequence(
#         attention_masks,
#         batch_first=True,
#         padding_value=0
#     )
#
#     # 填充coords_raw (用0)
#     padded_coords = pad_sequence(
#         coords_raw,
#         batch_first=True,
#         padding_value=0.0
#     )
#
#     return {
#         'input_ids': padded_input_ids,
#         'attention_mask': padded_masks,
#         'coords_raw': padded_coords,
#         'lengths': torch.tensor(lengths)
#     }