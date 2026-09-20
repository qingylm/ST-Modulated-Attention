"""
SubsequentProcessing: 后续处理组件。

仅包含 LearnableSpacetimeNormalizer。
注意: 本包**不再**提供 SpacetimeAttentionWithCache —— 那份实现是
CoreAttention/SpacetimeAttentionWithCache.py 的早期草稿（省略号未填充、
self.W_q_s 等从未定义、space_norm_type='rms' 是非法枚举会静默跳过归一化、
模块底部还有 test_normalizer() 的导入副作用），已删除以免同类名两份实现
造成误用。需要注意力层请从 Logic.CoreAttention 导入。
"""

from .LearnableSpacetimeNormalizer import LearnableSpacetimeNormalizer

__all__ = [
    'LearnableSpacetimeNormalizer',
]
