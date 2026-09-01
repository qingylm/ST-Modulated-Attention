import copy
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class SpacetimeDataAugmentor:
    def __init__(self,
                 time_scale_range=(0.8, 1.2),
                 time_shift_range=(-60, 60),
                 synonym_prob=0.2,
                 window_size=3,
                 window_stride=2):
        self.time_scale_range = time_scale_range
        self.time_shift_range = time_shift_range
        self.synonym_prob = synonym_prob
        self.window_size = window_size
        self.window_stride = window_stride

        # ---- 尝试加载 nlpaug（高级增强），若失败则降级 ----
        self.use_nlpaug = False
        self.aug_bert = None
        try:
            import nlpaug.augmenter.word as naw
            # 尝试加载模型，可能因版本问题失败
            try:
                self.aug_bert = naw.ContextualWordEmbsAug(
                    model_path='bert-base-chinese',
                    action="substitute",
                    aug_p=0.1
                )
                self.use_nlpaug = True
                print("✅ 已加载 nlpaug BERT 增强器")
            except Exception as e:
                print(f"⚠️ nlpaug BERT 模型加载失败：{e}，将使用内置同义词替换")
                self.use_nlpaug = False
        except ImportError:
            print("ℹ️ nlpaug 未安装，使用内置同义词替换")
            self.use_nlpaug = False

        # ---- 简易同义词词典 ----
        self.synonym_dict = {
            "今天": ["今日", "当天"],
            "天气": ["气候", "天象"],
            "怎么样": ["如何", "怎样"],
            "阳光明媚": ["晴空万里", "阳光灿烂"],
            "适合": ["适宜", "合适"],
            "出游": ["出行", "游玩"],
            "公园": ["园林", "绿地"],
            "Hello": ["Hi", "Hey"],
            "there": ["here", "you"],
        }

    def _parse_time(self, ts_str: str) -> datetime:
        try:
            return datetime.fromisoformat(ts_str.replace(' ', 'T'))
        except:
            return datetime.now()

    def _format_time(self, dt: datetime) -> str:
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    def _jitter_time(self, conv: List[Dict]) -> List[Dict]:
        if len(conv) < 2:
            return copy.deepcopy(conv)
        times = [self._parse_time(msg['timestamp']) for msg in conv]
        shift_seconds = random.uniform(*self.time_shift_range)
        base_time = times[0] + timedelta(seconds=shift_seconds)
        scale = random.uniform(*self.time_scale_range)
        scaled_times = [base_time]
        for i in range(1, len(times)):
            delta_seconds = (times[i] - times[i-1]).total_seconds()
            scaled_delta = delta_seconds * scale
            scaled_times.append(scaled_times[-1] + timedelta(seconds=scaled_delta))
        new_conv = copy.deepcopy(conv)
        for i, msg in enumerate(new_conv):
            msg['timestamp'] = self._format_time(scaled_times[i])
        return new_conv

    def _synonym_replace(self, conv: List[Dict]) -> List[Dict]:
        """同义词替换：优先使用 nlpaug，否则使用内置词典"""
        new_conv = copy.deepcopy(conv)
        if self.use_nlpaug and self.aug_bert is not None:
            # 使用 nlpaug 对整个文本进行增强
            for msg in new_conv:
                text = msg['content']
                augmented_text = self.aug_bert.augment(text)
                msg['content'] = augmented_text if augmented_text else text
            return new_conv
        else:
            # 内置词典替换
            for msg in new_conv:
                text = msg['content']
                for word, syns in self.synonym_dict.items():
                    if word in text and random.random() < self.synonym_prob:
                        text = text.replace(word, random.choice(syns), 1)
                msg['content'] = text
            return new_conv

    def _sliding_window_split(self, conv: List[Dict]) -> List[List[Dict]]:
        if len(conv) <= self.window_size:
            return [conv]
        windows = []
        for start in range(0, len(conv) - self.window_size + 1, self.window_stride):
            window = copy.deepcopy(conv[start:start + self.window_size])
            if window:
                base_ts = self._parse_time(window[0]['timestamp'])
                for msg in window:
                    dt = self._parse_time(msg['timestamp'])
                    new_dt = base_ts + (dt - base_ts)
                    msg['timestamp'] = self._format_time(new_dt)
            windows.append(window)
        return windows

    def augment(self, conv: List[Dict], count: int = 5) -> List[List[Dict]]:
        results = []
        windows = self._sliding_window_split(conv)
        for w in windows:
            results.append(w)
        for _ in range(count // 2 + 1):
            seed = random.choice(windows) if windows else conv
            temp = self._jitter_time(seed)
            temp = self._synonym_replace(temp)
            results.append(temp)
            temp2 = self._synonym_replace(seed)
            results.append(temp2)
        return results[:count]