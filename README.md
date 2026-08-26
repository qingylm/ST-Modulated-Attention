# Spacetime_Modulated Attention
将闵可夫斯基时空注入Transformer注意力机制

## 概述
Spacetime-Modulated Attention (STMA) 是一个受狭义相对论启发的注意力机制，它将输入的每个Token映射为四维时空坐标 (x,y,z,t)，并用闵可夫斯基内积替代标准点积，同时引入光锥因果掩码和物理正则化，使模型能够感知并利用事件之间的因果关系和时空距离。

该算法已从零实现为完整的PyTorch训练框架，包含数据管道、滑动窗口缓存、物理损失函数和完整的训练循环

## 特性
闵可夫斯基注意力：用 Q⋅K=Qspace⋅Kspace−α⋅Qtime⋅Ktime 计算相似度。

光锥因果掩码：强制未来Token被硬掩蔽，类空（超光速）关联被软惩罚。

物理正则化损失：惩罚类空间隔，鼓励时间方差，保持坐标稳定。

滑动窗口缓存：支持无限长序列，复杂度 O(L⋅W)，其中 WW 为窗口大小。

可学习时空嵌入：为每个Token动态生成 (x,y,z,t)，或使用固定嵌入。

完整数据管道：自动将对话日志转换为时空坐标，支持时间戳解析。

## 算法原理
### 时空坐标建模
每个Token被表示为四维向量 (x,y,z,iw)，其中：

    x,y,z 表示语义空间位置（可学习或固定）。

    w 表示时间（实部），通过 iw 在闵可夫斯基内积中实现符号翻转。

### 注意力分数计算
<img width="472" height="78" alt="图片" src="https://github.com/user-attachments/assets/25d7a7f6-01ce-4f78-951a-ff03fa61ef03" />
​

其中 α 是可学习的耦合系数。
### 光锥掩码
因果约束：若 tj>ti（未来），则 Logitsij=−∞

类空惩罚：若 (xi−xj)**2+(yi−yj)**2+(zi−zj)**2>(ti−tj)**2，则减去可学习偏置 b。
### 物理正则化
<img width="688" height="98" alt="图片" src="https://github.com/user-attachments/assets/08a3b948-c4f9-4068-bec3-114eb6273ee6" />

### 架构组件

<img width="836" height="615" alt="图片" src="https://github.com/user-attachments/assets/92a145da-8923-4c5c-953c-c07dad2eacbf" />
