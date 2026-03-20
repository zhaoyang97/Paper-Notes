# DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime

**会议**: CVPR 2026  
**arXiv**: [2603.10538](https://arxiv.org/abs/2603.10538)  
**代码**: 待确认（作者声明 acceptance 后公开）  
**领域**: 场景图生成 / 视觉场景理解  
**关键词**: panoptic scene graph generation, real-time inference, bidirectional relation prediction, token pruning, low-latency  

## 一句话总结
DSFlash 通过合并分割与关系预测 backbone、双向关系预测头、动态 patch 剪枝等策略，将全景场景图生成速度提升至 RTX 3090 上 56 FPS，同时在 PSG 数据集上达到 mR@50=30.9 的 SOTA 性能。

## 背景与动机
场景图（Scene Graph）将图像结构化为节点（实例）和边（关系），在 VQA、推理、图像描述等任务中有广泛应用。现有 PSGG 方法几乎不关注延迟，一次推理往往数百毫秒，难以部署到边缘设备或实时系统。DSFormer 虽达到 SOTA 性能但推理耗时 458 ms，且使用两个独立 backbone（MaskDINO + ResNet），资源浪费严重。本文的核心洞察是：两阶段方法可以通过共享 backbone 特征、减少前向次数、剪枝无关 token 等手段实现极低延迟，同时不损失甚至提升场景图质量。

## 核心问题
如何在不牺牲场景图质量的前提下，让全景场景图生成达到实时级别的推理速度？

## 方法详解

### 整体框架
DSFlash 采用两阶段设计：第一阶段用冻结的 EoMT（Encoder-only Mask Transformer）分割模型提取分割 mask 与特征；第二阶段复用 EoMT 的中间特征（从 block 2/5/8/11 抽取 patch token 并拼接为 768×40×40 特征张量），通过 mask embedding 编码主体/客体位置，经轻量 Transformer neck 后由关系预测头输出关系类别。

### 关键设计

1. **Merged Backbone**：不再使用独立的分割与关系预测 backbone，而是直接抽取 EoMT 内部特征，省去了一次完整 backbone 前向推理。EoMT 全程冻结，训练时仅训 neck 和 head，单张 GTX 1080 不到 24 小时即可完成训练。

2. **双向关系预测（Gated Bidirectional Prediction）**：对于一对 mask (S₀, S₁)，原 DSFormer 需要两次前向分别预测 S₀→S₁ 和 S₁→S₀ 方向的关系。DSFlash 设计了一个门控分裂机制——将编码后的特征 x 通过 sigmoid 门控分成 t→ 和 t← 两个分支，共享同一个 MLP 关系头分别预测两个方向。训练时通过翻转 mask 顺序计算 consistency loss（MSE），确保模型对输入顺序等变。推理时只需一次前向即可得到双向预测。

3. **Mask-based Dynamic Patch Pruning**：在 mask embedding 阶段，与主体和客体 mask 均无重叠的 patch 不含有用的定位信息，直接丢弃后送入 neck。因为重叠率本就需要计算，剪枝几乎零开销。

4. **Raw-resolution Segmentation Masks**：不再将 EoMT 输出的 160×160 mask logits 上采样到原图分辨率再下采样，而是直接在低分辨率上计算 patch 重叠比例，省去了昂贵的双线性插值。

5. **Token Merging (ToMe-SD)**：在 backbone attention 层前合并相似 token，attention 后再 unmerge，降低注意力计算量，在老旧 GPU 上效果尤其明显（GTX 1080 延迟从 230ms 降至 173ms）。

### 损失函数 / 训练策略
- 关系分类：Binary Cross Entropy
- 双向一致性：MSE consistency loss（Eq. 7），约束翻转输入后中间特征应交换
- DeiT III 风格数据增强（随机翻转、颜色抖动、灰度/模糊/solarization 三选一）
- AdamW，lr=1e-5，cosine schedule + warmup，梯度裁剪 norm≤1，训练 20 epoch
- 每 5 个正样本采 1 个负样本

## 实验关键数据

| 方法 | mR@50 | 延迟 (ms) | 参数量 |
|------|-------|-----------|--------|
| DSFormer | 30.70 | 458 | 330M |
| REACT | 19.00 | 19 | 43M |
| HiLo-L | 19.08 | 427 | 230M |
| **DSFlash-L** | **30.90** | 50 | 340M |
| DSFlash-B* | 28.50 | 23 | 116M |
| DSFlash-S* | 25.05 | **18** | **40M** |

- DSFlash-L 在 mR@50 上超越 DSFormer（30.9 vs 30.7），延迟仅为其 1/9
- DSFlash-S* 仅 40M 参数、18ms 延迟（56 FPS），性能仍优于 REACT 和 HiLo

### 消融实验要点
- 统一 backbone 将延迟从 458ms 降至 41ms（-91%），但 mR@50 从 30.7 降至 25.0
- 高效 mask embedding：延迟 37ms（-10%），mR@50 不变
- 门控双向预测：延迟 29ms（-22%），mR@50 从 25.0 提至 28.8（额外监督信号带来性能提升）
- 跳过 mask 上采样：延迟 23ms（-21%），mR@50=28.5（轻微下降）
- mR@50 与分割模型的 Panoptic Quality 相关系数高达 0.99

## 亮点
- 实现了首个真正实时的全景场景图生成系统，GTX 1080 上也能以 ~6 FPS 运行
- 双向关系预测设计精巧，通过一次前向同时输出两个方向，还借助 consistency loss 提升质量
- 整体设计简洁实用：冻结 backbone + 轻量 neck + 共享 head，训练成本极低
- 对评估协议的严谨态度值得肯定：严格遵循 SingleMPO 避免多 mask 膨胀 R@k

## 局限性 / 可改进方向
- Backbone 冻结意味着关系预测无法反向影响特征提取，可能限制上限
- PSG 数据集偏小（49k 图像），在更大数据集上的表现未知
- 低分辨率 mask 对小目标的分割精度可能不足
- 双向预测共享 MLP head，可能在谓词方向性强的关系上有信息混淆
- 作者提到主客体混淆是常见失败模式，可考虑对比学习解决

## 与相关工作的对比
- vs DSFormer：继承其 mask embedding 和 strictly decoupled 思想，但通过 backbone 合并和双向预测将延迟降低 9×
- vs REACT：REACT 用 YOLOv8 做 bbox 检测而非全景分割，DSFlash 在 PSGG 设定下性能高出 12 个 mR@50 点
- vs HiLo：一阶段方法，性能（19.08 mR@50）远逊于 DSFlash，延迟也更高

## 启发与关联
- 冻结 backbone + 复用中间特征的思路可以推广到其他两阶段视觉任务
- 双向预测 + consistency loss 的设计思路可借鉴到检测中的方向关系建模
- 动态 patch 剪枝利用任务先验（mask 覆盖）实现零开销加速，适用于类似的 mask-conditioned 架构

## 评分
- 新颖性: ⭐⭐⭐⭐ 双向预测和 mask-based 剪枝在 PSGG 中是新的，系统级优化很到位
- 实验充分度: ⭐⭐⭐⭐ 多 GPU 延迟评估、详尽消融、公平评估协议
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，对评估问题的讨论很有价值
- 价值: ⭐⭐⭐⭐ 将 PSGG 带入实时领域，实用性强，对资源受限场景特别有意义
