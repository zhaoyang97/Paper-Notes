---
title: >-
  [论文解读] SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction
description: >-
  [ACL 2025][自监督学习][手语表示学习] 提出 SHuBERT（Sign Hidden-Unit BERT），将语音自监督学习模型 HuBERT 的 masked cluster prediction 范式迁移到手语视频——对手部、面部、身体姿态四个流分别聚类并同时预测 masked 帧的聚类标签，在约 984 小时 ASL 视频上预训练后，在翻译/孤立识别/指拼检测多任务上达到公开数据 SOTA。
tags:
  - "ACL 2025"
  - "自监督学习"
  - "手语表示学习"
  - "自监督预训练"
  - "多流聚类预测"
  - "masked prediction"
  - "HuBERT"
---

# SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction

**会议**: ACL 2025  
**arXiv**: [2411.16765](https://arxiv.org/abs/2411.16765)  
**代码**: [http://shubert.pals.ttic.edu](http://shubert.pals.ttic.edu)  
**领域**: 自监督学习 / 手语处理  
**关键词**: 手语表示学习, 自监督预训练, 多流聚类预测, masked prediction, HuBERT

## 一句话总结
提出 SHuBERT（Sign Hidden-Unit BERT），将语音自监督学习模型 HuBERT 的 masked cluster prediction 范式迁移到手语视频——对手部、面部、身体姿态四个流分别聚类并同时预测 masked 帧的聚类标签，在约 984 小时 ASL 视频上预训练后，在翻译/孤立识别/指拼检测多任务上达到公开数据 SOTA。

## 研究背景与动机

**领域现状**：手语处理（翻译/识别）传统上依赖任务特定模型。预训练方法有两种：监督预训练（需要大量标注数据，如 6600 小时）和自监督预训练（如 MAE），但现有自监督方法要么学习的是上下文无关的帧/片段表示，要么只建模部分模态（如仅手部）。

**现有痛点**：(1) 手语数据稀缺，标注成本极高；(2) 手语是多通道的——手部、面部表情、身体姿态同时承载语义信息，单通道模型丢失关键信息；(3) 现有自监督方法（如 SSVP-SLT 的 MAE）计算量巨大（64×A100 训练 14 天）且仅处理 128 帧/8 秒片段，无法建模长程依赖。

**核心矛盾**：需要一种既能处理手语多通道特性、又能建模长程上下文、还计算高效的自监督表示学习方法。

**本文目标**：为手语视频学习统一的、上下文的、多通道的自监督表示。

**切入角度**：语音领域的 HuBERT 已成功通过 masked cluster prediction 学习上下文语音表示。手语和语音共享类似挑战（无预定义 token、变长单位、无显式边界），因此将 HuBERT 范式适配到多流手语输入。

**核心 idea**：四流输入（左手/右手/面部/身体姿态）→ 每流独立 k-means 聚类 → masked prediction 同时预测四个流的聚类标签 → 一个预训练模型适用于多个下游任务。

## 方法详解

### 整体框架
视频 → MediaPipe 提取关键点 → 裁剪手部/面部/身体姿态 → DINOv2 提取特征 → 四流 k-means 聚类生成伪标签 → Transformer encoder 做 masked cluster prediction → 微调到下游任务。

### 关键设计

1. **四流特征预处理**:

    - **左手/右手**：MediaPipe 检测手部关键点 → 裁剪+缩放到 224×224 → DINOv2（经手部数据 fine-tune）提取 384 维特征
    - **面部**：MediaPipe 检测面部 → 保留嘴部和眼部区域、其余灰化 → 高斯模糊（隐私保护）→ DINOv2（经面部数据 fine-tune）提取 384 维特征
    - **身体姿态**：7 个上身关键点（鼻、肩、肘、手腕），归一化为 14 维向量
    - 设计动机：手部 keypoint 估计在捕获手形方面不够准确，DINOv2 特征更好；面部处理在语言信息保留和隐私保护间取得平衡

2. **自监督训练（Masked Cluster Prediction）**:

    - 功能：预测 masked 帧的四流聚类标签
    - 核心思路：四流特征各自线性投影到 256 维 → 拼接为 1024 维/帧 → random span masking（span=3 帧≈200ms，约一个指拼字母的时长）→ 12 层 Transformer encoder → 四个线性分类头分别预测 masked 位置的 k-means 聚类标签（k=256）
    - 设计动机：每流独立聚类但联合预测，让模型学习跨流依赖。随机 masking 比通道 masking 和时间 masking 更有效

3. **多任务微调**:

    - **翻译（SLT）**：SHuBERT + ByT5 decoder，两阶段训练（YouTube-ASL 预训练→目标数据集微调）
    - **孤立识别（ISLR）**：SHuBERT + 线性分类头
    - **指拼检测**：SHuBERT + 二分类头（是否在指拼）
    - 使用学习的层权重加权组合所有 Transformer 层输出

### 训练策略
- 预训练：984 小时 ASL 视频，8×A6000 GPU，约 7 天，400K 步
- Adam optimizer，peak lr=5e-4，cosine schedule + linear warmup
- 86M 参数（12 层 Transformer，d=768，h=12）

## 实验关键数据

### 主实验（手语翻译，公开数据）

| 方法 | 自监督 | 预训练时长 | How2Sign BLEU↑ | OpenASL BLEU↑ | FLEURS-ASL BLEU↑ |
|------|--------|-----------|----------------|--------------|-------------------|
| Uthus 2023 | × | 984h | 12.4 | - | - |
| SSVP (Rust 2024) | ✓ | 1054h | 15.5 | - | - |
| Tanzer 2024 | × | 3207h | 15.4 | - | 4.4 |
| Uni-Sign | × | 984h | 14.9 | 23.1* | - |
| **SHuBERT** | **✓** | **984h** | **16.2** | **23.2** | **4.7** |

*Uni-Sign 预训练含 >72% OpenASL 测试集，不完全可比。

### 消融实验

| 配置 | How2Sign BLEURT |
|------|----------------|
| Random masking (默认) | **49.9** |
| Channel masking | 48.7 |
| Time masking | 49.1 |
| Frozen SHuBERT | 49.1 |
| Fine-tuned SHuBERT | 49.9 |

### 关键发现
- **一个预训练模型多任务 SOTA**：同一个 SHuBERT 在翻译/ISLR/指拼检测上都达到公开数据 SOTA，证明了表示的通用性
- **计算效率优势**：8×A6000 训练 7 天 vs SSVP 的 64×A100 训练 14 天（约 50× 计算量差异），得益于多流特征+compact 表示
- **冻结 SHuBERT 也很强**：frozen 设置下翻译质量仅微降（BLEURT 49.1 vs 49.9），说明预训练表示质量极高
- **多流联合建模必要**：四个流共同预测优于单独建模，跨流依赖（如手形+面部表情联合表达否定）被有效捕获
- **自然手语 > 翻译手语**：在包含自然 ASL 的 OpenASL 上提升最大（+10 BLEU vs baseline），说明预训练在 domain-similar 数据上效果更好

## 亮点与洞察
- **HuBERT→SHuBERT 的模态迁移**：将语音自监督范式适配到视觉手语是自然但非平凡的迁移——关键创新是多流聚类+联合预测替代了语音中的单流 HuBERT
- **隐私友好的面部表示**：灰化+模糊面部但保留嘴部/眼部区域，在隐私和语言信息保留间取得平衡
- **DINOv2 作为手语特征提取器**：对 DINOv2 做任务特定 continued pretraining（5M 手部/面部裁剪），比 keypoint 更准确

## 局限与展望
- 仅在 ASL 上验证，跨手语（DGS/BSL/CSL）泛化需探索
- 86M 参数 base 模型，scaling up 可能进一步提升
- 依赖 MediaPipe 手部检测（约 95% 准确率），检测失败需插值处理
- 未探索与辅助损失（对比学习、多任务联合训练）的结合
- 面部模糊可能丢失微妙的非手动标记（如眉毛运动）

## 相关工作与启发
- **vs SSVP-SLT (MAE)**：SSVP 用 MAE 重建图像像素，计算量巨大（64×A100×14天）且只处理 128 帧。SHuBERT 用聚类预测代替像素重建，高效得多
- **vs SignBERT+**: SignBERT+ 只建模手部姿态，缺少面部和身体；且预训练数据含下游测试数据。SHuBERT 四流联合建模，预训练/测试数据完全分离
- **vs HuBERT (语音)**：SHuBERT 是 HuBERT 在手语的自然扩展，核心区别是多流聚类+random span masking

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ HuBERT→手语的迁移设计精巧，多流聚类预测是针对手语特性的成功适配
- 实验充分度: ⭐⭐⭐⭐⭐ 三大任务六个 benchmark，详细消融，与私有数据方法对比
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，动机明确，图表优秀
- 价值: ⭐⭐⭐⭐⭐ 手语处理的基础模型突破，计算高效，开源可复现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](../../ICCV2025/self_supervised/scaling_languagefree_visual_representation_learning.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](../../NeurIPS2025/self_supervised/adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](../../NeurIPS2025/self_supervised/m-grpo_stabilizing_self-supervised_reinforcement_learning_for_multimodal_underst.md)
- [\[ACL 2025\] Improving Low-Resource Morphological Inflection via Self-Supervised Objectives](improving_low-resource_morphological_inflection_via_self-supervised_objectives.md)
- [\[ACL 2025\] WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning](whispa_semantically_and_psychologically_aligned_whisper_with_self-supervised_con.md)

</div>

<!-- RELATED:END -->
