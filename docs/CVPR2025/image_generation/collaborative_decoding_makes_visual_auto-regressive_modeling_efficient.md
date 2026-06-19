---
title: >-
  [论文解读] Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient
description: >-
  [CVPR 2025][图像生成][视觉自回归] 提出 CoDe（协同解码），将 VAR 的多尺度推理分解为大模型草稿（低频小尺度）+ 小模型精修（高频大尺度）的协作流程，实现 1.7× 加速、50% 显存降低，FID 仅从 1.95 微增至 1.98。 领域现状：Visual Auto-Regressive (VAR) 模…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "视觉自回归"
  - "下一尺度预测"
  - "协同解码"
  - "推理加速"
  - "KV缓存优化"
---

# Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient

**会议**: CVPR 2025  
**arXiv**: [2411.17787](https://arxiv.org/abs/2411.17787)  
**代码**: [https://github.com/czg1225/CoDe](https://github.com/czg1225/CoDe)  
**领域**: 扩散模型 / 模型压缩  
**关键词**: 视觉自回归, 下一尺度预测, 协同解码, 推理加速, KV缓存优化

## 一句话总结
提出 CoDe（协同解码），将 VAR 的多尺度推理分解为大模型草稿（低频小尺度）+ 小模型精修（高频大尺度）的协作流程，实现 1.7× 加速、50% 显存降低，FID 仅从 1.95 微增至 1.98。

## 研究背景与动机

**领域现状**：Visual Auto-Regressive (VAR) 模型用下一尺度预测替代传统 GPT 式下一 token 预测，在 10 步内并行解码生成 256×256 图像，实现了质量和速度的双重突破。但 VAR 的渐进放大策略导致总 token 序列长达 680（是传统 AR 的 2.7 倍），KV 缓存消耗约占总显存的 80%（70GB 中 57GB 用于 KV 缓存）。

**现有痛点**：序列长度主要集中在最后几个尺度——最后一个尺度占 38% 的 token。自注意力计算随序列长度二次增长。用更小的 VAR 模型虽然快但质量下降明显（VAR-d20 FID 从 1.95 升到 2.61）。现有 AR 加速方法（如投机解码）针对下一 token 预测范式，不适用于 VAR 的下一尺度预测。

**核心矛盾**：VAR 在大尺度（高分辨率 token map）上计算开销最大，但参数需求最小——实验发现 2B 和 0.3B 模型在最后一个尺度上性能几乎相同。同时，小尺度和大尺度的生成模式完全不同（低频 vs 高频），单模型同时学习两者存在优化干扰。

**本文目标** 如何利用 VAR 大尺度参数冗余和跨尺度生成模式互斥这两个特性，显著提升推理效率。

**切入角度**：既然大尺度不需要大模型、小尺度不需要小模型，那就让大模型只做小尺度（草稿），小模型只做大尺度（精修）。每个模型专注自己的尺度范围后通过专门微调进一步消除跨尺度干扰。

**核心 idea**：将 VAR 的 10 步多尺度推理分成大模型前 N 步（低频草稿）和小模型后 K-N 步（高频精修），配合专门微调消除训练干扰。

## 方法详解

### 整体框架
使用预训练 VAR-d30（2B）作为 drafter，VAR-d16（0.3B）作为 refiner。Drafter 生成前 N 个尺度的 token map $R_L$（低频全局结构），释放 KV 缓存后，refiner 以 $R_L$ 为前缀续写后 K-N 个尺度的 token map $R_H$（高频细节）。两个模型分别在各自负责的尺度上进行专门微调。最终用残差式量化函数和多尺度 VQVAE 解码器重建图像。

### 关键设计

1. **大小模型协同解码**

    - 功能：在保持质量的前提下用小模型替代大模型处理计算密集的大尺度
    - 核心思路：Drafter（2B）负责前 N 步（计算稀疏但参数需求高），refiner（0.3B）负责后 K-N 步（计算密集但参数需求低）。关键发现支撑：最后 3 步占总推理时间的 64%，但小模型在这些步骤上仅比大模型性能差一点点。Refiner 在最后一步比 drafter 快 4.6×。Drafter 完成后释放 KV 缓存，refiner 仅需维护自己较少的 KV 缓存，显存大幅降低
    - 设计动机：傅里叶分析证实前 3 个尺度主要生成低频成分，后 3 个尺度生成高频成分，两者需要完全不同的能力

2. **专门微调（Specialized Fine-Tuning）**

    - 功能：消除跨尺度训练干扰，让每个模型在其负责的尺度上更精准
    - 核心思路：Drafter 用 CSE 损失在前 N 个尺度上微调（5% 原训练 epoch，lr=1e-6）。Refiner 用 KL 散度知识蒸馏从大模型学习，动态权重 $\lambda_{ep}$ 从 1 线性衰减到 0，逐渐将学习重心从全部 token 转移到精修 token（25% epoch，lr=1e-5）。实验证实仅微调精修尺度时如果意外影响到全局建模能力会导致 FID 从 3.30 暴涨到 21.93
    - 设计动机：预训练时单模型同时学低频和高频存在互相干扰，专门微调让每个模型只优化自己的尺度

3. **灵活的加速-质量 trade-off**

    - 功能：通过调节草稿步数 N 在速度和质量之间灵活权衡
    - 核心思路：N 越小，drafter 做的越少，refiner 做的越多，加速越大但质量略降。N=9（1.2× 加速）FID=1.94 甚至优于原模型；N=8（1.7×）FID=1.98；N=6（2.9×）FID=2.27。即使 training-free（不微调），CoDe 也优于同加速比的小 VAR 模型
    - 设计动机：不同应用场景对速度和质量的需求不同，灵活权衡增加实用性

### 损失函数 / 训练策略
Drafter: CSE 损失对齐生成分布与真值标签。Refiner: KL 散度蒸馏从大模型学习，动态权重逐渐聚焦精修尺度。4 × NVIDIA L20 GPU，batch 1024（梯度累积），AdamW 优化器。

## 实验关键数据

### 主实验（ImageNet-256 类条件生成）

| 方法 | 步数 | 加速↑ | 显存↓ | FID↓ | IS↑ |
|------|------|------|------|------|-----|
| VAR-d30 (原始) | 10 | 1.0× | 39.2GB | 1.95 | 301 |
| VAR-d24 | 10 | 1.7× | 25.1GB | 2.11 | 311 |
| **CoDe N=8** | 8+2 | **1.7×** | **21.0GB** | **1.98** | 302 |
| VAR-d20 | 10 | 2.8× | 17.8GB | 2.61 | 301 |
| **CoDe N=6** | 6+4 | **2.9×** | **19.9GB** | **2.27** | 297 |
| DiT-XL/2 | 50 | 0.2× | 11.4GB | 2.26 | 239 |
| LlamaGen-XXL | 384 | <0.1× | 42.6GB | 2.34 | 254 |

### 消融实验

| 配置 | N=6 FID | N=8 FID | 说明 |
|------|---------|---------|------|
| Training-free CoDe | 2.42 | 2.10 | 直接用预训练模型协作 |
| **+ 专门微调** | **2.27** | **1.98** | 微调消除干扰，显著提升 |

### 关键发现
- CoDe N=9 的 FID (1.94) 甚至优于原始 VAR-d30 (1.95)，说明专门微调消除了跨尺度干扰后模型更精准
- 同等加速比下 CoDe 显著优于直接用小 VAR 模型（N=8 FID 1.98 vs d24 FID 2.11）
- KV 缓存显存从 28.7GB 降到 4.1GB（bs=64），是主要的显存节省来源
- 最后一步 refiner 比 drafter 快 4.6×，验证了大尺度参数冗余的观察
- CoDe 是目前最快达到 FID<2 的方法

## 亮点与洞察
- **"大尺度不需要大模型"的观察**简单但有力：通过替换实验定量证明了 2B 和 0.3B 模型在最大尺度上性能相当，为协同解码提供了坚实的理论基础
- **低频/高频的尺度互斥性**通过傅里叶分析和扰动实验双重验证，解释了为什么单模型学所有尺度不是最优的
- **方法极其简单且实用**：不改架构、不改训练流程，仅在推理时切换模型+轻量微调

## 局限与展望
- 需要维护两个模型（2B + 0.3B），总参数量增加，虽推理时不同时在 GPU 上
- 专门微调仍需一定训练成本（drafter 5% epoch + refiner 25% epoch）
- 仅在 ImageNet-256 类条件生成上验证，text-to-image 等更复杂任务未测试
- 修改了采样超参（top-k 从 900 降到 600 + 温度 1.1），可能影响公平比较

## 相关工作与启发
- **vs 直接用小 VAR**: CoDe N=8 (1.98 FID, 1.7× 加速) 显著优于 VAR-d24 (2.11 FID, 1.7× 加速)，因为前几步仍用大模型保底
- **vs DiT-XL/2**: CoDe N=6 达到 2.27 FID 同时 15× 更快，展示了 VAR 范式在效率上的巨大优势
- **vs 投机解码 (LANTERN, SJD)**: 这些方法针对 next-token 预测，不适用于 VAR 的 next-scale 预测。CoDe 是专为 VAR 设计的加速方案

## 评分
- 新颖性: ⭐⭐⭐⭐ 观察驱动的简单方法，核心洞察（大尺度参数冗余+尺度互斥）有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 效率分析极其详细（时间/显存/吞吐量），消融和定性结果充分
- 写作质量: ⭐⭐⭐⭐⭐ 两个观察→方法设计的逻辑链清晰，图表优秀
- 价值: ⭐⭐⭐⭐ 最快达到 FID<2 的方法，对 VAR 部署有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)
- [\[CVPR 2025\] HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation](hmar_efficient_hierarchical_masked_auto-regressive_image_generation.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](../../NeurIPS2025/image_generation/distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)
- [\[CVPR 2025\] Visual-ERM: Reward Modeling for Visual Equivalence](visual-erm_reward_modeling_for_visual_equivalence.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)

</div>

<!-- RELATED:END -->
