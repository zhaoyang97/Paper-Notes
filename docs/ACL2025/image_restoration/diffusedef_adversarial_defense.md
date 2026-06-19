---
title: >-
  [论文解读] DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising
description: >-
  [ACL 2025][图像恢复][对抗防御] DiffuseDef 在编码器与分类器之间插入一个扩散去噪层，训练时学习预测隐状态噪声，推理时对隐表示加噪→迭代去噪→集成平均，以即插即用的方式大幅提升文本分类模型在黑盒和白盒对抗攻击下的鲁棒性。 - 领域现状： 预训练语言模型 (PLM) 在文本分类、自然语言推理等任务上性能优…
tags:
  - "ACL 2025"
  - "图像恢复"
  - "对抗防御"
  - "扩散模型"
  - "文本对抗攻击"
  - "隐状态去噪"
  - "集成鲁棒性"
---

# DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising

**会议**: ACL 2025  
**arXiv**: [2407.00248](https://arxiv.org/abs/2407.00248)  
**代码**: [https://github.com/Nickeilf/DiffuseDef](https://github.com/Nickeilf/DiffuseDef)  
**机构**: Imperial College London (LAMA Lab)
**领域**: 图像复原  
**关键词**: 对抗防御, 扩散模型, 文本对抗攻击, 隐状态去噪, 集成鲁棒性

## 一句话总结

DiffuseDef 在编码器与分类器之间插入一个扩散去噪层，训练时学习预测隐状态噪声，推理时对隐表示加噪→迭代去噪→集成平均，以即插即用的方式大幅提升文本分类模型在黑盒和白盒对抗攻击下的鲁棒性。

## 研究背景与动机

- **领域现状**: 预训练语言模型 (PLM) 在文本分类、自然语言推理等任务上性能优异，但仍极易受到对抗文本攻击的威胁——攻击者仅需替换几个同义词或引入拼写变体即可使模型预测翻转。现有防御方法可分为对抗训练、集成和去噪三类，各有明显局限。
- **现有方法痛点**:
    - **对抗训练 (FreeLB++, InfoBERT 等)**: 假设训练时与测试时的攻击类型相似，容易过拟合到特定攻击方式，泛化性差
    - **集成方法 (RanMask, SAFER 等)**: 需要对每个输入变体做完整前向传播 (10 个集成 = 10x FLOPS)，推理效率极低
    - **去噪方法 (RMLM 等)**: 在文本/标签层面去噪，可能大幅改变干净文本的语义表示，导致干净样本性能下降
- **核心矛盾**: 如何在不损失干净文本分类精度的前提下，高效地去除对抗扰动并泛化到未知攻击类型
- **本文切入角度**: 扩散模型天然擅长预测和去除噪声，而对抗扰动可以类比为隐空间中的"噪声"——将扩散去噪从**图像像素空间**迁移到 **NLP 隐状态空间**，同时利用集成但仅在轻量扩散层执行以兼顾效率
- **核心 idea**: 用即插即用的扩散层在隐状态空间迭代去噪对抗扰动，并通过多重采样集成增强鲁棒性

## 方法详解

### 整体框架

两阶段训练 + 三步推理:

1. **对抗训练阶段**: 用 FreeLB++ 或 RSMI 等方法训练编码器 + 分类器，获得基础鲁棒性
2. **扩散训练阶段**: 冻结编码器和分类器，仅训练扩散层 (单层 Transformer + 时间嵌入) 学习预测隐状态中注入的随机高斯噪声
3. **推理阶段**: 编码器提取隐状态 $h$ → 采样 $k$ 个噪声向量加入生成 $k$ 个含噪变体 → 扩散层对每个变体执行 $t'$ 步反向扩散去噪 → 平均所有去噪隐状态 → 分类器输出最终预测

### 关键设计

1. **扩散层去噪器 (Diffusion Denoiser)**

    - **结构**: 单层 Transformer 编码器 + 正弦时间嵌入，仅增加约 10M 参数 (BERT 为 110M)
    - **训练**: 对干净隐状态 $h$ 按前向扩散公式加噪 $h_t = \sqrt{\bar\alpha_t} h + \sqrt{1 - \bar\alpha_t} \epsilon$，扩散层学习预测噪声 $\epsilon_\theta(h_t, t)$，损失为 MSE: $L = \mathbb{E}_{t,h,\epsilon}[\|\epsilon - \epsilon_\theta(h_t, t)\|^2]$
    - **设计动机**: 对抗扰动在隐空间表现为偏移/噪声，扩散去噪天然适合消除这类扰动；在隐空间而非输入空间操作避免了文本离散性带来的困难

2. **加噪-去噪-集成推理管线 (Noise-Denoise-Ensemble Pipeline)**

    - **加噪**: 采样 $k=10$ 个独立高斯噪声向量，对隐状态 $h$ 执行 1 步前向扩散，生成 $k$ 个含噪变体 $H_{t'} = [h_{t'}^0, ..., h_{t'}^k]$
    - **去噪**: 每个变体经 $t'=5$ 步反向扩散 (DDPM)，逐步减去扩散层预测的噪声
    - **集成**: 对所有去噪隐状态求平均 $\text{avg}(H_0)$，送入分类器输出最终标签
    - **设计动机**: 加噪引入随机性使攻击者无法找到稳定的脆弱词；集成仅在扩散层 (10M 参数) 上做，无需重跑编码器 (110M 参数)，效率远高于传统全模型集成

3. **即插即用兼容设计 (Plug-and-Play Compatibility)**

    - **解耦**: 扩散训练阶段编码器和分类器完全冻结，扩散层独立训练，因此可叠加在任何已有的对抗训练方法之上
    - **灵活性**: 实验验证了 FreeLB++、RSMI、对抗数据增强等多种基座方法均可通过加装 DiffuseDef 进一步提升鲁棒性
    - **设计动机**: 避免方法间的耦合限制，使 DiffuseDef 成为通用的鲁棒性增强模块

### 训练与推理超参

| 超参 | AGNews / QNLI | IMDB |
|------|--------------|------|
| 最大训练时间步 $t$ | 30 | 10 |
| 推理去噪步数 $t'$ | 5 | 5 |
| 集成数量 $k$ | 10 | 10 |
| $\beta$ schedule | 线性, $10^{-4}$ → $0.02$ | 同左 |
| 对抗训练 epoch | 10 | 10 |
| 扩散训练 epoch | 100 | 100 |

## 实验关键数据

### 主实验：黑盒攻击鲁棒性 (BERT backbone, AUA%)

| 方法 | Clean% | TextFooler | TextBugger | BERT-Attack | 平均 #Query |
|------|--------|-----------|-----------|-------------|-----------|
| Fine-tuned | 94.4 | 10.2 | 25.4 | 27.1 | ~366 |
| FreeLB++ | 95.0 | 54.7 | 56.5 | 44.6 | ~415 |
| RSMI | 94.3 | 52.6 | 56.7 | 55.4 | ~701 |
| ATINTER | 94.2 | 68.0 | 59.0 | 81.0 | ~295 |
| **DiffuseDef-FreeLB++** | **94.8** | **84.5** | **86.0** | **84.6** | **~920** |
| **DiffuseDef-RSMI** | **93.8** | **82.7** | **83.3** | **84.4** | **~951** |

> AGNews 数据集。DiffuseDef 在几乎不损失干净准确率的情况下，AUA 平均提升约 30 个百分点，攻击所需查询数增加 2-3 倍。IMDB 和 QNLI 上也有类似趋势。

### 白盒攻击鲁棒性 (AUA%)

| 方法 | AGNews T-PGD | AGNews SemAttack | IMDB T-PGD | IMDB SemAttack |
|------|-------------|-----------------|-----------|---------------|
| Fine-tuned | 8.8 | 41.5 | 3.0 | 1.3 |
| FreeLB++ | 19.6 | 58.2 | 15.5 | 3.4 |
| **DiffuseDef-FreeLB++** | **59.4** | **68.1** | **50.3** | **28.2** |
| RSMI | 79.6 | n/a | 43.3 | n/a |
| **DiffuseDef-RSMI** | **81.7** | n/a | **48.4** | n/a |

### 消融实验 (AGNews, AUA%)

| 配置 | TextFooler | TextBugger | BERT-Attack |
|------|-----------|-----------|-------------|
| DiffuseDef (完整) | 84.5 | 86.0 | 84.6 |
| 去掉集成 | 64.2 | 65.8 | 54.9 |
| 去掉扩散去噪 | 57.1 | 58.4 | 47.7 |
| 去掉对抗训练 | 80.3 | 80.9 | 80.5 |

> 扩散去噪和集成是鲁棒性提升的主要来源；去掉对抗训练影响相对较小，说明 DiffuseDef 不依赖特定对抗训练方法。

### 效率对比

| 方法 | 参数量 | 推理 FLOPS | 训练时间 |
|------|-------|-----------|---------|
| Fine-tuned BERT | 110M | 46G | 1x |
| FreeLB++ | 110M | 46G | 10.5x |
| RanMask (k=10) | 110M | 459G | 1.2x |
| SAFER (k=10) | 110M | 459G | 1x |
| DiffuseDef (t'=1, k=10) | 120M | 96G | 1.1x |
| DiffuseDef (t'=5, k=10) | 120M | 267G | 1.1x |

> DiffuseDef 的推理 FLOPS 仅为同等集成数传统方法 (459G) 的 21%~58%，训练时间也远低于 FreeLB++ (10.5x)。

### 隐空间距离分析

| 方法 | L2 距离 | Cosine 距离 |
|------|--------|------------|
| FreeLB++ | 12.53 | 0.35 |
| DiffuseDef-FreeLB++ | 10.66 | 0.27 |
| RSMI | 9.72 | 0.24 |
| DiffuseDef-RSMI | 8.61 | 0.21 |

> 应用 DiffuseDef 后，对抗隐状态与干净隐状态的距离显著缩小，说明去噪+集成有效将对抗表示拉近干净表示。

## 亮点与洞察

- **隐空间去噪 vs. 输入/标签空间去噪**: 不同于 RMLM (文本层面) 和 Yuan et al. (标签层面)，DiffuseDef 直接在最终隐层去噪，避免了文本离散性问题且更高效
- **扰动 = 噪声的类比**: 将对抗扰动视为隐空间噪声是一个优雅的抽象，使得扩散模型的去噪机制自然适配对抗防御
- **集成效率突破**: 传统集成需 10x FLOPS，DiffuseDef 仅需 ~2x-6x，因为集成只发生在轻量扩散层
- **攻击难度倍增**: 随机加噪使得模型输出具有非确定性，攻击者需 2-3 倍查询才能找到脆弱词

## 局限与展望

- 仅在分类任务 (AGNews, IMDB, QNLI) 上验证，生成任务 (翻译、摘要) 和序列标注任务的适用性未探索
- 推理 FLOPS 仍高于非集成方法 (如 FreeLB++)，对延迟敏感场景有一定限制
- 去噪步数 $t'$ 和集成数 $k$ 需按数据集调优，缺乏自适应机制
- 对语义改写类攻击 (整句换述而非词级扰动) 的防御效果未验证
- 扩散层固定为单层 Transformer，更深或不同架构的扩散去噪器尚未探索

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-----------|------|
| 新颖性 | 8 | 首次将扩散去噪应用于 NLP 隐状态空间进行对抗防御，思路新颖且有效 |
| 有效性 | 9 | 三个数据集、五种攻击下全面 SOTA，AUA 平均提升 ~30%，消融实验完整 |
| 实用性 | 7 | 即插即用设计好，但推理 FLOPS 增加和超参调优是实际部署的障碍 |
| 表达质量 | 8 | 论文结构清晰，公式推导完整，消融和效率分析全面 |
- **vs ATINTER**: ATINTER 用 T5 重写对抗文本，DiffuseDef 直接在表示空间操作更高效
- **vs CV 中的 DiffPure**: DiffPure 对整个输入图像做扩散净化，DiffuseDef 只在最后一层隐状态做

## 评分
- 新颖性: ⭐⭐⭐⭐ 扩散去噪用于NLP对抗防御的思路新颖
- 实验充分度: ⭐⭐⭐⭐ 3数据集+5种攻击+多基线+白盒+效率分析
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法框图直观
- 价值: ⭐⭐⭐⭐ 即插即用的防御方法，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising](../../ICCV2025/image_restoration/idf_iterative_dynamic_filtering_networks_for_generalizable_image_denoising.md)
- [\[ICLR 2026\] Are Deep Speech Denoising Models Robust to Adversarial Noise?](../../ICLR2026/image_restoration/are_deep_speech_denoising_models_robust_to_adversarial_noise.md)
- [\[CVPR 2025\] AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](../../CVPR2025/image_restoration/adversarial_diffusion_compression_for_real-world_image_super-resolution.md)
- [\[ACL 2025\] A Self-Denoising Model for Robust Few-Shot Relation Extraction](a_self-denoising_model_for_robust_few-shot_relation_extraction.md)
- [\[ICML 2025\] ε-VAE: Denoising as Visual Decoding](../../ICML2025/image_restoration/epsilon-vae_denoising_as_visual_decoding.md)

</div>

<!-- RELATED:END -->
