---
title: >-
  [论文解读] SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders
description: >-
  [ECCV 2024][视频理解][骨架动作识别] SA-DVAE 首次将特征解耦引入骨架零样本动作识别，通过双头 VAE 将骨架特征分离为语义相关和语义无关两个独立部分，仅用语义相关部分与文本对齐，配合对抗性总相关惩罚增强解耦效果，在 NTU RGB+D 60/120 和 PKU-MMD 三个基准上达到 SOTA。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "骨架动作识别"
  - "零样本学习"
  - "特征解耦"
  - "变分自编码器"
  - "跨模态对齐"
---

# SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders

**会议**: ECCV 2024  
**arXiv**: [2407.13460](https://arxiv.org/abs/2407.13460)  
**代码**: [https://github.com/pha123661/SA-DVAE](https://github.com/pha123661/SA-DVAE)  
**领域**: 视频理解  
**关键词**: 骨架动作识别, 零样本学习, 特征解耦, 变分自编码器, 跨模态对齐

## 一句话总结

SA-DVAE 首次将特征解耦引入骨架零样本动作识别，通过双头 VAE 将骨架特征分离为语义相关和语义无关两个独立部分，仅用语义相关部分与文本对齐，配合对抗性总相关惩罚增强解耦效果，在 NTU RGB+D 60/120 和 PKU-MMD 三个基准上达到 SOTA。

## 研究背景与动机

骨架动作识别因对外观和背景变化的鲁棒性而受到关注，但标注数据昂贵且费时。零样本学习（ZSL）提供了一种替代方案——通过语义信息（类名、描述等）桥接已见类和未见类。

**现有方法的痛点**：所有现有方法（ReViSE、CADA-VAE、SynSE、JPoSE、SMIE）都直接将骨架特征与文本特征对齐到共享空间。然而它们忽略了一个根本性的不对称问题：

- **同一类别的骨架序列变化巨大**：不同演员的体型、动作幅度不同，不同摄像机角度也会带来显著差异。例如"穿鞋"这个动作，不同人的骨架序列差异极大
- **而文本标签是固定不变的**：每个类别只有一个标签（如"wear a shoe"）
- 这种多对一的不对称性使得强行将所有骨架特征对齐到同一文本表示非常困难

**核心矛盾**：骨架特征中既包含与类别语义相关的信息（做什么动作），也包含与语义无关的风格信息（谁在做、从哪个角度拍）。强行对齐会让语义无关的噪声污染共享空间。

**本文的切入角度**：受图像 ZSL 中 SDGZSL 方法启发——视觉嵌入可以被分解为语义一致和语义无关的部分——提出将骨架特征也进行类似的解耦。

**核心 idea**：通过双头编码器将骨架潜在表示分离为语义相关部分 $z_x^r$ 和语义无关部分 $z_x^v$，仅用 $z_x^r$ 与文本特征 $z_y$ 对齐，同时用对抗性总相关惩罚确保二者统计独立。

## 方法详解

### 整体框架

系统由三大组件构成：(1) 两个模态特定的特征提取器（骨架用 Shift-GCN/ST-GCN，文本用 Sentence-BERT/CLIP）；(2) 跨模态对齐模块（双 VAE + 特征解耦 + 对抗判别器）；(3) 三个分类器用于 ZSL/GZSL 推理。训练时在已见类上学习对齐，推理时用解耦后的语义相关特征识别未见类。

### 关键设计

1. **双头骨架编码器与特征解耦**:

    - 功能：将骨架特征 $f_x$ 编码为两个独立的潜在向量——语义相关 $z_x^r$ 和语义无关 $z_x^v$
    - 核心思路：骨架编码器 $E_x$ 设计为双头网络，一个头输出 $z_x^r \sim \mathcal{N}(\mu_x^r, \Sigma_x^r)$，另一个输出 $z_x^v \sim \mathcal{N}(\mu_x^v, \Sigma_x^v)$。完整骨架潜在表示为拼接 $z_x = z_x^v \oplus z_x^r$。文本编码器 $E_y$ 输出单头特征 $z_y$。两个 VAE 的损失为：
    $\mathcal{L}_x = \mathbb{E}[\log p_\theta(f_x|z_x)] - \beta_x D_{KL}(q_\phi(z_x^r|f_x) \| p(z_x^r)) - \beta_x D_{KL}(q_\phi(z_x^v|f_x) \| p(z_x^v))$
    - 设计动机：t-SNE 可视化验证了设计的有效性——$z_x^r$ 展示出清晰的类别聚类，而 $z_x^v$ 的类别分离不明显，说明语义无关信息被成功剥离

2. **跨模态对齐损失（Cross-Alignment Loss）**:

    - 功能：通过交叉重建建立两个模态潜在空间之间的对应关系
    - 核心思路：
    $\mathcal{L}_C = \|D_y(z_x^r) - f_y\|_2^2 + \|D_x(z_x^v \oplus z_y) - f_x\|_2^2$
   第一项要求仅用语义相关部分 $z_x^r$ 就能重建文本特征；第二项要求用语义无关部分 $z_x^v$ 加上文本特征 $z_y$ 来重建骨架特征
    - 设计动机：这种交叉重建巧妙地强制了解耦——$z_x^r$ 必须包含足够的语义信息来重建文本，$z_x^v$ 则补充文本无法提供的风格信息（体型、角度等）来完整重建骨架

3. **对抗性总相关惩罚（Adversarial Total Correlation Penalty）**:

    - 功能：确保 $z_x^r$ 和 $z_x^v$ 统计独立，防止信息泄漏
    - 核心思路：训练判别器 $D_T$ 判断 $z_x^v \oplus z_x^r$ 是否来自同一骨架特征：
    $\mathcal{L}_T = \log D_T(z_x) + \log(1 - D_T(\tilde{z}_x))$
   其中 $\tilde{z}_x$ 通过在 batch 内随机打乱 $z_x^v$ 的索引并与原始 $z_x^r$ 拼接生成。$D_T$ 最大化此损失，$E_x$ 最小化——对抗博弈推动两部分特征趋向独立
    - 设计动机：仅靠 VAE 的 KL 正则化不足以保证两个子空间独立。总相关惩罚提供更强的约束，显著减少特征冗余，使域分类器不偏向已见类

### 损失函数 / 训练策略

- 总损失：$\mathcal{L} = \mathcal{L}_{VAE} + \lambda_1 \mathcal{L}_C + \lambda_2 \mathcal{L}_T$
- VAE 和判别器交替训练：先训 VAE $n_d$ 次，再训 $D_T$ 一次
- 采用 cyclical annealing 策略缓解 KL 散度消失问题：每个 epoch 前 1/3 样本 $\lambda_2' = 0$，之后线性增加到 $\lambda_2$
- $\lambda_1$ 第一个 epoch 为 0，之后为 1（先学好重建再强制对齐）
- GZSL 推理采用双分类器策略：已见类分类器 $C_s$（用原始 $f_x$）+ 未见类分类器 $C_u$（用 $z_x^r$）+ 域分类器 $C_d$（逻辑回归融合两者概率）

## 实验关键数据

### 主实验（ZSL）

| 数据集 | 划分 | 本文 | 之前SOTA (SMIE) | 提升 |
|--------|------|------|-----------------|------|
| NTU-60 | 55/5 | 82.37% | 77.98% | +4.39% |
| NTU-60 | 48/12 | 41.38% | 40.18% | +1.20% |
| NTU-120 | 110/10 | 68.77% | 65.74% | +3.03% |
| NTU-120 | 96/24 | 46.12% | 45.30% | +0.82% |

### 主实验（GZSL Harmonic Mean）

| 数据集 | 划分 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| NTU-60 | 55/5 | 66.27% | 59.02% (SynSE) | +7.25% |
| NTU-60 | 48/12 | 42.56% | 36.33% (SynSE) | +6.23% |
| NTU-120 | 110/10 | 60.42% | 54.94% (SynSE) | +5.48% |
| NTU-120 | 96/24 | 44.50% | 41.04% (SynSE) | +3.46% |

### 消融实验（随机划分，NTU-60 ZSL）

| 配置 | 准确率 | 说明 |
|------|--------|------|
| Naive alignment | 69.26% | 无解耦基线 |
| FD (仅特征解耦) | 82.21% | +12.95%，核心贡献 |
| SA-DVAE (FD+TC) | 84.20% | +1.99%，TC 进一步提升 |

### 关键发现

- **特征解耦是核心贡献**：仅 FD 就在 NTU-60 随机划分上提升 +12.95%，效果显著
- **TC 惩罚主要改善 GZSL**：TC 略降已见类准确率但大幅提升未见类（NTU-60 GZSL H 从 70.71%→75.27%），减少了域分类器对已见类的偏见
- GZSL 任务上改进比 ZSL 更大（+7.25% vs +4.39%），因为解耦帮助域分类器更好地区分已见/未见类
- 已见类分类器用原始 $f_x$ 优于用 $z_x^r$（因为风格信息有时也有助于分类）

## 亮点与洞察

- **问题洞察精准**：首次指出骨架 ZSL 中多对一不对称性的根本问题，观察到同类骨架序列的巨大变异是对齐的主要障碍
- **交叉重建损失的设计很巧妙**：通过 $z_x^r → f_y$ 和 $(z_x^v, z_y) → f_x$ 的不对称重建，自然地将语义相关和风格信息分离
- **简单高效**：所有编码器/解码器/分类器都是单层 MLP，判别器仅两层，训练成本低（NTU-60 约 4.6 小时/单卡 RTX 3090）
- **不依赖词性标注**：相比 SynSE/JPoSE 需要 PoS tags，SA-DVAE 直接使用简单类名

## 局限与展望

- 骨架特征提取器（Shift-GCN/ST-GCN）是单独预训练后冻结的，未与 VAE 联合端到端训练，可能限制了特征质量
- 解耦维度 $z_x^r$ 和 $z_x^v$ 需要手动调参（如 160 vs 8），自适应维度分配可能更好
- 仅使用类名作为文本（未用丰富的动作描述），与 LLM 生成的详细描述结合可能进一步提升
- 未探索多视角骨架数据增强来增加 $z_x^v$ 的多样性
- 随机划分实验仅三次取平均，统计显著性稍弱

## 相关工作与启发

- **vs CADA-VAE**: SA-DVAE 的直接前身，但 CADA-VAE 不做特征解耦，强制全部骨架特征对齐文本；SA-DVAE 在 NTU-60 55/5 上 ZSL 超 +5.53%
- **vs SynSE/JPoSE**: 依赖词性（PoS）标注来分别对齐动词/名词，增加了预处理复杂度；SA-DVAE 更简洁且效果更好
- **vs SDGZSL**: 图像 ZSL 中的特征解耦方法，依赖类级属性；SA-DVAE 将此思路迁移到骨架 ZSL，直接用文本描述替代预定义属性

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将特征解耦引入骨架 ZSL，问题动机清晰，但 VAE + 解耦的技术方案本身不算全新
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、ZSL/GZSL 两种协议、固定/随机划分，且有详细消融
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，架构图直观，t-SNE 可视化极具说服力
- 价值: ⭐⭐⭐⭐ 解耦思路具有通用性，可迁移到其他跨模态零样本学习任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](../../ICCV2025/video_understanding/frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[ECCV 2024\] CrossGLG: LLM Guides One-Shot Skeleton-Based 3D Action Recognition in a Cross-Level Manner](crossglg_llm_guides_one-shot_skeleton-based_3d_action_recognition_in_a_cross-lev.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[ECCV 2024\] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)
- [\[ECCV 2024\] FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition](finepseudo_improving_pseudo-labelling_through_temporal-alignablity_for_semi-supe.md)

</div>

<!-- RELATED:END -->
