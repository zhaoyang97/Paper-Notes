---
title: >-
  [论文解读] A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1
description: >-
  [NeurIPS 2025][多模态VLM][黑盒迁移攻击] 提出 M-Attack，通过对源图像做随机裁剪后与目标图像在嵌入空间做局部-全局/局部-局部匹配，配合多 CLIP 模型集成，使对抗扰动自然聚集在语义关键区域形成清晰的语义细节，在 GPT-4.5/4o/o1 等商业黑盒 LVLM 上实现 >90% 的定向攻击成功率。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "黑盒迁移攻击"
  - "局部语义匹配"
  - "对抗扰动"
  - "LVLM安全"
  - "模型集成"
---

# A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1

**会议**: NeurIPS 2025  
**arXiv**: [2503.10635](https://arxiv.org/abs/2503.10635)  
**代码**: [https://github.com/VILA-Lab/M-Attack](https://github.com/VILA-Lab/M-Attack)  
**领域**: 多模态VLM / 对抗攻击  
**关键词**: 黑盒迁移攻击, 局部语义匹配, 对抗扰动, LVLM安全, 模型集成

## 一句话总结

提出 M-Attack，通过对源图像做随机裁剪后与目标图像在嵌入空间做局部-全局/局部-局部匹配，配合多 CLIP 模型集成，使对抗扰动自然聚集在语义关键区域形成清晰的语义细节，在 GPT-4.5/4o/o1 等商业黑盒 LVLM 上实现 >90% 的定向攻击成功率。

## 研究背景与动机

**领域现状**：基于迁移的定向对抗攻击是攻击黑盒商业 LVLM 的主要手段。现有方法（AttackVLM、SSA-CWA、AnyAttack、AdvDiffVLM）通常在白盒代理模型上生成对抗扰动，期望扰动能迁移到未知的商业模型上。

**现有痛点**：这些方法生成的扰动大多呈现类均匀分布，缺乏清晰的语义结构。作者通过实证分析发现两个关键失败模式：(1) 扰动的经验累积分布函数几乎与均匀分布重叠，说明扰动均匀散布在整个图像上而没有聚焦于语义关键区域；(2) 即使商业 LVLM 检测到了扰动的"异样"，也只能给出"模糊的"、"抽象的"描述，而非具体的语义名词，表明扰动中缺乏可被模型解读的语义信息。

**核心矛盾**：传统的全局-全局特征匹配方法虽然在嵌入空间中能快速拉近相似度，但正因如此导致过拟合——相似度迅速饱和，限制了进一步学习精细语义的空间。而局部匹配由于每步的随机裁剪引入了随机性，收敛更慢但能捕获更细粒度的语义细节。

**本文目标** 如何让对抗扰动在局部区域编码清晰的目标语义，使商业黑盒 LVLM 不仅能"看到"扰动，还能准确地将其解码为目标语义。

**切入角度**：作者观察到商业 LVLM 无论训练数据和架构如何差异，都会优先提取图像中的语义特征。因此，只要扰动本身带有足够清晰的语义信息，就有可能跨模型迁移。

**核心 idea**：将全局匹配改为随机裁剪后的局部匹配，让扰动在重叠的中心区域自然聚合出丰富的目标语义。

## 方法详解

### 整体框架

M-Attack 包含两个核心组件：(1) Local Matching (LM)——每步对源图像做随机裁剪+缩放后，与目标图像（全局或局部）在嵌入空间做余弦相似度对齐；(2) Model Ensemble (ENS)——使用多个白盒 CLIP 模型的集成来避免对单一模型过拟合。输入是一对源-目标图像，输出是在 $\ell_\infty$ 约束下的对抗图像，使黑盒 LVLM 将其描述为目标图像的内容。

### 关键设计

1. **局部-全局/局部-局部匹配 (Local Matching)**:

    - 功能：通过随机裁剪在源图像上生成局部区域，与目标图像在嵌入空间对齐
    - 核心思路：每个优化步骤 $i$，对当前对抗图像做随机裁剪（scale 范围 $[0.5, 1.0]$），缩放到原始尺寸后计算与目标图像的余弦相似度作为损失。由于不同步骤的裁剪区域相互重叠（一致性条件 $\hat{x}_i \cap \hat{x}_j \neq \emptyset$）又不完全相同（多样性条件 $|\hat{x}_i \cup \hat{x}_j| > |\hat{x}_i|$），扰动在图像中心区域被反复优化从而聚合出清晰的语义，而边缘区域则贡献多样化的细节
    - 设计动机：全局匹配相似度快速饱和导致过拟合，而局部匹配引入的随机性使收敛更慢但能捕获更精细的语义结构，从而大幅提升迁移性

2. **模型集成 (Model Ensemble)**:

    - 功能：利用多个白盒代理模型提取共享语义，提升对未知黑盒模型的迁移性
    - 核心思路：使用 ViT-B/16、ViT-B/32 和 ViT-g-14 三个 CLIP 变体，对每步的匹配损失取平均 $\mathcal{M} = \mathbb{E}_{f_{\phi_j} \sim \phi}[\text{CS}(f_{\phi_j}(\hat{x}_i^s), f_{\phi_j}(\hat{x}_i^t))]$。不同模型因 patch size 不同而具有互补的感受野——小 patch 模型捕获精细细节，大 patch 模型保留整体结构
    - 设计动机：单模型容易过拟合其特定的嵌入空间，模型集成能提取多个模型共享的语义特征，这些共享语义更有可能迁移到未知的商业模型

3. **KMRScore 评估指标**:

    - 功能：提供更客观、可复现的攻击成功率评估
    - 核心思路：为每张图像手动标注多个语义关键词，设置三档匹配阈值（0.25/0.5/1.0）对应 KMR_a/KMR_b/KMR_c，然后用 GPT-4o 做半自动匹配。KMR_c 要求所有关键词都匹配，是最严格的度量
    - 设计动机：之前的评估方法依赖主观的"语义主体"定义，人为偏差大且不可复现

### 训练策略

使用 I-FGSM 优化，步长 $\alpha=1$（Claude 为 0.75），扰动预算 $\epsilon=16$（$\ell_\infty$ 范数），总迭代步数 300。每步对源图像做随机裁剪后用 PGD 风格梯度更新。

## 实验关键数据

### 主实验

| 方法 | GPT-4o ASR | Gemini-2.0 ASR | Claude-3.5 ASR | $\ell_1$↓ | $\ell_2$↓ |
|------|-----------|----------------|----------------|-----------|-----------|
| AttackVLM (B/32) | 0.02 | 0.00 | 0.00 | 0.036 | 0.041 |
| SSA-CWA (Ensemble) | 0.09 | 0.04 | 0.05 | 0.059 | 0.060 |
| AnyAttack (Ensemble) | 0.42 | 0.48 | 0.23 | 0.048 | 0.052 |
| **M-Attack (Ours)** | **0.95** | **0.78** | **0.29** | **0.030** | **0.036** |

### 消融实验（不同匹配方式对比）

| 匹配方式 | GPT-4o ASR | Gemini-2.0 ASR | Claude-3.5 ASR |
|---------|-----------|----------------|----------------|
| Global-to-Global | 0.05 | 0.05 | 0.01 |
| Local-to-Global | 0.93 | 0.83 | 0.22 |
| Local-to-Local | 0.95 | 0.78 | 0.26 |

### 关键发现

- 局部匹配相比全局匹配在 GPT-4o 上的 ASR 从 5% 暴涨到 95%，提升幅度惊人
- M-Attack 在扰动不可感知性上（$\ell_1$/$\ell_2$ 范数）反而优于所有对比方法，说明局部聚合产生了更高效的扰动
- 不同扰动预算 $\epsilon$ 下（4/8/16），M-Attack 始终保持最优，且在 $\epsilon=8$ 时就已在 GPT-4o 上达到 82% ASR
- 严格的 KMR_c 指标（需要所有关键词匹配）揭示之前方法的 ASR 被严重高估——它们在 KMR_c 下基本为 0

## 亮点与洞察

- **极致简洁却有效的设计**：M-Attack 的核心操作就是"随机裁剪 + 缩放 + 余弦相似度对齐"，没有引入任何复杂模块，却比所有已有方法高出一个量级。这种"less is more"的思路非常值得借鉴
- **对失败案例的深入分析驱动了方法设计**：作者先分析了为什么已有方法失败（扰动呈均匀分布 + 语义模糊），然后针对性地设计了局部匹配来解决语义缺失问题。这种"从失败中找灵感"的研究范式很有参考价值
- **KMRScore 评估方法**的多阈值设计可以迁移到其他攻击评估任务中，提供更细粒度的成功率度量

## 局限与展望

- Claude 系列模型的攻击成功率显著低于 GPT 和 Gemini（29% vs 95%/78%），作者未深入分析原因，可能与 Claude 的额外安全机制有关
- 方法依赖 CLIP 作为代理模型，没有探索非 CLIP 架构的代理模型（如 SigLIP、EVA）对迁移性的影响
- 仅在 224x224 分辨率下实验，高分辨率场景下的效果未验证
- 防御侧视角缺失——没有讨论如何检测或防御此类攻击

## 相关工作与启发

- **vs AttackVLM**: AttackVLM 使用全局特征匹配，本文证明全局匹配在嵌入空间中快速饱和导致过拟合，用局部匹配替代后效果提升约 20 倍
- **vs AnyAttack**: AnyAttack 使用自监督预训练生成对抗样本，虽然比 AttackVLM 好但仍远不及 M-Attack，其扰动仍偏向均匀分布
- **vs SSA-CWA**: SSA-CWA 使用频谱增强和锐度感知优化，但本质仍是全局匹配，改进有限

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心idea（局部裁剪匹配）虽然简单但洞察深刻，从失败分析到方法设计的逻辑链完整
- 实验充分度: ⭐⭐⭐⭐⭐ 在 7 个商业 LVLM 上测试，包括推理模型 o1 和 Claude-3.7-thinking，消融细致
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，从动机到方法到实验的叙事流畅
- 价值: ⭐⭐⭐⭐⭐ 对 LVLM 安全领域有重要警示意义，90%+ 的攻击成功率表明当前商业模型的对抗鲁棒性堪忧

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model](../../ACL2025/multimodal_vlm/internlm-xcomposer25-reward_a_simple_yet_effective_multi-modal_reward_model.md)
- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)
- [\[ICCV 2025\] Visual Interestingness Decoded: How GPT-4o Mirrors Human Interests](../../ICCV2025/multimodal_vlm/visual_interestingness_decoded_how_gpt-4o_mirrors_human_interests.md)
- [\[ICLR 2026\] BaseReward: A Strong Baseline for Multimodal Reward Model](../../ICLR2026/multimodal_vlm/basereward_a_strong_baseline_for_multimodal_reward_model.md)
- [\[NeurIPS 2025\] AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining](adalrs_lossguided_adaptive_learning_rate_search_for_efficien.md)

</div>

<!-- RELATED:END -->
