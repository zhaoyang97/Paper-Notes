---
title: >-
  [论文解读] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition
description: >-
  [AAAI 2026][视频理解][少样本动作识别] 提出 TS-FSAR 框架，通过 α-距离相关性捕获帧间非线性依赖关系并结合任务特定匹配矩阵进行 query-support 匹配，同时用适配后的冻结 CLIP 引导侧网络训练，在 SSv2-Full 等时序敏感数据集上大幅超越先前方法。 领域现状：少样本动作识别（FSA…
tags:
  - "AAAI 2026"
  - "视频理解"
  - "少样本动作识别"
  - "距离相关性"
  - "CLIP微调"
  - "任务特定匹配"
  - "侧网络调优"
---

# Task-Specific Distance Correlation Matching for Few-Shot Action Recognition

**会议**: AAAI 2026  
**arXiv**: [2512.11340](https://arxiv.org/abs/2512.11340)  
**代码**: 无  
**领域**: 视频理解 / 少样本学习  
**关键词**: 少样本动作识别, 距离相关性, CLIP微调, 任务特定匹配, 侧网络调优

## 一句话总结

提出 TS-FSAR 框架，通过 α-距离相关性捕获帧间非线性依赖关系并结合任务特定匹配矩阵进行 query-support 匹配，同时用适配后的冻结 CLIP 引导侧网络训练，在 SSv2-Full 等时序敏感数据集上大幅超越先前方法。

## 研究背景与动机

**领域现状**：少样本动作识别（FSAR）旨在仅从少量标注样本学习识别新类别的动作。主流方法分为两条路线：设计更好的度量（如集合匹配）和高效适配大规模预训练模型（如 CLIP）。集合匹配类方法如 HyRSM（Hausdorff 距离）和 TSAM（最优传输）已取得不错效果。

**现有痛点**：(1) **度量局限**：现有集合匹配方法普遍使用余弦相似度来构建帧间关系矩阵，但余弦相似度近似等价于 Pearson 相关系数，只能捕获线性依赖关系，无法建模更复杂的非线性依赖。(2) **匹配范式**：现有方法仅使用实例级信息进行匹配，忽略了任务特定的上下文线索。(3) **CLIP 适配**：参数高效的侧网络微调（如 EMP-Net 的 skip-fusion 层）虽然节省显存，但新引入的层在有限数据下难以优化。

**核心矛盾**：视频中帧间存在复杂的非线性动态关系（尤其在时序敏感数据集如 SSv2 中），但余弦相似度无法捕获；侧网络微调节省显存但训练不稳定，尤其在静态图像主导的数据集上更依赖预训练权重。

**本文目标** (1) 设计能捕获线性+非线性帧间关系的匹配度量；(2) 引入任务特定信息进行更准确的匹配；(3) 改善侧网络在有限数据下的训练效果。

**切入角度**：用 α-距离相关性替代余弦相似度来度量帧间依赖（它可以捕获任意阶的统计依赖），用任务原型生成匹配矩阵来加权帧间关系的重要性，用适配后的冻结 CLIP 输出分布引导侧网络学习。

**核心 idea**：用 α-距离相关性建模帧间非线性依赖，用任务原型驱动的匹配矩阵进行任务特定匹配，用 CLIP 知识蒸馏引导侧网络训练。

## 方法详解

### 整体框架

TS-FSAR 包含三个组件。(1) Ladder Side Network (LSN)：轻量侧网络接收冻结 CLIP 的中间特征，输出用于度量计算的帧级 token。(2) Task-Specific Distance Correlation Matching (TS-DCM)：由帧间 α-距离相关性（IF-DαC）和任务特定匹配（TSM）两部分组成，计算 query-support 相似度。(3) Guiding LSN with Adapted CLIP (GLAC)：用适配后的冻结 CLIP 输出分布对齐 LSN 输出分布，稳定训练。

### 关键设计

1. **帧间 α-距离相关性（Inter-Frame α-Distance Correlation, IF-DαC）**:

    - 功能：计算 query 和 support 视频帧之间的全面依赖关系矩阵
    - 核心思路：对 LSN 输出的第 $i$ 帧 support 特征 $\mathbf{V}_\mathcal{S}^i$ 和第 $j$ 帧 query 特征 $\mathbf{V}_\mathcal{Q}^j$，将每列视为随机变量的一个观测，计算 α 次方欧氏距离矩阵 $\hat{a}_{kl} = \|\mathbf{x}_k - \mathbf{x}_l\|^\alpha$，经双中心化得到 α-D 矩阵 $\mathbf{A}^i, \mathbf{B}^j$，然后 α-距离相关性为 $m_{ij} = \text{tr}(\mathbf{A}^i \mathbf{B}^j) / \sqrt{\text{tr}(\mathbf{A}^i \mathbf{A}^i) \text{tr}(\mathbf{B}^j \mathbf{B}^j)}$。参数 $\alpha \in (0,2)$ 调节对不同尺度依赖的灵敏度，实验选 $\alpha = 0.8$。
    - 设计动机：距离相关性的核心性质是：当且仅当两个随机变量独立时距离相关性为零，可以捕获任意（包括非线性）依赖关系。相比之下，余弦相似度（≈Pearson 相关系数）只对线性关系敏感。SSv2 等时序敏感数据集中动作的区分依赖于细粒度的非线性时间模式。

2. **任务特定匹配（Task-Specific Matching, TSM）**:

    - 功能：生成一个编码帧间关系相对重要性的匹配矩阵，实现任务感知的匹配
    - 核心思路：首先构建查询特定的任务原型 $\mathbf{p}^\mathcal{T} = \tilde{\mathbf{v}}^\mathcal{Q} + \frac{1}{N_\mathcal{S}} \sum \tilde{\mathbf{v}}_i^\mathcal{S}$（query 和 support 类 token 的平均融合），然后将原型输入可学习线性层生成器 $\mathcal{G}(\cdot)$，输出 $T \times T$ 的匹配矩阵 $\mathbf{M}^{task}$。最终相似度为 $\langle \mathbf{M}^{task}, \mathbf{M}^{IF\text{-}D^\alpha C} \rangle$（内积）。
    - 设计动机：IF-DαC 矩阵给出了所有帧对之间的依赖强度，但不同帧对的重要性应该因任务而异。通过任务原型驱动的匹配矩阵来加权这些关系，使得匹配关注当前任务最相关的帧间模式。实验证明简单平均融合优于拼接和交叉注意力。

3. **GLAC 模块（Guiding LSN with Adapted CLIP）**:

    - 功能：利用适配后的冻结 CLIP 的输出分布来引导 LSN 的训练
    - 核心思路：对 LSN 侧，将视频级 α-D 矩阵均值作为表示 $\widetilde{\mathbf{A}}_{\alpha\text{-}D}$，与可学习的类别 α-D 原型权重做内积后 softmax 得到预测分布 $\mathbf{p}$。对 CLIP 侧，将冻结 CLIP 的帧级 CLS token 通过 MHSA 适配器建模帧间依赖，平均后与文本嵌入做余弦相似度得到引导分布 $\mathbf{q}$。训练时最小化 KL 散度 $\text{KL}(\mathbf{p} \| \mathbf{q})$ 加上双分支的交叉熵损失。
    - 设计动机：LSN 是全新引入的轻量网络，参数随机初始化，在少样本条件下训练困难。用预训练的 CLIP 知识（经适配器适配到视频域后）作为"教师"来引导 LSN 学习，使 LSN 的 α-D 特征更可靠，从而改善下游的距离相关性估计。

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{LSN} + \lambda_1 \mathcal{L}_{TS\text{-}DCM} + \lambda_2 \mathcal{L}_{GLAC}$，其中 $\mathcal{L}_{LSN}$ 是 LSN 输出与文本嵌入的视觉-语言对齐交叉熵，$\mathcal{L}_{TS\text{-}DCM}$ 是 episodic 匹配的交叉熵，$\mathcal{L}_{GLAC} = \text{KL}(\mathbf{p} \| \mathbf{q}) + \text{CE}(\mathbf{p}, y) + \text{CE}(\mathbf{q}, y)$。使用 AdamW 优化器，余弦学习率调度。

## 实验关键数据

### 主实验

| 数据集 | 设置 | TS-FSAR | 之前SOTA | 提升 |
|--------|------|---------|---------|------|
| SSv2-Full | 1-shot | **75.1** | 66.7 (D2ST) | **+8.4** |
| SSv2-Full | 5-shot | **83.5** | 81.9 (D2ST) | +1.6 |
| SSv2-Small | 1-shot | 60.5 | 60.5 (TSAM) | 持平 |
| SSv2-Small | 5-shot | **70.3** | 69.3 (D2ST) | +1.0 |
| HMDB51 | 1-shot | **85.0** | 84.5 (TSAM) | +0.5 |
| HMDB51 | 5-shot | **88.9** | 88.9 (TSAM) | 持平 |
| UCF101 | 1-shot | **98.7** | 98.3 (TSAM) | +0.4 |
| Kinetics | 1-shot | **96.3** | 96.2 (TSAM) | +0.1 |

### 消融实验

| 配置 | SSv2-Full 1-shot | HMDB51 1-shot | 说明 |
|------|-----------------|---------------|------|
| Zero-shot CLIP | 37.0 | 75.9 | 基线 |
| + LSN | 67.1 | 77.7 | 侧网络微调的增益 |
| + LSN + IF-DαC | 71.4 | 82.1 | α-距离相关性提升 4.3% |
| + LSN + IF-DαC + TSM | 73.8 | 83.4 | 任务特定匹配再提升 2.4% |
| + 完整 TS-FSAR | **75.1** | **85.0** | GLAC 引导再提升 1.3-1.6% |

### 关键发现

- SSv2-Full 上的巨大提升（+8.4%）由两个因素叠加：(1) 该数据集有细粒度时序变化，α-DC 能捕获非线性模式；(2) 其训练集约为其他数据集的 10 倍，提供更充分的 LSN 训练监督
- 非线性度量（α-DC, DC, HSIC）一致优于余弦相似度，α-DC 最优
- GLAC 在静态数据集（HMDB51）上的提升更显著，验证了 LSN 训练不足主要影响静态任务
- 效率：14M 参数，~9GB 显存，0.42s/episode，比 CLIP-FSAR（89M, ~20GB）高效得多

## 亮点与洞察

- 将距离相关性引入少样本动作识别的度量设计，是对余弦相似度的重要升级
- 任务特定匹配的思路——用任务原型生成帧间重要性权重——简单有效，可推广到其他集合匹配场景
- GLAC 模块解决了侧网络微调中常见的"新层训练困难"问题，是一个通用的技巧
- SSv2-Full 上 8.4% 的 1-shot 提升非常显著，说明在时序敏感任务上非线性度量有巨大潜力

## 局限与展望

- 在静态数据集（UCF101, Kinetics）上提升有限，这些数据集更依赖 CLIP 预训练知识而非时序建模
- α-DC 计算涉及 $(P+1) \times (P+1)$ 的距离矩阵及其双中心化，计算开销较大
- LSN 维度固定为 256，未探索更灵活的架构设计
- 任务原型的融合策略（简单平均）可能在多 shot 场景下不够精细

## 相关工作与启发

- **vs TSAM**: TSAM 使用最优传输做匹配但仍基于余弦相似度构建代价矩阵；TS-FSAR 用 α-DC 替代余弦相似度并加入任务特定匹配
- **vs EMP-Net**: 同样使用侧网络微调 CLIP，但 EMP-Net 缺少训练引导机制；TS-FSAR 的 GLAC 模块带来 1.1%~8.2% 的额外提升
- **vs DeepBDC**: DeepBDC 在少样本图像分类中使用距离相关性；TS-FSAR 扩展到视频理解，引入帧间 α-DC 和任务特定匹配

## 评分

- 新颖性: ⭐⭐⭐⭐ α-距离相关性的引入新颖，任务特定匹配设计合理，GLAC 模块解决了实际痛点
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个数据集全面评测，消融非常详细，包括度量对比、原型策略、效率分析
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学公式准确，消融分析逻辑性强
- 价值: ⭐⭐⭐⭐ 在时序敏感的少样本动作识别上有重要突破，α-DC 作为度量可推广到更多匹配任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition](../../CVPR2025/video_understanding/temporal_alignment-free_video_matching_for_few-shot_action_recognition.md)
- [\[CVPR 2026\] MPL: Match-guided Prototype Learning for Few-shot Action Recognition](../../CVPR2026/video_understanding/mpl_match-guided_prototype_learning_for_few-shot_action_recognition.md)
- [\[CVPR 2026\] VideoNet: A Large-Scale Dataset for Domain-Specific Action Recognition](../../CVPR2026/video_understanding/videonet_a_large-scale_dataset_for_domain-specific_action_recognition.md)
- [\[ECCV 2024\] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](../../ECCV2024/video_understanding/efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](../../ICCV2025/video_understanding/beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)

</div>

<!-- RELATED:END -->
