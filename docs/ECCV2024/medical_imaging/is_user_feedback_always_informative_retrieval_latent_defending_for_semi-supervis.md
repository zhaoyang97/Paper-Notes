---
title: >-
  [论文解读] Is User Feedback Always Informative? Retrieval Latent Defending for Semi-Supervised Domain Adaptation without Source Data
description: >-
  [ECCV 2024][医学图像][半监督域适应] 发现用户反馈在域适应中并非总是有益——偏向纠正错误预测的"负偏反馈"(NBF)会导致现有半监督域适应方法性能下降，并提出 Retrieval Latent Defending 方法，通过在每个 mini-batch 中加入伪标签防御样本来平衡监督信号。
tags:
  - "ECCV 2024"
  - "医学图像"
  - "半监督域适应"
  - "负偏反馈"
  - "Source-Free"
  - "用户反馈"
  - "伪标签"
---

# Is User Feedback Always Informative? Retrieval Latent Defending for Semi-Supervised Domain Adaptation without Source Data

**会议**: ECCV 2024  
**arXiv**: [2407.15383](https://arxiv.org/abs/2407.15383)  
**代码**: [有](https://github.com/junha1125/RLD-SemiSDA)  
**领域**: 医学图像 / 域适应  
**关键词**: 半监督域适应, 负偏反馈, Source-Free, 用户反馈, 伪标签

## 一句话总结

发现用户反馈在域适应中并非总是有益——偏向纠正错误预测的"负偏反馈"(NBF)会导致现有半监督域适应方法性能下降，并提出 Retrieval Latent Defending 方法，通过在每个 mini-batch 中加入伪标签防御样本来平衡监督信号。

## 研究背景与动机

深度学习模型在部署域（目标域）常因域偏移（domain shift）而性能退化。半监督域适应（SemiSDA）利用少量标注目标数据和大量未标注数据进行适应。但现有工作存在一个**被忽视的关键问题**：

**反馈偏差假设**：所有现有 SemiSDA 方法假设标注数据是目标域的**随机子集**。然而在实际 ML 产品中，用户反馈来源于**交互纠错**——用户更倾向于在模型出错时提供反馈。例如放射科医生更可能记录模型误诊的 X 光片，因为诊断准确性直接关系患者生存。

**负偏反馈 (Negatively Biased Feedback, NBF)**：这一现象得到认知心理学支持——人类对负面事件（即模型错误预测）反应更强烈。由此产生的标注数据集在每个类别内的分布是**有偏的**——集中在模型决策边界附近的困难样本区域，而非均匀分布。

**NBF的意外影响**：直觉上，纠正更多错误的反馈应该带来更好的适应效果。但实验发现，在 NBF 条件下使用 AdaMatch 等 SOTA SemiSDA 方法，性能反而**下降**（如 DomainNet-126 上从 67.6% 降到 64.5%）。原因是：SemiSDA 方法的适应过程中，决策边界会被有偏分布的标注数据**主导**，导致对真实类别聚类的判定边界偏移。

**source-free 约束**：考虑数据隐私和边缘设备限制，源数据不可用，进一步加大了适应难度。

这是**首个**揭示并分析用户反馈偏差对域适应影响的研究。

## 方法详解

### 整体框架

Retrieval Latent Defending (RLD) 是一个**即插即用**的方法，可与任意 SemiSDA baseline 组合。核心思路是：在每个训练 mini-batch 中，除了有偏的标注数据和未标注数据外，额外加入从候选库中检索的"防御样本"(latent defending samples)，平衡监督信号的分布。

### 关键设计

1. **Negatively Biased Feedback (NBF) 的建模与验证**：

    - 通过 blobs 合成数据集的模拟实验可视化 NBF 影响：随机反馈(RF)在类聚类中均匀分布，NBF 则集中于决策边界附近
    - Pseudo-labeling 在 RF 下达到 91.7% 准确率，NBF 下仅 88.1%
    - 原因分析：标注数据的分布对模型适应后的决策边界贡献显著（图示红色箭头），有偏分布导致次优决策边界
    - 设计动机：这揭示了现有 SemiSDA 方法**对标注数据位置分布敏感**的隐含假设

2. **候选库生成 (Candidate Bank Generation)**：每个 epoch 开始前执行一次。

    - 冻结当前模型，为所有未标注数据 $X_t^{ulb}$ 生成伪标签
    - $\hat{y}_{ulb}^n = \arg\max_c [f_\theta(x_{ulb}^n)]_c$
    - 在每个类别内仅保留 softmax 概率 **top-p%** 的样本（默认 p=40%）
    - 过滤步骤避免引入不准确的伪标签
    - 随着适应进行，模型改善 → 伪标签质量提高 → 候选库逐步精化
    - 设计动机：利用高置信度伪标签样本代理真实的均匀分布，为有偏的标注数据提供平衡力量

3. **防御样本选择 (Defending Sample Selection)**：每个迭代中执行。

    - 对 mini-batch 中每个标注数据 $(x_{lb}^b, y_{lb}^b)$，从候选库中随机选择 $k$ 个同类伪标签样本（默认 k=3）
    - 选择策略：class-aware random selection（消融实验验证最优）
    - 效果：被选中的防御样本在特征空间中"环绕"有偏的标注数据，形成更均衡的类别表征
    - 设计动机：防止监督信号过度依赖有偏的标注样本位置

### 损失函数 / 训练策略

总损失由三部分组成：

$$\mathcal{L}_{total} = \underbrace{\mathcal{L}_{sup} + \mathcal{L}_{unsup}}_{\text{baseline}} + \underbrace{\frac{1}{k \cdot B} \sum_{b=1}^{k \cdot B} \mathcal{H}(\hat{y}_{LD}^b, f_\theta(x_{LD}^b))}_{\text{retrieval latent defending}}$$

- $\mathcal{L}_{sup}$：标注数据的交叉熵损失
- $\mathcal{L}_{unsup}$：未标注数据的一致性正则化损失（具体形式取决于 baseline）
- RLD 损失：防御样本的伪标签交叉熵
- 超参数：$k=3$（每个标注样本配 3 个防御样本），$p=0.4$（top-40% 过滤率），所有实验统一设置
- mini-batch 中减少未标注样本比例（从 1:7 改为 1:4），给防御样本和标注数据更多权重

## 实验关键数据

### 主实验 — DomainNet-126 (ResNet-50, 378 feedback)

| 方法 | RF | NBF | NBF w/ RLD | 提升 |
|------|-----|-----|-----------|------|
| AdaMatch | 67.6 | 64.5 | **72.0** | +7.5 |
| FixMatch | 67.6 | 63.4 | **73.2** | +9.8 |
| FreeMatch | 73.8 | 72.0 | **74.8** | +2.8 |
| FlexMatch | 73.3 | 71.4 | **74.7** | +3.3 |
| CDAC | 68.3 | 64.6 | **73.2** | +8.6 |

NBF 普遍导致 1-4% 的性能下降，RLD 不仅恢复还**超越 RF** 的性能。

### 消融实验 — 防御样本选择策略 (FreeMatch, ResNet-50)

| 选择策略 | 准确率 | baseline (无 RLD) |
|---------|-------|------------------|
| 随机（不分类） | 74.1 | 72.0 |
| **随机（class-aware）** | **74.8** | 72.0 |
| K-means 聚类中心 | 74.6 | 72.0 |
| 余弦距离最远 | 74.0 | 72.0 |

### 医学图像 — MIMIC-CXR-V2 (PA→AP, DenseNet-121)

| 方法 | RF (AUROC) | NBF | NBF w/ RLD | NBF-CE | NBF-CE w/ RLD |
|------|-----------|-----|-----------|--------|---------------|
| Source model | .7738 | - | - | - | - |
| Pseudo-Label | .7850 | .7691 | **.7884** | .7639 | **.7875** |

NBF-CE（医生更可能反馈高置信错误预测）比 NBF 更严重（.7691→.7639），但 RLD 均能有效缓解。

### 关键发现

- **NBF 的普遍性**：在图像分类、语义分割、医学图像诊断三个任务上，NBF 均导致现有 SemiSDA 方法性能下降
- **RLD 的通用性**：与 7 种 SemiSDA/SemiSL baseline 组合均有效，无需修改 baseline 核心策略
- **反馈量少时优势更大**：每类仅 1 个反馈时，RLD 提升高达 +4.9%（vs 每类 15 个时 +1.2%），说明在实际应用中（反馈稀少）价值更大
- **负反馈 vs 正反馈**：有趣的是，结合 RLD 后，NBF 反而比纯正面反馈（PBF）效果更好——因为 NBF 包含了模型缺陷的新知识
- **mini-batch 比例**：减少未标注样本（1:4 优于 1:7），优先可靠信息更有利

## 亮点与洞察

1. **问题发现比方法更重要**：NBF 是一个被整个 SemiSDA 领域忽视的实际问题，认知心理学+ML 的跨学科分析令人信服
2. **方法极简但有效**：RLD 只需在 mini-batch 中加入高置信伪标签样本，无需修改任何 baseline 算法
3. **医学场景的实际意义**：放射科医生反馈模式确实符合 NBF 假设，naive 适应可能导致模型性能下降，这对患者安全有直接影响
4. **NBF-CE 场景**：医生更可能反馈高置信错误的进一步分析，增加了工作的深度和实际相关性

## 局限与展望

1. 候选库每 epoch 更新一次，计算开销随数据集增大而增加
2. 伪标签质量依赖当前模型，适应初期可能不可靠
3. 只考虑了反馈是标注形式，实际中还有 thumbs up/down、评分等其他反馈形式
4. 实验中 NBF 是模拟的（随机选错误预测），未在真实用户交互环境中验证
5. 防御样本的数量 k 和过滤率 p 对所有实验统一设置，缺少自适应调节机制

## 相关工作与启发

- 与主动域适应（ActiveDA）的区别：ActiveDA 由机器选择最有信息量的样本请求标注，本文关注用户自发提供的有偏反馈
- 伪标签方法（FixMatch、FreeMatch）在 NBF 下失效尤为严重，因为有偏标注数据会引导伪标签分布偏移
- 课程学习和自适应阈值的思想与 RLD 减少未标注样本、优先可靠信息的策略相通
- 未来可结合不确定性估计来自适应调整防御样本的选择和权重

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次发现并系统分析 NBF 现象，问题定义极具洞察力和实际价值
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个任务领域、七种 baseline、两种架构、多种反馈量，消融全面
- **写作质量**: ⭐⭐⭐⭐ — 模拟实验的可视化分析直观清晰，问题阐述循序渐进
- **实用价值**: ⭐⭐⭐⭐⭐ — 直接解决实际 ML 产品部署中的问题，医学场景尤其有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Semi-supervised Deep Transfer for Regression without Domain Alignment](../../ICCV2025/medical_imaging/semi-supervised_deep_transfer_for_regression_without_domain_alignment.md)
- [\[ECCV 2024\] Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation](alternate_diverse_teaching_for_semi-supervised_medical_image_segmentation.md)
- [\[CVPR 2025\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](../../CVPR2025/medical_imaging/semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](../../CVPR2026/medical_imaging/tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[ECCV 2024\] Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals](brain_netflix_scaling_data_to_reconstruct_videos_from_brain_signals.md)

</div>

<!-- RELATED:END -->
