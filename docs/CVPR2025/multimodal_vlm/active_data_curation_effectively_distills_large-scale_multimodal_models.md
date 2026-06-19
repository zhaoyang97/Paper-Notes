---
title: >-
  [论文解读] Active Data Curation Effectively Distills Large-Scale Multimodal Models
description: >-
  [CVPR 2025][多模态VLM][知识蒸馏] 提出 ACID（主动数据筛选即隐式蒸馏）和 ACED（结合显式蒸馏），证明用大模型作为参考来主动筛选训练数据是一种比传统知识蒸馏更有效的多模态模型压缩方式，两者互补结合后在 27 个零样本任务上以更少推理 FLOPs 达到 SOTA。 将 CLIP/SigLIP 等大规模多…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "知识蒸馏"
  - "主动数据筛选"
  - "对比学习"
  - "CLIP压缩"
  - "推理效率"
---

# Active Data Curation Effectively Distills Large-Scale Multimodal Models

**会议**: CVPR 2025  
**arXiv**: [2411.18674](https://arxiv.org/abs/2411.18674)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 知识蒸馏, 主动数据筛选, 对比学习, CLIP压缩, 推理效率

## 一句话总结

提出 ACID（主动数据筛选即隐式蒸馏）和 ACED（结合显式蒸馏），证明用大模型作为参考来主动筛选训练数据是一种比传统知识蒸馏更有效的多模态模型压缩方式，两者互补结合后在 27 个零样本任务上以更少推理 FLOPs 达到 SOTA。

## 研究背景与动机

将 CLIP/SigLIP 等大规模多模态模型部署到边缘设备面临推理成本高的困难，需要压缩出小而强的模型。知识蒸馏（KD）是经典压缩手段——让小模型匹配大模型的输出分布。当前 SOTA（TinyCLIP、MobileCLIP）使用了复杂的多教师集成、合成字幕、权重继承、数据增强等策略。

然而存在两个关键gap：

**KD 方法越来越复杂**：多种损失函数组合、bespoke 架构、复杂训练流程，可复现性差

**数据筛选领域的"共识"限制了想象力**：现有主动数据筛选方法假设参考模型应小于/等于学习模型（为效率考虑），且认为用大模型筛选的数据对小模型不适用（capacity gap）

本文的核心洞察是：**主动数据筛选本质上是一种隐式蒸馏**——通过大参考模型选择"容易但对学生有挑战性"的数据来训练小模型，等价于优化一种结合了模型预测和真实标签的新型蒸馏目标。这在理论和实验上都打破了"大模型不能为小模型选数据"的共识。

## 方法详解

### 整体框架

统一目标函数：$\mathcal{L}_{full} = \mathcal{L}_{softmax/sigmoid}[\mathcal{B}_{CE}] + \lambda \cdot \mathcal{L}_{dist}[\mathcal{B}_{KD}]$

通过设置不同的 $\lambda$ 和数据采样策略可恢复所有方法变体：
- IID-Baseline：$\lambda=0$，随机采样
- ACID：$\lambda=0$，主动筛选采样（隐式蒸馏）
- Softmax-KD：$\lambda>0$，随机采样
- ACED：$\lambda>0$，主动筛选采样 + 显式蒸馏

### 关键设计

1. **ACID：主动筛选即隐式蒸馏**:

    - 功能：用预训练的大参考模型 $\theta_{ref}$ 从超级批次 $\mathcal{S}$（大小 $B$）中筛选出高信息量的小批次 $\mathcal{B}$（大小 $b$）用于训练
    - 核心思路：两种筛选评分策略——
        - Easy-reference：$s^{easy\_ref} = -\mathcal{L}(\mathcal{B}|\theta_{ref})$，优先选参考模型觉得"容易"的样本
        - Learnability：$s^{learn} = \mathcal{L}(\mathcal{B}|\theta) - \mathcal{L}(\mathcal{B}|\theta_{ref})$，选"参考模型容易但学生困难"的样本
    - **理论贡献**：作者证明 easy-reference 筛选的期望训练目标等价于：
    $\mathcal{E}_{easy-ref} = \frac{1}{Z}\sum_{x \in \mathcal{D}} KD[p(x) \cdot y(x); q(x)]$
      即一种结合参考模型预测 $p$ 与真实标签 $y$ 的隐式蒸馏目标。模型预测和标签的噪声来源不同（模型欠拟合 vs 标注错误），仅保留二者一致的目标实现"互相去噪"
    - 设计动机：与传统 active learning 相反，ACID 应使用比学生更大的参考模型——这在蒸馏视角下自然合理

2. **联合批次采样（Joint Batch Sampling）**:

    - 功能：批次内样本选择相互依赖（对比学习中每个样本的损失取决于批次内其他样本），需联合采样
    - 核心思路：使用 blocked Gibbs sampling 迭代构建批次。每轮从剩余候选中按条件评分采样一个 chunk（$b/n$），附加到当前批次，共 $n$ 轮
    - 设计动机：独立采样忽略批次内交互，联合采样确保选出的批次整体信息量最大化
    - 关键超参：filtering ratio $f = 1 - b/B$（默认0.8，即从5倍大的超级批次中筛选）

3. **ACED：结合隐式与显式蒸馏**:

    - 功能：将 ACID 与 Softmax-KD 结合，同时利用两种互补的知识转移途径
    - 核心思路：ACIDistill 策略——用 H-ACID 采样单个批次，同时计算对比损失和蒸馏损失
    - 设计动机：虽然 ACID 整体优于 KD，但在 Cars、DTD 等细粒度任务上不如 KD（可能因数据筛选过滤掉了部分有用样本）。ACID 隐式蒸馏和 KD 显式蒸馏优化不同目标，二者互补

### 损失函数 / 训练策略

- 对比损失：默认使用 sigmoid 变体（SigLIP 风格），更可扩展
- 蒸馏损失：教师-学生对比概率矩阵的交叉熵（Eq. 3）
- 评估协议 StableEval：系统分析 34 个评测的跨种子方差，选出 27 个高可靠性评测组成标准集，保证实验比较的可信度

## 实验关键数据

### 主实验

| 方法 | 样本量 | 推理 GFLOPs | 27任务平均 | ImageNet | COCO | Flickr |
|------|--------|-------------|-----------|----------|------|--------|
| MobileCLIP-S0 | 13B* | 3.70 | 63.6 | 67.8 | 49.6 | 76.7 |
| **ACED-F0** | 13B | **3.30** | **64.0** | **68.5** | **51.0** | **79.5** |
| MobileCLIP-S1 | 13B* | 7.64 | 67.9 | 72.6 | 53.0 | 80.0 |
| **ACED-F1** | 13B | **7.14** | **69.7** | **74.9** | **55.6** | **84.7** |
| MobileCLIP-S2 | 13B* | 10.81 | 69.8 | 74.4 | 54.4 | 81.8 |
| **ACED-F2** | 13B | **10.29** | **70.9** | **76.9** | **58.3** | **85.3** |

- ACED-F1 甚至用比 MobileCLIP-S2 少 34% 的 GFLOPs 就超越其 ImageNet 性能
- ACED 还在 LiT-Decoder 设置（冻住视觉编码器 + 训练文本解码器）中超越 SigLIP

### 消融实验

| 配置 | StableEval 平均 | 说明 |
|------|----------------|------|
| IID-Baseline | 基线 | 随机采样训练 |
| Softmax-KD (L teacher) | 优于IID | 传统蒸馏 |
| I-ACID (L ref) | 优于KD | 隐式蒸馏 |
| H-ACID (L ref) | > I-ACID | hard 例子优先更优 |
| ACED-ACIDistill | **最优** | ACID + KD 互补 |

- 参考模型 scaling：存在最优学生-参考容量比（Ti→B, S→L, B→g），超过后性能饱和
- ACID 在所有参考模型规模上均优于 IID，而 Softmax-KD 只在大教师时有效
- 跨数据集：无论参考/教师在 WebLI-curated++ 还是 WebLI 上训练，ACID 均显著优于 KD
- 跨蒸馏目标：ACID 优于 Softmax-KD、Sigmoid-KD、Feature-Matching KD 及其组合

### 关键发现

- **大模型可以有效地为小模型选数据**——打破了 active learning 和 data curation 领域的长期共识
- ACID 对小学生模型（Ti, S）效果更显著，KD 对大学生模型（B）更好
- 两种蒸馏方式真正互补：在个别 benchmark 上 ACID 不如 KD（4/27），组合后全面超越

## 亮点与洞察

- **理论贡献突出**：严格推导了 active curation 等价于隐式蒸馏，将数据筛选和模型压缩两个独立领域统一起来
- **方法极简**：不需要特殊架构、不需要合成数据、不需要权重继承、不需要数据增强——只需选好训练数据
- **StableEval 评估协议**：系统化地选择可靠 benchmark 组合，值得其他工作借鉴
- **scaling 分析系统**：从参考模型大小、训练数据、学生规模、蒸馏目标多个维度全面消融

## 局限与展望

- ACID 需要在超级批次上前向传播参考模型，增加了训练时计算开销（虽然推理时完全无额外成本）
- filtering ratio、参考模型选择等超参需调优
- 只在对比式 VLM（CLIP/SigLIP）上验证，对生成式多模态模型（如 LLaVA）是否适用未知
- 理论推导基于 softmax 变体，sigmoid 变体的理论保证较弱

## 相关工作与启发

- RHO-Loss 开创了基于参考模型的可教性评分，但限于小参考模型；本文扩展至大参考模型并给出理论解释
- 与 curriculum learning 的联系：ACID 本质上构建了一种动态课程，根据学生当前状态自适应选择"最有用"的样本
- 对数据筛选领域的启示：参考模型与学习模型之间的 capacity gap 不是阻碍而是优势

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 数据筛选=隐式蒸馏的理论洞察新颖且有深度
- 实验充分度: ⭐⭐⭐⭐⭐ 多维度 scaling 分析、27 个 benchmark、多种方法对比
- 写作质量: ⭐⭐⭐⭐⭐ 统一框架清晰优雅，可视化丰富
- 价值: ⭐⭐⭐⭐⭐ 对多模态模型压缩和数据筛选两个领域都有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models](img-diff_contrastive_data_synthesis_for_multimodal_large_language_models.md)
- [\[ACL 2025\] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relations?](../../ACL2025/multimodal_vlm/cordial_can_multimodal_large_language_models_effectively_understand_coherence_re.md)
- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[ACL 2025\] Scalable Vision Language Model Training via High Quality Data Curation](../../ACL2025/multimodal_vlm/scalable_vision_language_model_training_via_high_quality_data_curation.md)
- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)

</div>

<!-- RELATED:END -->
