---
title: >-
  [论文解读] OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection
description: >-
  [ECCV 2024][目标检测][Zero-shot Keypoint Detection] 提出 OpenKD 模型，从模态（视觉+文本）、语义（seen vs. unseen）、语言（多样化文本）三个维度开放 prompt 多样性，通过多模态 prototype set、辅助关键点-文本插值和 LLM 文本解析，实现通用的 zero- and few-shot keypoint detection，在 Animal Pose、AwA、CUB、NABird 上取得 SOTA。
tags:
  - "ECCV 2024"
  - "目标检测"
  - "Zero-shot Keypoint Detection"
  - "few-shot learning"
  - "CLIP"
  - "提示学习"
  - "LLM Parsing"
---

# OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection

**会议**: ECCV 2024  
**arXiv**: [2409.19899](https://arxiv.org/abs/2409.19899)  
**代码**: [https://github.com/AlanLuSun/OpenKD](https://github.com/AlanLuSun/OpenKD)  
**领域**: 关键点检测 / 目标检测  
**关键词**: Zero-shot Keypoint Detection, few-shot learning, CLIP, Multimodal Prompting, LLM Parsing

## 一句话总结

提出 OpenKD 模型，从模态（视觉+文本）、语义（seen vs. unseen）、语言（多样化文本）三个维度开放 prompt 多样性，通过多模态 prototype set、辅助关键点-文本插值和 LLM 文本解析，实现通用的 zero- and few-shot keypoint detection，在 Animal Pose、AwA、CUB、NABird 上取得 SOTA。

## 研究背景与动机

关键点检测是计算机视觉的基础任务（姿态估计、动作识别、细粒度分类等），但现有全监督方法只能预测固定物种的固定关键点集合，泛化到新物种/新关键点需要重新标注大量数据。Few-shot keypoint detection (FSKD) 通过视觉 prompt（带标注的支持图像）实现少样本检测；Zero-shot keypoint detection (ZSKD) 借助 CLIP 等 VLM 通过文本 prompt 实现零样本检测。

**现有痛点**：

**模态单一**：大多数方法只支持视觉 prompt 或文本 prompt，不能同时利用两种模态的互补优势

**语义封闭**：模型无法处理未见过的文本 prompt（如训练时见过 "eye" 但测试时遇到 "knee"），novel keypoint 检测性能极差

**语言刻板**：现有 ZSKD 仅支持模板化的简单文本（如 "the nose of a cat"），无法处理自然语言风格多样的提问（如 "Can you detect the nose and ears of a cat?"）

**核心 idea**：OpenKD 从三个维度"开放 prompt 多样性"——用多模态 prototype set 支持模态多样性，用 LLM 推理辅助关键点文本插值支持语义多样性，用 LLM 文本解析支持语言多样性。

## 方法详解

### 整体框架

OpenKD 基于 episodic training，每个 episode 包含 support set（视觉/文本 prompt）和 query set（待检测图像）。推理流程：1) CLIP 编码器提取图像和文本特征 → 2) 残差适配网络微调特征 → 3) 构建多模态关键点 prototype set → 4) prototype 与 query 特征做 correlation → 5) 解码得到 heatmap → 6) 融合多模态 heatmap 得到最终预测。

### 关键设计

1. **基于 CLIP 的多模态特征提取与投影**:

    - 功能：使用 CLIP RN50 作为共享 backbone，提取支持/查询图像特征和文本特征
    - 核心思路：原始 CLIP 图像编码器通过 attention pooling 只保留 classification token，丢弃了空间信息。作者通过 $\mathbf{X}' = \mathbf{X}\mathbf{W}_v\mathbf{W}_o$ 复用 CLIP attention pooling 的 V/O 投影矩阵来获取 projected image tokens，保留空间位置信息
    - 设计动机：关键点检测需要精确的空间定位，不能只用全局特征；同时复用 CLIP 的投影矩阵可以拉近图像 token 与文本特征的模态间距

2. **残差特征适配（Residual Feature Adaptation）**:

    - 功能：使用两个轻量适配网络 $\mathcal{A}_v$ 和 $\mathcal{A}_t$ 以残差方式分别微调图像和文本特征
    - 核心思路：$\mathbf{X}^s := \mathbf{X}^s + \mathcal{A}_v(\mathbf{X}^s)$，$\mathbf{t}_n := \mathbf{t}_n + \mathcal{A}_t(\mathbf{t}_n)$
    - 设计动机：CLIP 的预训练特征是 image-level 对齐而非 keypoint-level 对齐，需要适配到关键点检测的细粒度任务空间

3. **多模态关键点 Prototype Set**:

    - 功能：将视觉和文本 prompt 统一转换为 keypoint prototype，构建 prototype set $\mathcal{T} = \mathcal{T}^v \cup \mathcal{T}^t$
    - 视觉 prototype (VKP)：对支持图像特征图 $\mathbf{X}^s$ 以关键点位置为中心用高斯加权求和，得到 $\mathbf{\Phi}_n \in \mathbb{R}^d$；K-shot 时取同类关键点平均 $\mathbf{\Psi}_n^v = \frac{1}{K}\sum_k \mathbf{\Phi}_{k,n}$
    - 文本 prototype (TKP)：直接用 CLIP 文本编码器编码关键点文本得到 $\mathbf{\Psi}_n^t$
    - 设计动机：将不同模态的 prompt 统一到共享 $d$ 维特征空间，使模型能灵活处理视觉 prompt、文本 prompt 或两者组合

4. **辅助关键点与文本插值（Auxiliary Keypoints & Texts Interpolation）**:

    - 功能：在视觉和文本两个域中生成辅助训练样本，大幅提升 novel keypoint 检测能力
    - **视觉插值**：在两个已知关键点之间按 $z=0.5$ 线性插值生成辅助关键点位置 $\hat{\mathbf{p}}$，用显著性检测过滤前景外的点
    - **文本插值**：利用 LLM（GPT-3.5）推理两个已知关键点之间可能存在的身体部位。使用 Chain of Thought (CoT) prompt 提升推理质量。重复 $R$ 次每次返回 3 个答案，构建候选文本池 $\{{\hat{t}_i}\}_{i=1}^{3R}$
    - **False Text Control (FTC) 选择策略**：从候选文本池中采样 top-$\eta$ 结果，但若辅助关键点的视觉特征 $\hat{\mathbf{\Phi}}$ 与候选文本特征 $\hat{\mathbf{t}}_i$ 的 cosine similarity 低于阈值 $\alpha$，则拒绝该文本
    - 设计动机：训练时见过的关键点类型有限（如只有 "eye", "nose"），无法泛化到 novel keypoint。通过插值生成中间关键点和对应文本，扩展模型的空间推理能力

5. **模态内/模态间对比学习**:

    - 功能：引入两个对比损失提升 prototype 的判别性
    - $\mathcal{L}_{tt}$（文本间对比）：随机采样两个物种的 TKP set，构建相似度矩阵 $\mathbf{J}$，优化相同类型关键点跨物种的不变性和不同类型关键点的区分度
    - $\mathcal{L}_{vt}$（视觉-文本对比）：将 VKP 对齐向 TKP，对 TKP 施加 stop gradient 避免质量更好的文本表征被拖低
    - 设计动机：实验发现文本 prototype 天然具有更好的聚类效果和更低方差，因此将视觉对齐向文本而非反向

6. **LLM 作为语言解析器**:

    - 功能：用 LLM 解析多样化自然语言文本 prompt，提取 keypoint 和 object 关键词，合成标准 prompt 格式
    - 例如输入 "Can you localize the left eye and nose of cat?"，LLM 解析出 "left eye", "nose", "cat"
    - GPT-3.5 解析准确率达 96%+，Vicuna 达 93%+

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \lambda_1 \mathcal{L}_{kp} + \lambda_2 \mathcal{L}_{tt} + \lambda_3 \mathcal{L}_{vt}$

- $\mathcal{L}_{kp}$：多组 heatmap 回归损失（MSE），分别监督视觉组和文本组 heatmap
- 默认 $\lambda_1=1$，$\lambda_2=\lambda_3=0.002$
- CLIP 文本编码器冻结，图像编码器微调最后两层，温度 $\tau=0.05$

## 实验关键数据

### 主实验

**1-shot Keypoint Detection（Animal Pose Dataset，5 个子问题平均 PCK@0.1）：**

| 方法 | Novel | Base |
|------|-------|------|
| ProtoNet | 15.47 | 37.73 |
| FSKD-D | 44.75 | 49.93 |
| **OpenKD** | **50.32** | **54.39** |
| **OpenKD+Text** | **63.19** | **64.93** |

**0-shot Keypoint Detection（Animal Pose Dataset，PCK@0.1）：**

| 方法 | Novel | Base |
|------|-------|------|
| CLAMP | 21.92 | 59.47 |
| CLAMP† (加辅助文本) | 59.84 | 59.51 |
| **OpenKD** | **63.37** | **65.59** |

**跨数据集 1-shot 结果（Novel keypoints）：**

| 方法 | Animal Pose | AwA | CUB | NABird |
|------|-------------|-----|-----|--------|
| FSKD-D | 44.75 | 64.76 | 77.89 | 56.04 |
| OpenKD | 50.32 | 66.71 | 78.39 | 53.35 |
| OpenKD+T | **63.19** | **79.02** | 73.29 | 53.40 |

### 消融实验

**辅助关键点与文本的效果（Animal Pose）：**

| 训练配置 | 1-shot Novel | 0-shot Novel | 说明 |
|---------|-------------|-------------|------|
| 仅主关键点 | 21.36 | 1.26 | 无法检测 novel |
| +辅助关键点 | 47.54 | 2.00 | 视觉插值大幅提升 1-shot |
| 仅主文本 | 16.18 | 25.60 | 基础 0-shot 能力 |
| +辅助文本 | 15.87 | **63.14** | 文本插值大幅提升 0-shot |
| 全部（ours） | **50.32** | **63.37** | 最佳组合 |

**对比学习消融（AwA）：**

| 配置 | 1-shot Novel | 0-shot Base | 说明 |
|------|-------------|-------------|------|
| 无 CL | 65.56 | 81.67 | baseline |
| +$\mathcal{L}_{tt}$ | 66.05 | 84.00 | 文本判别性提升 |
| +$\mathcal{L}_{tt}$+$\mathcal{L}_{vt}$(stop grad) | **66.71** | **84.32** | 最佳 |

### 关键发现

- **辅助文本插值贡献最大**：0-shot Novel 从 25.60 跃升至 63.14（+37.54%），证明 LLM 推理的辅助文本对泛化至关重要
- **文本 prompt 优于视觉 prompt（base keypoints）**：0-shot base > 1-shot base，因为文本特征聚类效果更好、方差更低
- **多模态组合互补**：1-shot+text 测试（63.19%）显著优于任一单模态，两种模态的弱点被互补
- **CoT 推理提升显著**：使用 CoT prompt 的文本插值在 AwA Novel 上比无 CoT 高 4.5%（78.30 vs. 73.80）
- **LLM 解析鲁棒**：GPT-3.5 对多样化文本的解析准确率 > 96%，性能下降仅 1.29%（Novel）

## 亮点与洞察

- **三维度开放 prompt 多样性**的问题定义非常有价值——明确了模态、语义、语言三个维度的不足，每个维度都有针对性解决方案
- **LLM 的双重角色**：既作为推理器（text interpolation）又作为解析器（diverse text parsing），巧妙利用了 LLM 的不同能力
- **Stop gradient 策略**：发现文本表征质量优于视觉后，单向对齐 VKP → TKP，避免"强带弱"变成"弱拖强"，这个洞察可迁移到其他多模态对齐场景
- **辅助文本 + FTC 选择**策略：通过视觉-文本相似度过滤低质量候选，平衡了召回率和精度

## 局限与展望

- **CUB 和 NABird 上提升有限**：鸟类关键点的辅助文本推理更难，LLM 对细粒度身体部位的推理质量下降
- **依赖 LLM API**：文本插值和解析都需要 LLM（GPT-3.5），增加了成本和延迟
- **仅使用 CLIP RN50**：未探索更强的 ViT-based CLIP 或更新的 VLM（如 SigLIP）
- **文本辅助关键点的空间位置不精确**：LLM 推理的文本与实际插值位置可能不完全匹配，FTC 仅是缓解而非解决

## 相关工作与启发

- **vs CLAMP**: CLAMP 也用 CLIP 做动物姿态估计，但只支持文本 prompt 且无法处理 novel text。OpenKD 的辅助文本插值策略（加到 CLAMP 后的 CLAMP† 版本）也能大幅提升其性能（Novel 21.92→59.84），证明策略的通用性
- **vs FSKD-D**: 纯视觉的 FSKD 方法，OpenKD 通过引入文本模态在 1-shot 上大幅超越（50.32 vs. 44.75），加文本后更是 63.19
- **vs ProtoNet / RelationNet**: 经典 few-shot learning 方法在 keypoint detection 上表现较弱，说明需要任务特定的设计

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性提出三维度 prompt 多样性开放，辅助文本插值和 LLM 解析的结合很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、FSKD/ZSKD/混合测试、详细消融、多种对比方法
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义好，但公式符号较多，需要仔细阅读
- 价值: ⭐⭐⭐⭐ 在 keypoint detection 领域推动了多模态和零样本的边界，LLM 辅助的思路有广泛迁移价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)
- [\[ICCV 2025\] UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement](../../ICCV2025/object_detection/upre_zero-shot_domain_adaptation_for_object_detection_via_unified_prompt_and_rep.md)
- [\[AAAI 2026\] PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts](../../AAAI2026/object_detection/promptmoe_generalizable_zero-shot_anomaly_detection_via_visually-guided_prompt_m.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](../../CVPR2026/object_detection/bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](../../CVPR2026/object_detection/gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)

</div>

<!-- RELATED:END -->
