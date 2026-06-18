---
title: >-
  [论文解读] Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning
description: >-
  [NeurIPS 2025][多模态VLM][上下文表征学习] 提出 In-Context Representation Learning（ICRL），首个训练无关框架，将非文本模态基础模型（FM）的表征注入纯文本 LLM 进行少样本推理，通过 PCA 文本注入和最优传输嵌入对齐两种策略实现跨模态知识利用。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "上下文表征学习"
  - "训练无关多模态推理"
  - "最优传输对齐"
  - "基础模型"
  - "少样本学习"
---

# Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.17552](https://arxiv.org/abs/2509.17552)  
**代码**: [GitHub](https://github.com/ztlmememe/LLMxFM_ICRL)  
**领域**: 多模态VLM  
**关键词**: 上下文表征学习, 训练无关多模态推理, 最优传输对齐, 基础模型, 少样本学习

## 一句话总结

提出 In-Context Representation Learning（ICRL），首个训练无关框架，将非文本模态基础模型（FM）的表征注入纯文本 LLM 进行少样本推理，通过 PCA 文本注入和最优传输嵌入对齐两种策略实现跨模态知识利用。

## 研究背景与动机

### 1. 领域现状

LLM 通过测试时计算和工具集成大幅增强能力，多模态 LLM（MLLM）如 LLaVA、InstructBLIP 通过训练投影层/微调使文本 LLM 理解图像等非文本模态。

### 2. 现有痛点

- 现有 MLLM 方法需要**额外有监督训练**来对齐模态（如训练投影层或微调 LLM 本身），计算成本高
- 需要专门的**配对数据集**，限制了对新领域/新模态的即时适应
- 多智能体框架仅利用外部模型的最终输出，浪费了中间表征的丰富信息

### 3. 核心矛盾

能否让纯文本 LLM **在推理时不需训练**就利用非文本基础模型的内部表征？

### 4. 本文目标

探索训练无关地将非文本 FM 表征集成到文本 LLM 的可行性，并分析其工作机制。

### 5. 切入角度

将传统 ICL 中的文本-标签对 $(x_i, y_i)$ 替换为 FM 表征-标签对 $(r_i, y_i)$，利用少样本学习能力使 LLM 适应性地处理非文本表征。

### 6. 核心 idea

用降维后的 FM 表征替代 ICL 中的文本输入，通过最优传输将分布对齐到 LLM 嵌入空间，实现零训练的跨模态推理。

## 方法详解

### 整体框架

ICRL 框架支持两个注入层级：
1. **文本层注入**：将 FM 表征降维后作为字符串嵌入 prompt
2. **嵌入层注入**：将 FM 表征投影后直接替换 LLM embedding 层的 token

### 关键设计

#### 模块1：PCA 文本层注入

**功能**：通过降维将高维 FM 表征嵌入 prompt 文本。

**核心思路**：对 FM 表征 $\mathbf{H} \in \mathbb{R}^{N \times d_{FM}}$ 做 PCA 降到 $d_{Reduced} \ll d_{FM}$ 维（默认 20 维），然后将降维后的向量转为字符串 $S_{pca}$ 融入 prompt。

**设计动机**：最简单直接的方式。虽然信息有损但实验证明 LLM 能有效理解数值字符串并用于推理。

#### 模块2：最优传输对齐（OT-Embed / OT-PCA）

**功能**：将投影后的 FM 表征分布对齐到 LLM 嵌入分布。

**核心思路**：使用随机初始化的线性投影器 $P: \mathbb{R}^{d_{FM}} \to \mathbb{R}^{d_{LLM}}$，然后通过最优传输对齐分布：

对每个维度 $j$ 计算偏移和缩放：
$$shift_j = \bar{v}_j - \bar{u}_j, \quad scale_j = \frac{\sigma_{t,j}}{\sigma_{p,j}}$$

最终对齐：
$$OT(\mathcal{D}_{proj}, \mathcal{D}_{tar}) = scale \cdot \mathbf{H}_{proj} + shift$$

两个变体：
- **OT-Embed**：目标分布为 LLM 对原始文本（如 SMILES）的嵌入
- **OT-PCA**：目标分布为 LLM 对 PCA 字符串的嵌入

**设计动机**：直接投影可能导致分布不匹配，OT 通过对齐均值和方差确保 FM 表征落入 LLM 嵌入的正常统计范围。

#### 模块3：线性投影器的理论保证

**功能**：证明不加非线性激活的随机线性投影器能保持几何关系。

**定理 1（范数集中）**：
$$|\|\mathbf{W}\mathbf{u} + \mathbf{b}\|^2 - (\|\mathbf{b}\|^2 + \|\mathbf{u}\|^2)| \leq \epsilon_1 (\|\mathbf{b}\|^2 + \|\mathbf{u}\|^2)$$
其中 $\epsilon_1 = O(\sqrt{\log(1/\delta_1)/d})$。

**定理 2（余弦相似度保持）**：
$$|\cos(\mathbf{W}\mathbf{u}, \mathbf{W}\mathbf{v}) - \cos(\mathbf{u}, \mathbf{v})| \leq \epsilon_2$$

**推论 1**：非线性激活（ReLU, sigmoid）会扭曲向量角度、膨胀相似度，导致信息丢失。

**设计动机**：理论上保证线性投影比 MLP 更适合保持 FM 表征的几何结构。

### 损失函数/训练策略

**完全无训练**：
- 投影器随机初始化，无需任何梯度更新
- OT 的 shift/scale 参数只需计算一次（<2秒 CPU）
- 推理时使用 Llama-3.1-70B-Instruct，标准 ICL 流程

## 实验关键数据

### 主实验：ICRL 各注入方法对比（RMSE ↓）

| 数据集 | ICL (文本) | PCA | Zero-Pad | Random Noise | Random Proj | OT-Embed | OT-PCA |
|------|------|------|------|------|------|------|------|
| ESOL | 1.16 | **1.11** | 1.73 | 1.41 | 1.69 | 1.19 | 1.24 |
| Caco2 | **0.83** | 0.95 | 1.04 | 1.03 | 1.03 | 0.89 | **0.88** |
| LD50 | **0.99** | 1.06 | 1.28 | 1.21 | 1.29 | 1.18 | 1.14 |
| AstraZeneca | **1.37** | **1.39** | 1.55 | 1.50 | 1.54 | 1.46 | 1.47 |

文本层 PCA 在多数数据集最优；嵌入层中 OT 系列最佳。

### ICRL+ICL 联合使用（Pearson r ↑）

| 数据集 | ICL baseline | Zero-Pad+ICL | Ran-Noi+ICL | OT-Embed+ICL | OT-PCA+ICL |
|------|------|------|------|------|------|
| ESOL | 0.465 | 0.526 | **0.540** | 0.508 | **0.542** |
| Caco2 | 0.411 | 0.410 | 0.420 | **0.429** | 0.394 |
| AqSolDB | 0.596 | **0.606** | 0.597 | 0.569 | 0.589 |

OT-PCA+ICL 在 ESOL 上提升 **16.6%** Pearson r。

### 消融实验

**激活函数影响**：

| 投影器类型 | ESOL RMSE | Caco2 RMSE |
|------|------|------|
| 线性（无激活） | **最优** | **最优** |
| +ReLU | 下降 | 下降 |
| +GELU | 下降 | 下降 |

**PCA 维度**：纯 ICRL 模式下增加 PCA 维度并不提升性能（甚至下降），说明 LLM 理解长数值序列有限。

**成本对比**：

| 方法 | 类型 | 资源 | 训练时间 | ESOL RMSE |
|------|------|------|------|------|
| MolecularGPT | 指令微调 | 4×A800 | <1天 | 1.471 |
| GIMLET | 预训练+微调 | 2-4 GPU | ~1天 | 1.132 |
| GPT-MolBERTa | 预训练+微调 | 2-4 GPU | ~2周 | **0.477** |
| **OT-PCA (Ours)** | **训练无关** | **仅CPU** | **~2秒** | 1.140 |

### 关键发现

1. **ICRL 表征间的高相似度会降低性能**：FM 特征空间狭窄导致映射后的嵌入高度相似，LLM 无法区分不同样本
2. **两种运行模式**：
    - 纯 ICRL：LLM 进入"任务学习模式"，依赖少样本范例进行预测
    - ICRL+文本：LLM 进入"任务检索模式"，注入的表征更像"暂停 token"增加思考时间
3. **更好的 FM 特征 + 文本不一定更好**：OT 方法单独最优，但加上文本后简单方法（如随机噪声）反而更好——因为 LLM 注意力主要集中在 SMILES 文本上

## 亮点与洞察

1. **首个训练无关的非文本模态集成框架**：完全不需要梯度更新，仅靠分布对齐即可使 LLM 利用 FM 表征
2. **理论支撑**：证明线性投影保持几何结构优于非线性 MLP，为投影器设计提供原则性指导
3. **最优传输对齐**：简洁高效（<2秒 CPU），每个 FM 表征仅占一个 token 位置
4. **双运行模式发现**：揭示了 ICRL 表征在有/无文本输入时扮演不同角色（信息载体 vs 暂停 token）
5. **跨模态泛化**：初步展示了在视觉（ViT）和语音（wav2vec2）上的可行性

## 局限与展望

1. ICRL 总体性能仍显著弱于有监督微调方法（如 GPT-MolBERTa ESOL 0.477 vs ICRL 1.140）
2. 目前主要在分子领域（SMILES）验证，文本天然具有分子结构信息；在纯视觉等模态的验证有限
3. 纯 ICRL 需要 >10 个样例才能超越随机预测，少样本效率有待提升
4. OT 对齐仅在均值/方差维度进行一阶矩匹配，更精细的分布对齐（如 Sinkhorn）可能带来改进
5. 缺乏对 LLM 内部注意力模式的深入分析（如哪些层对 ICRL 表征最敏感）

## 相关工作与启发

- **与 Vector-ICL 的关系**：Vector-ICL 需要预训练/微调来训练投影器，ICRL 完全训练无关，是其轻量级替代方案
- **与 Flamingo/LLaVA 的关系**：这些 MLLM 需要大规模配对数据训练，ICRL 适用于配对数据稀缺的场景（如分子性质预测、蛋白质等）
- **与暂停 token 的联系**：ICRL 在有文本输入时表征被当作暂停 token 使用，呼应了 "Think before you speak" 的发现
- **对实际应用的启发**：在缺乏特定领域预训练多模态 LLM 的场景（如传感器数据、生物医学），ICRL 提供了即时可用的轻量方案

## 评分

⭐⭐⭐⭐ (4/5)

新颖的问题设定（训练无关跨模态推理），理论分析（线性投影保持几何）和机制分析（双运行模式）有深度。实验虽在分子领域为主但全面系统。主要不足是性能差距仍大，且"训练无关"的优势在不同场景下需权衡。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[NeurIPS 2025\] What Can RL Bring to VLA Generalization? An Empirical Study](what_can_rl_bring_to_vla_generalization_an_empirical_study.md)
- [\[ACL 2025\] Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates](../../ACL2025/multimodal_vlm/adversarial_compositionality_clip.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](training-free_online_video_step_grounding.md)

</div>

<!-- RELATED:END -->
