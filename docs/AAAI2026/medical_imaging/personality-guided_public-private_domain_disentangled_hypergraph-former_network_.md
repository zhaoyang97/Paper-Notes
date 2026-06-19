---
title: >-
  [论文解读] Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection
description: >-
  [AAAI 2026][医学图像][抑郁检测] 提出 P3HF 框架，通过人格引导的特征门控、时序感知的超图-Transformer（Hypergraph-Former）架构和事件级公私域解耦三大创新，在多事件多模态抑郁检测任务上实现约 10% 的准确率和 F1 提升。 抑郁症影响全球约 3.8% 的人口…
tags:
  - "AAAI 2026"
  - "医学图像"
  - "抑郁检测"
  - "多模态融合"
  - "超图神经网络"
  - "人格引导"
  - "领域解耦"
---

# Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection

**会议**: AAAI 2026  
**arXiv**: [2511.12460](https://arxiv.org/abs/2511.12460)  
**代码**: [https://github.com/hacilab/P3HF](https://github.com/hacilab/P3HF)  
**领域**: 医学图像  
**关键词**: 抑郁检测, 多模态融合, 超图神经网络, 人格引导, 领域解耦

## 一句话总结

提出 P3HF 框架，通过人格引导的特征门控、时序感知的超图-Transformer（Hypergraph-Former）架构和事件级公私域解耦三大创新，在多事件多模态抑郁检测任务上实现约 10% 的准确率和 F1 提升。

## 研究背景与动机

抑郁症影响全球约 3.8% 的人口，是 15-29 岁人群的第四大死因。心理健康资源的严重短缺促使研究者开发自动化抑郁检测技术。现有多模态方法面临三个关键问题：

**个体差异建模不足（Problem 1）**：抑郁表现因人而异，传统深度学习采用统一建模策略，忽略个体在表情模式和沟通风格上的异质性。离散的人口统计标签（如性别、年龄）无法捕捉细粒度的个体差异。

**时序依赖缺失（Problem 2）**：超图神经网络通过超边连接多个节点来建模高阶跨模态关系，但超边本质上是无序集合，无法显式捕捉节点间的时序关系。而抑郁症状表达往往具有重要的时间依赖性。

**跨事件泛化困难（Problem 3）**：根据 Bandura 的交互决定论，个体抑郁表现受个人因素、环境和行为的交互影响。单一事件场景无法全面捕捉抑郁的复杂表现，存在两类信息——跨事件共享的公共域信息（核心抑郁特征）和事件特异的私有域信息（个体化响应）。

## 方法详解

### 整体框架

P3HF 包含三个核心模块：(1) 人格引导的特征门控，利用 LLM 生成人格描述来引导音视频特征提取；(2) Hypergraph-Former 架构，在超图中引入位置编码和注意力机制；(3) 公私域解耦机制，通过对比学习区分跨事件共享信息和事件特异信息。

输入：样本 $S$ 由 $K$ 个事件组成 $\{E_1, E_2, \ldots, E_K\}$，每个事件包含视觉 $V_k$ 和音频 $A_k$ 两种模态。每个样本附带抑郁标签、大五人格评分和人口统计信息。

### 关键设计

#### 1. **人格引导的特征门控（Personality-guided Feature Gating）**

**核心思路**：利用 LLM 将离散的个体属性（性别、年龄、家乡、大五人格分数）转化为连续的上下文描述，为每个个体生成自适应的特征调制权重。

- **人格描述生成**：将人口统计和人格特质作为 prompt 输入 GPT-4（temperature=0），生成描述性文本，再用 BERT 编码为 768 维特征。
- **特征对齐**：对视觉 $V_k$、音频 $A_k$ 和人格 $P$ 分别通过 Bi-LSTM 统一维度到 $D_1$。
- **门控机制**：通过可学习线性变换生成门控权重 $W_{\text{gate}} = \sigma(\mathbf{W}_p \tilde{P} + \mathbf{b}_p)$，结合残差连接调制音视频特征：$A_k|P = \tilde{A}_k + \tilde{A}_k \odot W_{\text{gate}}$。
- **设计动机**：传统方法用离散标签（如 one-hot 编码性别）表示个体属性，粒度太粗。通过 LLM 生成的自然语言描述可以捕捉更细微的人格特征组合，而门控机制允许模型根据不同个体动态调整特征表示。

#### 2. **Hypergraph-Former 架构**

**核心思路**：在超图网络中引入位置编码和自注意力，同时建模高阶跨模态关系和时序依赖。

- **位置编码注入**：对人格引导后的特征添加正弦位置编码 $\hat{A}_k = A_k|P + \text{PE}(A_k|P)$，注入时序顺序信息。
- **超图构建**：每个事件构建超图 $\mathcal{H} = (\mathcal{V}, \mathcal{E})$，节点集 $\mathcal{V}$ 包含同一事件的所有音频和视觉特征（共 $2T_k$ 个节点）。超边通过滑动窗口（大小 $w$）构建：
    - 同模态超边（实线）：窗口内同一模态的所有节点相连，增强模态内局部特征
    - 跨模态超边（虚线）：窗口内一个模态的每个节点与另一模态的所有节点相连，建模局部跨模态交互
    - 共产生 $(T_k - w + 1) \times (2 + 2w)$ 条超边
- **超图卷积**：$X^{(l+1)} = \sigma(\mathbf{D}_v^{-1/2} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^T \mathbf{D}_v^{-1/2} X^{(l)} \mathbf{\Theta}^{(l)})$
- **自注意力增强**：超图处理后施加多头自注意力，捕捉超越局部连接的全局依赖
- **最终拼接**：音频和视觉注意力增强特征拼接 $H_k = \oplus(A_k^{(\text{att})}, V_k^{(\text{att})})$
- **设计动机**：传统超图无法编码时序关系，而抑郁症状的时序演化至关重要。滑动窗口策略捕捉局部时序一致性，最优窗口大小约 11 个时间步。

#### 3. **公私域解耦（Public-Private Domain Disentanglement）**

**核心思路**：通过对抗训练学习跨事件不变的公共域特征，通过 HSIC 独立性约束学习事件特异的私有域特征。

- **公共编码器**：共享的编码器提取跨事件的公共特征 $Pub_k = \text{Pub-Enc}(H_k)$
- **私有编码器**：每个事件独立的编码器提取私有特征 $Pri_k = \text{Pri}_k\text{-Enc}(H_k)$
- **对抗训练（公共域）**：训练判别器从公共特征预测事件标签，公共编码器试图欺骗判别器（最优判别准确率约 1/3 表示成功学习事件不变表示）
- **HSIC 约束（私有域）**：最小化不同事件私有特征间的 Hilbert-Schmidt 独立性准则 $\mathcal{L}_{HSIC} = \sum_{i \neq j} \text{HSIC}(Pri_i, Pri_j)$，确保私有编码器捕捉独立的事件特异特征
- **特征整合**：拼接平均公共特征和所有私有特征 $I = \bigoplus\{\frac{1}{K}\sum_{k=1}^K Pub_k, Pri_1, \ldots, Pri_K\}$

### 损失函数 / 训练策略

总训练目标：$\mathcal{L}_{\text{main}} = \alpha \mathcal{L}_{\text{dep}} + \beta \mathcal{L}_{\text{adv}} + \gamma \mathcal{L}_{\text{HSIC}}$，约束 $\alpha + \beta + \gamma = 1$。

- $\mathcal{L}_{\text{dep}}$：NLL 抑郁分类损失
- $\mathcal{L}_{\text{adv}}$：对抗损失（MinMax 优化）
- $\mathcal{L}_{\text{HSIC}}$：私有域独立性损失

采用交替训练策略：判别器最小化 $\mathcal{L}_{\text{disc}}$，主模型最小化 $\mathcal{L}_{\text{main}}$。最优损失权重为 $\beta = \gamma = 0.1$。

## 实验关键数据

### 主实验

在 MPDD-Young 数据集上评估（多事件、多模态、多维个体信息标注），包含 3 个任务事件（自我介绍、两个文本朗读）。

| 方法 | 二分类 ACC | 二分类 w-F1 | 三分类 ACC | 三分类 w-F1 |
|------|-----------|------------|-----------|------------|
| NUSD (2023) | 63.01 | 60.64 | 57.19 | 55.44 |
| STA-DRN (2024) | 64.14 | 62.23 | 58.93 | 57.34 |
| TBN (2019) | 66.21 | 64.77 | 61.76 | 60.23 |
| IA fusion (2022) | 68.41 | 67.23 | 62.87 | 61.39 |
| MGLRA (2024) | 70.37 | 68.93 | 61.35 | 59.78 |
| DepMamba (2025) | 72.56 | 71.44 | 67.85 | 66.23 |
| **P3HF (Ours)** | **82.17** | **81.39** | **76.29** | **74.61** |

二分类提升 9.61%/9.95%，三分类提升 8.44%/8.38%。

### 消融实验

| 配置 | 二分类 ACC | 二分类 w-F1 | 三分类 ACC | 三分类 w-F1 |
|------|-----------|------------|-----------|------------|
| w/o 视觉 | 77.52 | 76.63 | 72.94 | 70.52 |
| w/o 音频 | 76.89 | 75.77 | 70.85 | 69.39 |
| w/o 域解耦 | 71.84 | 70.17 | 66.53 | 65.72 |
| w/o 公共域 | 75.34 | 74.38 | 70.30 | 68.19 |
| w/o 私有域 | 78.15 | 77.02 | 74.01 | 71.32 |
| w/o 人格信息 | 76.68 | 75.41 | 71.55 | 69.24 |
| w/ 数值嵌入 | 80.61 | 78.77 | 75.32 | 73.34 |
| **完整模型** | **82.17** | **81.39** | **76.29** | **74.61** |

架构比较（融合模块）：

| 架构 | 二分类 ACC | 三分类 ACC |
|------|-----------|-----------|
| Cross-Attention | 75.82 | 69.15 |
| Directed GCN | 77.55 | 72.41 |
| Undirected GAT | 80.07 | 74.23 |
| Hypergraph | 79.68 | 73.86 |
| **Hypergraph-Former** | **82.17** | **76.29** |

### 关键发现

1. 音频特征比视觉特征更重要（移除音频导致更大性能下降），符合心理学中韵律线索在心理健康评估中更为关键的理论
2. 公共域编码器比私有域更关键（移除导致 6.83% vs 4.02% 下降），验证人格特质主导跨事件模式的假设
3. LLM 生成的人格描述（82.17%）优于传统数值嵌入（80.61%），证明语言描述能捕捉更丰富的个体差异
4. t-SNE 可视化清晰展示了解耦效果：最优配置 $\beta = \gamma = 0.1$ 下公共特征收敛、私有特征分离

## 亮点与洞察

1. **LLM 作为个体建模桥梁**：巧妙利用 GPT-4 将离散人格标签转化为连续语义描述，是将 LLM 用于心理计算的创新尝试
2. **超图 + Transformer 的有机融合**：通过位置编码和注意力机制补全超图在时序建模上的短板，比简单堆叠更优雅
3. **交互决定论的计算化**：将社会认知理论中的个人-环境-行为三元交互转化为公私域解耦的技术方案
4. **10%+ 的巨大提升**：在竞争激烈的抑郁检测任务上实现如此显著的提升非常少见

## 局限与展望

1. 仅在 MPDD-Young 一个数据集上验证，缺少 DAIC-WOZ 等其他主流数据集的实验
2. 依赖 GPT-4 生成人格描述，增加了部署成本和隐私风险
3. 仅使用音频和视觉两种模态，未考虑文本模态（如言语内容）
4. 滑动窗口大小需要调优，自适应窗口机制可能更好
5. MPDD 数据集规模相对较小，需要更大规模验证

## 相关工作与启发

- **DialogueGCN** 和 **MS2-GNN** 建立了图网络在多模态心理健康分析中的基础
- **MISA** 的模态不变/特异分离思想与本文的公私域解耦相似，但本文将其推广到事件级
- **DepMamba** 作为最强基线，表明 Mamba 架构在时序建模上有潜力，但缺乏高阶关系建模

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（三重创新，组合新颖且有理论支撑）
- 实验充分度: ⭐⭐⭐⭐（消融全面但数据集单一）
- 写作质量: ⭐⭐⭐⭐（组织清晰，公式规范）
- 价值: ⭐⭐⭐⭐（在抑郁检测领域有重要方法论贡献）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Cross-Modal Guided Visual Synthesis for Data-Efficient Multimodal Depression Recognition](../../CVPR2026/medical_imaging/cross-modal_guided_visual_synthesis_for_data-efficient_multimodal_depression_rec.md)
- [\[AAAI 2026\] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)
- [\[AAAI 2026\] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)
- [\[CVPR 2026\] Adaptive Confidence Regularization for Multimodal Failure Detection](../../CVPR2026/medical_imaging/adaptive_confidence_regularization_for_multimodal_failure_detection.md)
- [\[AAAI 2026\] DiA-gnostic VLVAE: Disentangled Alignment-Constrained Vision Language Variational AutoEncoder for Robust Radiology Reporting with Missing Modalities](dia-gnostic_vlvae_disentangled_alignment-constrained_vision_language_variational.md)

</div>

<!-- RELATED:END -->
