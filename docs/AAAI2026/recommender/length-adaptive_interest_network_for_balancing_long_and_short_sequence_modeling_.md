---
title: >-
  [论文解读] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction
description: >-
  [AAAI 2026][推荐系统][CTR预测] 提出LAIN框架，通过将序列长度作为显式条件信号注入CTR模型，缓解长序列用户与短序列用户之间的性能不均衡问题，包含谱长度编码器、长度条件提示和长度调制注意力三个轻量级即插即用模块。 核心矛盾：长短序列用户的性能失衡 现代推荐系统中，用户行为序列表现出显著的长度异质性——从稀…
tags:
  - "AAAI 2026"
  - "推荐系统"
  - "CTR预测"
  - "序列长度自适应"
  - "注意力极化"
  - "长短序列建模"
---

# Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction

**会议**: AAAI 2026  
**arXiv**: [2601.19142](https://arxiv.org/abs/2601.19142)  
**代码**: [FuxiCTR](https://github.com/reczoo/FuxiCTR)  
**领域**: 推荐系统  
**关键词**: CTR预测, 序列长度自适应, 注意力极化, 长短序列建模, 推荐系统

## 一句话总结

提出LAIN框架，通过将序列长度作为显式条件信号注入CTR模型，缓解长序列用户与短序列用户之间的性能不均衡问题，包含谱长度编码器、长度条件提示和长度调制注意力三个轻量级即插即用模块。

## 研究背景与动机

### 核心矛盾：长短序列用户的性能失衡

现代推荐系统中，用户行为序列表现出显著的长度异质性——从稀疏的短期交互到丰富的长期历史。主流CTR模型（如SIM、SDIM、TWIN）采用两阶段架构（GSU + ESU）来建模长序列，取得了不错效果。然而，作者发现了一个**反直觉现象**：

**性能失衡**：增加最大输入序列长度时，长序列用户（>200）性能提升，但短序列用户（<100）的AUC反而显著下降

**数据失衡**：短序列用户占57%的用户群体，但仅贡献14.3%的训练样本；而25%的长序列用户却主导了68%的训练数据

这导致模型隐式偏向优化高频长序列用户，而忽略了占多数的短序列用户（通常是新用户或低活跃用户），影响平台增长。

### 两大根因诊断

**注意力极化（Attention Polarization）**：
softmax注意力在长序列上会过度集中于少数显著行为（Gini系数随序列长度急剧增大），对长序列用户有利（过滤噪声），但对短序列用户有害——仅有几个行为时，注意力会过度集中在一两个item上，导致有限信号的利用不充分。

**长度信号缺失（Length Signal Deficiency）**：
现有模型将行为序列视为同质事件集，完全忽略序列长度这一重要先验信息。实际上，短序列用户的点击率更低（8.28% vs 9.41%）、行为不稳定性更高（变异系数0.515 vs 0.268）、兴趣多样性更低（平均2.5 vs 4.6种类别）。用一套参数处理3次交互和300次交互的用户是不合理的。

### 梯度冲突的理论分析

从优化角度来看，短序列用户需要平滑、分散的注意力来最大化有限信号利用，而长序列用户需要稀疏、集中的注意力来聚焦关键兴趣。这导致梯度信号冲突：$\langle\nabla\mathcal{L}_{\text{short}},\nabla\mathcal{L}_{\text{long}}\rangle<0$，降低泛化性能。

## 方法详解

### 整体框架

LAIN是一个轻量级、即插即用的长度感知增强框架，将序列长度$L$作为条件信号注入CTR模型。参数分解为共享参数和自适应参数：$\theta=\{\theta_{\text{shared}},\theta_{\text{prompt}}(L)\}$，其中后者由长度$L$的可学习函数动态生成，从而解耦不同长度用户的梯度流。

### 关键设计

#### 1. 谱长度编码器（Spectral Length Encoder, SLE）

**功能**：将离散的序列长度$L$编码为连续的稠密向量表示。

**核心思路**：使用可学习的傅里叶基函数将长度映射到连续空间，避免离散化导致的信息损失：

$$\mathbf{f}_{\text{fourier}}=[\sin(L\cdot\boldsymbol{\omega});\cos(L\cdot\boldsymbol{\omega})]\in\mathbb{R}^{2d_f}$$

其中$\boldsymbol{\omega}\in\mathbb{R}^{d_f}$是可学习的频率参数。再通过MLP投射得到共享嵌入：

$$\mathbf{h}_{\text{len}}=\text{MLP}(\text{LayerNorm}(\text{Linear}(\mathbf{f}_{\text{fourier}})))\in\mathbb{R}^{d}$$

**设计动机**：受NeRF中位置编码的启发（Tancik et al. 2020），傅里叶特征能自然地捕捉长度的连续语义，同时保持对不同尺度长度变化的敏感性。

#### 2. 长度条件提示（Length-Conditioned Prompting, LCP）

**功能**：从长度嵌入生成$k$个可学习的提示token，注入行为序列中。

**核心思路**：从$\mathbf{h}_{\text{len}}$生成提示矩阵：

$$\mathbf{P}(L)=\text{reshape}(\text{MLP}_{\text{prompt}}(\mathbf{h}_{\text{len}}))\in\mathbb{R}^{k\times d}$$

将提示token预置到短期和长期行为序列前端：

$$\mathbf{S}'_{\text{short}}=[\mathbf{P};\mathbf{S}_{\text{short}}], \quad \mathbf{S}'_{\text{long}}=[\mathbf{P};\mathbf{S}_{\text{long}}]$$

**设计动机**：LCP有效地注入了全局用户状态信息，扩展了参数空间，使不同长度的用户可以获得不同的条件信号，避免参数共享导致的干扰。

#### 3. 长度调制注意力（Length-Modulated Attention, LMA）

**功能**：根据序列长度自适应地调整注意力机制的行为。包含两个子模块。

**查询-键条件化（Query-Key Conditioning）**：将长度嵌入拼接到每个查询和键向量中：

$$\mathbf{Q}'=W_q([\mathbf{Q};\mathbf{e}_{\text{len}}]), \quad \mathbf{K}'=W_k([\mathbf{K};\mathbf{e}_{\text{len}}])$$

**Softmax温度缩放**：定义长度感知的温度系数：

$$\tau=1+\text{sigmoid}(-\beta(L-L_0))\cdot\gamma$$

注意力权重变为：$\alpha_{ij}=\frac{\exp(\frac{\mathbf{Q}'_i\cdot\mathbf{K}'_j}{\sqrt{d}\cdot\tau})}{\sum_j\exp(\frac{\mathbf{Q}'_i\cdot\mathbf{K}'_j}{\sqrt{d}\cdot\tau})}$

**设计动机**：当$L$较小时，$\tau$增大，平滑注意力分布，减少过极化；当$L$较大时，$\tau$缩小，促进对显著item的聚焦。$\beta$和$\gamma$是可学习参数，模型能自动找到最佳平衡点。

### 损失函数 / 训练策略

LAIN端到端训练，使用标准的二元交叉熵损失：

$$\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}[y_i\log\hat{y}_i+(1-y_i)\log(1-\hat{y}_i)]$$

额外参数开销不到1.5%，推理时间增加约2.3%。$k$=2~4，序列长度从$L$增加到$L+k$，总体复杂度仍为$\mathcal{O}(L^2d)$。

## 实验关键数据

### 主实验

| 数据集 | 指标 | TWIN | TWIN+LAIN | 提升 |
|--------|------|------|-----------|------|
| EBNeRD-small | GAUC | 0.6930 | 0.7012 | +1.19% |
| EBNeRD-small | AUC | 0.6993 | 0.7074 | +1.15% |
| KuaiVideo | AUC | 0.6918 | 0.6976 | +0.84% |
| MicroVideo1.7M | AUC | 0.7158 | 0.7233 | +1.05% |
| MicroVideo1.7M | Logloss | 0.4164 | 0.4097 | -1.63% |

LAIN在5个骨干模型（DIN、DIEN、SIM、SDIM、TWIN）和3个数据集上**一致提升**，AUC最高提升1.15%，Logloss最大下降2.25%。

### 分长度评估（TWIN, MicroVideo1.7M）

| 序列长度 | AUC提升 | Logloss下降 |
|----------|---------|-------------|
| 0-100（短） | +1.08% | -2.17% |
| 100-200（中） | +0.50% | -1.26% |
| 200+（长） | +1.58% | -1.74% |

**关键发现**：短序列用户获得显著提升（+1.08% AUC），同时长序列用户不但没有牺牲反而也有提升（+1.58%），实现了双赢。

### 消融实验

| 配置 | AUC | Logloss | 说明 |
|------|-----|---------|------|
| LAIN (Full) | 0.7233 | 0.4097 | 完整模型 |
| w/o LCP | 0.7228 | 0.4107 | 去掉长度条件提示 |
| w/o Query-Key Conditioning | 0.7195 | 0.4148 | 去掉QK条件化 |
| w/o Temperature Scaling | 0.7212 | 0.4111 | 去掉温度缩放 |
| w/o LMA | 0.7189 | 0.4157 | 去掉整个长度调制注意力 |
| w/o Short-term Branch | 0.7226 | 0.4137 | 不在短期分支应用LAIN |

### 关键发现

1. LMA（长度调制注意力）是最关键的组件，移除后AUC下降最大（-0.44%）
2. LAIN将注意力的Gini系数从0.346-0.353降低到0.318-0.322，方差减少50.6%，显著缓解了注意力极化
3. 12种超参数配置都优于基线，说明方法鲁棒

## 亮点与洞察

1. **问题发现有价值**：首次系统性地识别和分析了CTR建模中的长度失衡问题，不只是发现现象，还深入诊断了注意力极化和长度信号缺失两个根因
2. **方案设计优雅**：三个模块各司其职、互相协作，从编码、注入、调制三个层面全方位注入长度意识，且即插即用、开销极低
3. **温度缩放的直觉**：通过sigmoid自动在"平滑"（短序列）和"聚焦"（长序列）之间切换，是一个简洁有效的设计
4. **实用性强**：额外参数<1.5%，推理开销~2.3%，工业部署友好

## 局限与展望

1. 仅在公开数据集上验证，工业规模的在线A/B测试结果未展示
2. 长度阈值（短/中/长的切分点）是手动设定的，可以探索自适应分组
3. 温度缩放公式相对简单（sigmoid形式），可以考虑更灵活的参数化
4. 未讨论在非CTR任务（如序列推荐、会话推荐）上的效果

## 相关工作与启发

- **SIM/SDIM/TWIN系列**：两阶段长序列建模架构的基础，LAIN作为增强插件在其上工作
- **PPR (Wu et al. 2024)**：从用户画像生成软提示用于冷启动，LAIN则从序列长度维度生成提示
- **DARE (Feng et al. 2025)**：解耦注意力和表示嵌入，但未条件化于序列长度
- 启发：条件信号（如用户活跃度、时间跨度等）的注入可能是一种通用的推荐模型增强范式

## 评分

- 新颖性: ⭐⭐⭐⭐ （问题发现新颖，方案是已有技术的组合但组合得当）
- 实验充分度: ⭐⭐⭐⭐⭐ （3数据集×5骨干+分长度评估+消融+参数敏感性+注意力分析）
- 写作质量: ⭐⭐⭐⭐⭐ （诊断-方案-验证逻辑清晰，图表丰富）
- 价值: ⭐⭐⭐⭐ （实用问题、轻量方案、工业可部署）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[AAAI 2026\] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search](crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)
- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[AAAI 2026\] When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation](when_top-ranked_recommendations_fail_modeling_multi-granular_negative_feedback_f.md)
- [\[AAAI 2026\] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)

</div>

<!-- RELATED:END -->
