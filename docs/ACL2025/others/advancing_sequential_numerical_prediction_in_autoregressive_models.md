---
title: >-
  [论文解读] Advancing Sequential Numerical Prediction in Autoregressive Models
description: >-
  [ACL 2025][数值预测] 提出Numerical Token Integrity Loss (NTIL)——一种双层级数值预测损失函数，在token级别用指数位置加权的EMD替代交叉熵以保持数值有序性，在序列级别通过可微数值构造进行整体数值偏差惩罚，在目标检测、文字检测、数学推理和时钟识别等任务上显著提升自回归模型的数值预测精度。
tags:
  - "ACL 2025"
  - "数值预测"
  - "Earth Mover's Distance"
  - "自回归模型"
  - "损失函数"
  - "多模态大语言模型"
---

# Advancing Sequential Numerical Prediction in Autoregressive Models

**会议**: ACL 2025  
**arXiv**: [2505.13077](https://arxiv.org/abs/2505.13077)  
**代码**: [GitHub](https://github.com/xfey/NTIL)  
**领域**: 其他  
**关键词**: 数值预测, Earth Mover's Distance, 自回归模型, 损失函数, 多模态大语言模型

## 一句话总结

提出Numerical Token Integrity Loss (NTIL)——一种双层级数值预测损失函数，在token级别用指数位置加权的EMD替代交叉熵以保持数值有序性，在序列级别通过可微数值构造进行整体数值偏差惩罚，在目标检测、文字检测、数学推理和时钟识别等任务上显著提升自回归模型的数值预测精度。

## 研究背景与动机

**领域现状**：自回归模型（LLM/MLLM）已成为序列生成任务的主流选择，广泛应用于VQA、目标检测、数学推理等需要精确数值输出的任务。标准训练方法使用交叉熵(CE)损失进行token-by-token优化。

**现有痛点**：
- **局限1（token级）**：CE将每个数字token视为独立类别，忽略数值之间的有序关系（如预测"2"和预测"9"对于ground truth "3"的CE损失相同，但"2"显然更接近正确答案）
- **局限2（序列级）**：CE逐token计算损失，无法捕获多token组成的整体数值误差（如预测"1.01"与"1.98"相比目标"0.98"，前者数值更接近但CE损失更高）

**核心矛盾**：自回归模型需要逐token生成数值，但传统CE损失完全忽略了数字token之间的ordinal关系和跨token的数值完整性，导致数值预测精度受限。

**本文目标** 设计一种能同时在token级别保持数值有序性、在序列级别保持数值完整性的训练损失函数，提升自回归模型在所有涉及数值输出的任务上的精度。

**切入角度**：将Earth Mover's Distance (EMD)引入自回归模型训练（首次），并结合可微数值构造实现跨token的序列级数值优化。

**核心 idea**：用EMD替代CE解决token级ordinal忽视问题，用可微数值重建解决序列级全局数值偏差问题，两层联合优化。

## 方法详解

### 整体框架

NTIL = 指数位置加权EMD（token级）+ 相对偏差度量 + 量级偏差度量（序列级），最终损失为三者的加权和：

$$\mathcal{L} = \mathbf{W_{exp}} \cdot \text{EMD} + \alpha \cdot \mathcal{L}_{relative} + \beta \cdot \mathcal{L}_{magnitude}$$

### 关键设计

1. **指数位置加权EMD（Exponential Position-Based Weighting）**:
    - 功能：在token级别替代CE，建模数字token之间的ordinal关系
    - 核心思路：EMD衡量将预测分布"搬运"到目标分布的最小代价，天然编码了数字间的距离关系；进一步引入指数位置加权W_exp = [(1+σ)^(n-i-1)]，使高位数字的误差获得更大惩罚
    - 设计动机：在十进制位值系统中，高位数字（如百位）比低位数字（如个位）对最终数值影响大得多，指数加权自然反映了这一性质

2. **可微数值构造（Differentiable Numerical Value Construction）**:
    - 功能：从离散的token预测分布中重建连续数值，使序列级损失可反向传播
    - 核心思路：用Gumbel-softmax近似argmax（低温度低噪声确保一致性），对预测的digit index按位值加权求和得到最终数值
    - 设计动机：直接对离散argmax求梯度不可行，Gumbel-softmax提供可微松弛

3. **双序列级偏差度量**:
    - 功能：从相对偏差和量级偏差两个角度惩罚整体数值误差
    - 核心思路：相对偏差 L_relative = |X-Y|/max(X,Y)+ε 提供归一化的比例误差；量级偏差 L_magnitude = log(max(X,Y)/min(X,Y)) 惩罚数量级差异
    - 设计动机：相对偏差处理同一数量级内的比例差异（如1 vs 10和1 vs 100的相对偏差相近），而量级偏差补充区分不同数量级的差异（log(10)=2.3 vs log(100)=4.6）

### 损失函数 / 训练策略

- 总损失 L = W_exp·EMD + α·L_relative + β·L_magnitude
- 仅对数值token应用NTIL，非数值token仍使用标准CE
- 超参数α和β为可调权重
- 可无缝集成到LLM（Baichuan2, Qwen2.5, LLaMA3, Yi, MiniCPM3）和MLLM（PaliGemma, LLaVA-1.5, Yi-VL, Qwen2-VL）的训练流程中

## 实验关键数据

### 主实验 - 图像定位 (Acc@0.5)

| 模型 | CE | EMD | NTIL (Ours) |
|---|---|---|---|
| PaliGemma (3b) | 0.785 | 0.789 | **0.795** |
| LLaVA-1.5 (7b) | 0.818 | 0.820 | **0.822** |
| Yi-VL (6b) | 0.733 | 0.740 | **0.744** |
| Qwen2-VL (2b) | 0.863 | 0.859 | **0.866** |
| Qwen2-VL (7b) | 0.860 | 0.855 | **0.862** |

### 文字检测 (Acc@0.5 avg)

| 模型 | CE | EMD | NTIL (Ours) |
|---|---|---|---|
| PaliGemma (3b) | 0.193 | 0.241 | **0.263** |
| Qwen2-VL (2b) | 0.720 | 0.718 | **0.732** |
| Qwen2-VL (7b) | 0.764 | 0.751 | **0.770** |
| LLaVA-1.5 (7b) | 0.675 | 0.690 | **0.698** |

### 时钟识别

| 模型 | CE Acc(%) | Ours Acc(%) | CE 时间偏差(min) | Ours 时间偏差(min) |
|---|---|---|---|---|
| LLaVA-1.5 (7b) | 95.1 | **98.3** | 8.52 | **4.14** |
| Yi-VL (6b) | 76.2 | **87.4** | 56.58 | **26.58** |
| Qwen2-VL (2b) | 81.3 | **85.3** | 32.34 | **24.66** |

### 消融实验

| Exp | Rel | Mag | PaliGemma-MathVista | LLaVA-MathVista | Yi-VL-Clock |
|---|---|---|---|---|---|
| ✗ | ✓ | ✓ | 0.137 | 0.166 | 0.834 |
| ✓ | ✗ | ✓ | 0.137 | 0.154 | 0.856 |
| ✓ | ✓ | ✗ | 0.142 | 0.143 | 0.876 |
| ✓ | ✓ | ✓ | **0.157** | **0.170** | **0.874** |

### 关键发现

- NTIL在5个MLLM和5个LLM上的4类任务中均一致优于CE和纯EMD
- 三个损失分量互补：去掉任一项都会在某些任务上降低性能
- 在小模型上提升更显著（如PaliGemma文字检测：0.193→0.263，提升36%）
- 数学推理任务提升相对温和，可能因为数值预测只是链条中一环
- 时钟识别任务上不仅准确率提升，时间偏差也大幅减小

## 亮点与洞察

- **首创性**：首次将EMD作为自回归模型的优化目标，首次提出跨多个time step的整体数值优化
- **位值系统的巧妙利用**：指数位置加权自然编码了十进制位值系统的重要性差异
- **通用性强**：方法与模型架构无关，可即插即用到任何LLM/MLLM
- **Gumbel-softmax桥接离散与连续**：巧妙解决了从离散token预测到连续数值重建的可微性问题
- **双度量互补**：相对偏差处理scale-invariant误差，量级偏差处理order-of-magnitude误差

## 局限与展望

- 仅处理十进制整数和简单浮点数，对更复杂的数值格式（科学计数法、分数等）的适用性未验证
- 序列级损失依赖Gumbel-softmax的温度参数，可能引入额外调参成本
- 在大模型和数学推理任务上提升较小，可能因为大模型本身数值预测能力已较强
- 未探索与其他解码策略（如beam search、采样）的交互效果
- 负数和特殊数值（0、NaN、Inf）的处理未讨论

## 相关工作与启发

- **vs Wasserstein GAN**：同样使用EMD，但WGAN用于生成器判别器训练稳定性，本文用于监督学习中的token级优化
- **vs 标准CE**：CE是分类损失，完全忽略类别间距离；NTIL通过EMD+序列级度量双重建模数值距离
- **vs 纯EMD**：实验显示纯EMD在部分任务上甚至劣于CE（如Qwen2-VL 7b定位），NTIL通过序列级约束修正了这一问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将EMD引入自回归数值预测 + 序列级可微数值构造，切入点新颖
- 实验充分度: ⭐⭐⭐⭐ 涵盖5个MLLM、5个LLM、4类任务，消融完整
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清晰（CE的两个局限很直观），方法描述条理分明
- 价值: ⭐⭐⭐⭐ 即插即用的通用改进方案，对所有涉及数值输出的自回归任务均有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Cautious Next Token Prediction](cautious_next_token_prediction.md)
- [\[ACL 2025\] On Support Samples of Next Word Prediction](on_support_samples_of_next_word_prediction.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](the_hidden_attention_of_mamba_models.md)
- [\[NeurIPS 2025\] Evolutionary Prediction Games](../../NeurIPS2025/others/evolutionary_prediction_games.md)
- [\[ACL 2025\] Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)

</div>

<!-- RELATED:END -->
