# Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection

**会议**: ACL 2025  
**arXiv**: [2506.00488](https://arxiv.org/abs/2506.00488)  
**代码**: [https://github.com/TSCenter/GLPN-LLM](https://github.com/TSCenter/GLPN-LLM)  
**领域**: NLP 理解  
**关键词**: Fake News Detection, Label Propagation, LLM Pseudo Labels, Multimodal Learning, Graph Neural Networks  

## 一句话总结

提出 GLPN-LLM 框架，通过 mask-based 全局标签传播机制有效整合 LLM 生成的伪标签，解决了 LLM 伪标签直接组合效果不佳的问题，在 Twitter/PHEME/Weibo 三个数据集上全面超越 SOTA。

## 研究背景与动机

### 多模态假新闻检测

社交媒体上假新闻的传播已成为严重社会问题，需要同时分析文本和图片的多模态检测方法。

### LLM 在假新闻检测中的困境

一个关键观察：**LLM（如 GPT-4o）单独用于假新闻检测的效果不如传统多模态检测模型**。例如在 Twitter 数据集上，LLM 的 F1 仅为 78.20，而 HMCAN 达到 82.57。直接将 LLM 预测与现有模型输出简单组合的效果也有限。

### 核心问题

如何有效地将 LLM 的能力整合到假新闻检测系统中？简单的直接组合不行，需要更巧妙的方法。

### 解决思路

Label Propagation（标签传播）技术**即使在伪标签准确率一般的情况下也能保持有效**（Sun et al., 2025），因此特别适合整合 LLM 生成的不完美伪标签。

## 方法详解

### 整体框架

GLPN-LLM 由三个核心模块组成：

1. **Multimodal Feature Extraction**：使用 CLIP 提取图文特征
2. **LLM-based Pseudo Label Generation**：GPT-4o 生成伪标签
3. **Global Label Propagation with Mask**：mask-based 全局标签传播

### 关键设计

#### 1. 多模态特征提取

使用 CLIP 的双编码器分别提取视觉特征 $v_i \in \mathbb{R}^{d_v}$ 和文本特征 $t_i \in \mathbb{R}^{d_t}$，拼接得到统一表示：
$$x_i = t_i \oplus v_i$$

#### 2. Cross-Modal Graph 构建

每个新闻项作为图的一个节点，基于五种相似度度量建边：
- **拼接特征相似度**：图文嵌入拼接后的余弦相似度
- **图-文交叉相似度**（双向）
- **图-图相似度**
- **文-文相似度**

当任一相似度超过阈值 $\theta = 0.95$ 时建立边，确保只连接强相关的新闻项。

#### 3. Mixed-Initiative Labeling（LLM 伪标签生成）

构造结构化 prompt 输入 LLM：
- 输入：`[cls] <prompt> [SEP] <cleaned Twitter text>`
- 输出：`[detection] ŷ [confidence] c`

LLM 输出两个信息：
1. **Detection Label ŷ**：true 或 fake
2. **Confidence Score c**：预测置信度

基于置信度过滤：只有高置信度伪标签才被选用。

#### 4. Label Integration 与 Global Random Mask (GRM)

**标签整合**：将标签信息（真实标签 / 高置信伪标签 / 零向量）作为 one-hot 编码拼接到节点特征中：

$$x_i' = x_i \oplus y_i'$$

分三种情况：
- 有标签的训练节点：使用真实标签
- 高置信度无标签节点：使用 LLM 伪标签
- 其他：零向量

**Global Random Mask 机制**（核心创新）：

训练时按 mask ratio $\rho$（默认 0.3）随机选择 $\rho \times N$ 个节点，将其标签嵌入替换为零向量：
$$y_i' = \tilde{y}_i \cdot m_i, \quad m_i \in \{0, 1\}$$

**为什么需要 GRM？** 防止标签泄露——如果节点的标签包含在其输入特征中，模型会直接利用该信息做预测，而不真正学习图结构和内容特征。GRM 确保被 mask 的节点只能通过邻居的标签传播获得标签信息。

训练时只对被 mask 节点计算损失；推理时使用所有可用标签信息。

#### 5. GCN 分类

将带标签信息的节点特征 $x_i'$ 输入 GCN 进行标签传播和分类，使用交叉熵损失 + Adam 优化器。

### 损失函数 / 训练策略

- 使用交叉熵损失，只对被 mask 的节点计算
- 每个 epoch 随机选择不同的 mask 子集
- Adam 优化器
- 推理时不做 mask，使用完整标签信息

## 实验关键数据

### 主实验

**三个 benchmark 数据集的 F1 结果**：

| 方法 | Twitter F1 | PHEME F1 | Weibo F1 |
|------|-----------|---------|---------|
| LLM (GPT-4o) | 78.20 | 76.87 | 81.75 |
| HMCAN | 82.57 | 83.49 | 87.20 |
| FCN-LP (CLIP) | 85.24 | 87.97 | 89.78 |
| FCN-LP (CLIP) + LLM | 85.97 | 89.21 | 89.85 |
| **GLPN-LLM (CLIP)** | **89.03** | **90.66** | **91.52** |

**GLPN-LLM (HMCAN) 结果**：

| 方法 | Twitter F1 | PHEME F1 | Weibo F1 |
|------|-----------|---------|---------|
| FCN-LP (HMCAN) | 84.04 | 84.50 | 88.11 |
| FCN-LP (HMCAN) + LLM | 85.10 | 84.60 | 88.75 |
| **GLPN-LLM (HMCAN)** | **86.86** | **86.87** | **91.46** |

**消融实验**（以 CLIP 为例）：

| 方法 | Twitter F1 | PHEME F1 | Weibo F1 |
|------|-----------|---------|---------|
| FCN-LP | 85.24 | 87.97 | 89.78 |
| GLPN (无 LLM) | 86.30 | 86.96 | 90.76 |
| GLPN-LLM | **89.03** | **90.66** | **91.52** |

### 关键发现

1. **LLM 单独表现显著弱于传统方法**：GPT-4o 的 F1 在 Twitter 上仅 78.20，而 HMCAN 达 82.57
2. **简单组合 LLM 效果有限**：FCN-LP + LLM 相比 FCN-LP 仅提升约 0.5-1.2%
3. **GLPN-LLM 大幅提升**：相比 FCN-LP + LLM，在 Twitter 上提升 3.06%，PHEME 上提升 1.45%，Weibo 上提升 1.67%
4. **GRM 模块是关键**：GLPN（有 GRM 无 LLM）vs FCN-LP 已有提升，说明全局标签传播机制本身有效
5. **LLM 伪标签通过传播产生价值**：GLPN-LLM vs GLPN 提升 2-4%，说明 LLM 伪标签在传播机制下被有效利用
6. **mask rate 的影响**：标签传播通过邻居标签而非自身标签提供信息，mask rate 过低导致标签泄露，过高导致信息不足

## 亮点与洞察

1. **精准定位问题**：清晰揭示了 LLM 在假新闻检测中"有用但不好用"的困境，并给出了优雅的解决方案
2. **Label Propagation 的巧妙应用**：LP 对噪声标签的鲁棒性正好匹配 LLM 伪标签的特性
3. **GRM 的设计直觉清晰**：类似于 masked language model 的思想——遮住标签让模型通过上下文（邻居）学习
4. **跨模态图构建的多相似度融合**：5 种相似度度量确保捕捉丰富的跨模态关系
5. **实验验证了一个重要直觉**：LLM 的价值不在于单点预测的准确性，而在于通过传播机制放大其整体判断能力

## 局限性 / 可改进方向

1. **LLM 只使用文本信息**：当前 GPT-4o prompt 只包含清理后的推文文本，未利用图片信息（可改用 GPT-4V 多模态分析）
2. **图构建阈值 $\theta = 0.95$ 较高**：可能遗漏一些有价值的弱关联
3. **仅在英语和中文数据集验证**：其他语言场景未探索
4. **LLM API 成本**：需要为每个样本调用 GPT-4o，大规模部署成本较高
5. **GCN 层数和架构较简单**：可尝试更高级的 GNN 架构（如 GAT、GraphSAGE）
6. **未讨论对抗样本鲁棒性**：对精心设计的对抗性假新闻的检测能力未知

## 相关工作与启发

- **Label Propagation 的复兴**：传统的半监督方法（Zhu & Ghahramani, 2002）在 GNN 时代焕发新生，结合 LLM 伪标签开辟了新方向
- **FCN-LP (Zhao et al., 2023)**：直接前作，本文的全局 mask 机制是其重要改进
- **LLM 能力的间接利用范式**：不直接用 LLM 做预测，而是将其输出作为辅助信号融入传统模型——这一范式可推广到其他任务
- **与 LACA 的思路对比**：LACA 用 LLM 生成数据，GLPN-LLM 用 LLM 生成伪标签并通过图传播利用，两者都在解决 LLM 单独效果不够好时如何利用 LLM 的问题

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-------------|------|
| 创新性 | 7 | GRM 机制新颖，但整体框架基于已有组件的组合 |
| 实验充分性 | 8 | 三个数据集、全面消融、参数敏感性分析 |
| 写作质量 | 7 | 结构清晰但部分公式推导较繁琐 |
| 实用价值 | 7 | 依赖 LLM API，部署成本较高 |
| 总分 | 7 | 有效利用 LLM 伪标签的实用方法，核心贡献在 GRM 机制 |
