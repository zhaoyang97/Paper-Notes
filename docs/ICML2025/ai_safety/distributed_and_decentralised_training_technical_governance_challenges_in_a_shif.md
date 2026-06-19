---
title: >-
  [论文解读] Distributed and Decentralised Training: Technical Governance Challenges in a Shifting AI Landscape
description: >-
  [ICML 2025][AI安全][分布式训练] 本文系统区分了分布式训练（multi-data centre）与去中心化训练（community-driven）两种新兴范式，分析了低通信训练算法（如 DiLoCo）如何使这两种范式成为可能，并深入讨论了它们对AI技术治理（计算结构化、能力扩散、可关停性）带来的挑战与机遇。
tags:
  - "ICML 2025"
  - "AI安全"
  - "分布式训练"
  - "去中心化训练"
  - "计算治理"
  - "低通信算法"
  - "AI政策"
---

# Distributed and Decentralised Training: Technical Governance Challenges in a Shifting AI Landscape

**会议**: ICML 2025  
**arXiv**: [2507.07765](https://arxiv.org/abs/2507.07765)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 分布式训练, 去中心化训练, 计算治理, 低通信算法, AI政策

## 一句话总结

本文系统区分了分布式训练（multi-data centre）与去中心化训练（community-driven）两种新兴范式，分析了低通信训练算法（如 DiLoCo）如何使这两种范式成为可能，并深入讨论了它们对AI技术治理（计算结构化、能力扩散、可关停性）带来的挑战与机遇。

## 研究背景与动机

当前前沿LLM训练高度依赖集中式大规模数据中心，单次训练所需功率已超过100兆瓦，预计到2030年将增长到5吉瓦以上。这种集中化设置曾被视为AI治理的天然切入点——易于检测、集中在少数实体手中。

然而，近期算法进展正在打破这一格局：

**能源瓶颈推动分布式**：建设单一超大规模集群面临许可和能源供给瓶颈，超大规模企业开始采用多数据中心训练（如GPT-4.5、Gemini-1.5已跨数据中心训练）

**去中心化AI兴起**：自2024年初以来，Prime Intellect、Nous Research等初创公司已累计融资约1.45亿美元，目标是通过社区贡献算力训练与前沿模型竞争的开源模型

**政策讨论滞后**：policy discourse中"distributed"和"decentralised"经常被混用，缺乏精确定义，严重阻碍了有效的治理讨论

本文的核心动机就是：为AI政策社区提供一个清晰的概念框架，帮助理解这两种训练范式的技术基础、区别和治理含义。

## 方法详解

### 整体框架

本文提出一个二维分析框架（见论文 Figure 1），按**参与方数量**和**计算位置数量**两个维度将训练模式分为四象限：

| 象限 | 参与方 | 计算位置 | 代表案例 |
|------|--------|---------|---------|
| 集中式 | 单一 | 单一 | Grok 3（xAI） |
| 分布式 | 单一 | 多地 | GPT-4.5（OpenAI） |
| 去中心化 | 多方 | 多地 | INTELLECT-1 |
| 假设场景 | 多方 | 单一 | 尚无实例 |

**核心术语定义**：

- **分布式训练（Distributed Training）**：跨多个物理上分散的计算池训练，但有中心化实体协调。更准确的名称可以是"多数据中心训练"或"地理分布式训练"
- **去中心化训练（Decentralised Training）**：利用社区贡献的计算资源，无中心化协调实体

### 关键设计

#### 1. 低通信数据并行训练算法

传统数据并行要求每个训练batch后同步梯度，通信开销巨大。以互联网连接为例：若单batch训练4秒、同步需60秒，GPU利用率仅6.25%。

**DiLoCo（Distributed Low Communication）及后续算法**的核心突破：

- **双优化器架构**：本地优化器（如SGD变体）处理本地更新，全局优化器在周期性同步后调整全局参数
- **同步频率降低500倍**：从每step同步降低到每500 steps同步一次
- 同样以互联网训练为例，每500步（~33分钟）同步一次，GPU利用率从6.25%提升到**97%以上**

关键算法进展汇总：

| 算法 | 核心机制 | 通信降低 | 关键特性 |
|------|---------|---------|---------|
| DiLoCo | 双优化器 + 稀疏同步 | ~500× | 异步鲁棒、数据异构鲁棒 |
| Async Local-SGD | 异步本地SGD | 显著 | 适应不稳定连接 |
| DeMo | 解耦动量优化 | 显著 | 减少传输数据量 |
| Streaming DiLoCo | 重叠通信与计算 | ~500× | 接近"免费午餐" |
| Eager Updates | 急切更新策略 | ~500× | 进一步优化DiLoCo |

这些算法的重要附加特性：

- **异步通信鲁棒性**：对部分副本无法参与同步的情况具有鲁棒性，适合不稳定的网络连接
- **数据分布异构鲁棒性**：当不同worker训练来自不同分布的数据时，性能不会退化——这是解锁私有数据的关键
- **随模型规模提升的效果**：Charles et al. (2025) 发现在某些设置下DiLoCo甚至优于传统数据并行训练，且优势随模型规模增大

#### 2. 点对点（P2P）通信架构

去中心化训练进一步消除了中心化瓶颈：

- **无中心同步节点**：模型同步通过节点间直接通信完成（peer-to-peer），而非通过中心参数服务器
- **无需信任的参与**：新节点可通过密码学证明无需中心授权即可加入训练（trustless participation）
- **容错与动态重组**：单GPU故障不会中断训练，网络自动重组并接纳替代GPU
- **架构更接近区块链**：如Ethereum的去中心化结构，而非传统数据并行

Prime Intellect 和 Nous Research 已成功在不同大洲的GPU上、以普通互联网速度预训练了100亿参数规模的模型（INTELLECT-1）。

#### 3. 推理模型与去中心化的天然契合

论文深入分析了为何推理模型（reasoning models）特别适合去中心化环境：

- **传统SSL训练**：前向与反向传播比例 1:1，每次前向传播都需要权重更新和同步
- **RL后训练**：前向与反向传播比例可达 **1000:1**，模型探索大量"思考轨迹"后才进行一次权重更新
- **硬件门槛降低**：生成思考轨迹的计算与内存需求低于完整反向传播，消费级硬件（如Apple M3 Ultra，512GB内存）即可运行DeepSeek-R1（671B参数）的推理
- **通信需求进一步降低**：只需传输思考轨迹和奖励值，无需传输完整梯度

### 损失函数 / 训练策略

本文为治理分析论文，未提出新的损失函数。但对训练策略的分析表明：

- DiLoCo使用**外部优化器**（outer optimizer）对全局参数进行周期性校正，保证在稀疏同步条件下模型收敛的稳定性
- 在去中心化场景中，各节点独立运行本地优化器，通过密码学验证（如TopLoc局部敏感哈希方案）确保计算贡献的可信性
- 后训练阶段RL的GRPO等策略天然适配低带宽环境，因为生成阶段是"仅推理"操作

## 实验关键数据

### 主实验

本文为position paper，不含传统实验。核心实证证据来自已有工作：

| 训练范式 | 模型 | 规模 | 通信条件 | 结论 |
|---------|------|------|---------|------|
| 去中心化预训练 | INTELLECT-1 | 10B | 跨大洲/互联网 | 可行性验证，性能尚未追平同规模模型 |
| 分布式训练 | GPT-4.5 | 未公开 | 多数据中心 | 前沿模型已采用 |
| 分布式训练 | Gemini-1.5 | 未公开 | 多数据中心 | 前沿模型已采用 |
| 先例：分布式计算 | Folding@Home | 28万GPU | 互联网 | 峰值算力突破 10^18 FLOPS |

### 消融实验

通信频率对GPU利用率的影响分析：

| 同步频率 | 连接方式 | GPU利用率 | 说明 |
|---------|---------|---------|------|
| 每1步（传统） | 高速互联/单集群 | ~80% | 传统数据并行，通信开销~20% |
| 每1步 | 互联网 | **6.25%** | 跨地域训练完全不可行 |
| 每500步（DiLoCo） | 互联网 | **>97%** | 低通信算法使跨地域训练可行 |

### 关键发现

1. **低通信算法的规模效应**：DiLoCo及后续方法的效果随模型规模增大而提升，暗示可跟上scaling paradigm
2. **去中心化初创生态爆发**：2024年初以来至少6家初创融资约1.45亿美元，目标是训练o3级别模型
3. **INTELLECT-1同步间隔**：每38分钟同步一次，且可刻意延长以混淆通信模式
4. **后训练更适合去中心化**：RL后训练的高前向/反向比自然降低通信需求

## 亮点与洞察

1. **概念澄清价值极高**：明确区分distributed vs. decentralised，为政策讨论奠定基础。此前两个术语在政策文献中频繁混用，导致治理措施缺乏精确指向
2. **双刃剑视角平衡**：既分析了治理挑战（计算结构化规避监管、能力扩散、不可关停），也承认了去中心化的正面价值（隐私保护训练释放更多数据、缓解权力集中）
3. **推理模型分析独到**：深刻洞察了reasoning model的RL后训练与去中心化训练的天然契合性，这一分析在现有文献中较为少见
4. **"marginal risk"框架**：引用Kapoor et al.的边际风险分析框架，主张如果去中心化AI显著落后于前沿且防御技术足够，政府干预未必必要

## 局限与展望

1. **缺乏定量分析**：对去中心化训练池可汇集的算力总量未给出具体估算，仅引用Folding@Home的粗略类比
2. **安全性讨论较浅**：对去中心化训练中模型安全对齐如何执行（如RLHF中human feedback的去中心化采集）缺乏讨论
3. **技术深度有限**：对DiLoCo等算法的分析停留在概述层面，未涉及收敛性理论保证、通信压缩的具体方案等技术细节
4. **博弈论视角缺失**：未分析超大规模企业与去中心化社区之间的战略互动——前者是否会主动采取措施抑制后者？
5. **国际治理维度不足**：跨国去中心化训练的管辖权冲突、国际协作机制等议题着墨较少
6. **实证案例有限**：INTELLECT-1和INTELLECT-2是目前仅有的去中心化大模型案例，且性能存疑，结论的外推性有待验证

## 相关工作与启发

- **DiLoCo系列** (Douillard et al., 2023; 2025)：低通信训练的算法基石，使跨地域训练成为可能
- **Sastry et al., 2024**：计算治理的理论框架，本文在此基础上讨论新范式对治理假设的冲击
- **Kapoor et al., 2024**：边际风险框架，为评估去中心化AI的风险-收益提供了方法论工具
- **Seferis & Fist, 2025**：计算结构化检测的具体技术方案
- **Hivemind** (Ryabinin et al., 2020)：PyTorch去中心化深度学习框架，INTELLECT-1的技术基础之一
- **联邦学习与隐私保护** (Sani et al., 2024)：去中心化预训练的数据隐私方向

**对后续研究的启发**：去中心化训练的治理挑战可能催生新的"链上AI治理"范式（结合区块链和AI安全），这一交叉方向值得关注。

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ★★★★☆ | 首次系统性地为AI政策社区明确两种训练范式的区别 |
| 技术深度 | ★★★☆☆ | 概述性质，算法细节欠缺，但政策分析到位 |
| 实用性 | ★★★★☆ | 对政策制定者和治理研究者直接有用 |
| 清晰度 | ★★★★★ | 写作优秀，概念定义清晰，论证结构严谨 |
| 综合评分 | ★★★★☆ | 高质量治理分析论文，填补了AI policy的重要空白 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI](../../NeurIPS2025/ai_safety/understanding_challenges_to_the_interpretation_of_disaggregated_evaluations_of_a.md)
- [\[ICML 2026\] Position: 'AI Alignment' Encompasses Competing Technical Priorities](../../ICML2026/ai_safety/ai_alignment_encompasses_competing_technical_priorities.md)
- [\[NeurIPS 2025\] Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification](../../NeurIPS2025/ai_safety/keep_it_real_challenges_in_attacking_compression-based_adversarial_purification.md)
- [\[ICML 2025\] Identifying and Understanding Cross-Class Features in Adversarial Training](identifying_and_understanding_cross-class_features_in_adversarial_training.md)
- [\[ECCV 2024\] Resilience of Entropy Model in Distributed Neural Networks](../../ECCV2024/ai_safety/resilience_of_entropy_model_in_distributed_neural_networks.md)

</div>

<!-- RELATED:END -->
