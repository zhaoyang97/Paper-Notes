# Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients

**会议**: ICLR2026  
**arXiv**: [2506.11024](https://arxiv.org/abs/2506.11024)  
**代码**: [https://github.com/snumprlab/fedmosaic](https://github.com/snumprlab/fedmosaic)  
**领域**: 联邦学习  
**关键词**: personalized federated learning, LoRA, model heterogeneity, multimodal, knowledge sharing

## 一句话总结
提出 FedMosaic 框架解决个性化联邦学习中的双重异构问题：RELA 通过梯度相似度度量任务相关性实现定制化聚合（解决数据异构），Co-LoRA 通过维度不变的 $P \in \mathbb{R}^{r \times r}, Q \in \mathbb{R}^r$ 模块实现跨异构架构（如 Llama vs Qwen）的知识共享（解决模型异构），在新提出的 40 任务多模态 PFL benchmark DRAKE 上大幅超越 SOTA。

## 研究背景与动机

1. **领域现状**：个性化联邦学习（PFL）让各客户端协作学习但保留隐私。现有 PFL 方法如 DITTO、FedDAT 通过双 adapter（本地+全局）设计处理数据异构，但假设所有客户端使用相同模型架构。
2. **现有痛点**：(a) **数据异构**——现有 benchmark 用同一数据集的 non-IID 切分模拟异构，不真实（真实世界中客户端做不同任务）；(b) **模型异构**——不同客户端有不同硬件，使用不同模型族（Llama vs Qwen）和规模（1B vs 3B），LoRA 矩阵维度不同无法直接平均；(c) 现有处理模型异构的方法（HETLoRA、FlexLoRA）假设基础架构相同只是 rank 不同，不支持真正异构的架构。
3. **核心矛盾**：LoRA 的 $A \in \mathbb{R}^{r \times d_I}$ 和 $B \in \mathbb{R}^{d_O \times r}$ 矩阵依赖模型特定的隐藏维度 $d_I, d_O$，不同架构维度不同→无法聚合。同时朴素平均不同任务的模型会产生参数干扰。
4. **本文要解决什么？** 同时处理数据异构（客户端做不同任务）和模型异构（客户端用不同架构/规模），在真实多模态 PFL 场景中实现有效协作。
5. **切入角度**：在 LoRA 中间插入仅依赖 rank $r$ 的维度不变模块 $P, Q$——只聚合这些模块就能跨架构传递知识。
6. **核心idea一句话**：用维度不变的 Co-LoRA 模块实现跨架构知识共享 + 用梯度相似度驱动的相关性聚合减少任务干扰。

## 方法详解

### 整体框架
每轮通信：各客户端本地用 Co-LoRA 微调 → 上传 $(P_i, Q_i)$ 模块和消毒梯度 $\tilde{g}_i$ → 服务端用 RELA 计算客户端间任务相关性 → 为每个客户端构建定制化全局 Co-LoRA $G_i$ → 下发回客户端冻结使用。推理时本地 LoRA 和全局 Co-LoRA 通过可学习门控 $\beta$ 自适应融合。

### 关键设计

1. **RELA（Relevance-Guided Aggregation）**:
   - 做什么：基于任务相关性为每个客户端定制全局模型，而非朴素平均
   - 核心思路：每个客户端从小预训练模型提取最后一层梯度 $g_i$，EMA 更新反映分布漂移→添加噪声+维度采样消毒→服务端计算余弦相似度矩阵 $S_{ij} = \cos(\tilde{g}_i, \tilde{g}_j)$→softmax 加权聚合：$G_i = \sum_j w_{ij} L_j$
   - 设计动机：朴素平均不相关任务的模型会产生参数干扰。只与相关客户端共享知识（类似任务的模型冲突更少）。EMA 梯度反映当前知识状态（考虑遗忘），比累计梯度更准

2. **Co-LoRA（Collaborative LoRA）**:
   - 做什么：在 LoRA 的 $A, B$ 之间插入仅依赖 rank 的模块，实现跨架构知识共享
   - 核心思路：$h_O = W_p h_I + B(PA h_I + Q)$，其中 $P \in \mathbb{R}^{r \times r}$, $Q \in \mathbb{R}^r$ 的维度只与 rank $r$ 有关→不同架构的 $P, Q$ 可以直接聚合。训练时冻结 $A, B$（保持对齐），只更新 $P, Q$
   - **块对齐（Block-wise Aggregation）**：CKA 分析发现不同深度模型的层在相对位置上对齐——将模型按相对深度分 $N_B$ 块，每块最后一层附 Co-LoRA，对应块之间聚合
   - **权重对齐（Weight Alignment）**：$A$ 矩阵用 L2 loss 在公共数据上对齐 $r$ 维表征；$B$ 矩阵用 CCA 找最大相关投影空间→投影到共享空间再反投影回来。正交约束保证表达能力最大化（Theorem 1：span 为 $r^2$ 维）
   - 设计动机：比联邦蒸馏（需要公共数据 logits）更轻量、更隐私安全。比 HETLoRA（只处理 rank 不同）更通用

3. **门控融合**:
   - 做什么：自适应平衡本地个性化知识和全局共享知识
   - 公式：$h_O = W_p h_I + (1-\tilde{\beta})h_L + \tilde{\beta}h_G$，$\tilde{\beta} = \sigma(\beta)$ 可学习
   - 设计动机：不同层、不同任务需要不同程度的全局知识——可学习门控自动调节

### 损失函数 / 训练策略
- A/B 对齐只在联邦训练前做一次（一次性开销）
- 通信成本低：只传 $P \in \mathbb{R}^{r \times r}$ 和 $Q \in \mathbb{R}^r$（比完整 LoRA 小很多）
- 隐私保护：梯度 EMA + 高斯噪声 + 随机维度采样

## 实验关键数据

### 主实验（DRAKE Benchmark，40 任务）

| 方法 | 同构设置 (Avg Acc) | 异构设置 (Avg Acc) | 说明 |
|------|-------------------|-------------------|------|
| Local only | 基线 | 基线 | 无联邦协作 |
| FedAvg | 低于 Local | 不适用 | 朴素平均有害 |
| DITTO | 中等 | 不适用 | 双 adapter 但不支持异构 |
| FedDAT | 中等偏上 | 不适用 | 同上 |
| HETLoRA | - | 中等 | 只处理 rank 异构 |
| **FedMosaic** | **最优** | **最优** | 显著超越所有方法 |

### 消融实验

| 配置 | Acc 变化 | 说明 |
|------|---------|------|
| Full FedMosaic | 最优 | 完整方法 |
| w/o RELA（用 FedAvg） | 下降 | 任务干扰 |
| w/o Co-LoRA（用 HETLoRA） | 下降 | 架构异构处理不足 |
| w/o 块对齐 | 下降 | 层对应错误 |
| w/o 权重对齐 | 下降 | 优化轨迹不一致 |
| w/o 门控 $\beta$ | 下降 | 全局/本地知识固定比例不佳 |

### 关键发现
- **FedAvg 在真实异构场景中比不协作还差**：朴素平均不相关任务的模型产生严重参数干扰
- **RELA 的选择性聚合至关重要**：只与相关客户端协作显著优于全局平均
- **Co-LoRA 成功跨架构传递知识**：Llama-1B ↔ Llama-3B 和 Llama-1B ↔ Qwen-3B 都能有效协作
- **CKA 验证层对齐假设**：不同深度模型的相对位置层有最高表征相似度，支持块对齐策略
- **通信开销极低**：只传 $P(r^2)$ 和 $Q(r)$ 参数 + 消毒梯度

## 亮点与洞察
- **维度不变模块是跨架构联邦学习的优雅解决方案**：$P \in \mathbb{R}^{r \times r}$ 只依赖 rank 不依赖隐藏维度——这个设计思想可以推广到任何需要跨架构知识传递的场景
- **梯度消毒三件套**：EMA 混合 + 高斯噪声 + 随机维度采样——每一步都有隐私理论支撑，整体方案实用且安全
- **DRAKE benchmark 填补多模态 PFL 评估空白**：40 个不同任务 + 分布漂移 + 多图像输入——比之前 non-IID MNIST 的评估真实得多
- **Theorem 1 的正交约束保证**：冻结正交 A/B 下，Co-LoRA 的权重更新空间维度为 $r^2$（最大可能）——从理论上保证了表达能力

## 局限性 / 可改进方向
- **DRAKE 任务数 40 偏少**：真实 agentic AI 场景可能有数百个任务类型，scalability 需验证
- **公共数据需求**：A/B 对齐需要公共数据（虽然很少量），在极端隐私场景下可能不可接受
- **只测试了 1B/3B 规模**：7B/13B 以上规模是否有效、通信成本是否仍然可控？
- **rank r 需要所有客户端统一**：如果客户端需要不同 rank 的 LoRA，Co-LoRA 目前不支持

## 相关工作与启发
- **vs HETLoRA/FlexLoRA**：它们只处理同一架构不同 rank 的异构，本质上假设 $d_I, d_O$ 相同。Co-LoRA 处理真正的架构异构（不同模型族、不同深度、不同维度）
- **vs FedMD/FedMKT**：联邦蒸馏通过公共数据 logits 传递知识，但 logit 提取对大模型计算昂贵、且有梯度反演隐私风险。Co-LoRA 只传小矩阵，更安全更高效
- **vs DITTO**：DITTO 保留本地/全局双 adapter，但全局用朴素平均。FedMosaic 用 RELA 做任务感知聚合 + Co-LoRA 支持异构

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 维度不变模块、梯度相关性聚合、块对齐/权重对齐的组合设计系统且新颖
- 实验充分度: ⭐⭐⭐⭐ 40 任务 benchmark 全面，消融充分，但缺少更大规模模型实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法推导严谨，但篇幅很长
- 价值: ⭐⭐⭐⭐⭐ 为异构联邦学习提供了可行的实用方案，DRAKE benchmark 有持续价值
