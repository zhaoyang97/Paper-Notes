# Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits

**会议**: NeurIPS 2025  
**arXiv**: [2511.20273](https://arxiv.org/abs/2511.20273)  
**代码**: [GitHub](https://github.com/Exploration-Lab/Beyond-Components)  
**领域**: LLM可解释性 / Transformer电路分析  
**关键词**: SVD interpretability, transformer circuits, singular vectors, mechanistic interpretability, directional masking

## 一句话总结
提出基于SVD奇异向量的方向级可解释性框架，通过对注意力头和MLP的增广矩阵统一SVD分解+可学习对角掩码（KL+L₁），发现单组件内存在正交低秩子函数叠加——IOI任务仅需~9%方向即可KLD=0.21复现模型行为。

## 研究背景与动机
1. **领域现状**: 机械可解释性将注意力头和MLP视为原子单元，用因果追踪/激活修补等方法对整个组件探测消融
2. **现有痛点**: 组件级视角隐含"功能=组件"假设，但单个头或MLP可能通过叠加复用多个子函数
3. **核心矛盾**: merullo等分析头间通信的低秩通道，但未深入单个组件内部的功能分解
4. **本文要解决**: 统一注意力(QK/OV)和MLP的线性表示，揭示组件内叠加的正交子函数，实现方向级归因
5. **切入角度**: 将bias折入权重矩阵得增广矩阵→SVD→奇异方向作为正交计算方向
6. **核心idea**: Transformer计算是分布式组合式的，重叠子函数嵌入共享子空间，可通过SVD方向独立操控

## 方法详解

### 整体框架
对每个组件构建增广矩阵（bias折入权重）→ SVD分解为正交方向 → 可学习对角掩码识别任务关键方向 → 方向级归因和干预

### 关键设计

1. **统一增广矩阵**: 将bias折入权重使QK/OV/MLP在同一框架下SVD可比。如QK交互: $[1,\mathbf{x}_i]\mathbf{W}_{aug}^{(QK)}[1,\mathbf{x}_j]^\top = \mathbf{q}_i\cdot\mathbf{k}_j^\top$
2. **SVD方向分解**: $\mathbf{W}_{aug} = \sum_k \sigma_k \mathbf{u}_k \mathbf{v}_k^\top$，每个奇异方向编码独立子函数
3. **可学习对角掩码**: $\Lambda = \text{diag}(\lambda_1,...,\lambda_R)$，优化 $\min KL(p_{orig}\|p_{masked}) + \alpha\|\Lambda\|_1$ 自动发现最小必要方向
4. **Logit Receptors**: 自然产生的logit空间方向，标量缩放即可控制模型预测

## 实验关键数据

### 方向级稀疏性（GPT-2 Small）
| 任务 | 保留方向 | KL散度 | 说明 |
|------|---------|--------|------|
| IOI | ~9% | 0.21 | 91%方向可丢弃 |
| GP | 稀疏 | 低 | 性别方向可独立操控 |
| GT | 稀疏 | 低 | 数值比较方向对齐 |

### 关键发现
| 发现 | 说明 |
|------|------|
| 组件内多功能复用 | Head 9.6不同方向分别编码实体分离/显著性/初始化 |
| 电路头方向激活更强 | IOI电路头的掩码权重显著高于非电路头 |
| Logit Receptors可控 | 标量干预即可切换性别预测 |
| MLP同样适用 | MLP层也展现方向级功能分解 |

## 亮点与洞察
- **范式转变**: 从"组件=功能"到"方向=功能"，未来可解释性应在奇异方向层面归因
- **增广矩阵统一框架**: bias折入使注意力和MLP在同一框架下可比
- **Logit Receptors**: 为模型编辑和控制提供新工具

## 局限性 / 可改进方向
- 仅GPT-2 Small验证，大模型scalability未知
- SVD线性假设忽略非线性激活影响
- 部分方向功能难以语言描述
- 仅IOI/GP/GT三个任务

## 相关工作与启发
- **vs ACDC电路发现**: 标准方法组件级消融，本文方向级——粒度更细
- **vs SAE**: SAE需训练自编码器，SVD直接分解更轻量
- **启发**: 方向级可控性为精确模型编辑提供新途径

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SVD方向级可解释性是全新视角
- 实验充分度: ⭐⭐⭐ 三个任务充分但仅GPT-2 Small
- 写作质量: ⭐⭐⭐⭐⭐ 统一线性框架设计精彩
- 价值: ⭐⭐⭐⭐ 对机械可解释性有重要启发
**领域**: LLM可解释性 / Transformer电路分析  
**关键词**: SVD interpretability, transformer circuits, singular vectors, mechanistic interpretability, directional masking, low-rank subfunctions

## 一句话总结
提出基于奇异向量的方向级可解释性框架，通过 SVD 分解 Transformer 注意力头与 MLP 的增广矩阵，配合可学习对角掩码（KL+L₁）优化，发现组件内部存在正交的低秩子函数——IOI 任务仅保留 ~9% 方向即可 KLD=0.21 复现模型行为，且 Head 9.6 内部沿不同奇异方向分别编码语义实体分离、实体显著性和序列初始化等独立计算原语。

## 研究背景与动机
1. **领域现状**：机械可解释性（mechanistic interpretability）通常将注意力头和 MLP 层视为不可分割的原子单元，使用因果追踪（causal tracing）、激活修补（activation patching）、归因分析（attribution）等方法对整个组件进行探测或消融（ablation）
2. **核心问题**：这种组件级视角隐含假设"功能与组件边界一一对应"，但实际上单个头或 MLP 可能通过叠加（superposition）方式复用（multiplex）多个子函数，组件主义掩盖了内部的细粒度计算结构
3. **前人工作局限**：merullo et al. 提出低秩视角分析头间通信（inter-component communication），展示注意力头通过 value 矩阵奇异方向在残差流中通信，但未深入单个组件内部的功能分解（intra-component decomposition）
4. **本文切入点**：将 SVD 奇异向量作为正交的"计算方向"，统一处理注意力的 QK/OV 变换和 MLP 的 in/out 投影，揭示组件内部叠加的独立子函数，并通过可学习掩码进行方向级归因

## 方法详解

### 关键设计
1. **SVD 分解**：对注意头/MLP 的增广权重矩阵进行 SVD
2. **可学习对角掩码**：KL+L₁正则化优化，识别任务关键的正交方向

## 实验关键数据
- IOI 任务: 稀疏度 **91.32%** 仅 KLD=0.21
- GT/GP 任务验证通用性
- “name mover”头跨多个奇异向量编码重叠子函数

## 亮点与洞察
- 细粒度方向级解释超越组件主义的假设

## 局限性 / 可改进方向
- 仅在 GPT-2 Small 上验证，缺乏大规模模型评估

## 评分
- 新颖性: ⭐⭐⭐⭐ SVD 方向级可解释性新颖
- 实验充分度: ⭐⭐⭐ 小模型验证充分但规模有限
- 写作质量: ⭐⭐⭐⭐⭐ 统一线性框架设计精彩
- 价值: ⭐⭐⭐⭐ 对机械可解释性研究有重要启发

**会议**: NEURIPS2025  
**arXiv**: [2511.20273](https://arxiv.org/abs/2511.20273)
