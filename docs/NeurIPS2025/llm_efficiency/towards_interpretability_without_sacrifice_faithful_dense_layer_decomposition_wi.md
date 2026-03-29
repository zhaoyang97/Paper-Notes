# Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders

**会议**: NeurIPS 2025  
**arXiv**: [2505.21364](https://arxiv.org/abs/2505.21364)  
**代码**: [GitHub](https://github.com/james-oldfield/MxD/)  
**领域**: LLM效率 / 可解释性  
**关键词**: mechanistic interpretability, sparse approximation, mixture of experts, tensor factorization, MLP decomposition

## 一句话总结
提出 Mixture of Decoders (MxD)，将 LLM 的 MLP 层分解为数万个稀疏激活的专家子层（layer-level sparsity），每个专家通过 Hadamard 乘积张量分解实现满秩线性变换，在稀疏性-准确性权衡上显著优于 Transcoders，同时保持可解释性。

## 研究背景与动机
1. **领域现状**：LLM 的 MLP 层表示是密集的——单个神经元编码多个概念，难以隔离特定特征。SAE 和 Transcoder 通过学习稀疏的过完备基来近似 MLP 层输出，使特征更可解释。
2. **现有痛点**：目前方法都采用**神经元级稀疏性**（限制隐层中非零元素数量），但这导致严重的**稀疏性-准确性权衡**——稀疏度越高，对原始 MLP 映射的重建误差越大。不忠实的重建意味着可能遗漏关键行为，且无法直接替换原始层做推理。
3. **核心矛盾**：可解释性要求高稀疏度，但高稀疏度的神经元级方法只能使用输出空间的低维子空间（$K$ 个非零隐层单元 → $K$ 维子空间），丢失了原始层的表达能力。
4. **本文要解决什么？** 在保持高稀疏度的同时，忠实地重建原始 MLP 层的功能。
5. **切入角度**：从神经元级稀疏性转向**层级稀疏性**——每次选择少量全秩线性变换（专家子层），每个专家的表达能力远强于单个神经元。
6. **核心idea一句话**：用 Hadamard 乘积参数化的张量分解构造大量参数高效的满秩专家子层，稀疏激活 $K$ 个即可忠实重建原始 MLP。

## 方法详解

### 整体框架
MxD 将原始 MLP 输出近似为 $N$ 个线性变换的稀疏加权组合：$\text{MxD}(\mathbf{x}) = \sum_{n=1}^N a_n(\mathbf{W}_n^\top\mathbf{z})$，其中 $\mathbf{z} = \phi(\mathbf{E}^\top\mathbf{x})$ 是密集隐层表示，$\mathbf{a} = \mathcal{S}(\mathbf{G}^\top\mathbf{x})$ 是稀疏专家系数（top-$K$），$\mathbf{W}_n$ 是第 $n$ 个专家的解码权重。

### 关键设计

1. **Hadamard 乘积张量分解**：
   - 做什么：参数高效地存储 $N$ 个全秩专家权重
   - 核心思路：$\boldsymbol{\mathcal{W}}(n,h,:) = \mathbf{c}_n * \mathbf{d}_h$，$\mathbf{C} \in \mathbb{R}^{N \times O}$ 是专家特定参数，$\mathbf{D} \in \mathbb{R}^{H \times O}$ 是共享变换。参数量从 $NHO$ 降到 $O(N+H)$
   - 等效前向传播：$\text{MxD}(\mathbf{x}) = (\mathbf{C}^\top\mathbf{a}) * (\mathbf{D}^\top\mathbf{z})$

2. **满秩保证（Lemma 1）**：
   - 做什么：证明每个专家权重满秩
   - 核心结论：$\mathbf{W}_n = \mathbf{D}\,\text{diag}(\mathbf{c}_n)$，只要 $\mathbf{c}_n$ 无零元素，$\text{rank}(\mathbf{W}_n) = \text{rank}(\mathbf{D})$
   - 设计动机：Transcoder 稀疏度 $K$ 时输出限于 $K$ 维子空间，MxD 是 $K$ 个满秩变换的和，表达力更强

3. **GLU 扩展**：
   - 做什么：推广到现代 LLM 的 Gated Linear Unit 架构
   - 核心思路：直接将 GLU 隐层 $\mathbf{z}_{\text{GLU}} = \psi(\mathbf{E}_{\text{GLU}}^\top\mathbf{x}) * (\mathbf{E}^\top\mathbf{x})$ 代入 MxD

### 损失函数 / 训练策略
- MSE 蒸馏损失（MxD 输出 vs 原始 MLP 输出）
- top-$K$ 路由，在 480M tokens OpenWebText 上训练
- $\mathbf{D}$ 初始化为零矩阵（渐进学习）

## 实验关键数据

### 主实验
4 个 LLM 上稀疏性-准确性边界（参数量匹配）：

| 方法 | 稀疏层级 | CE Loss 增量 | 可解释性 |
|------|---------|-------------|---------|
| Transcoder | 神经元级 | 较大 | 好 |
| Skip Transcoder | 神经元级+跳跃 | 中等 | 好 |
| **MxD** | **层级** | **显著更小** | **同等水平** |

MxD 在所有稀疏度水平上 Pareto 支配 Transcoder。

### 消融实验
| 配置 | 结果 | 说明 |
|------|------|------|
| GELU vs ReLU | GELU 显著更好 | 匹配原始激活函数重要 |
| 满秩 vs 低秩 MoE | 满秩更好 | 验证 Lemma 1 的实际价值 |
| 不同 $N$ | 更多专家→更好 | 张量分解使数万专家可行 |

### 关键发现
- **MxD 在稀疏性-准确性边界全面优于 Transcoder**：相同参数量和稀疏度下 CE loss 增量更小
- **可解释性无牺牲**：34 个 sparse probing 和 steering 任务上与 Transcoder 水平相当
- **满秩是关键**：即使 $K$=32 也能高保真重建，因为每个活跃专家贡献满秩变换

## 亮点与洞察
- **层级稀疏性的范式转变**：原子解释单元从神经元升级为完整线性变换，更接近"功能模块"概念
- **Hadamard 分解兼得效率和满秩**：$(\mathbf{C}^\top\mathbf{a}) * (\mathbf{D}^\top\mathbf{z})$ 实现极简，理论保证满秩
- **统一 MLP 和 GLU 分解**：不依赖特定稀疏假设

## 局限性 / 可改进方向
- **实验最大到 3B**：10B+ 效果未验证
- **逐层独立训练**：多层联合或端到端训练未探索
- **可解释性评估有限**：缺少人工评估和深入 mechanistic 分析
- **改进方向**：多层联合训练、更大模型验证、模型编辑/安全控制应用

## 相关工作与启发
- **vs SAE**: SAE 做后验分析（额外推理开销），MxD 直接替换层
- **vs Transcoder**: 神经元级稀疏 vs 层级稀疏，后者表达力更强
- **vs 传统 MoE**: 传统 MoE 专家少（几十个），MxD 通过张量分解扩展到数万个

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 层级稀疏性视角和 Hadamard 分解满秩保证都是全新贡献
- 实验充分度: ⭐⭐⭐⭐ 4个模型×多种稀疏度，Pareto 分析+可解释性评估
- 写作质量: ⭐⭐⭐⭐⭐ 理论-方法-实验链条清晰，Table 1 总结到位
- 价值: ⭐⭐⭐⭐⭐ 为 LLM 可解释性提供了更忠实且实用的层分解工具
