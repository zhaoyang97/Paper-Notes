---
title: >-
  [论文解读] Generalized Linear Mode Connectivity for Transformers
description: >-
  [NeurIPS 2025 Oral][linear mode connectivity] 提出统一对称性框架（置换、半置换、正交、可逆变换四级层次），首次在 Vision Transformer 和 GPT-2 上实现零/近零 barrier 的线性模式连通性（LMC），并扩展至多模型融合与异构宽度对齐。
tags:
  - "NeurIPS 2025 Oral"
  - "linear mode connectivity"
  - "model merging"
  - "permutation symmetry"
  - "orthogonal symmetry"
  - "ViT"
  - "GPT-2"
---

# Generalized Linear Mode Connectivity for Transformers

**会议**: NeurIPS 2025 Oral  
**arXiv**: [2506.22712](https://arxiv.org/abs/2506.22712)  
**代码**: 有（论文中提供链接）  
**领域**: 模型融合 / Transformer 理论  
**关键词**: linear mode connectivity, model merging, permutation symmetry, orthogonal symmetry, ViT, GPT-2  
**作者**: Alexander Theus, Alessandro Cabodi, Sotiris Anagnostidis, Antonio Orvieto, Sidak Pal Singh, Valentina Boeva (ETH Zürich, MPI 等)

## 一句话总结

提出统一对称性框架（置换、半置换、正交、可逆变换四级层次），首次在 Vision Transformer 和 GPT-2 上实现零/近零 barrier 的线性模式连通性（LMC），并扩展至多模型融合与异构宽度对齐。

## 研究背景与动机

**Linear Mode Connectivity (LMC)** 描述的是独立训练的神经网络在参数空间中可通过线性插值路径连接、且路径上 loss 不显著升高的现象。形式化定义为插值 barrier：

$$\mathcal{B}[\theta_A, \theta_B] = \sup_{\lambda \in [0,1]} \left[ \mathcal{L}[\lambda \theta_A + (1-\lambda)\theta_B] - (\lambda \mathcal{L}[\theta_A] + (1-\lambda)\mathcal{L}[\theta_B]) \right]$$

当 $\mathcal{B} \approx 0$ 时即为线性模式连通。LMC 对模型融合 (model merging)、联邦学习、集成学习等应用有重要意义。

**已有工作的局限**：

1. Git Re-Basin 等方法主要利用**离散置换对称性**（神经元重排序）来对齐模型
2. 在 MLP 和 CNN（VGG、ResNet）上取得了成功，但对 Transformer 效果有限
3. Transformer 的 multi-head attention、LayerNorm/RMSNorm、QK/OV circuit 具有比置换更丰富的对称性
4. 仅靠置换对称性在 Transformer 上仍存在不可忽略的插值 barrier
5. Git Re-Basin 在 ResNet-20 CIFAR-10 上需要 **32× 宽度增加**才能达到零 barrier，实用性受限

**核心问题**：如何系统地利用 Transformer 架构中所有的对称性来实现 LMC？

## 方法详解

### 1. 四级对称性层次框架

论文将网络对称性组织为严格包含的四层结构 $\mathcal{S}_1 \subset \mathcal{S}_2 \subset \mathcal{S}_3 \subset \mathcal{S}_4$：

| 层级 | 类别 | 变换结构 | 适用场景 |
|------|------|----------|----------|
| $\mathcal{S}_1$ | Permutation（置换） | 置换矩阵 | GELU、sigmoid、softmax、tanh |
| $\mathcal{S}_2$ | Semi-permutation（半置换） | 稀疏随机矩阵 | ReLU、LayerNorm、Multi-Head Attention |
| $\mathcal{S}_3$ | Orthogonal（正交） | 正交矩阵 | RMSNorm |
| $\mathcal{S}_4$ | Invertible（可逆） | 满秩矩阵 | Attention QK/OV circuit |

- **置换对称性**：逐元素激活函数（如 GELU）的输出仅依赖对应位置的输入，允许神经元重排序
- **半置换对称性**：分段线性函数（如 ReLU）满足 $f(x) = f(\alpha x) + f((1-\alpha)x)$，允许稀疏加权混合；MHA 的各 head 独立求和，且 OV 部分可做线性分解 ($\alpha$ 加权)
- **正交对称性**：RMSNorm 在正交变换下不变，保范数；LayerNorm 可改写为 RMSNorm + centering 矩阵 $M$ + scale 参数的形式
- **可逆对称性**：Attention 的 QK circuit ($W^Q(W^K)^\top$) 和 OV circuit ($W^V W^O$) 内部可插入任意可逆变换而不改变输出

### 2. Transformer 组件级对称性分析

**Feed-forward 层**：$\text{FF}(\mathbf{x}) = W_2 \phi(W_1 \mathbf{x} + b_1) + b_2$
- 当 $\phi$ 为 GELU 时仅有置换对称性
- 当 $\phi$ 为 ReLU 时可扩展至半置换对称性

**Multi-head Attention**：
- **Intra-head（头内）**：QK 和 OV circuit 具有可逆对称性，任意可逆变换可被代数消去，论文直接用 circuit 乘积作为规范表示
- **Inter-head（头间）**：各 head 独立求和 → 置换对称；OV 满足线性分解 → 半置换对称

**Residual Stream**：
- LayerNorm 可改写为 $\text{LayerNorm}(\mathbf{Z}) = \text{RMSNorm}(\mathbf{Z}\mathbf{M}) \cdot \text{diag}(\alpha)\sqrt{D} + \mathbf{1}_N \beta^\top$
- 吸收 $M$ 和 scale 后，残差路径具有正交对称性
- 正交矩阵可为**长方形** ($M \geq N$)，支持宽度异构对齐

### 3. 三种对齐策略

**Weight Matching（权重匹配，无数据）**：
- FF 层：逐层求解 SOBLAP（sum of bilinear assignments），匈牙利算法近似
- Attention 层：基于 QK/OV circuit 的 Frobenius 范数构建 cost matrix，线性分配求解 head 级置换
- Residual stream：Procrustes 问题 SVD 闭式解 $\mathbf{O} = U V^\top$
- **优点**：完全无数据，计算高效

**Activation Matching（激活匹配）**：
- 在共享数据集上比较中间层激活进行对齐（沿用已有方法）

**Learned Matching（学习匹配，端到端优化）**：
- 引入可学习隐式矩阵 $Z_{\text{FF}}, Z_H, Z_O$
- 每步前向传播投影到对应对称类：$P_{\text{FF}} = \text{ProjPerm}(Z_{\text{FF}})$（匈牙利 + STE），$O = \text{ProjOrth}(Z_O)$（SVD，完全可微）
- Weight matching 解作为初始化（关键，identity init 效果差很多）
- $\lambda \sim \mathcal{U}(0.4, 0.6)$，在插值模型上计算 task loss 并反向传播

### 4. 多模型融合

**Universe Matching**：迭代构建共享参考模型 $U^{(t)}$，所有模型对齐到该参考后平均，多轮迭代收敛

**Learned Refinement**：扩展到 M-way，采样 $\lambda \sim \text{Dirichlet}(\alpha \mathbf{1}_M), \alpha=0.1$，直接最小化多模型 barrier

## 实验关键数据

### 表1：两模型对齐 Loss Barrier 对比（越低越好）

| 方法 | CIFAR-10 | CIFAR-100 | Tiny ImageNet | Tiny Shakespeare | BookCorpus |
|------|----------|-----------|---------------|------------------|------------|
| Vanilla averaging | 1.69±0.07 | 2.46±0.04 | 2.84±0.02 | 2.02±0.12 | 4.34±0.09 |
| Activation matching | 1.27±0.13 | 2.11±0.17 | 1.86±0.10 | 1.43±0.16 | 4.05±0.13 |
| Weight matching (本文) | 0.36±0.01 | 0.69±0.21 | 0.47±0.04 | 0.34±0.01 | 1.56±0.02 |
| Learned matching (仅置换) | 0.45±0.02 | 0.53±0.07 | 0.29±0.02 | 0.63±0.17 | 1.60±0.04 |
| **Learned matching (本文)** | **0.00±0.00** | **0.00±0.00** | **0.00±0.00** | **0.02±0.00** | **0.42±0.01** |

### 表2：模型架构与训练配置

| 配置项 | ViT (CIFAR-10/100) | ViT (Tiny ImageNet) | GPT-2 (Tiny Shakespeare) | GPT-2 (BookCorpus) |
|--------|---------------------|---------------------|--------------------------|---------------------|
| Transformer 层数 | 6 | 8 | 6 | 6 |
| 注意力头数 | 8 | 8 | 4 | 8 |
| Embedding dim | 256 | 384 | 256 | 512 |
| MLP hidden dim | 512 | 768 | 1024 | 2048 |
| 训练 epochs | 150 | 150 | 100 | 5 |
| 优化器 | AdamW | AdamW | AdamW | AdamW |
| 硬件 | 1× RTX 2060 | 1× RTX 4090 | 1× RTX 4090 | 4× RTX 4090 |

## 亮点与洞察

1. **理论贡献突出**：首次系统化地将 Transformer 的所有对称性组织为层次化框架，统一了此前散落的对齐方法
2. **里程碑结果**：首次在标准大小（无需宽度膨胀）的 ViT 和 GPT-2 上实现零 barrier LMC
3. **Gap 分析精彩**：Weight matching 和 learned matching 的差距**仅集中在残差流的正交矩阵** $O$ 上；learned matching 的修正量很小（eigen-angles 集中在 $0 \bmod 2\pi$ 附近），置换部分基本不变 → 说明更好的无数据正交估计可能弥合大部分差距
4. **宽度异构对齐**：利用长方形正交矩阵，可以对齐不同宽度的 Transformer，且插值路径仍维持零/近零 barrier
5. **多模型扩展自然**：三模型在 simplex 上的 loss surface 可视化清晰展示了 learned matching 产生最广泛的近零偏差区域
6. **Weight matching 已经很强**：即使不用训练数据，weight matching 也大幅领先 activation matching 和 vanilla averaging，说明正交对称性的利用是关键

## 局限性

1. **语言模型规模偏小**：GPT-2 实验仅用了 6 层、最大 dim=512 的缩小版，未在 GPT-2 full / LLaMA 等大模型上验证可扩展性
2. **BookCorpus barrier 非零（0.42）**：可能反映 NLP 模型学到了根本不同的泛化策略（lexical-overlap vs. syntactic cues），而非对齐不足
3. **需要 RMSNorm 重参数化**：吸收 LayerNorm 的 centering 会改变网络架构表示（虽然功能等价），对某些架构可能不便
4. **仅限同任务模型**：当前只考虑在相同任务上预训练的模型对齐，跨任务场景（如 ZipIt! 方向）尚未探索
5. **视觉 benchmark 偏小**：CIFAR-10/100 和 Tiny ImageNet，未在 ImageNet-1K 等更大规模上验证
6. **Soft permutation 潜力未完全挖掘**：doubly stochastic 松弛可改善测试时性能，但系统研究尚不充分

## 相关工作与启发

### 直接相关工作

- **Git Re-Basin** (Ainsworth et al.)：提出 weight matching / activation matching / STE 三种对齐方法，本文的统一框架将其作为特例
- **OT Fusion** (Singh & Jaggi)：用最优传输做神经元对齐，本文扩展了对称类别
- **SliceGPT** (Ashkboos et al.)：发现 Transformer 残差路径的正交对称性用于剪枝，本文将其用于 LMC
- **Transformer Circuits** (Elhage et al.)：提出 QK/OV circuit 的概念，本文利用其可逆对称性

### 启发方向

1. **无数据正交估计改进**：论文 Section 6 分析表明 gap 主要在 $O$ 矩阵 → 更好的结构先验或谱方法可能在无数据情况下逼近 learned matching 的性能
2. **联邦学习应用**：weight matching 无需共享数据即可大幅降低 barrier，天然适合联邦场景
3. **模型编辑 / Continual Learning**：如果不同阶段训练的模型在同一 basin 中，可通过对齐后插值实现知识组合
4. **大模型 LMC 验证**：最有价值的后续工作之一是在 LLaMA-scale 模型上验证广义 LMC 是否依然成立

## 评分

- 新颖性：⭐⭐⭐⭐⭐（首次统一四类对称性并在 Transformer 上实现零 barrier LMC）
- 理论深度：⭐⭐⭐⭐⭐（严格的对称性层次化分析，Procrustes/匈牙利/SVD 的融合）
- 实验质量：⭐⭐⭐⭐（消融充分，但规模偏小；gap 分析是加分项）
- 实用价值：⭐⭐⭐⭐（weight matching 无数据且高效，直接可用于联邦学习/模型融合）
- 总体评价：⭐⭐⭐⭐⭐（该方向的里程碑式工作，首次将 LMC 从 MLP/CNN 推进到 Transformer）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Functional Equivalence in Attention: A Comprehensive Study with Applications to Linear Mode Connectivity](../../ICML2026/others/functional_equivalence_in_attention_a_comprehensive_study_with_applications_to_l.md)
- [\[ICLR 2026\] Do We Really Need Permutations? Impact of Model Width on Linear Mode Connectivity](../../ICLR2026/others/do_we_really_need_permutations_impact_of_model_width_on_linear_mode_connectivity.md)
- [\[NeurIPS 2025\] Scalable Inference of Functional Neural Connectivity at Submillisecond Timescales](scalable_inference_of_functional_neural_connectivity_at_submillisecond_timescale.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[NeurIPS 2025\] Sheaf Cohomology of Linear Predictive Coding Networks](sheaf_cohomology_of_linear_predictive_coding_networks.md)

</div>

<!-- RELATED:END -->
