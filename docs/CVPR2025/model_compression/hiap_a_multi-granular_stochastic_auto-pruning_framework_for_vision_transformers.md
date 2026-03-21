# HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers

**会议**: CVPR 2025  
**arXiv**: [2603.12222](https://arxiv.org/abs/2603.12222)  
**代码**: 待确认  
**领域**: 模型压缩  
**关键词**: Vision Transformer, 结构化剪枝, Gumbel-Sigmoid, 多粒度, 自动架构搜索

## 一句话总结
HiAP 提出了一种多粒度自动剪枝框架，通过在宏观（attention heads、FFN blocks）和微观（intra-head dimensions、FFN neurons）两级部署可学习 Gumbel-Sigmoid 门控，在单阶段端到端训练中自动发现最优子网络，无需手工重要性排序或后处理阈值。

## 研究背景与动机
1. **领域现状**：ViT 剪枝方法已取得显著进展，但大部分方法在单一粒度操作。微观剪枝（ViT-Slim 剪 intra-head 维度和 FFN channel）能减少 FLOPs，宏观剪枝（UPDP 剪整个 block）能减少内存带宽开销。
2. **现有痛点**：（1）单粒度剪枝无法同时优化内存带宽（需要宏观剪枝）和计算量（需要微观剪枝）；（2）现有可微搜索方法依赖后处理阈值（post-hoc magnitude thresholding），需要专家知识和多阶段流程。
3. **核心矛盾**：现代硬件的真正瓶颈往往是 HBM 访问而非纯计算——即使微观剪枝减少了 FLOPs，保留所有 layer 结构仍然要加载每一层的权重矩阵和 attention map，导致实际加速有限。
4. **本文要解决什么？** 如何在单阶段训练中让模型自主发现跨粒度的最优剪枝策略，同时满足硬件资源约束？
5. **切入角度**：将剪枝形式化为预算感知的学习问题，用 Gumbel-Sigmoid 连续松弛取代离散决策，通过温度退火自然收敛到二值门控。
6. **核心idea一句话**：宏观+微观双层 Gumbel-Sigmoid 门控 + 可微 MACs 代价 + 结构可行性约束 = 单阶段自动剪枝。

## 方法详解

### 整体框架
输入预训练 ViT（如 DeiT-Small），为每个 Transformer block 部署双层门控：宏观门控控制是否保留整个 attention head ($g_{l,h}$) 和 FFN block ($b_l$)；微观门控控制 active 结构内部的 intra-head 维度 ($d_{l,h,j}$) 和 FFN neuron ($c_{l,k}$)。训练过程中门控采用 Gumbel-Sigmoid 连续松弛，温度 $\tau$ 从 2.0 退火到 0.5，最终门控自然硬化为 0/1，物理截断权重矩阵得到可直接部署的紧凑子网络。

### 关键设计

1. **层级门控机制（Hierarchical Gating）**:
   - 做什么：宏观层面控制 head/block 的保留/丢弃，微观层面控制存活结构内部的维度裁剪
   - 核心思路：$\text{Head}'_{l,h}(X) = g_{l,h} [\text{softmax}(\frac{QK^\top}{\sqrt{D_h}})(V \odot d_{l,h})]$，$\text{FFN}'_l(X) = b_l[(\phi(XW_1) \odot c_l)W_2]$，宏观门为0时完全跳过，微观门对存活结构做 channel-wise masking
   - 设计动机：宏观剪枝节省内存带宽（跳过整个 head/block 的矩阵加载），微观剪枝在保持结构的前提下节省计算量，两者互补

2. **可微分代价建模（Differentiable Cost Modeling）**:
   - 做什么：将 MACs 精确分解为关于门控的线性函数
   - 核心思路：$\mathbb{E}[C(\mathcal{G})] = \sum_{l,h}(C_1 \cdot \mathbb{E}[g_{l,h}] + C_2 \sum_j \mathbb{E}[g_{l,h} \cdot d_{l,h,j}]) + \sum_{l,k} C_3 \cdot \mathbb{E}[b_l \cdot c_{l,k}]$，将代价解耦为 $\mathcal{L}_{\text{macro}}$（惩罚 $C_1$）和 $\mathcal{L}_{\text{micro}}$（惩罚 $C_2, C_3$），独立超参数控制
   - 设计动机：线性分解使优化器能清晰地将硬件惩罚归因到个别结构，显式惩罚空 head 的结构开销

3. **结构可行性约束（Feasibility Penalty）**:
   - 做什么：防止 layer collapse（整层被贪心剪掉导致梯度断流）
   - 核心思路：$\mathcal{L}_{f,\text{head}} = \sum_l \text{ReLU}(k_{\min} - \sum_h g_{l,h})^2$，保证每层至少保留 $k_{\min}$ 个 head，类似约束保证最少 FFN 神经元和 attention 维度
   - 设计动机：这是可微架构搜索中一种常见失效模式的解药

4. **单阶段训练 + 温度退火**:
   - Gumbel-Sigmoid：$\hat{z} = \sigma((\alpha + \epsilon)/\tau)$，早期 $\tau$ 高时门控行为类似 stochastic dropout，网络学习鲁棒分布式表示；$\tau$ 降低后门控自然收敛到二值
   - 训练结束后直接以 0.5 阈值硬化门控，物理截断权重矩阵，无需二次 fine-tuning

### 损失函数
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{macro}} \mathcal{L}_{\text{macro}} + \lambda_{\text{micro}} \mathcal{L}_{\text{micro}} + \mathcal{L}_{\text{feasibility}}$$

其中 $\mathcal{L}_{\text{task}}$ 包含 cross-entropy 和知识蒸馏 loss（$\alpha_{KD}=0.7, T=4.0$）。

## 实验关键数据

### 主实验（ImageNet-1K, DeiT-Small）

| 方法 | Params (M) | FLOPs (G) | Top-1 Acc (%) | Δ Acc |
|------|-----------|-----------|---------------|-------|
| Dense Baseline | 22.1 | 4.6 | 79.85 | — |
| ViT-Slim | 15.6 | 3.1 | 79.90 | +0.05 |
| GOHSP | 14.4 | 3.0 | 79.98 | +0.13 |
| **HiAP** | 15.0 | 3.1 | 79.10 | -0.75 |
| ViT-Slim | 13.5 | 2.8 | 79.50 | -0.35 |
| **HiAP** | 12.3 | 2.5 | 77.95 | -1.90 |

### 消融实验（CIFAR-10, ViT-Tiny）

| 方法 | MACs (M) | 减少率 | Acc (%) |
|------|---------|--------|---------|
| Dense | 174.0 | 0% | 90.50 |
| Uniform-Ratio | 116.6 | 33% | 86.63 |
| $\ell_1$-Structured (FFN) | 116.5 | 33% | 87.15 |
| **HiAP (Moderate)** | 116.3 | 33% | **87.56** |
| $\ell_1$-Structured (FFN) | 87.3 | 50% | 86.80 |
| **HiAP (Aggressive)** | 87.1 | 50% | **87.25** |

### 关键发现
- HiAP 在精度上不如 GOHSP 和 ViT-Slim（在3.1G FLOPs 配置下差 0.75-0.88%），但**无需多阶段流程**
- 网络自主发现的剪枝策略：早期先大胆剪宏观结构（前10 epoch 将 6 个 head 减到 2-4 个），后期微调微观维度
- 最后一层（L=12）的 FFN block 被完全识别为冗余并移除
- 33% 剪枝实际延迟从 5.57ms 降到 3.86ms（1.44× 加速），物理截断确保不依赖稀疏计算引擎
- 解耦损失验证：网络先关闭空 head（消除 $C_1$ 结构开销），再精细裁剪微观维度（优化 $C_2, C_3$）

## 亮点与洞察
- **宏观+微观双层门控统一框架**：首次将 block/head 级剪枝和 dimension/neuron 级剪枝放入同一个可微优化框架中，且有理论证明（Lemma 1）双层搜索空间严格超过单层
- **无后处理的单阶段训练**：温度退火使门控自然从 soft 过渡到 hard，无需二次 fine-tuning 或手动阈值设定
- **可迁移思路**：Gumbel-Sigmoid 门控 + 可微代价建模的框架可以推广到 LLM 剪枝（attention heads + FFN experts in MoE）

## 局限性 / 可改进方向
- 精度与 SOTA 方法有差距：在 3.1G FLOPs 下 79.10% vs GOHSP 79.98%，说明简化流程带来了性能代价
- 优化目标是期望 MACs 而非校准延迟/能耗，实际加速因硬件/kernel 而异
- 仅在 DeiT-Small 验证，缺少更大模型（如 DeiT-Base、Swin）和下游任务的实验
- 可以与 token pruning、量化等技术组合获得更大压缩比

## 相关工作与启发
- **vs ViT-Slim**: ViT-Slim 只做微观剪枝（intra-head + FFN channel），需要 $\ell_1$ 稀疏 + 排序阈值。HiAP 增加了宏观剪枝维度且完全自动化，但精度略低
- **vs GOHSP**: GOHSP 用图排序 + 优化求解器，精度更高但流程复杂。HiAP 用端到端梯度优化替代离线求解器
- **vs UPDP**: UPDP 仅做 block 级剪枝（遗传算法），HiAP 的宏观层面包含 head 和 block 两个粒度

## 评分
- 新颖性: ⭐⭐⭐⭐ 多粒度 Gumbel-Sigmoid 门控统一框架设计巧妙
- 实验充分度: ⭐⭐⭐ 只有 DeiT-Small 一个模型在 ImageNet，数据点偏少
- 写作质量: ⭐⭐⭐⭐ 理论分析扎实，框架图清晰
- 价值: ⭐⭐⭐ 精度不如 SOTA，主要贡献在简化流程而非刷新性能
