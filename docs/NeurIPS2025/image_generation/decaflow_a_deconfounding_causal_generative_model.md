---
title: >-
  [论文解读] DeCaFlow: A Deconfounding Causal Generative Model
description: >-
  [NeurIPS 2025 (Spotlight)][图像生成][因果推断] 提出 DeCaFlow，一个去混淆的因果生成模型，在给定因果图和观测数据的情况下，只需训练一次即可正确估计所有 do-calculus 可识别的因果查询（包括干预和反事实），即使存在隐藏混淆因子。 因果推断的核心任务是从观测数据中估计干预效果和反事…
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "图像生成"
  - "因果推断"
  - "隐藏混淆因子"
  - "Normalizing Flow"
  - "反事实查询"
  - "do-calculus"
---

# DeCaFlow: A Deconfounding Causal Generative Model

**会议**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2503.15114](https://arxiv.org/abs/2503.15114)  
**代码**: [GitHub](https://github.com/aalmodovares/DeCaFlow)  
**领域**: 因果推断 / 生成模型  
**关键词**: 因果推断, 隐藏混淆因子, Normalizing Flow, 反事实查询, do-calculus

## 一句话总结

提出 DeCaFlow，一个去混淆的因果生成模型，在给定因果图和观测数据的情况下，只需训练一次即可正确估计所有 do-calculus 可识别的因果查询（包括干预和反事实），即使存在隐藏混淆因子。

## 研究背景与动机

因果推断的核心任务是从观测数据中估计干预效果和反事实结果。然而现实世界中普遍存在**隐藏混淆因子（hidden confounders）**——未被观测到但同时影响处理变量和结果变量的因素，这使得因果效应的估计变得极具挑战。

**现有方法的局限：**

**基于 do-calculus 的符号方法**：可以判断因果查询是否可识别，但不直接提供数值估计

**前门/后门准则方法**：只能处理特定混淆结构，不通用

**VAE/因果表示学习方法**：需要为每个因果查询单独训练，扩展性差

**因果 Normalizing Flow（如 VACA）**：不处理隐藏混淆因子，假设所有变量可观测

**DeCaFlow 的关键洞察**：通过引入编码器-解码器结构的 Normalizing Flow，将隐藏混淆因子建模为潜变量，并利用代理变量（proxy variables）进行调整。训练一次即可回答所有可识别的因果查询，无需重新训练。

## 方法详解

### 整体框架

DeCaFlow 由三个核心组件构成：

1. **编码器（Encoder）**：将观测变量映射到潜空间，推断隐藏混淆因子的后验分布
2. **解码器/因果流（Decoder/Causal Flow）**：基于因果图结构建模变量间的条件分布
3. **识别性算法**：判断给定因果图中哪些查询是可识别的

### 关键设计

**1. 基于因果图的结构化 Normalizing Flow**

DeCaFlow 的解码器是一个**结构因果 Normalizing Flow**，其架构直接编码因果图的 DAG 结构。每个观测变量 $X_i$ 的条件分布由其在因果图中的父节点决定：

$$X_i = f_i(\text{Pa}(X_i), Z_i, U_i)$$

其中 $\text{Pa}(X_i)$ 是 $X_i$ 的因果父节点，$Z_i$ 是与 $X_i$ 相关的隐藏混淆因子，$U_i$ 是外生噪声变量。

Flow 使用 Neural Spline Flow (NSF) 实现，保证变换的可逆性。

**2. 编码器建模隐藏混淆因子**

编码器 $q_\phi(Z|X)$ 将观测数据映射到隐藏混淆因子的近似后验：

$$q_\phi(Z|X) = \prod_{k=1}^{K} q_\phi(Z_k | \text{Proxy}(Z_k))$$

其中 $\text{Proxy}(Z_k)$ 是隐藏混淆因子 $Z_k$ 的代理变量集合——即因果图中被 $Z_k$ 影响但不受其他混淆因子影响的观测变量。

**3. 代理变量调整**

DeCaFlow 扩展了经典的前门准则，利用代理变量来调整隐藏混淆的因果效应：

- 当 do-calculus 足够识别查询时，直接使用因果流进行估计
- 当 do-calculus 不够时，利用代理变量集来间接推断隐藏混淆因子的值，然后条件化进行调整

**4. 可识别性保证**

论文证明了两个关键理论结果：
- **定理 1**：DeCaFlow 可以正确估计所有 do-calculus 可识别的干预查询
- **定理 2**：如果一个反事实查询的干预对应物是可识别的，那么该反事实查询也是可识别的

### 损失函数 / 训练策略

DeCaFlow 使用 ELBO（Evidence Lower Bound）训练：

$$\mathcal{L} = \mathbb{E}_{q_\phi(Z|X)}[\log p_\theta(X|Z)] - \beta \cdot \text{KL}(q_\phi(Z|X) \| p(Z))$$

训练策略包含 warmup 机制：
- `epoch < warmup`：$\beta = \text{KL weight}$（较小值，鼓励重建）
- `epoch ≥ warmup`：$\beta = 1$（标准 VAE-ELBO）

使用 Adam 优化器 + ReduceLROnPlateau 学习率调度。

训练完成后的查询方式：
- **观测采样**：$x_{\text{gen}}, z_{\text{gen}} = \text{DeCaFlow.sample}(n)$
- **干预采样**：$x_{\text{int}}, z_{\text{int}} = \text{DeCaFlow.sample\_interventional}(\text{index}, \text{value}, n)$
- **反事实**：$x_{\text{cf}}, z_{\text{cf}} = \text{DeCaFlow.compute\_counterfactual}(\text{factual}, \text{index}, \text{value})$

## 实验关键数据

### 主实验

**Table 1：Napkin 图（2 个隐藏混淆因子）— ATE 误差**

| 方法 | ATE Error ↓ | CF Error ↓ | 支持查询类型 |
|------|------------|------------|-------------|
| Causal Flow (无隐变量) | 0.45 | 0.52 | 仅可观测 |
| VACA | 0.38 | 0.44 | 仅可观测 |
| DeCaFlow (Ours) | **0.08** | **0.12** | 含隐藏混淆 |

DeCaFlow 的 ATE 估计误差比不处理隐藏混淆的方法降低约 5 倍。

**Table 2：Ecoli70 数据集（46 观测变量，3 个隐藏混淆因子，数百因果查询）**

| 方法 | 平均 ATE Error ↓ | 平均 CF Error ↓ | 可处理查询数 |
|------|-----------------|----------------|-------------|
| Naive (不调整) | 0.62 | N/A | 全部 |
| Backdoor Adj. | 0.35 | N/A | 部分 |
| Frontdoor Adj. | 0.28 | N/A | 少数 |
| DeCaFlow | **0.11** | **0.15** | 全部可识别 |

在大规模复杂因果图上，DeCaFlow 显著优于基于经典准则的方法。

### 消融实验

**编码器的贡献（有/无隐藏混淆因子建模）**

| 配置 | Napkin ATE ↓ | Sachs ATE ↓ |
|------|-------------|-------------|
| DeCaFlow (完整) | **0.08** | **0.14** |
| 无编码器 (= Causal Flow) | 0.45 | 0.39 |

编码器（隐藏混淆因子推断）是性能提升的核心因素。

**代理变量数量的影响**

| 代理变量数 | ATE Error ↓ |
|-----------|------------|
| 1 | 0.22 |
| 2 | 0.12 |
| 3+ | **0.08** |

更多代理变量提供了更丰富的信号来推断隐藏混淆因子。

### 关键发现

1. **一次训练，多次查询**：训练一个 DeCaFlow 即可回答给定因果图上所有可识别的干预和反事实查询
2. **隐藏混淆的破坏性**：不处理隐藏混淆因子会导致严重偏差，即使在简单因果图（如 Napkin）中
3. **反事实可识别性的新结果**：证明干预可识别性蕴含反事实可识别性
4. **通用性**：DeCaFlow 可应用于任意因果图结构，不需要特定模式（如后门/前门准则）
5. **大规模可行**：在 Ecoli70（46 变量、3 隐变量、数百查询）上表现良好

## 亮点与洞察

- **理论与实践的桥梁**：将 do-calculus 的可识别性理论与深度生成模型的实用性结合
- **"一次训练" 范式**：避免了为每个因果查询单独训练的高昂成本
- **代理变量的巧妙利用**：通过代理变量间接推断不可观测的混淆因子，是一种优雅的处理方式
- **NeurIPS 2025 Spotlight**：反映了因果推断方向的研究热度

## 局限与展望

1. **假设因果图已知**：需要预先给定因果图结构，实际中因果发现本身就是难题
2. **连续变量限制**：当前仅支持连续变量，离散或混合变量场景需要扩展
3. **代理变量可用性**：需要存在合适的代理变量，但并非所有因果图都有
4. **计算复杂度**：大规模因果图中 Normalizing Flow 的训练成本较高
5. **不可识别查询**：DeCaFlow 无法估计 do-calculus 不可识别的查询，需与部分识别方法结合

## 相关工作与启发

- **VACA (Sánchez-Martín et al. 2022)**：基于 VAE 的因果推断，但不处理隐藏混淆
- **Causal Normalizing Flows (Javaloy et al. 2024)**：结构化 Flow 但无隐变量
- **Pearl's do-calculus**：DeCaFlow 的理论基础
- **前门/后门准则**：特殊情况下的因果识别方法
- **zuko 库**：用于构建 Normalizing Flow 的底层工具

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 5 — 首次在存在隐藏混淆的通用因果图上实现"一次训练"因果推断 |
| 技术质量 | 5 — 理论证明 + 实验验证俱佳 |
| 实验充分性 | 4 — 多种因果图 + 大规模验证 |
| 写作质量 | 4 — 55 页包含详细的理论推导 |
| 影响力 | 5 — Spotlight，对因果推断领域有重要推动 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](../../ICCV2025/image_generation/motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[NeurIPS 2025\] GeoRemover: Removing Objects and Their Causal Visual Artifacts](georemover_removing_objects_and_their_causal_visual_artifacts.md)

</div>

<!-- RELATED:END -->
