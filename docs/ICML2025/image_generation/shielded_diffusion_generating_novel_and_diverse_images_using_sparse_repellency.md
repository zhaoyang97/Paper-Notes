---
title: >-
  [论文解读] Shielded Diffusion: Generating Novel and Diverse Images using Sparse Repellency
description: >-
  [ICML2025][图像生成][扩散模型多样性] 提出 SPELL（Sparse Repellency）方法，在扩散模型生成过程中添加稀疏排斥项：，将采样轨迹推离参考图像集合（受保护图像或已生成图像），以免训练：方式提升输出多样性并防止复制训练集。 文本到图像扩散模型在部署时面临两大问题： 多样性不足：使用 Classif…
tags:
  - "ICML2025"
  - "图像生成"
  - "扩散模型多样性"
  - "稀疏排斥引导"
  - "训练后引导"
  - "图像保护"
  - "去重生成"
---

# Shielded Diffusion: Generating Novel and Diverse Images using Sparse Repellency

**会议**: ICML2025  
**arXiv**: [2410.06025](https://arxiv.org/abs/2410.06025)  
**代码**: 待确认  
**领域**: 扩散模型 / 图像生成  
**关键词**: 扩散模型多样性, 稀疏排斥引导, 训练后引导, 图像保护, 去重生成

## 一句话总结

提出 SPELL（Sparse Repellency）方法，在扩散模型生成过程中添加**稀疏排斥项**，将采样轨迹推离参考图像集合（受保护图像或已生成图像），以**免训练**方式提升输出多样性并防止复制训练集。

## 研究背景与动机

文本到图像扩散模型在部署时面临两大问题：

**多样性不足**：使用 Classifier-Free Guidance (CFG) 的模型对同一 prompt 反复生成时，往往产出高度相似的图像，缺乏真正的多样性

**训练集泄漏**：模型可能直接复制训练集中的图像，引发版权和隐私风险

现有方法要么需要重训模型，要么采用"生成后丢弃"策略（计算浪费），要么使用全局密集的粒子引导（Particle Guidance），在每个时间步对所有样本施加扰动，导致图像质量下降。

**核心洞察**：能否设计一种**按需触发、稀疏介入**的后处理引导机制——仅在扩散轨迹即将落入"屏蔽区"时才施加校正，且校正主要集中在生成的早期阶段？

## 方法详解

### 核心框架：稀疏排斥（SPELL）

给定参考图像集 $\{z_k\}_{k=1}^K$ 和保护半径 $r > 0$，定义屏蔽区域为各参考图像周围的 L2 球：

$$S = \bigcup_{k=1}^K B_k, \quad B_k = \{x \in \mathcal{X} : \|x - z_k\|_2 \leq r\}$$

### 轨迹校正机制

在反向扩散的每个时间步 $t$，利用去噪网络预测终态：

$$\hat{x}_0 = D_{\theta^*}(t, x_t)$$

若 $\hat{x}_0$ 落入某个屏蔽球 $B_k$ 内，则施加最小校正将其推出：

$$\delta_k(\hat{x}_0) = \frac{(\hat{x}_0 - z_k) \cdot r}{\|\hat{x}_0 - z_k\|_2} - (\hat{x}_0 - z_k)$$

### 稀疏聚合公式

汇总所有屏蔽点的校正，通过 ReLU 实现自然稀疏：

$$\Delta = \sum_{k=1}^K \sigma_{\text{relu}}\left(\frac{r}{\|\hat{x}_0 - z_k\|_2} - 1\right) \cdot (\hat{x}_0 - z_k)$$

- 当 $\|\hat{x}_0 - z_k\|_2 \geq r$ 时，ReLU 输出为 0，**不施加任何干预**
- 仅当预测终态过于接近某参考图像时才触发校正
- 实际中，每个时间步通常只有极少数（典型为 1 个）屏蔽点处于活跃状态

### 理论推导：DPS 视角

SPELL 可理解为 Diffusion Posterior Sampling 的特殊情形。通过贝叶斯准则：

$$\nabla_{x_t} \log p_t(x_t \mid x_0 \notin S) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_{0|t}(x_0 \notin S \mid x_t)$$

修正后的反向 SDE 为：

$$d\mathbf{X}_t = \left[f(t, \mathbf{X}_t) - g(t)^2 \tilde{s}_t(\mathbf{X}_t, S)\right] dt + g(t) dB_t$$

SPELL 用 ReLU 硬截断代替了 DPS 中基于高斯的软引导，避免了难以调节的似然尺度超参数。

### 两种使用模式

| 模式 | 参考集来源 | 应用场景 |
|------|-----------|---------|
| **静态屏蔽** | 受保护的训练集图像 | 防止生成训练集近似副本 |
| **动态批内排斥** | 当前批次 + 历史批次的预测终态 | 提升同 prompt 多图的多样性 |

批内排斥时，屏蔽点动态更新为各轨迹的当前预测终态 $z_{k,t} = D_{\theta^*}(t, x_t^{(k)})$。

### 过补偿因子

引入放大系数 $\lambda$（论文推荐 $\lambda = 1.6$），可提前结束排斥，使轨迹更早跳出屏蔽区：

$$\Delta' = \lambda \cdot \Delta$$

## 实验关键数据

### 多样性提升（Table 1 精选）

| 模型 | Recall ↑ | Vendi Score ↑ | FID ↓ |
|------|----------|---------------|-------|
| Latent Diffusion | 0.236 | 2.527 | 9.50 |
| + SPELL | **0.289** (+22%) | **2.695** (+7%) | 9.55 |
| SD3-Medium | 0.379 | 3.749 | 20.10 |
| + SPELL | **0.483** (+27%) | **4.711** (+26%) | 35.17 |
| EDMv2 | 0.589 | 11.645 | 3.38 |
| + SPELL | **0.600** (+2%) | **11.806** (+1%) | 3.46 |
| MDTv2 | 0.623 | 12.546 | 4.88 |
| + SPELL | **0.634** (+2%) | **12.772** (+2%) | 4.38 |

- 所有模型的多样性指标均一致提升
- Precision 仅有轻微下降或不变，FID 影响边际
- SPELL 的 diversity-precision Pareto 前沿优于 Particle Guidance、Interval Guidance、CADS

### 稀疏性分析

- 排斥校正幅度通常 **不超过扩散 score 的 5%**，最大不超过 35%
- 在 $t = 0.8$ 时仅 40% 的轨迹有非零排斥项，$t = 0.6$ 时降至 21%
- 排斥主要集中在 $t \in [0.6, 1.0]$（生成早期），后期几乎为零

### 大规模图像保护（Table 2）

| 模型 | 落入屏蔽区比例 ↓ | Precision | 每张耗时 |
|------|-----------------|-----------|---------|
| EDMv2 (无 SPELL) | 7.60% | 0.792 | 2.43s |
| + SPELL-1 | 1.08% | 0.792 | 4.63s |
| + SPELL-10 | **0.16%** | 0.768 | 13.54s |

屏蔽全部 120 万张 ImageNet-1k 训练图像后，近副本生成率从 7.6% 降至 0.16%，Precision 几乎不变。

## 亮点与洞察

1. **优雅的稀疏设计**：ReLU 门控使排斥项天然为零，仅在必要时激活；无需为每对粒子计算交互能量
2. **免训练、即插即用**：适用于任意预训练扩散模型（RGB/VAE 空间、有/无 CFG、text/class 条件）
3. **单参数控制**：仅需调节保护半径 $r$，即可平滑控制多样性-精度权衡
4. **可扩展至百万级屏蔽集**：配合近似最近邻搜索，可屏蔽 120 万张图像
5. **跨批次一致性**：通过累积历史生成图像为参考集，即使 batch size 较小也能保证大量图像间的多样性

## 局限与展望

1. **屏蔽重叠问题**：当多个屏蔽球重叠且轨迹恰好落入重叠中心时，排斥力可能相互抵消；严格保证需要二次规划
2. **L2 距离局限**：当前在 VAE 潜空间中用 L2 距离度量相似性，不一定反映语义相似性；可考虑在 DINOv2 等语义空间中操作
3. **期望近似**：SPELL 作用于条件期望 $\mathbb{E}[X_0 | x_t]$ 而非 $p_{0|t}$ 的真实样本，在使用 probability flow ODE 采样器时理论保证减弱
4. **大规模屏蔽的计算开销**：百万级屏蔽集需要 CPU 端近似最近邻搜索，单张生成时间从 2.4s 增至 13.5s

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将排斥引导从密集交互简化为稀疏 ReLU 门控，几何直觉清晰
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖 6 种扩散模型、2 种任务设定、百万级扩展实验、详细消融
- 写作质量: ⭐⭐⭐⭐ — 理论推导与几何解释结合良好，图文并茂
- 价值: ⭐⭐⭐⭐ — 为扩散模型部署中的多样性和版权保护提供了实用工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior](../../ICML2026/image_generation/initialization_is_half_the_battle_generating_diverse_images_from_a_guidance_pote.md)
- [\[ICCV 2025\] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](../../ICCV2025/image_generation/loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [\[CVPR 2026\] Refracting Reality: Generating Images with Realistic Transparent Objects](../../CVPR2026/image_generation/refracting_reality_generating_images_with_realistic_transparent_objects.md)
- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](../../ICCV2025/image_generation/less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[ACL 2025\] Synthia: Novel Concept Design with Affordance Composition](../../ACL2025/image_generation/synthia_novel_concept_design_with_affordance_composition.md)

</div>

<!-- RELATED:END -->
