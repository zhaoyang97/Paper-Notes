---
title: >-
  [论文解读] LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching
description: >-
  [NeurIPS 2025][图像生成][反事实解释] 提出LeapFactual，一种基于条件流匹配(CFM)的反事实解释算法，通过"起飞-降落"(Leap)机制在扁平化和结构化潜在空间之间建立桥梁，生成可靠且分布内的反事实样本，即使在学习决策边界与真实边界不一致时也能有效工作。 反事实解释(CE)通过回答"输入需要做什么…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "反事实解释"
  - "条件流匹配"
  - "可靠性"
  - "模型无关"
  - "信息混合"
---

# LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching

**会议**: NeurIPS 2025  
**arXiv**: [2510.14623](https://arxiv.org/abs/2510.14623)  
**代码**: [GitHub](https://github.com/caicairay/LeapFactual)  
**领域**: 可解释AI / 反事实解释  
**关键词**: 反事实解释, 条件流匹配, 可靠性, 模型无关, 信息混合

## 一句话总结

提出LeapFactual，一种基于条件流匹配(CFM)的反事实解释算法，通过"起飞-降落"(Leap)机制在扁平化和结构化潜在空间之间建立桥梁，生成可靠且分布内的反事实样本，即使在学习决策边界与真实边界不一致时也能有效工作。

## 研究背景与动机

反事实解释(CE)通过回答"输入需要做什么改变才能改变模型的预测？"来提供模型可解释性。现有方法存在三大核心问题：

**基于优化(Opt)的方法**的困境：这类方法在生成模型的潜在空间中优化潜在向量使其跨越决策边界。但当从源类到目标类距离较远时（尤其是多分类场景），梯度需要穿越多个类别区域，导致**梯度消失**，生成的反事实样本停留在决策边界附近，不能真正体现目标类的特征。此外，这类方法必须要求分类器可微分。

**基于条件生成模型(CGM)的方法**的问题：将分类器输出作为生成模型的条件，通过替换条件来生成反事实。但这类方法的潜在空间是**不连续的**——不同类别的潜在空间相互分离，无法进行有意义的插值或从反事实样本反向追溯到决策边界。

**可靠性问题**：现有方法生成的反事实样本通常位于**学习到的**决策边界附近，而非**真实的**决策边界。当模型学到的决策边界偏离真实边界时，生成的反事实样本可能既不在数据分布内，也不能准确代表目标类。

作者的核心洞察是：可以用流匹配在扁平化潜在表示（类别信息与残差信息耦合）和结构化潜在表示（类别信息作为外部条件）之间建立连续的、可逆的映射，从而同时享受两种范式的优势。

## 方法详解

### 整体框架

LeapFactual在潜在空间中引入一个新维度，沿正向可以剥离类别信息（"起飞"），沿反向可以注入类别信息（"降落"）。具体来说，利用条件流匹配学习一个从结构化表示 $Z_0$（不含类别信息）到扁平化表示 $Z_1$（含类别信息）的连续映射，以及其逆映射。

### 关键设计

1. **CE-CFM训练目标**: 标准I-CFM假设源分布和目标分布独立耦合，但在反事实场景中 $Z_0$ 和 $Z_1$ 通过共同父节点残差信息 $R$ 相关联。作者将条件项重新定义为 $h := (z_0, z_{1,c})$，其中 $z_{1,c} \sim q(z_1|c)$ 是特定类别 $c$ 的潜在向量。训练目标为：

$$\mathcal{L}_{\text{CE-CFM}}(\psi) := \mathbb{E}_{t, q(h), p_t(z|h)} \| v_\psi(t, z, c) - u_t(z|h) \|^2$$

通过将类别信息 $c$ 显式作为网络条件，利用高斯 $Z_0$ 的信息瓶颈效应压缩 $Z_1$ 中的类别信息。理论证明(Theorem 1)：$Z_0$ 是 $Z_1$ 的压缩表示，压缩损失的信息恰好是作为条件提供的类别信息 $C$。

2. **Leap机制（起飞-降落传输）**: 一次Leap由两步组成：(a) **Lift（起飞）**：沿反向积分 $\int_1^t \gamma_{\text{lift}} v_\psi(\tau, z^{y_c}(\tau), y_c) d\tau$ 从 $Z_1$ 到 $Z_0$，移除当前类别信息；(b) **Land（降落）**：沿正向积分 $\int_0^t \gamma_{\text{land}} v_\psi(\tau, z^{\hat{y}_c}(\tau), \hat{y}_c) d\tau$ 从 $Z_0$ 到 $Z_1$，注入目标类别信息。通过调节步长 $\gamma$ 实现三种操作模式。

3. **信息混合与信息注入**: 

    - **信息混合(Blending)**：设 $\gamma_b = \gamma_{b,\text{lift}} = \gamma_{b,\text{land}} < 1$，反事实样本混合源类和目标类特征，产生局部反事实。混合在目标类被到达时自动停止。
    - **信息注入(Injection)**：设 $\gamma_{i,\text{lift}} < \gamma_{i,\text{land}}$，在已到达目标类后继续注入目标类信息，使反事实样本更深入目标类的数据分布。这是生成"可靠"反事实的关键——让样本不仅跨越学习边界，还接近真实决策边界。

### 损失函数 / 训练策略

训练阶段仅需优化CE-CFM目标函数。流匹配模型可以很轻量（如4层MLP），训练后用于推理。推理阶段通过组合 $N_b$ 次混合Leap和 $N_i$ 次注入Leap生成反事实样本。建议从小步长、多次Leap开始。

## 实验关键数据

### 主实验

**Morpho-MNIST反事实生成质量**

| 方法 | ACC↑ | AUC↑ | D(Area)↓ | D(Thickness)↓ | D(Height)↓ |
|------|------|------|----------|---------------|------------|
| Opt-based | 0.828 | 0.881 | 0.248 | 0.172 | 0.062 |
| CGM-based | 0.942 | 0.998 | 0.256 | 0.086 | 0.029 |
| LeapFactual | 0.987 | 0.999 | **0.167** | **0.081** | **0.027** |
| LeapFactual_R | **0.991** | **1.000** | 0.230 | 0.090 | 0.030 |

LeapFactual同时在正确性和相似性指标上领先，LeapFactual_R（含信息注入）进一步提升正确性。

### 消融实验

**Galaxy10数据集 - 可靠反事实用于模型改进**

| 训练配置 | CE比例 | ACC↑ | AUC↑ | 说明 |
|----------|--------|------|------|------|
| 基线(20%数据) | - | 0.811 | 0.977 | - |
| 基线(100%数据) | - | 0.853 | 0.981 | - |
| + 标准CE | 100% | 0.797 | 0.974 | 性能下降！CE在学习边界附近 |
| + 可靠CE | 10% | 0.816 | 0.978 | 性能提升 |
| + 可靠CE | 100% | **0.824** | **0.979** | 接近100%数据基线 |

标准反事实作为训练数据会损害性能，而可靠反事实持续改善模型——验证了可靠性的重要价值。

**FFHQ高分辨率实验（1024×1024, 非可微分类器CLIP）**

| Nb | ACC↑ | SSIM↑ | LPIPS↓ |
|----|------|-------|--------|
| 5 | 0.706 | 0.564 | 0.149 |
| 10 | 0.957 | 0.538 | 0.171 |
| 20 | 0.993 | 0.525 | 0.180 |
| 随机配对 | - | 0.070 | 0.555 |

### 关键发现

- LeapFactual**模型无关**：无需分类器可微分，可用CLIP代理人类标注，扩展CE到公民科学等需要人工标注的领域
- 可靠反事实不仅解释性更强，还可作为**数据增强**改善模型性能，这是标准反事实做不到的
- 信息混合过程中可追踪分类器预测变化路径，观察类别切换过程（如从蓝色经过黄色到达红色）

## 亮点与洞察

- **理论支撑扎实**：通过d-separation论证和信息论定理证明了CE-CFM的合理性，而非纯经验方法
- **统一框架**：同时具备Opt方法的连续潜在空间和CGM方法的结构化表示，解决了两类方法各自的痛点
- **可靠性概念新颖**：区分"跨越学习边界"和"接近真实边界"，通过信息注入实现后者
- 玩具实验的可视化非常直观，清晰展示了信息替换、混合、注入三种模式的区别

## 局限与展望

- 高维潜在空间（如扩散模型、归一化流）中流匹配模型的训练成本显著增加
- 仅聚焦视觉数据，虽然理论上可推广到其他模态，但未实际验证
- 未探索OT-CFM等更高效的流匹配变体（因为最优传输需要修改分类器预测上的传输映射）
- 潜在空间纠缠和不平衡数据集场景有待探索

## 相关工作与启发

- 可靠反事实作为数据增强的发现值得深入，或许可以与主动学习结合，识别模型最不确定的区域
- 流匹配在其他可解释性任务（如概念编辑、属性操控）中可能有类似应用
- 模型无关特性使其可用于黑盒API服务的解释

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将条件流匹配引入反事实解释，可靠性概念和Leap机制设计精巧
- 实验充分度: ⭐⭐⭐⭐ 涵盖基准数据集、真实天文数据集、高分辨率人脸，但基线对比方法偏少
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机阐述清晰，理论推导完整，可视化出色
- 价值: ⭐⭐⭐⭐ 解决了反事实解释的核心可靠性问题，可靠CE用于数据增强的发现有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] V-CECE: Visual Counterfactual Explanations via Conceptual Edits](v-cece_visual_counterfactual_explanations_via_conceptual_edits.md)
- [\[NeurIPS 2025\] Improving Posterior Inference of Galaxy Properties with Image-Based Conditional Flow Matching](improving_posterior_inference_of_galaxy_properties_with_image-based_conditional_.md)
- [\[ICCV 2025\] Looking in the Mirror: A Faithful Counterfactual Explanation Method for Interpreting Deep Image Classification Models](../../ICCV2025/image_generation/looking_in_the_mirror_a_faithful_counterfactual_explanation_method_for_interpret.md)
- [\[NeurIPS 2025\] Flow Matching Neural Processes](flow_matching_neural_processes.md)
- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](counterfactual_identifiability_via_dynamic_optimal_transport.md)

</div>

<!-- RELATED:END -->
