---
title: >-
  [论文解读] EasyInv: Toward Fast and Better DDIM Inversion
description: >-
  [ICML 2025][LLM效率][DDIM Inversion] 提出 EasyInv，通过在反演过程中周期性地将当前 latent 状态与前一步 latent 状态加权聚合（类卡尔曼滤波），增强初始 latent 的影响力、抑制噪声累积误差，在不需要迭代优化的前提下达到与迭代方法相当甚至更好的反演质量，同时推理速度提升约 3 倍。
tags:
  - "ICML 2025"
  - "LLM效率"
  - "DDIM Inversion"
  - "扩散模型"
  - "图像编辑"
  - "卡尔曼滤波"
  - "推理加速"
---

# EasyInv: Toward Fast and Better DDIM Inversion

**会议**: ICML 2025  
**arXiv**: [2408.05159](https://arxiv.org/abs/2408.05159)  
**代码**: [potato-kitty/EasyInv](https://github.com/potato-kitty/EasyInv)  
**领域**: 扩散模型/图像反演  
**关键词**: DDIM Inversion, 扩散模型, 图像编辑, 卡尔曼滤波, 推理加速

## 一句话总结

提出 EasyInv，通过在反演过程中周期性地将当前 latent 状态与前一步 latent 状态加权聚合（类卡尔曼滤波），增强初始 latent 的影响力、抑制噪声累积误差，在不需要迭代优化的前提下达到与迭代方法相当甚至更好的反演质量，同时推理速度提升约 3 倍。

## 研究背景与动机

DDIM Inversion 是扩散模型进行真实图像编辑的基础技术：给定一张真实图像，通过反转去噪过程将其映射回噪声空间，再从该噪声出发进行可控的去噪生成/编辑。然而，DDIM Inversion 本质上是一种近似过程，反演时每一步预测的噪声 $\varepsilon_t^*$ 与采样时期望的噪声 $\varepsilon_t$ 之间存在偏差，误差会逐步累积，导致重建质量下降。

现有改进方法主要聚焦于**迭代优化噪声估计**：

- **ReNoise**：在每个时间步内反复加噪→去噪，用最终去噪结果作为下一次迭代的输入。但面对特殊输入时会输出全黑图像。
- **Fixed-Point Iteration (Pan et al.)**：从不动点理论出发，在每步混合相邻迭代的噪声估计。但在精度有限的模型（如 SD-V1.4）上效果不佳，且需多次前向传播，效率低。
- **Null-Text Inversion / PTI**：在重建分支中优化条件向量，计算开销大。

这些方法共同的问题是：**(1) 计算效率低**——每个时间步需要多次模型前向推理；**(2) 鲁棒性不足**——当模型精度（如 float16）或模型能力有限时，迭代优化噪声不一定收敛到好的结果，累积误差反而可能更严重。

EasyInv 从根本上转换了视角：**不去迭代优化每步的噪声，而是直接增强初始 latent 状态在整个反演过程中的影响力**。

## 方法详解

### 整体框架

EasyInv 的核心思想非常简洁：在标准 DDIM Inversion 的基础上，在选定的时间步 $\bar{t}$ 处，将当前反演得到的 latent 状态 $\mathbf{z}_{\bar{t}}^*$ 与前一步的 latent 状态 $\mathbf{z}_{\bar{t}-1}^*$ 进行加权聚合：

$$\mathbf{z}_{\bar{t}}^{*} = \eta \cdot \mathbf{z}_{\bar{t}}^{*} + (1-\eta) \cdot \mathbf{z}_{\bar{t}-1}^{*}$$

其中 $\eta$ 是权衡参数。这一操作不需要额外的模型前向传播，因此几乎不增加计算开销。

**算法流程**（Algorithm 1）：

1. 输入：反演算法 $Inv()$、总反演步数 $T$、初始 latent $z_0$、选定步集合 $\bar{t}$、参数 $\eta$
2. 对每个时间步 $t = 0 \to T-1$：
    - 执行标准反演：$z_{t+1} = Inv(z_t, t)$
    - 若 $t \in \bar{t}$：$z_{t+1} = \eta \cdot z_{t+1} + (1-\eta) \cdot z_t$
3. 输出：反演后的噪声 latent $z_T$

### 关键设计

#### 1. 从噪声优化到 latent 聚合的视角转换

论文首先推导出反演过程的展开式。令 $\bar{\alpha}_t = \sqrt{\alpha_t / \alpha_{t-1}}$，$\bar{\beta}_t$ 为对应的噪声系数，则：

$$\mathbf{z}_t^* = \bar{\alpha}_t \mathbf{z}_{t-1}^* + \bar{\beta}_t \varepsilon_t^*$$

递归展开到初始状态：

$$\mathbf{z}_t^* = \left(\prod_{i=1}^{t} \bar{\alpha}_i\right) \mathbf{z}_0^* + \sum_{i=1}^{t} \left(\bar{\beta}_i \prod_{j=i+1}^{t} \bar{\alpha}_j\right) \varepsilon_i^*$$

这表明 $\mathbf{z}_t^*$ 是初始 latent $\mathbf{z}_0^*$ 与一系列噪声项的加权和。之前的方法专注于优化每步噪声 $\varepsilon_i^*$，但 EasyInv 选择**直接增强 $\mathbf{z}_0^*$ 的权重**，通过混入前一步的 latent（其中包含更多初始信息）来压制噪声累积。

#### 2. 卡尔曼滤波理论支撑

论文将 EasyInv 与卡尔曼滤波建立了理论联系。在经典卡尔曼滤波中，系统通过融合预测值 $\bar{x}_k$ 和观测值 $y_k$ 来获得最优估计：

$$\tilde{x}_k = \bar{x}_k + K(y_k - H\bar{x}_k)$$

将其应用到反演问题中：

- **预测值** $\bar{x}_k \to \mathbf{z}_{\bar{t}}^*$：当前步 DDIM Inversion 的输出
- **观测值** $y_k \to \mathbf{z}_{\bar{t}-1}^*$：前一步的 latent 状态（因为 $z_t - z_{t-1}$ 的差异可视为高斯噪声）
- **卡尔曼增益** $K \to \eta$：论文将其简化为经验常数

由此得到：$\mathbf{z}_{\bar{t}}^* = \eta \cdot \mathbf{z}_{\bar{t}}^* + (1-\eta) \cdot \mathbf{z}_{\bar{t}-1}^*$

这本质上是一个**简化的卡尔曼滤波器**，省略了动态计算卡尔曼增益的步骤，用固定的 $\eta$ 代替。

#### 3. 噪声分布保持特性

从聚合操作的角度看，$\mathbf{z}_{\bar{t}}^* - \mathbf{z}_{\bar{t}-1}^*$ 的分布与原始 DDIM Inversion 的过渡分布保持相同方差方向，但均值被缩小了 $\eta$ 倍。这意味着 EasyInv **保持了原始方法的噪声模式，只是应用了一个重缩放因子**，不会引入分布外的扰动。

#### 4. 超参数选择

- **聚合时间步 $\bar{t}$**：选择在 $0.05T < \bar{t} < 0.25T$ 范围内的步骤执行聚合操作。反演初期距离初始 latent 最近，此阶段注入纠正信号效果最好。
- **权衡参数 $\eta$**：默认设为 0.5，即当前状态和前一步状态各占一半权重。
- **总步数 $T$**：默认 50 步。

### 损失函数 / 训练策略

EasyInv 是一种**无需训练的 (training-free) 推理时方法**。它不涉及任何损失函数或梯度更新，仅在标准 DDIM Inversion 流程中插入了简单的线性加权操作。这使得它可以**即插即用地与任意现有反演方法组合**（如与 DirectInv 结合，即论文中的 "Ours+DirectInv"）。

## 实验关键数据

### 主实验

在 COCO 2017 测试/验证集上采样 2,298 张图像，使用 SD V1.4，GTX 3090 GPU，float16 精度：

| 方法 | LPIPS↓ | SSIM↑ | PSNR↑ | 推理时间↓ |
|------|--------|-------|-------|-----------|
| DDIM Inversion | 0.328 | 0.621 | 29.717 | 5s |
| ReNoise | 0.316 | 0.641 | 31.025 | 16s |
| Fixed-Point Iteration | 0.373 | 0.563 | 29.107 | 14s |
| **EasyInv (Ours)** | **0.321** | **0.646** | **30.189** | **5s** |

### 消融实验

| 配置 | LPIPS↓ | SSIM↑ | PSNR↑ | 时间↓ | 说明 |
|------|--------|-------|-------|-------|------|
| Full Precision (float32) | 0.321 | 0.646 | 30.184 | 9s | 全精度基线 |
| Half Precision (float16) | 0.321 | 0.646 | 30.189 | 5s | 半精度无质量损失，速度提升44% |

### 下游编辑任务对比

| 反演方法 | 编辑方法 | Structure Dist.↓ | PSNR↑ | LPIPS↓ | SSIM↑ |
|----------|----------|------------------|-------|--------|-------|
| DDIM | P2P | 69.43 | 17.87 | 208.80 | 71.14 |
| DirectInv | P2P | 11.65 | 27.22 | 54.55 | 84.76 |
| **Ours+DirectInv** | P2P | **11.58** | **27.30** | **53.52** | **84.80** |
| DDIM | PnP | 28.22 | 22.28 | 113.46 | 79.05 |
| DirectInv | PnP | 24.29 | 22.46 | 106.06 | 79.68 |
| **Ours+DirectInv** | PnP | **22.88** | **22.56** | **102.34** | **80.27** |

### 关键发现

1. **速度优势显著**：EasyInv 推理时间与 vanilla DDIM Inversion 相同（5s），比迭代方法快约 3 倍
2. **低精度鲁棒**：float16 下质量完全不降，而 Fixed-Point Iteration 在低精度/弱模型上会崩溃
3. **即插即用**：与 DirectInv 组合后，在所有下游编辑任务上都获得一致提升
4. **收敛更快**：latent 状态可视化显示，在去噪中间步骤 EasyInv 的 latent 已更接近原图

## 亮点与洞察

1. **极简设计，理论扎实**：核心操作只有一行代码（线性插值），但通过展开式分析和卡尔曼滤波视角给出了清晰的理论解释
2. **视角转换的力量**：从"优化每步噪声"转向"增强初始 latent 影响力"，是一个优雅的思路转换
3. **效率与质量的帕累托改进**：同时在质量（SSIM 最优）和速度（最快）上取得优势
4. **即插即用的通用性**：不修改底层模型或反演框架，可叠加到任何现有反演方法之上
5. **对半精度友好**：在实际部署中半精度是默认选择，EasyInv 在半精度下表现稳定是重要的实用优势

## 局限与展望

1. **固定卡尔曼增益**：$\eta$ 设为常数 0.5，未能根据每步的误差动态调整。若引入自适应 $\eta(t)$，可能进一步提升性能
2. **聚合步范围的经验性**：$0.05T < \bar{t} < 0.25T$ 的选择缺乏严格理论推导，不同数据/模型下可能需要重新调参
3. **仅验证了 SD 系列模型**：未在 DiT 等新架构上验证，泛化性存疑
4. **编辑质量上限**：在 CLIP Similarity 等编辑保真度指标上改进幅度较小
5. **理论分析可深化**：卡尔曼滤波类比仍较粗糙，若能给出最优 $\eta$ 的闭式解或误差界，理论贡献会更强

## 相关工作与启发

- **DDIM Inversion (Couairon et al., 2023)**：本工作的基础框架
- **ReNoise (Garibi et al., 2024)**：迭代加噪/去噪的反演优化方法
- **Fixed-Point Iteration (Pan et al., 2023)**：将反演噪声优化形式化为不动点问题
- **PNPInversion / DirectInv (Ju et al., 2024)**：在重建分支计算 latent 间距离进行校正
- **Null-Text Inversion (Mokady et al., 2023)**：优化空文本嵌入来改善重建
- **启发**：简单的 latent 空间插值操作（类似 EMA 思想）在扩散模型的各个环节都可能有效，EasyInv 将其推广到反演过程中

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 视角转换新颖，但核心操作（线性插值）过于简单 |
| 理论深度 | 3.5 | 卡尔曼滤波类比有启发性，但分析停留在类比层面 |
| 实验充分性 | 4 | 主实验+精度对比+下游任务，覆盖全面 |
| 实用价值 | 4.5 | 即插即用、零额外开销、半精度友好，部署价值高 |
| 写作质量 | 3.5 | 公式推导清晰，但部分段落冗长 |
| **综合** | **4** | 简单有效的方法，工程价值突出 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](../../ACL2025/llm_efficiency/smarter_better_faster_longer_a_modern_bidirectional_encoder_for_fast_memory_effi.md)
- [\[NeurIPS 2025\] Advancing Expert Specialization for Better MoE](../../NeurIPS2025/llm_efficiency/advancing_expert_specialization_for_better_moe.md)
- [\[NeurIPS 2025\] SkyLadder: Better and Faster Pretraining via Context Window Scheduling](../../NeurIPS2025/llm_efficiency/skyladder_better_and_faster_pretraining_via_context_window_scheduling.md)
- [\[ICML 2025\] Mixture of Lookup Experts](mixture_of_lookup_experts.md)
- [\[ICML 2025\] MoH: Multi-Head Attention as Mixture-of-Head Attention](moh_multi-head_attention_as_mixture-of-head_attention.md)

</div>

<!-- RELATED:END -->
