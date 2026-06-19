---
title: >-
  [论文解读] Earth-Adapter: Bridge Geospatial Domain Gaps with Mixture of Frequency Adaptation
description: >-
  [AAAI 2026][模型压缩][遥感语义分割] 提出 Earth-Adapter，首个针对遥感图像**伪影问题**设计的参数高效微调 (PEFT) 方法，通过频率引导的混合适配器 (MoA) 将特征分解为高低频子空间、独立优化后动态聚合，在遥感语义分割 (SS)、域自适应 (DA) 和域泛化 (DG) 三个设定中均超越基线 Rein。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "遥感语义分割"
  - "参数高效微调"
  - "频域分解"
  - "混合适配器"
  - "伪影缓解"
---

# Earth-Adapter: Bridge Geospatial Domain Gaps with Mixture of Frequency Adaptation

**会议**: AAAI 2026  
**arXiv**: [2504.06220](https://arxiv.org/abs/2504.06220)  
**代码**: [有](https://github.com/VisionXLab/Earth-Adapter)  
**领域**: 模型压缩  
**关键词**: 遥感语义分割, 参数高效微调, 频域分解, 混合适配器, 伪影缓解

## 一句话总结

提出 Earth-Adapter，首个针对遥感图像**伪影问题**设计的参数高效微调 (PEFT) 方法，通过频率引导的混合适配器 (MoA) 将特征分解为高低频子空间、独立优化后动态聚合，在遥感语义分割 (SS)、域自适应 (DA) 和域泛化 (DG) 三个设定中均超越基线 Rein。

## 研究背景与动机

视觉基础模型 (VFMs) 如 DINOv2 在自然图像上表现出色，但在遥感 (RS) 语义分割任务中结合现有 PEFT 方法时性能显著下降。作者发现根本原因在于：

**VFM 特征中普遍存在的伪影 (artifacts)**。通过 PCA 可视化 DINOv2-L 的特征图，可以观察到明显的冗余伪影。关键区别在于：

- **自然图像**中伪影通常围绕前景物体（如人、动物），干扰相对有限
- **遥感图像**由于俯视视角，缺少集中主体，包含多个共存的多尺度目标（如大规模农田和碎片化道路网络），导致**伪影几乎无处不在**，严重干扰像素级特征提取

现有 PEFT 方法（LoRA、VPT、AdaptFormer 等）均未解决这一问题。例如 LoRA 在 P2V (DA) 上仅获 17.8% mIoU，甚至低于冻结 DINOv2-L。

核心策略：**分而治之 (Divide and Conquer)** — 先用频域分解隔离伪影相关信息，再独立优化不同频率子空间。

## 方法详解

### 整体框架

Earth-Adapter 嵌入到 DINOv2-L (冻结) + Mask2Former 的架构中，由两个核心组件构成：

1. **MoA (Mixture of Adapters)**: 包含空间适配器 + 低频适配器 + 高频适配器
2. **Router (动态路由器)**: 根据原始特征自适应分配三个适配器的聚合权重

优化目标从标准微调的 $\arg\min_\theta$ 扩展为 $\arg\min_{\theta,\epsilon,\xi}$，其中 $\epsilon$ 和 $\xi$ 分别是适配器和路由器参数，冻结 VFM 骨干。

### 关键设计

#### 混合适配器 (MoA)

给定骨干第 $i$ 层特征 $\mathbf{F}_i \in \mathbb{R}^{(hw) \times c}$：

**空间适配器**：直接对空间特征进行低秩投影：
$$\Delta \mathbf{F}_i^{spatial} = \text{Adapter}_1^i(\mathbf{F}_i^T)$$
适配器结构：$\text{Adapter} = W_{up}(\text{ReLU}(W_{down}(\cdot)))$，标准 bottleneck。

**频率分解**：将空间特征 reshape 为 $(C, H, W)$，施加 2D DFT，用固定截止频率 $\rho$ 和频率掩码 $\mathbf{M}$ 分离高低频：
$$\mathbf{F}_i^{low} = \mathcal{FT}^{-1}(\mathbf{M} \odot \mathcal{FT}(\mathbf{F}_{spatial}))$$
$$\mathbf{F}_i^{high} = \mathcal{FT}^{-1}((1-\mathbf{M}) \odot \mathcal{FT}(\mathbf{F}_{spatial}))$$

频率掩码定义：以频域中心为参考，距中心 $\leq \rho \frac{H}{2}$ 的区域为低频，其余为高频。

**低频/高频适配器**：分别对低频和高频特征独立处理：
$$\Delta \mathbf{F}_i^{low} = \text{Adapter}_2^i(\mathbf{F}_i^{low}), \quad \Delta \mathbf{F}_i^{high} = \text{Adapter}_3^i(\mathbf{F}_i^{high})$$

"分"的核心逻辑：高频信号捕获局部细节（伪影主要集中在此），低频信号编码全局结构。分离后可针对性地处理伪影。

#### 动态路由器 (Router)

通过通道注意力机制学习最优特征组合权重：
$$\mathbf{w}_i = \text{Softmax}(R_\xi(\mathbf{F}_i))$$

最终特征调整（"治"）：
$$\Delta \mathbf{F}_i = \alpha_i \sum_{k=1}^{3} \mathbf{w}_i^{(k)} \Delta \mathbf{F}_i^{(k)}$$

其中 $\alpha_i$ 为可学习缩放参数（初始值小），$k \in \{spatial, low, high\}$。

通过残差连接融合冻结特征和精炼特征：$\bar{\mathbf{F}_i} = \mathbf{F}_i + \Delta \mathbf{F}_i$。

### 损失函数 / 训练策略

- **SS 和 DG**：标准 Mask2Former 端到端监督训练，CE loss
- **DA**：使用 DACS 自训练框架，EMA 教师生成目标域伪标签，混合源域和目标域训练
- 优化器：AdamW，骨干 lr=1e-5，解码器和 PEFT 参数 lr=1e-4
- 仅训练适配器参数 $\epsilon$、路由器参数 $\xi$ 和解码器参数 $\theta$，骨干完全冻结

## 实验关键数据

### 主实验

**域自适应 (DA) + 域泛化 (DG)**（4 个 RS 数据集：Potsdam, Vaihingen, LoveDA, iSAID）：

| 方法 | P2V DA | V2P DA | R2U DA | U2R DA | DA 均值 | P2V DG | V2P DG | R2U DG | U2R DG | DG 均值 |
|------|--------|--------|--------|--------|---------|--------|--------|--------|--------|---------|
| Frozen DINOv2-L | 21.0 | 7.2 | 21.5 | 11.8 | 15.4 | 57.9 | 49.4 | 57.1 | 42.7 | 51.8 |
| LoRA | 17.8 | 18.8 | 24.5 | 26.0 | 15.7 | 20.3 | 25.6 | 29.3 | 21.9 | 24.3 |
| VPT | 66.2 | 59.3 | 55.3 | 48.0 | 57.2 | 59.2 | 52.3 | 57.4 | 44.9 | 53.5 |
| Rein (baseline) | 60.2 | 60.9 | 52.8 | 26.0 | 50.0 | 60.8 | 52.5 | 55.8 | 43.4 | 53.1 |
| **Earth-Adapter** | **67.7** | **62.2** | **55.9** | **50.0** | **59.0** | **64.9** | **55.1** | **59.0** | **45.7** | **56.2** |

DA 均值超越 Rein **+9.0%**，DG 均值超越 Rein **+3.1%**。U2R DA 上提升最为惊人：+24.0%。

**语义分割 (SS)**：

| 方法 | Potsdam | Vaihingen | LoveDA | iSAID | 均值 |
|------|---------|-----------|--------|-------|------|
| Rein | 76.2 | 70.8 | 54.9 | 68.4 | 67.6 |
| **Earth-Adapter** | **76.7** | **71.7** | **56.9** | **69.8** | **68.8** |

SS 均值超越 Rein **+1.2%**。

### 消融实验

适配器组合消融（DG 设定）：

| Spatial | + HF | + LF | P2V DG | V2P DG | 均值 |
|---------|------|------|--------|--------|------|
| 1 | - | - | 61.0 | 52.8 | 56.9 |
| 3 | - | - | 64.0 | 52.7 | 58.4 (+1.5) |
| **1** | **1** | **1** | **64.9** | **55.1** | **60.0 (+3.1)** |

单纯增加空间适配器数量（3 个）只带来 +1.5%，而 1 空间+1 高频+1 低频带来 +3.1%，证明**频域分解比简单增加适配器数量更有效**。

效率对比：

| 方法 | 可训练参数 | DA 均值 mIoU |
|------|-----------|-------------|
| Full Fine-Tune | 304.2M | 15.4 |
| Rein | 3.0M | 50.0 |
| **Earth-Adapter** | **2.6M~9.6M** | **59.0** |

不同骨干消融：DINOv2-S/B/L 均有一致提升；在遥感预训练 VFM (MTP, ScaleMAE, DOFA) 上也有效。

### 关键发现

- **LoRA 和 Full Fine-Tune 在 RS 中灾难性失败**：LoRA DA 均值仅 15.7%，Full Fine-Tune 15.4%，说明自然图像的 PEFT 策略直接迁移到 RS 行不通
- **频域分解是关键**：高频中隔离伪影、低频中保留语义结构，分治策略显著优于同构适配器堆叠
- **DINOv2 优于 RS 预训练 VFM**：RS 预训练模型训练数据规模有限，DINOv2 + Earth-Adapter 效果更好
- **冻结 DINOv2 的 DG 能力不弱**：DG 设定下 Frozen 已达 51.8%，优于 LoRA/AdaptFormer，说明不当微调反而损害泛化

## 亮点与洞察

1. **首个针对 RS 伪影的 PEFT 方法**：精准定位了 VFM 在遥感场景中性能下降的原因——高维特征中的伪影干扰
2. **频率引导的分治策略**：DFT 分离 + 独立适配 + 动态路由，简洁优雅且有效
3. **极致的参数效率**：仅 2.6M~9.6M 可训练参数即可在 DA 上获得 +9.0% 的巨大提升
4. **统一三个设定**：一个方法同时适用于 SS、DA 和 DG，展示了良好的通用性

## 局限与展望

- 频率截止参数 $\rho$ 固定，未自适应调整，不同场景的最优频率分界点可能不同
- 仅在遥感图像上验证，自然图像中伪影特征不同，方法的泛化范围有待验证
- 路由器权重可视化和定量分析在论文中较少，MoA 的决策过程可解释性不足
- 依赖 DINOv2 作为骨干，未探索其他 VFM（如 InternViT、SigLIP）的情况

## 相关工作与启发

- **Rein** (Wei et al. 2024)：首个将 PEFT 用于 DG 的工作，是本文的直接基线
- **DINOv2 + Register Tokens** (Darcet et al. 2024)：通过注册 token 缓解伪影，但效果有限（DG +1.7% vs Earth-Adapter +3.1%）
- **DACS** (Tranheden et al. 2021)：DA 训练框架，本文沿用其自训练范式
- **频域方法在视觉中的复兴**：DFT 分解在风格迁移、域适应、特征去噪等方向持续发力

## 评分

- 新颖性: ⭐⭐⭐⭐ — 精准定位 RS 伪影问题 + 频率引导的 MoA 设计原创性强
- 技术深度: ⭐⭐⭐⭐ — DFT 分解 + 多适配器路由 + 跨域训练框架整合到位
- 实验充分度: ⭐⭐⭐⭐⭐ — 12 个基准、3 种设定、多种骨干、多种 PEFT 对比，极为全面
- 写作质量: ⭐⭐⭐⭐ — 可视化丰富（PCA、预测图对比），动机清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FAST: Topology-Aware Frequency-Domain Distribution Matching for Coreset Selection](../../CVPR2026/model_compression/fast_topology-aware_frequency-domain_distribution_matching_for_coreset_selection.md)
- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](../../ICLR2026/model_compression/freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)
- [\[AAAI 2026\] Group Orthogonal Low-Rank Adaptation for RGB-T Tracking](group_orthogonal_low-rank_adaptation_for_rgb-t_tracking.md)
- [\[ACL 2026\] TalkLoRA: Communication-Aware Mixture of Low-Rank Adaptation for Large Language Models](../../ACL2026/model_compression/talklora_communication-aware_mixture_of_low-rank_adaptation_for_large_language_m.md)

</div>

<!-- RELATED:END -->
