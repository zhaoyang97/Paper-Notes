---
title: >-
  [论文解读] Learning Robust Stereo Matching in the Wild with Selective Mixture-of-Experts
description: >-
  [ICCV 2025][3D视觉][立体匹配] 提出 SMoEStereo，通过在冻结的视觉基础模型(VFM)中集成变秩MoE-LoRA和变核MoE-Adapter，结合轻量决策网络选择性激活MoE模块，实现了场景自适应的鲁棒立体匹配，在跨域和联合泛化上达到SOTA。 立体匹配是计算机视觉中的核心任务…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "立体匹配"
  - "视觉基础模型"
  - "混合专家"
  - "LoRA"
  - "跨域泛化"
---

# Learning Robust Stereo Matching in the Wild with Selective Mixture-of-Experts

**会议**: ICCV 2025  
**arXiv**: [2507.04631](https://arxiv.org/abs/2507.04631)  
**代码**: [GitHub](https://github.com/cocowy1/SMoE-Stereo)  
**领域**: 3D视觉  
**关键词**: 立体匹配, 视觉基础模型, 混合专家, LoRA, 跨域泛化

## 一句话总结

提出 SMoEStereo，通过在冻结的视觉基础模型(VFM)中集成变秩MoE-LoRA和变核MoE-Adapter，结合轻量决策网络选择性激活MoE模块，实现了场景自适应的鲁棒立体匹配，在跨域和联合泛化上达到SOTA。

## 研究背景与动机

立体匹配是计算机视觉中的核心任务，广泛应用于自动驾驶、机器人导航和增强现实。近年来基于学习的方法虽然在标准基准上表现出色，但**跨域泛化能力有限**：

**核心挑战**：

**域偏移**：不同数据集间存在显著的场景差异和不平衡的视差分布

**噪声特征图**：域偏移可能导致嘈杂和扭曲的特征图，损害模型鲁棒性

**为什么用VFM？** 视觉基础模型（如DINOv2、SAM、DepthAnything）在大规模多样数据上预训练，具备提取鲁棒通用特征的能力。但直接应用存在两个关键问题：

**零样本性能有限**：VFM擅长语义信息提取（分割/分类），但缺乏为精确相似性度量所需的判别性特征

**固定微调不灵活**：固定秩的LoRA或固定CNN解码器对所有输入采用统一处理，无法适应野外场景的异质性

**本文方案**：让LoRA的秩和CNN Adapter的核大小都变为动态可选的MoE专家，实现场景特定的特征适配。

## 方法详解

### 整体框架

以RAFT-Stereo为骨干，用VFM替换其特征提取器，在ViT块中嵌入两种MoE模块 + 决策网络：

**VFM特征提取** → **MoE LoRA层**（注意力部分）+ **MoE Adapter层**（MLP部分）→ 浅层CNN压缩 → 相关性体积 → GRU迭代更新视差

### MoE-LoRA层（变秩专家）

每个MoE-LoRA层包含 $M$ 个LoRA专家，每个专家对应不同矩阵秩 $r_i$：

$$x_{\text{out}} = W_{q,k,v} x_{\text{in}} + \sum_{i=1}^{M} R_L(x_{\text{in}}) \cdot E_L^i(x_{\text{in}})$$

其中每个专家 $E_L^i(x) = W_i^{\text{up}} W_i^{\text{down}} x$，路由器通过Top-k选择最优专家：

$$R(x) = \text{Topk}\left(\frac{\exp(W^{\text{router}} x / \tau)}{\sum_k \exp(W^{\text{router}} x / \tau)}, k\right)$$

**设计动机**：不同场景需要不同低秩子空间——简单场景用低秩，复杂纹理场景用高秩。

### MoE-Adapter层（变核专家）

嵌入多个CNN Adapter专家，使用不同卷积核大小 $k$ 捕获不同感受野的局部几何结构：

$$E_A^j(x) = \text{Conv}_{1\times1}^{\text{up}}(\text{Conv}_{k\times k}^j(\text{Conv}_{1\times1}^{\text{down}}(x)))$$

**互补设计**：CNN分支强调细粒度局部几何细节，LoRA路径建模长程交互，D1误差降低最高30%。

### 决策网络（选择性激活）

轻量MLP预测每个MoE层的二值激活决策，使用Gumbel Softmax实现端到端训练：

$$\mathcal{L}_{\text{usage}} = \left(\frac{1}{L}\sum_{l=1}^L M_L^l - \gamma\right)^2 + \left(\frac{1}{L}\sum_{l=1}^L M_A^l - \gamma\right)^2$$

超参数 $\gamma \in (0,1]$ 控制计算预算。

### 损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{disp}} + \lambda_1 \mathcal{L}_{\text{blc}} + \lambda_2 \mathcal{L}_{\text{usage}}$$

- $\mathcal{L}_{\text{disp}}$：L1视差损失，指数加权叠代预测
- $\mathcal{L}_{\text{blc}}$：MoE专家平衡损失，防止负载不均

## 实验

### 跨域泛化（主实验）

| 方法 | KIT12 Bad3.0 | KIT15 Bad3.0 | Middle Bad2.0 | ETH3D Bad1.0 |
|:---|:---:|:---:|:---:|:---:|
| RAFT-Stereo | 5.1 | 5.7 | 12.6 | 3.3 |
| LoS | 4.4 | 5.5 | 19.6 | 3.1 |
| Former-RAFT‡ (DAM) | 3.9 | 5.1 | 8.1 | 3.3 |
| **SMoEStereo (DAMV2)** | **4.22** | **4.86** | **7.05** | **2.10** |

### 消融实验：组件有效性

| 消融设置 | 关键发现 |
|:---|:---|
| 固定LoRA vs MoE-LoRA | MoE-LoRA通过场景自适应秩选择提升性能 |
| 无Adapter vs MoE-Adapter | 注入归纳偏置显著改善几何特征提取 |
| 全MoE vs 选择性MoE | 决策网络减少冗余计算同时保持精度 |
| VFM容量对比 | ViT-Base已达满意效果，参数量仅为ViT-Large的1/3 |

### 效率对比

| 方法 | 容量 | 显存(GB) | 时间(s) | 额外参数(M) |
|:---|:---:|:---:|:---:|:---:|
| Former-RAFT (DAM)† | ViT-Large | 4.1 | 0.47 | 6.9 |
| **SMoEStereo (DAM)** | ViT-Base | **1.9** | **0.18** | **2.86** |

### 关键发现

1. SMoEStereo在4个基准上均达到SOTA跨域泛化，同一模型无需数据集特定适配
2. 相比VFM-LoRA基线，D1误差最高降低30%
3. 决策网络可灵活控制计算预算（$\gamma$从0.3到1.0），适应不同资源约束
4. DrivingStereo恶劣天气评估：平均D1从5.0降至4.3

## 亮点与洞察

1. **异质MoE设计**：与传统同质MoE不同，LoRA专家用变秩、Adapter专家用变核，巧妙利用了秩和感受野两个维度的互补性
2. **选择性激活**：决策网络的二值决策机制实现了精度-效率的帕累托最优
3. **即插即用**：SMoE可作为插件模块集成到大多数立体网络中

## 局限性

1. 路由器训练需要足够多样的训练数据才能学习有意义的专家分配策略
2. 更极端的域偏移（如水下、医学立体图像）未验证
3. MoE引入的额外参数虽少但增加了模型复杂度

## 相关工作

- **鲁棒立体匹配**：CFNet, CREStereo++, LoS, Selective-IGEV
- **VFM微调**：LoRA, AdaptFormer, VPT
- **MoE**：Sparse MoE, LoRA-MoE融合

## 评分

- 新颖性：⭐⭐⭐⭐ — 首次将异质MoE+选择性激活应用于VFM立体匹配
- 技术深度：⭐⭐⭐⭐ — 变秩LoRA+变核Adapter+决策网络设计完整
- 实验完整性：⭐⭐⭐⭐⭐ — 多基准、多VFM、多消融、效率对比全面
- 实用价值：⭐⭐⭐⭐ — 代码开源，效率优势明显，适合实际部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)
- [\[ICCV 2025\] BANet: Bilateral Aggregation Network for Mobile Stereo Matching](banet_bilateral_aggregation_network_for_mobile_stereo_matching.md)
- [\[ICCV 2025\] Stereo Any Video: Temporally Consistent Stereo Matching](stereo_any_video_temporally_consistent_stereo_matching.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](../../CVPR2025/3d_vision/defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[ICCV 2025\] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)

</div>

<!-- RELATED:END -->
