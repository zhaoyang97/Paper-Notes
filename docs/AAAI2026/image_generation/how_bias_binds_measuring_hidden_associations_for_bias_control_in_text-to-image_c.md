---
title: >-
  [论文解读] How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions
description: >-
  [AAAI 2026][图像生成][文本到图像生成] 首次研究文本到图像生成中的**组合语义绑定偏见**问题，提出Bias Adherence Score (BA-Score)量化物体-属性绑定如何激活偏见，并设计免训练的Context-Bias Control (CBC)框架，通过token嵌入解耦和残差注入实现组合生成中超过10%的去偏改善。
tags:
  - "AAAI 2026"
  - "图像生成"
  - "文本到图像生成"
  - "偏见控制"
  - "语义绑定"
  - "组合生成"
  - "公平性"
---

# How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions

**会议**: AAAI 2026  
**arXiv**: [2511.07091](https://arxiv.org/abs/2511.07091)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 文本到图像生成, 偏见控制, 语义绑定, 组合生成, 公平性

## 一句话总结

首次研究文本到图像生成中的**组合语义绑定偏见**问题，提出Bias Adherence Score (BA-Score)量化物体-属性绑定如何激活偏见，并设计免训练的Context-Bias Control (CBC)框架，通过token嵌入解耦和残差注入实现组合生成中超过10%的去偏改善。

## 研究背景与动机

文本到图像（T2I）扩散模型（如Stable Diffusion）常从训练数据中捕获虚假相关性，导致生成结果带有性别、种族等偏见。现有去偏研究的关键盲点：

**仅关注单物体提示**：如"a headshot of an assistant"，现有去偏方法对此类简单提示有效

**忽视组合语义绑定**：当提示变为"a headshot of an assistant wearing a pink hat"时，"pink hat"这个上下文本身携带女性偏见倾向，现有方法无法有效处理

作者的核心发现非常有趣：

- 通过CLIP文本编码器分析embedding相似度，发现"pink"与"woman"原型embedding高度相关，而"hat"则与"man"更相关
- **组合效果**：当"pink"和"hat"组合为"pink hat"时，偏见被放大，推动生成更偏向女性特征
- 使用SOTA去偏方法（SelfDisc、DGDebias、FairQueue）处理组合提示时，产生严重的视觉质量下降：不真实的风格、缺失职业特征、甚至违反安全检查器

这揭示了一个根本性挑战：**减少偏见不能破坏必要的语义关系**。

## 方法详解

### 整体框架

CBC框架包含三个核心步骤：
1. Token语义偏见解耦 — 分离敏感属性成分
2. BA-Score计算 — 量化偏见倾向初始化
3. Token残差注入 — 动态平衡去噪过程中的偏见

### 关键设计

#### 1. **Token语义偏见解耦（Token Semantic Bias Decoupling）**

核心思路：通过Schmidt正交化将上下文token的embedding投影到与敏感属性正交的方向。

对于token embedding $c$ 和敏感属性embedding $s_k$：
$$c^* = c - r_k = c - \frac{\langle c, s_k \rangle}{\langle s_k, s_k \rangle} s_k$$

- $c^*$ 是属性正交embedding：消除了与敏感属性的依赖信息
- $r_k$ 是属性特定残差向量：保留了属性信息，留作后续注入使用

与直接操纵潜空间或使用损失引导的先前方法不同，CBC在生成开始**之前**就解耦了影响，使模型从属性正交的输入embedding形成纯净的注意力关系。

#### 2. **Bias Adherence Score (BA-Score)**

量化上下文token和主体对偏见的贡献百分比。

给定主体embedding $c_m$、上下文token集合 $C = \{c_i\}$、原型embedding $p_k$：
$$B_{m,k} = \frac{\sum_{i=1}^{M} \mathbf{I}_{i \neq m} \exp((\cos(c_m, c_i) + \cos(p_k, c_i))/\tau)}{\sum_{i=1}^M \exp((\cos(c_m, c_i) + \cos(p_k, c_i))/\tau)}$$

BA-Score定义为最大偏离度：$B_m = \max_k |\pi - B_{k,m}|$，其中 $\pi = 0.5$ 代表平衡目标。

高BA-Score意味着主体和上下文在某个属性组上贡献严重不平衡，暗示可能存在强虚假相关，也预示去偏后更不容易保质量。

#### 3. **Token残差注入（Token Residual Injection）**

在扩散去噪的每一步：
- 使用潜在空间BA-Score（基于潜向量与属性聚类中心的距离）测量当前偏见倾向
- 当检测到偏向某属性组 $s_k$ 时，注入其他属性的平均残差embedding：
$$c_t^* = \delta_r \cdot \bar{r} + (1-\delta_r) \cdot c_{t-1}^*$$

其中 $\bar{r} = \frac{1}{K-1}\sum_{j \neq k} r_j$，$\delta_r = 0.2$ 控制注入强度。

**注意力重缩放机制**：为了防止残差注入影响其他组合属性和物体的注意力计算，对被注入token的注意力向量进行缩放：
$$\mathcal{M}_i^* = w(t) \delta_c \mathcal{M}_i$$

其中 $w(t) = 1 - t/T$ 是时间衰减函数，$\delta_c = 2$ 是缩放因子。

### 损失函数 / 训练策略

CBC是**完全免训练**的框架。使用预训练的Stable Diffusion v1.5生成512×512图像，引导尺度7.5，50步采样。BA-Score用于初始化偏见测量，后续步骤通过潜空间聚类中心动态调整。

## 实验关键数据

### 主实验

**基线提示（无组合，如"a head of a [profession]"）**

| 方法 | FD↓ | VQA↑ | AFS↑ | 说明 |
|------|-----|------|------|------|
| SD-1.5 | 0.69 | 0.74 | 0.44 | 原始模型有明显偏见 |
| SelfDisc | 0.62 | 0.61 | 0.47 | 概念编辑效果有限 |
| DGDebias | 0.68 | 0.64 | 0.43 | 分布引导效果有限 |
| FairQueue | 0.03 | 0.33 | 0.49 | 偏见低但质量严重下降 |
| **CBC (Ours)** | **0.04** | **0.68** | **0.80** | 低偏见且高质量 |

**组合提示（如"a head of a [profession] wearing a [color] [object]"）**

| 方法 | FD↓ | VQA↑ | AFS↑ | 说明 |
|------|-----|------|------|------|
| SD-1.5 | 0.69 | 0.63 | 0.41 | 组合使偏见加剧 |
| SelfDisc | 0.42 | 0.56 | 0.58 | 在组合场景失效 |
| DGDebias | 0.73 | 0.62 | 0.35 | 严重过度校正 |
| FairQueue | 0.04 | 0.45 | 0.60 | 偏见低但VQA很低 |
| **CBC (Ours)** | **0.04** | **0.62** | **0.75** | 最佳综合表现 |

### 消融实验

| 配置 | FD↓ | VQA↑ | AFS↑ | 说明 |
|------|-----|------|------|------|
| CBC完整 ($\delta_c$=2, $\delta_r$=0.2) | 0.04 | 0.62 | 0.75 | 最优配置 |
| 无BA-Score初始化 | 0.16 | 0.58 | 0.68 | AFS下降7%，BA-Score初始化至关重要 |
| 语义相似度初始化 | 0.19 | 0.53 | 0.64 | AFS下降11%，简单相似度不够 |
| $\delta_c=1$ | 0.08 | 0.61 | 0.73 | 注意力缩放不足 |
| $\delta_c=5$ | 0.11 | 0.59 | 0.71 | 过度缩放 |
| $\delta_r=0.5$ | 0.12 | 0.58 | 0.70 | 残差注入过多 |

### 关键发现

1. **组合偏见放大效应**：添加"wearing a scarf"可让assistant的偏见从0.6飙升到0.9以上
2. **颜色驱动偏见**：帽子颜色变化即可改变偏见方向——绿色帽子在多个白领职业中显著增大偏见
3. **解耦的副作用**：同时解耦太多token（如5个）会导致语义丢失——模型可能不再生成人类
4. **关键发现**：保留主体（assistant），只解耦关联属性和物体的偏见效果最好
5. **虚假相关的连锁**：成功去偏后仍可能出现其他虚假相关（如"pink hat"总是伴随户外绿地场景）

## 亮点与洞察

- **问题定义的创新性极高**：首次系统研究组合语义绑定中的偏见传播，这是现有去偏方法的根本盲点
- **BA-Score的设计**巧妙地量化了上下文对偏见的贡献，并同时考虑了token与主体的相关性和与偏见原型的相关性
- **Token解耦的实验分析**揭示了深刻的语义纠缠现象：由于cross-attention中的token information leakage，单独解耦某个词可能无效，需要考虑相邻token的联合效应
- **"去偏可能损害可靠性"**这一观点非常有深度：当模型被强制生成它的训练分布中极少见的组合时，必然质量下降

## 局限与展望

- 目前仅关注性别偏见，种族、年龄等其他敏感属性未涉及
- 只在SD-1.5上验证，更大模型（SDXL、SD3、Flux）是否有相同问题未知
- BA-Score依赖CLIP text encoder的embedding质量，可能继承其自身偏见
- 缺乏大规模benchmark：当前实验基于有限的职业和属性组合
- 注入策略相对简单（线性混合），可能不够精细

## 相关工作与启发

- 是SelfDisc、FairQueue等去偏方法的重要补充——指出了它们在组合场景中的根本缺陷
- BA-Score的设计灵感来源于softmax注意力机制和对比学习
- Token解耦与FreeCustom的multi-reference self-attention和token merging思路相关
- 发现的"偏见连锁"现象（去掉一个偏见可能暴露另一个）对整个AI公平性领域都有启示
- 未来方向：利用非敏感属性的关联来提升真实感，而非盲目消除所有相关性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 问题定义极具原创性，BA-Score和组合偏见分析填补了重要空白
- 实验充分度: ⭐⭐⭐⭐ — 组合场景分析深入，但数据集和基础模型覆盖有限
- 写作质量: ⭐⭐⭐⭐ — 问题阐述清晰，Figure的设计辅助理解
- 价值: ⭐⭐⭐⭐⭐ — 揭示了T2I去偏的根本性挑战，对该领域有重要启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models](../../NeurIPS2025/image_generation/fairimagen_post-processing_for_bias_mitigation_in_text-to-image_models.md)
- [\[CVPR 2025\] Bias for Action: Video Implicit Neural Representations with Bias Modulation](../../CVPR2025/image_generation/bias_for_action_video_implicit_neural_representations_with_bias_modulation.md)
- [\[CVPR 2026\] Elucidating the SNR-t Bias of Diffusion Probabilistic Models](../../CVPR2026/image_generation/dcw_snr_t_bias_diffusion.md)
- [\[CVPR 2025\] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](../../CVPR2025/image_generation/implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] MixFlow Training: Alleviating Exposure Bias with Slowed Interpolation Mixture](../../CVPR2026/image_generation/mixflow_training_alleviating_exposure_bias_with_slowed_interpolation_mixture.md)

</div>

<!-- RELATED:END -->
