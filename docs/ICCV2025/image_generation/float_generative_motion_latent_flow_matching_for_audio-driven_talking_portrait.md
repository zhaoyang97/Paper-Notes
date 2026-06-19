---
title: >-
  [论文解读] FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait
description: >-
  [ICCV 2025][图像生成][流匹配] 提出 FLOAT，基于流匹配（Flow Matching）的音频驱动说话肖像生成方法，在正交运动潜空间中用 Transformer 架构预测向量场，实现高效（~10 步采样）、时序一致的高质量说话视频生成，并支持语音驱动的情绪增强和测试时头部姿态编辑。 音频驱动的说话肖像生成旨在…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "流匹配"
  - "说话肖像生成"
  - "运动潜空间"
  - "正交基"
  - "语音情绪增强"
---

# FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait

**会议**: ICCV 2025  
**arXiv**: [2412.01064](https://arxiv.org/abs/2412.01064)  
**代码**: [GitHub](https://deepbrainai-research.github.io/float/)  
**领域**: 图像生成  
**关键词**: 流匹配, 说话肖像生成, 运动潜空间, 正交基, 语音情绪增强

## 一句话总结

提出 FLOAT，基于流匹配（Flow Matching）的音频驱动说话肖像生成方法，在正交运动潜空间中用 Transformer 架构预测向量场，实现高效（~10 步采样）、时序一致的高质量说话视频生成，并支持语音驱动的情绪增强和测试时头部姿态编辑。

## 研究背景与动机

音频驱动的说话肖像生成旨在从单张人像和驱动音频合成自然的说话视频，包括准确的唇同步、节奏性的头部运动和细粒度的表情变化。当前方法面临以下挑战：

**扩散模型的效率瓶颈**：EMO 等方法基于 Stable Diffusion 进行像素级视频生成，虽然质量优秀但需要数十分钟生成几秒视频；迭代采样的特性导致帧间闪烁和不一致。

**对辅助面部先验的过度依赖**：许多方法需要 2D 关键点、3D 网格或边界框作为中间表示，这些强空间先验限制了头部运动的多样性和保真度。

**像素空间的局限**：在像素/VAE 潜空间中直接生成视频帧，缺乏对运动的显式建模，难以保证时序一致性。

**情绪控制的缺失**：现有方法多数不支持情绪感知的运动生成，而人在说话时自然地通过语调反映情绪。

本文的核心洞察：
- **运动潜空间优于像素潜空间**：在学习到的正交运动潜空间中进行采样，天然保证时序一致性
- **流匹配优于扩散**：沿直线路径以恒速传输，大幅减少采样步数（10步 vs 50步）
- **正交基结构**：使运动可编辑，测试时可直接操控头部姿态

## 方法详解

### 整体框架

FLOAT 由两个阶段组成：
1. **运动潜空间自编码器（预训练）**：学习将图像编码为身份潜变量 + 运动潜变量的正交分解
2. **流匹配运动采样器**：用 Transformer 预测从高斯噪声到目标运动潜变量的向量场，由音频+情绪+参考运动条件驱动

推理时：采样运动潜变量 → 加上身份潜变量 → 解码为视频帧。

### 关键设计

1. **正交运动潜空间**：采用 LIA（Latent Image Animator）架构，将源图像 $S$ 编码为：
$$w_S = w_{S \to r} + w_{r \to S}$$
其中 $w_{S \to r} \in \mathbb{R}^d$ 是身份潜变量，$w_{r \to S} = \sum_{m=1}^M \lambda_m(S) \cdot \mathbf{v}_m$ 是运动潜变量。$V = \{\mathbf{v}_m\}_{m=1}^M$ 是学习到的正交基（$M=20$个方向，$d=512$维），$\lambda_m(S)$ 是源相关的运动系数。

正交性的优势：测试时可через内积 $\lambda_k(\hat{D}) = \langle w_{r \to \hat{D}}, \mathbf{v}_k \rangle$ 提取任意运动方向的系数，独立编辑后重新合成。

2. **面部组件感知损失 $\mathcal{L}_{comp-lp}$**：高分辨率下，牙齿、眼球等小区域的细节容易被大尺度动态淹没。提出针对面部组件的感知损失，显著提升牙齿、眼球等精细运动的保真度，无需依赖 SD 等预训练模型。

3. **Transformer 流匹配向量场预测器**：基于 DiT 架构但做了关键改进——**逐帧条件调制**（Frame-wise AdaLN）：

    - 每个第 $l$ 帧的输入用其对应的第 $l$ 帧条件通过 AdaLN 调制：
    $\gamma_i^l \times \text{LN}(X_t^l) + \beta_i^l$
    - 调制后的特征通过带掩码的自注意力层建模相邻 $2T$ 帧的时序关系
    - 这将"条件注入"与"时序建模"解耦，比用 cross-attention 同时做两件事效果更好

4. **OT-based 流匹配**：采用最优传输（OT）直线路径连接高斯噪声 $x_0$ 和目标运动潜变量 $w_{r \to D^{1:L}}$：
$$\mathcal{L}_{OT}(\theta) = \|v_t((1-t)x_0 + t \cdot w_{r \to D^{1:L}}; \theta) - (w_{r \to D^{1:L}} - x_0)\|^2$$

直线路径+恒速传输使得仅需约 10 步 ODE 求解即可生成高质量运动。

5. **语音驱动情绪增强**：直接用预训练的语音情绪预测器提取 7 类情绪的 softmax 概率 $w_e \in \mathbb{R}^7$（angry, disgust, fear, happy, neutral, sad, surprise），作为额外条件输入向量场预测器。

推理时使用**增量 CFV（Classifier-Free Vector field）**独立控制音频和情绪的引导强度：
$$\tilde{v}_t \approx v_t(\cdot|_{\{a, w_e\}}) + \gamma_a[v_t(\cdot|_{w_e}) - v_t(\cdot|_{\{a, w_e\}})] + \gamma_e[v_t(\cdot) - v_t(\cdot|_{w_e})]$$

支持情绪重定向：可将模糊的语音情绪替换为明确的 one-hot 编码。

### 损失函数 / 训练策略

总目标：
$$\mathcal{L}_{total}(\theta) = \lambda_{OT} \mathcal{L}_{OT}(\theta) + \lambda_{vel} \mathcal{L}_{vel}(\theta)$$

- $\mathcal{L}_{OT}$：OT 流匹配主损失
- $\mathcal{L}_{vel} = \|\Delta v_t - \Delta u_t\|$：速度一致性损失，促进时序平滑
- $\lambda_{OT} = \lambda_{vel} = 1$
- 驱动条件：Wav2Vec2.0 音频特征 + 语音情绪 + 源运动潜变量 + 流时间步嵌入
- 对 $w_r$、$w_e$、$a^{1:L}$ 分别以 0.1 概率 dropout（用于 CFV）
- 窗口长度 $L=50$ 帧（2.4 秒），前文 $L'=10$ 帧用于平滑过渡
- Adam 优化器，batch size 8，lr $10^{-5}$，单 A100 训练约 2 天

## 实验关键数据

### 主实验

HDTF / RAVDESS 数据集上的定量对比：

| 方法 | FID↓ | FVD↓ | CSIM↑ | E-FID↓ | P-FID↓ | LSE-D↓ | LSE-C↑ |
|------|------|------|-------|--------|--------|--------|--------|
| SadTalker | 71.95/119.43 | 339.06/376.29 | 0.644/0.644 | 1.914/3.500 | 1.456/2.045 | 7.947/7.273 | 7.305/4.748 |
| Hallo | 25.36/57.65 | 197.20/375.56 | **0.869/0.860** | 1.039/2.492 | 0.037/0.050 | 7.792/7.613 | 7.582/4.795 |
| EchoMimic | 33.55/81.84 | 296.76/320.22 | 0.823/0.805 | 1.234/3.201 | **0.023/0.047** | 8.903/8.161 | 6.242/4.144 |
| **FLOAT** | **21.10/31.68** | **162.05/166.36** | 0.843/0.810 | **1.229/1.367** | 0.032/0.031 | **7.290/6.994** | **8.222/5.730** |

FLOAT 在大多数指标上取得最优，尤其在视频质量（FID/FVD）和唇同步（LSE-D/LSE-C）上优势明显。

### 消融实验

| 配置 | FID↓ | FVD↓ | E-FID↓ | LSE-D↓ | NFEs↓ |
|------|------|------|--------|--------|-------|
| Cross-Attention (替代 AdaLN) | 21.87 | 162.70 | 1.452 | 7.757 | 10 |
| Diffusion ($\epsilon$-pred) | 21.19 | 161.67 | 1.213 | 9.922 | 50 |
| Diffusion ($x_0$-pred) | 21.70 | 162.85 | 1.278 | 9.048 | 50 |
| **FLOAT (Flow Matching)** | **21.10** | **162.05** | **1.229** | **7.290** | **10** |

附加条件的效果：

| 配置 | FID↓ | FVD↓ | E-FID↓ | P-FID↓ |
|------|------|------|--------|--------|
| FLOAT (base) | 21.10/31.68 | 162.05/166.36 | 1.229/1.367 | 0.032/0.031 |
| + 3DPose | **19.72/29.72** | **126.66/112.89** | **0.926/1.152** | **0.012/0.016** |
| - S2E (去掉语音情绪) | 21.24/32.04 | 155.03/166.87 | 1.254/1.502 | 0.031/0.025 |

### 关键发现

- **流匹配 vs 扩散**：在相同架构下，流匹配只需 10 NFEs 即可达到甚至超越 50 NFEs 扩散模型的效果，LSE-D 从 9.0+ 大幅降至 7.29
- **Frame-wise AdaLN vs Cross-Attention**：AdaLN 的条件-时序解耦设计在表情生成（E-FID）和唇同步（LSE-D）上明显优于 Cross-Attention
- **语音情绪贡献**：去掉 S2E 后 E-FID 和 P-FID 轻微退化，说明语音情绪提供有价值的运动先验
- **3DPose 条件**：加入头部姿态参数后所有指标显著提升，展现框架的可扩展性
- **效率优势**：FLOAT 的前向 FPS 远超扩散方法（约 5x）

## 亮点与洞察

- **运动潜空间的选择**：放弃 SD 的像素 VAE 潜空间，转向运动语义潜空间（LIA），获得时序一致性的天然优势
- **正交基的可编辑性**：$\lambda$-control 可以在测试时独立调整头部方向等运动维度，无需额外训练
- **流匹配的高效性**：OT-based 直线路径使 10 步采样即可获得高质量结果
- **增量 CFV**：分别控制音频和情绪的引导强度，提供细粒度的生成控制
- **语音→情绪→运动**的自然链路：无需用户额外输入情绪标签，从语音中自动提取

## 局限与展望

- 正交基维度 $M=20$ 可能不足以捕获所有精细运动模式
- 情绪识别依赖外部预训练模型，其准确性直接影响运动质量
- 训练数据仅包含约 250 个身份，对罕见人种或极端表情的泛化能力有限
- LIA 解码器在极大头部旋转时可能出现伪影
- 512×512 分辨率，对更高分辨率的扩展需要重新训练
- 仅在英语语音数据上训练和评估

## 相关工作与启发

- **VASA-1** [Xu et al., 2024]：同样采用运动潜空间，但 FLOAT 的正交结构使运动可编辑
- **EMO** [Tian et al., 2024]：SD-based 像素级生成，质量好但效率低
- **LIA** [Wang et al., 2022]：运动自编码器基础架构，本文在此基础上引入正交约束和面部组件损失
- **DiT** [Peebles & Xie, 2023]：AdaLN 条件注入机制的灵感来源
- 启发：生成模型的潜空间选择比模型架构更重要；正交结构带来自然的可编辑性

## 评分

- **新颖性**: ⭐⭐⭐⭐ 流匹配+正交运动潜空间+语音情绪的组合新颖实用
- **实验充分度**: ⭐⭐⭐⭐⭐ 对比全面、消融充分、应用展示丰富（姿态编辑/情绪重定向/多条件）
- **写作质量**: ⭐⭐⭐⭐ 公式推导清晰，架构图信息量大
- **价值**: ⭐⭐⭐⭐⭐ 效率与质量的最佳平衡点，正交基编辑性是独特卖点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](domain_generalizable_portrait_style_transfer.md)
- [\[ICML 2026\] From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection](../../ICML2026/image_generation/from_talking_to_singing_a_new_challenge_for_audio-visual_deepfake_detection.md)
- [\[ICCV 2025\] Deeply Supervised Flow-Based Generative Models](deeply_supervised_flow-based_generative_models.md)

</div>

<!-- RELATED:END -->
