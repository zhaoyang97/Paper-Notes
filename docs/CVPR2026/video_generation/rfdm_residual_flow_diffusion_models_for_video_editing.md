---
title: >-
  [论文解读] RFDM: Residual Flow Diffusion Models for Video Editing
description: >-
  [CVPR 2026][视频生成][指令式视频编辑] RFDM 把一个 2D 图像编辑（I2I）扩散模型改造成逐帧自回归的视频编辑模型——通过把当前帧的扩散噪声均值"平移"到上一帧的预测上，让模型只去学相邻帧之间的残差而非整帧，从而在**不增加任何额外算力**、可处理任意长度视频的前提下，做到媲美 3D 时空模型的时序一致性与编辑保真度。
tags:
  - "CVPR 2026"
  - "视频生成"
  - "指令式视频编辑"
  - "自回归扩散"
  - "残差流"
  - "曝光偏置"
  - "时序一致性"
---

# RFDM: Residual Flow Diffusion Models for Video Editing

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Salehi_RFDM_Residual_Flow_Diffusion_Models_for_Video_Editing_CVPR_2026_paper.html)  
**代码**: 无  
**领域**: 视频编辑 / 扩散模型  
**关键词**: 指令式视频编辑, 自回归扩散, 残差流, 曝光偏置, 时序一致性

## 一句话总结
RFDM 把一个 2D 图像编辑（I2I）扩散模型改造成逐帧自回归的视频编辑模型——通过把当前帧的扩散噪声均值"平移"到上一帧的预测上，让模型只去学相邻帧之间的残差而非整帧，从而在**不增加任何额外算力**、可处理任意长度视频的前提下，做到媲美 3D 时空模型的时序一致性与编辑保真度。

## 研究背景与动机
**领域现状**：指令式视频编辑（只给一句自然语言 prompt，如"把人去掉""把它变成梵高风格"）目前主流有两条路：一是基于大型 3D 时空视频模型（如 EVE）做编辑，质量高但要海量数据和算力；二是把图像模型（T2I / I2I）零样本或微调地搬到视频上（如 Fairy、VidToMe、TokenFlow），靠跨帧特征对齐 / 时空注意力维持一致性。

**现有痛点**：几乎所有方法都依赖**非因果**的时序机制——必须一次性吃进固定长度的整段视频，且跨所有帧的时空注意力让算力随帧数暴涨。这在视频流、手机等资源受限场景几乎不可用。自回归视频生成本可解决变长输入和效率问题，但在视频编辑上几乎没人探索，且自身效率仍不够实时。

**核心矛盾**：图像 I2I 模型最便宜，但逐帧独立地跑 I2I 会因扩散随机性和输入帧抖动产生**帧间不一致**（前一帧把液体变粉、下一帧又变回来）；而要一致性就得上重型时空注意力，又把图像模型的效率优势丢光了。一致性与效率被绑死成了 trade-off。

**本文目标**：在保持 I2I 模型算力的同时，让逐帧编辑结果时序一致，并且天然支持任意长度的视频。

**切入角度**：视频相邻帧高度冗余——大部分像素（背景、共同出现的物体）只需从上一帧"搬过去"，真正要改的只是少量新区域。如果把编辑任务重新表述成"预测相邻帧之间的残差"，模型就不必每帧从纯噪声重生成整张图。

**核心 idea**：把 I2I 扩散的前向过程的**噪声均值从 0 平移到上一帧预测 $\hat{y}_{t-1}$**，使去噪过程聚焦于帧间变化（残差），并用上一帧预测作为条件实现因果自回归——这就是 Residual Flow Diffusion Model（RFDM）。

## 方法详解

### 整体框架
RFDM 要解决的是"把单帧 I2I 模型变成因果、变长、零额外开销的视频编辑器"。给定输入视频帧序列 $X=\{x_t\}$ 和指令 $p$，模型逐帧自回归地产出编辑后的 $Y=\{y_t^0\}$，编辑第 $t$ 帧时以**自己在 $t-1$ 帧的预测** $\hat{y}_{t-1}$ 为条件。整条 pipeline 只在标准 I2I 扩散上动了两处：① 把上一帧预测拼接进去当条件（不加任何计算量）；② 把前向加噪过程改写成残差形式，让噪声均值朝 $\hat{y}_{t-1}$ 偏移。训练时为缓解自回归特有的曝光偏置，用 Diffusion Forcing 而非 Teacher Forcing 来采样 $\hat{y}_{t-1}$。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入帧 x_t + 指令 p"] --> B["上一帧预测条件<br/>拼接 ŷ_(t-1) 作为输入"]
    B --> C["残差流前向过程<br/>噪声均值平移到 ŷ_(t-1)"]
    C --> D["去噪网络 ŷ_θ<br/>DDIM + CFG，预测 y_t^0"]
    D -->|t=0 从纯噪声 I2I 起步| D
    D --> E["输出 ŷ_t"]
    E -->|作为下一帧的 ŷ_(t-1) 回灌| B
    F["Diffusion Forcing<br/>缓解曝光偏置"] -.训练时采样 ŷ_(t-1).-> C
```

### 关键设计

**1. 上一帧预测条件化：把 I2I 因果化且零额外开销**

要让逐帧编辑前后一致，最直接的办法是让第 $t$ 帧"知道"第 $t-1$ 帧编辑成了什么样。RFDM 在保留标准 I2I 前向过程 $y_t^s = \alpha^s y_t^0 + \sigma^s \epsilon$（$\alpha^s,\sigma^s$ 为噪声调度，对数信噪比 $\lambda^s=\log[\alpha^s/\sigma^s]$ 随扩散步 $s$ 单调下降）的同时，把上一帧预测 $\hat{y}_{t-1}$ 作为**额外输入通过通道拼接**喂给去噪网络，指令 $p$ 仍走 cross-attention。训练目标变成

$$\arg\min_\theta \big\| \hat{y}_\theta(y_t^s, \hat{y}_{t-1}, x_t, p, \lambda^s) - y_t^0 \big\|,\quad \hat{y}_{-1}=0 .$$

第一帧因为没有前序（$\hat{y}_{-1}=0$）就退化成普通 I2I，从纯噪声起步；之后每帧把自己的输出回灌成下一帧的条件，整个序列因此变成**因果自回归**：编辑第 $t$ 帧只依赖 $\le t$ 的信息，于是支持变长、可流式。关键在于这个条件只是多拼一个张量，**没有引入任何跨帧注意力**，算力与单帧 I2I 完全一致。

**2. 残差流前向过程：让扩散只学帧间变化**

光做条件化还不够——模型仍在每帧从纯噪声把整张图重建一遍，没利用相邻帧的冗余。RFDM 借鉴图像逆问题里的残差思路，把要生成的量从"整帧 $y_t^0$"换成"帧间残差" $m_t^0 = \hat{y}_{t-1} - y_t^0$。具体做法是把前向加噪过程的**高斯均值从 0 平移到 $\hat{y}_{t-1}$**：

$$q(y_t^s \mid y_t^0, \hat{y}_{t-1}) = \mathcal{N}\big(\alpha^s y_t^0 + \sigma^s \hat{y}_{t-1},\ (\sigma^s)^2 I\big) = \mathcal{N}\big(\gamma^s y_t^0 + \sigma^s m_t^0,\ (\sigma^s)^2 I\big),$$

其中 $\gamma^s = \sqrt{1-(\sigma^s)^2} + \sigma^s$（flow 模型下 $\gamma^s\equiv 1$，扩散模型下大多数时刻也接近 1）。等价地，采样写成

$$y_t^s = \alpha^s y_t^0 + \sigma^s \hat{y}_{t-1} + \sigma^s \epsilon,\quad \epsilon\sim\mathcal{N}(0,I)\ \Longleftrightarrow\ y_t^s = \alpha^s y_t^0 + \sigma^s\hat{\epsilon},\quad \hat{\epsilon}\sim\mathcal{N}(\hat{y}_{t-1}, I).$$

也就是说，噪声不再以 0 为中心，而是以上一帧预测为中心——这条 Markov 链把目标帧"运输"到上一帧预测的带噪版本。这样残差被**显式嵌进了带噪输入里**：网络可以把像素分成两类——① 在 $\hat{y}_{t-1}$ 里已经编辑好的区域（背景、共同出现的运动物体），只需平移；② 需要按 $x_t$ 和 $p$ 新编辑的区域，才真正去噪生成。去噪过程因此聚焦于帧间的"少量改动"，而不是每帧从头来过，这正是时序一致性的来源。

**3. Diffusion Forcing：消除自回归的曝光偏置**

自回归模型有个经典毛病——曝光偏置：训练时喂的 $\hat{y}_{t-1}$ 和推理时模型自己产生的 $\hat{y}_{t-1}$ 分布不一致，导致推理时视频质量随帧数逐渐退化。最朴素的 Teacher Forcing 直接用 ground-truth 帧 $y_{t-1}^0$ 当条件，但作者实验证明这样训出的模型推理时**退化很快**，因为它过度依赖"干净的真值"，而真值分布与模型自己那不完美的预测分布之间存在 gap。RFDM 改用 Diffusion Forcing：训练时给**不同帧采样不同的随机噪声水平**，并用去噪网络在 $t-1$ 帧的预测去采样 $\hat{y}_{t-1}$。这样条件帧仍然"干净"，但它来自模型自己的训练分布，与推理分布更接近，从而把训练-推理 gap 收窄。推理时则完全顺序地逐帧编辑，用上一帧的干净输出条件下一帧。

### 损失函数 / 训练策略
训练用简单的 MSE 重建损失（见 Algorithm 1）：对一段视频 clip，采样 $K$ 个有序帧索引，从第一帧（$\hat{y}_{-1}=0$）开始，每帧用 $y_t^s = \alpha^s y_t^0 + \sigma^s \hat{y}_{t-1} + \sigma^s\epsilon$ 构造带噪输入，过去噪网络得到 $\hat{y}_t$，对所有帧累加 $\text{MSE}(\hat{y}_t, y_t^0)$ 并反传。骨干用 SD1.5（RFDM1.5）与 SD3.5-M（RFDM3.5）两种；数据是 Señorita-2M（首个大规模真实世界视频编辑数据集，含全局/局部风格迁移、物体移除/添加/替换 5 类任务，200 万对视频），自定义 80/5/15 的 train/val/test 划分。8×A100 训 45k 步，batch 8、梯度累积 2、lr 1e-4、FusedAdam。推理用 DDIM 采样 + CFG（同时跑无条件、仅 $x_t$、$x_t{+}p$ 三路，系数 $\omega_x,\omega_{xp}$）。

## 实验关键数据

### 主实验
在 Señorita 测试集（16 帧生成，跨全局/局部风格迁移 + 物体移除三类任务平均）上对比，RFDM 在保真度（ViDreamSim↓、DVS↑、MLLM-Judge↑）和效率上全面领先同类 I2I 方法，并以 ~13× 更小的显存逼近闭源 3D 模型 EVE：

| 方法 | 因果 | DVS↑ | MLLM-Judge↑ | ViDreamSim↓ | TempCon↓ | 延迟(s)↓ | 显存(GB)↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Fairy | ✗ | 0.40 | 3.98 | 0.29 | 0.042 | 13 | 77 |
| VidToMe | ✗ | 0.37 | 1.77 | 0.59 | 0.014 | 86 | 9 |
| TokenFlow | ✗ | 0.29 | 3.23 | 0.48 | 0.010 | 128 | 11 |
| RAVE | ✗ | 0.34 | 3.54 | 0.42 | 0.017 | 92 | 9 |
| RFDM1.5 | ✓ | 0.43 | 6.60 | 0.23 | 0.010 | **8** | **2** |
| RFDM3.5 | ✓ | **0.48** | **7.37** | **0.20** | 0.009 | 13 | 6 |

（Fairy* 为用 RFDM 预训练 UNet 复现的版本。）在 TGVE / TGVE+ 两个 benchmark 上，RFDM3.5 在 6 个指标中拿下 4 个，且所有 RFDM 变体都拿到最高的 CLIPFrame（时序一致性最强）；延迟比 Fairy 持平、显存约 1/13，比其他 baseline 快约 4×。相比 EVE，RFDM3.5 的 PickScore 略低，但 EVE 用的是 4.4B 的 3D 骨干（RFDM3.5 为 2.5B）和 34M 训练视频（Señorita 仅 2M）。⚠️ 表中数字以原文 Table 1 为准。

### 消融实验
在 SD1.5 骨干、Señorita 全局风格迁移验证集上消融（默认设置：自回归帧数 3、残差流形式）：

| 配置 | TempCon↓ | ErrAccu↓ | 说明 |
|------|:---:|:---:|------|
| 自回归帧数 = 0 | 0.068 | 0.21 | 退化为逐帧独立，一致性最差 |
| 自回归帧数 = 1 | 0.013 | 0.12 | 加入前帧条件即大幅改善 |
| 自回归帧数 = 3（默认） | 0.009 | 0.07 | 一致性与误差累积俱佳 |
| 自回归帧数 = 5 | 0.007 | 0.07 | 略好但开销更大 |
| 输入条件 = 仅 $x$ | 0.027 | 0.14 | 不给 $\hat{y}_{t-1}$ 明显变差 |
| 输入条件 = $x,\hat{y}_{t-1}$ | 0.009 | 0.07 | 加上一帧预测条件 |
| 预测形式 = 整帧 | 0.009 | 0.09 | 误差累积更高 |
| 预测形式 = 残差流 | 0.009 | 0.07 | 误差累积更低 |

另有 DAVIS 跟踪实验直接验证残差流的价值：把预测形式从"整帧"换成"残差流"，分割跟踪 J&F 从 29.1 大幅升到 **43.6**，说明残差形式让编辑后的物体在帧间更稳定可追踪。曝光偏置消融里 Teacher Forcing 与 Diffusion Forcing 的 TempCon 接近，但 Diffusion Forcing 的 ViDreamSim 更低（0.35 vs 0.38），保真度更好。

### 关键发现
- **残差流主要降"误差累积"而非"逐帧一致性"**：整帧 vs 残差流的 TempCon 几乎一样（0.009），但 ErrAccu 从 0.09 降到 0.07、DAVIS J&F 从 29.1 跳到 43.6——残差形式真正缓解的是自回归长程漂移。
- **上一帧条件化是一致性的主开关**：自回归帧数从 0→1，TempCon 直接从 0.068 砍到 0.013，说明"让当前帧看到前一帧"是最关键的一步，多看几帧只是边际改善。
- **新基准更能区分模型**：作者指出仅靠 CLIP 文本相似度的旧基准无法衡量"保真度"（不该改的区域别动），于是基于 Señorita 真值提出 ViDreamSim（$\frac{1}{T}\sum_t d(\bar{y}_t^0, y_t^0)$，逐帧对比真值测保真）和 Error Accumulation（$\frac{1}{T-1}\sum_t d(\bar{y}_t^0, \bar{y}_0^0)$，测后续帧相对首帧的分布漂移），$d$ 取 DreamSim 类感知距离。VidToMe 在 TempCon 上更平滑却在 ViDreamSim/MLLM-Judge 上垫底，正说明"平滑"不等于"忠实"。

## 亮点与洞察
- **"平移噪声均值"是极其轻量的改造**：不动网络结构、不加注意力，只把前向过程的高斯中心从 0 挪到上一帧预测，就把"整帧生成"重述成"残差生成"，效率和图像模型完全一致——这种用数据分布而非算力换一致性的思路很值得迁移。
- **残差被显式塞进带噪输入**：网络天然学会"已编辑区域只平移、新区域才生成"的像素二分策略，这是把视频时序冗余编码进扩散过程的一种优雅方式。
- **把 Diffusion Forcing 用来治自回归编辑的曝光偏置**：用模型自己的训练分布采样条件帧，而非 ground-truth，缓解训练-推理 gap，这个 trick 可迁移到任何逐帧自回归扩散生成任务。
- **基准设计本身是贡献**：明确指出 CLIP 文本相似度衡量不了保真度，并用真值数据造出 ViDreamSim / Error Accumulation 两个可区分"忠实 vs 平滑"的指标。

## 局限与展望
- 作者承认相比 EVE 这类闭源 3D 模型，RFDM 的 PickScore 略低；EVE 用了更大骨干（4.4B vs 2.5B）和远多的训练数据（34M vs 2M），公平对比受限，且 EVE 闭源无法直接测其算力。
- ⚠️（自己观察）残差流主要降误差累积、对逐帧 TempCon 提升有限；纯自回归逐帧仍可能在极长视频上慢慢漂移，论文实验多在 16 帧规模，更长序列的稳定性需进一步验证。
- 方法依赖 Señorita 这种"配对真值"数据训练，目前覆盖风格迁移和物体移除；对需要大幅几何/结构改动的编辑（如物体替换、大运动）效果未充分展示。
- 第一帧仍走纯 I2I，首帧质量会传导到整段视频，首帧失真可能被自回归放大。

## 相关工作与启发
- **vs Fairy / VidToMe（同 I2I 骨干，非因果）**：它们靠跨帧特征合并或时空 cross-attention 拉一致性，需要一次性吃整段视频、显存高（Fairy 77GB）；RFDM 不加任何跨帧注意力、因果自回归、显存仅 2–6GB，且保真度（ViDreamSim/MLLM-Judge）显著更好。VidToMe 输出更平滑却偏离真值，反衬 RFDM 更忠实。
- **vs EVE 等 3D 时空 V2V**：3D 模型质量天花板高但算力随帧数暴涨、需海量数据；RFDM 用 2D 骨干 + 残差流，以 orders-of-magnitude 更低的延迟逼近其质量，并独立于视频长度扩展。
- **vs Teacher Forcing 训练**：传统自回归用真值帧当条件训练，RFDM 用 Diffusion Forcing 从模型自身分布采样条件帧，专门压制曝光偏置导致的长程退化。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把噪声均值平移、用残差流把 I2I 自回归化的 formulation 简洁且新，但建立在 Diffusion Forcing、残差扩散等已有思想之上
- 实验充分度: ⭐⭐⭐⭐ 三基准 + 七组消融 + DAVIS 跟踪验证，效率/质量对比完整；更长视频规模略欠
- 写作质量: ⭐⭐⭐⭐ 动机—公式—算法链条清晰，残差流推导给得明白
- 价值: ⭐⭐⭐⭐ 在资源受限、变长、流式场景下用图像级算力做一致视频编辑，落地价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FlowPortal: Residual-Corrected Flow for Training-Free Video Relighting and Background Replacement](flowportal_residual-corrected_flow_for_training-free_video_relighting_and_backgr.md)
- [\[CVPR 2026\] FlowDirector: Training-Free Flow Steering for Precise Text-to-Video Editing](flowdirector_training-free_flow_steering_for_precise_text-to-video_editing.md)
- [\[CVPR 2026\] Accelerating Autoregressive Video Diffusion via History-Guided Cache and Residual Correction](accelerating_autoregressive_video_diffusion_via_history-guided_cache_and_residua.md)
- [\[CVPR 2026\] Flowception: Temporally Expansive Flow Matching for Video Generation](flowception_temporally_expansive_flow_matching_for_video_generation.md)
- [\[CVPR 2026\] CamDirector: Towards Long-Term Coherent Video Trajectory Editing](camdirector_towards_long-term_coherent_video_trajectory_editing.md)

</div>

<!-- RELATED:END -->
