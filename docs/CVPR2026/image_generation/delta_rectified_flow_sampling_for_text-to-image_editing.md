---
title: >-
  [论文解读] Delta Rectified Flow Sampling for Text-to-Image Editing
description: >-
  [CVPR 2026][图像生成][整流流] DRFS 把 DDS「相减抵消共同信息」的思路搬进整流流（Rectified Flow）的速度场蒸馏采样，再加一个随时间衰减的偏移项把目标隐变量拉回正确轨迹，在不改架构、免反演、免训练的前提下解决了 RFDS 编辑时的过度平滑问题，并在 PIE 基准上取得最优的编辑保真度与可控性。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "整流流"
  - "蒸馏采样"
  - "文本编辑"
  - "免反演"
  - "能量函数"
---

# Delta Rectified Flow Sampling for Text-to-Image Editing

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Beaudouin_Delta_Rectified_Flow_Sampling_for_Text-to-Image_Editing_CVPR_2026_paper.html)  
**代码**: https://github.com/Harvard-AI-and-Robotics-Lab/DeltaRectifiedFlowSampling  
**领域**: 图像生成 / 文本到图像编辑 / 整流流  
**关键词**: 整流流、蒸馏采样、文本编辑、免反演、能量函数

## 一句话总结
DRFS 把 DDS「相减抵消共同信息」的思路搬进整流流（Rectified Flow）的速度场蒸馏采样，再加一个随时间衰减的偏移项把目标隐变量拉回正确轨迹，在不改架构、免反演、免训练的前提下解决了 RFDS 编辑时的过度平滑问题，并在 PIE 基准上取得最优的编辑保真度与可控性。

## 研究背景与动机
**领域现状**：文本引导图像编辑（T2I editing）有两条主线。一条是**非能量方法**（如 RF-Inversion、FlowEdit、FTEdit），靠两条速度场（一条做反演、一条做生成）加上注意力注入、隐变量平均等启发式技巧来保背景；其中 FlowEdit 干脆免去显式反演，直接用源/目标速度的偏移量估计目标隐变量。另一条是**能量优化方法**：SDS 和 DDS 把编辑写成一个由冻结扩散先验导出的能量函数最小化问题，RFDS（Rectified Flow Distillation Sampling）把这套思路从扩散模型的噪声残差扩展到整流流的速度场，实现即插即用编辑。

**现有痛点**：RFDS 用 $\phi=\phi_{tgt}$ 直接套到编辑上时会**过度平滑**——背景和高频细节被无意改动，保真度受损。作者诊断出根因：RFDS 的能量 $E_{RFDS}=\mathbb{E}\,\|v_\theta(x_t,t,\phi)-\dot{x}_t\|^2$ 的梯度对「该编辑的区域」和「该保留的区域」一视同仁，于是在本该不变的区域也产生非零梯度，把那里的高频细节给破坏掉。RFDS 自己的补救（iRFDS：先优化噪声把图反演回去）又要额外算力。

**核心矛盾**：能量优化方法的梯度无法区分编辑/保留区域，导致编辑强度和背景保真不可兼得。

**本文目标**：在整流流框架里设计一个能量函数，让优化只惩罚源/目标的差异、自动放过共有信息；同时把「目标隐变量评估点偏离正确轨迹」造成的模型-数据失配也修掉。

**切入角度**：作者注意到 DDS 之所以能保背景，是因为它最小化的是**源、目标两条去噪轨迹之差**，共有信息被相减抵消。把这个「相减」原则平移到 RF 的速度场上，就能让背景区域梯度归零。

**核心 idea**：用「目标残差 − 源残差」代替「目标残差」当能量（residual subtraction），再叠加一个随时间衰减的偏移项 $c_t(x_0^{tgt}-x_0^{src})$ 把评估点拉回目标轨迹，从而既消过度平滑、又精准对齐目标分布。

## 方法详解

### 整体框架
DRFS 是一个**免反演、免训练、不改架构**的蒸馏采样优化器：输入是源图 $x_0^{src}$ 加源/目标提示 $(\phi_{src},\phi_{tgt})$，输出是编辑后的图 $x_0^{tgt}$。它把待编辑图本身当作可优化参数（$\Theta=x_0^{tgt}$，初始化为 $x_0^{src}$），在一个 RF 速度场先验下用近似梯度反复更新这张图，直到它语义对齐目标提示又保住背景。

整条流程是一个优化循环（Algorithm 1）：每步从降序时间表 $\{\tau_j\}$ 取一个 $t$、采一个高斯噪声 $\varepsilon$，构造源隐变量 $x_t^{src}=(1-t)x_0^{src}+t\varepsilon$ 和**带偏移的目标隐变量** $\hat{x}_t^{tgt}=(1-t)x_0^{tgt}+t\varepsilon+c_t(x_0^{tgt}-x_0^{src})$，再用两者速度之差算出梯度去更新 $x_0^{tgt}$。其中两个核心贡献——「残差相减的能量函数」和「时变偏移项 $c_t$」——共同决定了梯度怎么走；理论部分进一步证明这套设计统一了 DDS 与 FlowEdit。这是纯采样/能量函数层面的改进（没有新增网络模块或多阶段 pipeline），故不另画框架图，用公式说清即可。

### 关键设计

**1. 残差相减能量：让背景梯度自动归零**

RFDS 的能量本质是 $\mathbb{E}\,\|r_{tgt}\|^2$，其中残差 $r=v_\theta(x_t,t,\phi)-\dot{x}_t$。问题在于它只看目标残差，背景区域也会被推着变。DRFS 借 DDS 的「相减」原则，把能量改成源、目标残差之差：

$$E=\mathbb{E}_{t,\varepsilon}\Big[\big\|v_\theta(x_t^{tgt})-v_\theta(x_t^{src})-(\dot{x}_t^{tgt}-\dot{x}_t^{src})\big\|^2\Big]=\mathbb{E}_{t,\varepsilon}\big[\|r_{tgt}-r_{src}\|^2\big]$$

这样优化只惩罚源/目标之间的**差异**，源图和目标图共有的信息（典型就是背景）在相减中被抵消，对应区域梯度趋零，自然不会被破坏。作者给了一个干净的 sanity check：当 $\phi_{tgt}=\phi_{src}$ 时「共有信息」就是整张图，优化理应什么都不改——此时初始 $x_0^{tgt}=x_0^{src}$ 故 $x_t^{tgt}=x_t^{src}$，能量恒为 $0$，与预期吻合。这一步是从 RFDS 单残差到「delta 残差」的关键，且和朴素「delta」只减条件预测不同，DRFS 减的是速度与数据动态之间的**完整残差**，从而抵消掉源/目标共享的分量、在梯度里留下一个 RF 特有的漂移项。

**2. 时变偏移项 $c_t$：把目标隐变量评估点拉回正确轨迹**

只做残差相减还有个隐患：直接用 $x_t^{tgt}=a_t x_0^{tgt}+b_t\varepsilon$ 去插值时，由于 $x_0^{tgt}$ 在优化途中处于源到目标编辑路径的中段，插出来的 $x_t^{tgt}$ 会偏离目标分布的前向后验，导致目标速度估得不准、编辑效果减弱、收敛变慢。DRFS 的补救是给目标隐变量加一个线性补偿，得到修正后的评估点：

$$\hat{x}_t^{tgt}=a_t x_0^{tgt}+b_t\varepsilon+c_t\,(x_0^{tgt}-x_0^{src}),\quad c_t\ge 0$$

偏移量 $c_t(x_0^{tgt}-x_0^{src})$ 沿着「源→目标」的方向把采样轨迹逐步推近目标分布，从而让 $v_\theta(\hat{x}_t^{tgt})$ 估得更准、在优化早期就减小模型-数据失配。代入后的 DRFS 梯度为（把网络 Jacobian 近似成单位阵、直接优化 $\Theta=x_0^{tgt}$）：

$$\nabla_\Theta E_{DRFS}=\mathbb{E}_{t,\varepsilon}\Big[w_{DRFS}(t)\big(v_\theta(\hat{x}_t^{tgt})-v_\theta(x_t^{src})\big)-(\dot{a}_t+\dot{c}_t)(x_0^{tgt}-x_0^{src})\Big]$$

其中 $w_{DRFS}(t)=2(a_t+c_t-\dot{a}_t-\dot{c}_t)$ 是时变权重。正是这个 $c_t$ 让 DRFS 成为「路径感知（path-aware）」方法——它显式利用编辑轨迹，而不是只在某个固定插值点上评估速度。

**3. $c_t$ 的形态与降序时间表：渐进偏移避免早期误差放大**

$c_t$ 不能随便取。作者从两个边界条件出发约束它：编辑后期 $t\to 0$ 时 $x_0^{tgt}$ 应已落在目标分布，不该再偏移，故要求 $c_t\propto t$ 使偏移随 $t$ 衰减；另一个初始条件是 $t\to 1$ 时 $c_t\to 0$。据此最终取 $c_t=\frac{k}{T}\,t\approx(1-t)t$（$k$ 为当前步、$T$ 为总步数），即偏移系数在整个优化过程中**从 0 渐增到 1**。这种渐进偏移的好处是：在早期高噪声步只用很小的偏移，避免误差被放大；越往后偏移越强，把图稳稳推向目标。配套地，DRFS 采用**降序时间表**（先大 $t$ 做粗粒度更新、后小 $t$ 做精修），实现 coarse-to-fine——早期大噪声步允许形状/姿态等大改动，末期小噪声步细修颜色纹理；相比之下随机时间表把粗细更新交错，容易引入可见瑕疵。轨迹分析进一步显示：取 $c_t=\eta t$ 时 $\eta$ 越大，编辑路径越直（用路径-弦长比 $SR=\sum_k\|x_{0,k+1}^{tgt}-x_{0,k}^{tgt}\|/\|x_{0,N}^{tgt}-x_{0,0}^{tgt}\|$ 度量，$SR=1$ 为完全直线）、单步更新幅度越大，于是取一个中间值即可兼顾对齐强度与保真。

**4. 统一视角：DDS 与 FlowEdit 都是 DRFS 的特例**

DRFS 的偏移系数 $c_t$ 像一个旋钮，把已有方法串成一条谱系。作者证明：当 $c_t=0$ 时，DRFS 能量 $E_{DRFS}$ 严格退化为 DDS 能量 $E_{DDS}$（借助速度场与噪声预测的等价关系 $\varepsilon_\theta=\frac{a_t}{\dot{b}_t a_t-\dot{a}_t b_t}(v_\theta-\frac{\dot{a}_t}{a_t}x)$ 改写即可对上）；当取 RF 参数化 $(a_t,b_t)=(1-t,t)$ 且 $c_t=t$ 时，DRFS 的编辑轨迹严格退化为免反演方法 FlowEdit——因为 FlowEdit 用来评估目标速度的那一项 $x_0^{tgt}(t)+x_t^{src}-x_0^{src}$ 恰好可以解读为 $c_t=t$ 时的 $\hat{x}_t^{tgt}$。于是 DRFS 把「基于分数的扩散优化（DDS）↔ 基于速度的整流流优化 ↔ 基于 ODE 的编辑（FlowEdit）」统一在一个能量框架下，并指出 DDS（$c_t=0$）和 FlowEdit（隐含 $c_t=t$）的偏移选择都是次优的，自己的 $c_t\approx(1-t)t$ 才是兼顾的甜点。

### 损失函数 / 训练策略
本质是无训练的隐变量优化：以 SGD 优化器、降序时间表、批大小 1（每步单个采样时间步）跑若干轮；源/目标 CFG 分别设为 6 和 16.5，权重取单位权重。底座用整流流模型 Stable Diffusion 3 / 3.5。整个过程不改任何网络参数、不需反演。

## 实验关键数据

### 主实验
在 PIE 基准（700 张多任务编辑图）上对比扩散类与整流流类方法，指标含结构距离、背景保持（PSNR/LPIPS/MSE/SSIM）与 CLIP 相似度（whole/edited）。

| 方法 | 模型 | 结构距离 ×10³ ↓ | PSNR ↑ | LPIPS ×10³ ↓ | MSE ×10⁴ ↓ | SSIM ×10² ↑ | CLIP-Edited ↑ |
|------|------|------|------|------|------|------|------|
| FlowEdit | SD3 | 27.24 | 22.13 | 105.46 | 87.34 | 83.48 | 23.67 |
| iRFDS | SD3 | 62.72 | 19.61 | 186.39 | 179.76 | 74.59 | 21.67 |
| **DRFS** | **SD3** | **23.05** | **23.38** | **93.81** | **67.49** | **84.85** | **23.83** |
| FlowEdit | SD3.5 | 12.73 | 26.59 | 56.17 | 33.84 | 89.34 | 23.00 |
| DNAEdit | SD3.5 | 14.19 | 26.66 | 74.57 | 32.76 | 88.63 | 22.71 |
| **DRFS** | **SD3.5** | **12.00** | **26.97** | **55.83** | **30.76** | **89.41** | **23.17** |

DRFS 拿下所有 SD3/SD3.5 方法中最高的编辑区 CLIP 相似度（23.83），语义对齐最好。与同为 RF 蒸馏路线的 iRFDS 相比，背景保持改善巨大：LPIPS 93.81 vs 186.39、MSE 67.49 vs 179.76、SSIM 84.85 vs 74.59——说明 DRFS 确实压住了过度平滑。在 SD3.5 上也几乎全面超过 FlowEdit、FTEdit、DNAEdit。

### 消融实验
偏移系数 $c_t$ 的三种取值（SD3，对应 DDS / 本文 / FlowEdit 三个特例）：

| 配置 | 结构距离 ×10³ ↓ | PSNR ↑ | LPIPS ×10³ ↓ | SSIM ×10² ↑ | CLIP-Edited ↑ | 说明 |
|------|------|------|------|------|------|------|
| $c_t=0$（≡DDS） | 8.35 | 28.63 | 44.66 | 90.52 | 22.53 | 背景保持最好，但编辑最弱 |
| $c_t\approx(1-t)t$（本文） | 23.05 | 23.38 | 93.81 | 84.85 | 23.83 | 编辑强度与保真平衡 |
| $c_t=t$（≡FlowEdit） | 37.28 | 20.71 | 143.06 | 80.27 | 23.21 | 过度编辑，保真最差 |

### 关键发现
- **$c_t$ 是编辑强度与背景保真的旋钮**：$c_t$ 越大梯度更新越大、轨迹越直、推向目标越猛，编辑更强但背景保持变弱；$c_t=0$ 几乎不动背景却编辑乏力。本文的 $c_t\approx(1-t)t$ 在两端之间取到最佳折中（CLIP-Edited 最高 23.83，同时 SSIM 84.85 远好于 $c_t=t$）。
- **渐进偏移胜过固定偏移**：从 0 渐增的 $c_t$ 避免了早期高噪声步的误差放大，这正是它优于 FlowEdit 隐含的恒定 $c_t=t$ 的原因。
- **降序时间表优于随机时间表**：coarse-to-fine 的更新顺序（先改形状姿态、后修颜色纹理）比随机交错更少瑕疵、更一致。

## 亮点与洞察
- **「相减抵消共有信息」从扩散搬到整流流**：DDS 在噪声残差上做相减，DRFS 在速度场残差上做相减，背景梯度自动归零——一个干净的迁移，且配了 $\phi_{tgt}=\phi_{src}\Rightarrow E=0$ 的 sanity check 自证逻辑闭环。
- **一个偏移系数把三种方法串成谱系**：$c_t=0\to$DDS、$c_t=t\to$FlowEdit、$c_t\approx(1-t)t\to$本文，把「能量优化」与「ODE 编辑」两套看似不同的范式统一在同一框架下，并据此论证已有方法的偏移选择是次优的——这种「先统一再指出甜点」的叙事很有说服力。
- **path-aware 的思路可迁移**：把「评估点要落在正确前向后验上」这个观察通用化，任何蒸馏采样类编辑/优化方法都可以引入随时间衰减的轨迹补偿项来减小模型-数据失配。

## 局限与展望
- **编辑强度与背景保真仍是 trade-off**：消融显示提高 CLIP 对齐必然牺牲一些源保真，$c_t$ 只是把这条 trade-off 曲线推得更优，并没有消除它；最优 $c_t$ 形态还是靠分析+经验选的。
- **底座局限在 SD3/SD3.5 整流流**：方法依赖 RF 速度场先验，对其他生成范式需重新推导等价关系。
- **批大小 1、单时间步采样**：梯度用单个 $(t,\varepsilon)$ 估期望，方差可能较大；增大批大小对效果/稳定性的影响文中未充分展开（⚠️ 以原文为准）。
- **改进思路**：让 $c_t$ 可学习或自适应于编辑类型（大尺度替换 vs 局部纹理改动可能需要不同偏移曲线），或按空间区域自适应地调 $c_t$，进一步解开编辑/保留的耦合。

## 相关工作与启发
- **vs DDS**：DDS 在扩散噪声残差上做源/目标相减保背景，DRFS 证明 DDS 就是自己 $c_t=0$ 的特例；DRFS 把它扩到整流流速度场，并加偏移项解决「评估点偏离轨迹」的失配，编辑更强。
- **vs FlowEdit**：FlowEdit 是首个免反演 RF 编辑、靠源/目标速度偏移直接估目标隐变量；DRFS 证明它等价于 $c_t=t$ 的特例，并指出恒定偏移会在早期放大误差，渐进偏移 $c_t\approx(1-t)t$ 更稳。
- **vs RFDS / iRFDS**：RFDS 把 SDS 扩到 RF 速度场但过度平滑；iRFDS 靠先反演噪声补救、要额外算力。DRFS 免反演、不改架构，直接在能量层面消掉过度平滑（LPIPS/MSE/SSIM 全面碾压 iRFDS）。

## 评分
- 新颖性: ⭐⭐⭐⭐ residual subtraction + 时变偏移项把 DDS/FlowEdit 统一进 RF 蒸馏框架，理论清晰
- 实验充分度: ⭐⭐⭐⭐ PIE 基准多指标 + $c_t$ 三特例消融，数据自洽；但主要绕 PIE，跨数据集仅在附录
- 写作质量: ⭐⭐⭐⭐ 公式推导扎实，「先统一再指出甜点」叙事顺畅
- 价值: ⭐⭐⭐⭐ 免反演免训练不改架构，即插即用，对 RF 编辑实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] NAMI: Efficient Image Generation via Bridged Progressive Rectified Flow Transformers](nami_efficient_image_generation_via_bridged_progressive_rectified_flow_transform.md)
- [\[CVPR 2026\] RecTok: Reconstruction Distillation along Rectified Flow](rectok_reconstruction_distillation_along_rectified_flow.md)
- [\[CVPR 2026\] CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)
- [\[CVPR 2026\] Coupled Diffusion Sampling for Training-Free Multi-View Image Editing](coupled_diffusion_sampling_for_training-free_multi-view_image_editing.md)
- [\[CVPR 2026\] FlowDC: Flow-Based Decoupling-Decay for Complex Image Editing](flowdc_flow-based_decoupling-decay_for_complex_image_editing.md)

</div>

<!-- RELATED:END -->
