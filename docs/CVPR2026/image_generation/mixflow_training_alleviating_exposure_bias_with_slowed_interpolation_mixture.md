---
title: >-
  [论文解读] MixFlow Training: Alleviating Exposure Bias with Slowed Interpolation Mixture
description: >-
  [CVPR 2026][图像生成][扩散模型] 针对扩散/流匹配模型「训练用真值插值、测试用自己生成的噪声数据」导致的曝光偏差，本文发现一个 Slow Flow 现象——采样时刻 $t$ 生成的噪声数据最接近的真值插值其实对应一个更高噪声（更慢）的时刻 $m_t \le t$，于是提出 MixFlow：训练时把输入插值换成「慢化时刻区间内的插值混合」，只改 5 行代码做后训练，就把 RAE 在 ImageNet 上做到 1.43 FID（无引导）/ 1.10 FID（有引导）。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散模型"
  - "流匹配"
  - "曝光偏差"
  - "训练-测试不一致"
  - "后训练"
---

# MixFlow Training: Alleviating Exposure Bias with Slowed Interpolation Mixture

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Li_MixFlow_Training_Alleviating_Exposure_Bias_with_Slowed_Interpolation_Mixture_CVPR_2026_paper.html)  
**代码**: 项目页 https://mixflowgen.github.io/  
**领域**: 图像生成 / 扩散模型  
**关键词**: 扩散模型, 流匹配, 曝光偏差, 训练-测试不一致, 后训练

## 一句话总结
针对扩散/流匹配模型「训练用真值插值、测试用自己生成的噪声数据」导致的曝光偏差，本文发现一个 Slow Flow 现象——采样时刻 $t$ 生成的噪声数据最接近的真值插值其实对应一个更高噪声（更慢）的时刻 $m_t \le t$，于是提出 MixFlow：训练时把输入插值换成「慢化时刻区间内的插值混合」，只改 5 行代码做后训练，就把 RAE 在 ImageNet 上做到 1.43 FID（无引导）/ 1.10 FID（有引导）。

## 研究背景与动机
**领域现状**：扩散与流匹配模型通过学习一个预测网络（预测噪声/数据/score/velocity）把噪声逐步转成数据。训练时，某个时刻 $t$ 喂给网络的输入是真值噪声数据 $x_t$，即噪声与数据的插值 $x_t = \alpha_t x_1 + \beta_t x_0$；测试时，喂进去的却是上一步**自己生成**的噪声数据。

**现有痛点**：这两种输入并不一致——这就是「曝光偏差 / 训练-测试不一致」。训练时网络从没见过自己生成的（带误差的）中间态，测试时却要在这些态上做预测，误差会沿采样轨迹不断累积（error accumulation、sampling drift），最终拉低生成质量。

**核心矛盾**：已有两条路线都不够好。改训练的（Input Perturbation 给真值加高斯扰动模拟测试误差）对扰动强度极其敏感，太大太小都更差；Self-Forcing 直接用生成数据当输入，但对标准多步扩散计算代价过高，只适合少步。改采样的（Epsilon Scaling 缩放预测噪声、Time-Shift Sampler 启发式平移采样时刻）都是推理期的启发式修正，校正不准。

**本文目标**：从训练侧，找到一种能真正贴近「测试时网络实际会遇到什么输入」的真值插值来做训练，而且要简单、几乎零额外开销。

**切入角度**：作者去量化「测试时生成的噪声数据，到底最像哪个真值时刻的插值」。在 ImageNet 上用 SiT-B、50 步采样、2 万张图统计后发现一个稳定现象——采样时刻 $t$ 生成的噪声数据，最近邻的真值插值落在一个**更高噪声**的时刻 $m_t \le t$（称为 slowed timestep，慢化时刻）；且 $t$ 越大，$m_t$ 与 $t$ 的偏差范围越大。直觉是「生成的数据比真值数据走得更慢」。

**核心 idea**：既然测试时刻 $t$ 的生成数据其实更像更高噪声时刻 $m_t$ 的真值插值，那训练时就别只喂时刻 $t$ 的插值，改喂「$[(1-\gamma)t,\,t]$ 这段慢化时刻区间内的插值混合」，让网络在每个训练时刻见过它测试时真正会遇到的那类（更慢的）输入。

## 方法详解

### 整体框架
MixFlow 不改网络结构、不改采样过程，只改**训练时喂给预测网络的输入插值和对应的损失**，是一个即插即用的后训练（post-training）方案。

标准训练：在训练时刻 $t$，取真值插值 $x_t=\alpha_t x_1+\beta_t x_0$ 作为输入，让网络 $u_\theta(x_t,t)$ 回归该时刻的真值速度 $u^*(x_t,t)$。

MixFlow 训练：在训练时刻 $t$，输入改成**慢化时刻 $m_t$ 处的插值** $x_{m_t}$（$m_t\le t$，更高噪声），但**喂给网络的时刻标签仍是 $t$**（不是 $m_t$），回归目标变成慢化时刻的真值速度 $u^*(x_{m_t},m_t)$。整套采样过程与标准训练完全相同，推理时也**无需计算 $m_t$**。

落到一次迭代上（Algorithm 1），只是在标准训练循环里插入两步采样、改一个插值和损失：采数据 $x_1$、采噪声 $x_0$、采训练时刻 $t\sim\mathrm{Beta}(2,1)$、采慢化时刻 $m_t\sim\mathcal{U}[(1-\gamma)t,\,t]$、算 $x_{m_t}=\beta_{m_t}x_0+\alpha_{m_t}x_1$、算损失 $\lVert u_\theta(x_{m_t},t)-u^*\rVert_2^2$ 后反传。官方实现里，接到 RAE 上仅改 5 行代码。

### 关键设计

**1. Slow Flow 现象：把曝光偏差量化成「时刻被拖慢」**

曝光偏差以前是个定性概念（训练/测试输入不一样），本文把它量化成一个可观测、可利用的规律。作者统计「采样时刻 $t$ 生成的噪声数据，最近邻的真值插值对应哪个时刻」，发现它系统性地落在 $m_t\le t$（更高噪声）一侧，即图 1 里阴影区始终压在对角线 $x=y$ 下方；而且 $t$ 越大，$m_t$ 的可能范围越宽。物理含义是：生成数据沿轨迹「走得偏慢」，到达 $t$ 时其实更像真值在更早（更噪）时刻的状态。这一现象是整个方法的地基——它告诉我们训练时该把输入往「更高噪声」方向偏，而不是像 Input Perturbation 那样无方向地加噪。

**2. 慢化插值混合（Slowed Interpolation Mixture）：用一段高噪声区间的插值替代单点插值**

标准训练每个时刻 $t$ 只用一个插值 $x_t$ 当输入，自然覆盖不到测试时实际遇到的「慢化态」。MixFlow 把单点输入换成一个**插值集合**

$$\mathcal{X}_t=\{x_{m_t}\mid x_{m_t}=\beta_{m_t}x_0+\alpha_{m_t}x_1,\ m_t\in\mathcal{M}_t\},$$

其中混合区间按 Slow Flow 现象取为

$$\mathcal{M}_t=[(1-\gamma)t,\ t].$$

区间长度 $\gamma t$ 与训练时刻线性相关——这正对应「$t$ 越大、慢化偏差越大」的观测，所以越靠近数据端给越宽的混合范围。$\gamma$ 控制往高噪声方向延伸多远，可经验选取或直接设为 1。这个设计的关键在于**只往高噪声一侧扩**（$m_t\le t$）：消融显示一旦把低噪声时刻 $\mathcal{U}[0,1]$ 也纳入混合，性能反而比 baseline 还差，因为低噪声插值不是测试时会遇到的状态，会污染训练。

**3. 慢化插值损失：输入用慢化插值、标签仍是训练时刻**

MixFlow 的损失在标准损失上多引入一个变量——慢化时刻 $m_t$：

$$\mathbb{E}_{t,x_0,x_1,m_t}\big[\lVert u_\theta(x_{m_t},t)-u^*(x_{m_t},m_t)\rVert_2^2\big].$$

对比标准损失 $\mathbb{E}_{t,x_0,x_1}[\lVert u_\theta(x_t,t)-u^*(x_t,t)\rVert_2^2]$，区别有两点：输入插值从 $x_t$ 换成更高噪声的 $x_{m_t}$；回归目标从 $u^*(x_t,t)$ 换成 $u^*(x_{m_t},m_t)$（慢化时刻的真值速度）。但网络拿到的**时刻条件仍是 $t$**，于是网络学到的是「当我被告知现在是时刻 $t$、但输入实际更像更慢的态时，该怎么预测」——这恰好对齐了测试时的真实处境。

**4. 训练时刻 Beta(2,1) 采样：让 (m_t, t) 对在全空间被均匀覆盖**

慢化时刻按条件分布 $m_t\sim\mathcal{U}[(1-\gamma)t,\,t]$ 采，其条件密度 $p(m_t|t)=\frac{1}{\gamma t}$ 随 $t$ 变化（区间越窄密度越高）。如果训练时刻 $t$ 还按常规 $\mathcal{U}[0,1]$ 采，那么各个 $(m_t,t)$ 对被采到的概率就不均匀，小 $t$ 区域被过采。作者要求所有输入对 $(x_{m_t},t)$ 被**均匀采样** $p(m_t,t)=p(m_{t'},t')$，由 $p(m_t,t)=p(m_t|t)p(t)$ 推得 $p(t)\propto t$，再结合归一化 $\int_0^1 p(t)=1$ 解出

$$t\sim\mathrm{Beta}(2,1)\quad(p(t)=2t).$$

消融证实：在 $m_t\sim\mathcal{U}[0,t]$ 前提下，$t$ 用 Beta(2,1) 比用 $\mathcal{U}[0,1]$ 明显更好（gFID 15.64 vs 16.57），说明「均匀覆盖 $(m_t,t)$ 对」这一推导确实带来增益，而非可有可无的工程细节。

### 损失函数 / 训练策略
- 训练目标即上式慢化插值损失，velocity 为预测目标；GVP 扩散设定下作者还换了一种真值速度目标 $u^*(x_t,t)$，发现能加速收敛（50 万步即可达到旧目标需 250 万步的效果）。
- 关键超参 $\gamma$：在 $\{0.3,0.5,0.7,0.8,0.9,1.0\}$ 上，$0.7\sim1.0$ 都好且接近，$\gamma=0.8$ 略优（默认值）；$\gamma=0$（退化为标准训练）甚至略差于 baseline。过度慢化（如 $m_t\in[0,0.2t]$）无益，因为标准模型在 $t$ 处不太可能被拖慢到那么早。
- 全程为后训练：在官方放出的预训练模型上接着训（SiT 后训 50 万步、RAE 后训 200 epoch、SD3.5 后训 1 万步），训练/评测超参沿用各原模型设置。

## 实验关键数据

### 主实验
ImageNet class-conditional 生成（second-order Heun，250 步，引导 1.5；RAE 用其自身设置 50 步 auto-guidance）：

| 模型 / 设置 | gFID w/o 引导 | gFID w/ 引导 |
|------|------|------|
| SiT-B/2 (256) | 17.97 | 4.46 |
| SiT-B/2 + MixFlow | **15.64** | **3.91** |
| SiT-XL/2 (256) | 9.35 | 2.15 |
| SiT-XL/2 + MixFlow | **7.87** | **1.99** |
| SiT-XL/2 (512) | 9.72 | 2.71 |
| SiT-XL/2 + MixFlow (512) | **7.99** | **2.53** |
| REPA-XL/2 | 6.90 | 1.65 |
| REPA-XL/2 + MixFlow | **6.28** | **1.59** |
| RAE-XL | 1.51 | 1.13 |
| RAE-XL + MixFlow | **1.43** | **1.10** |

与曝光偏差专门方法对比（SiT-B 基座）：MixFlow 改进幅度最大；Input Perturbation 需精调扰动强度，Time-Shift / Epsilon Scaling 是推理期启发式校正、不如直接改训练。

采样步数越少增益越大（gFID w/o 引导，SiT-B/2）：

| 步数 | 250 | 50 | 20 |
|------|------|------|------|
| 提升量 (Gain) | 2.33 | 2.34 | 2.58 |

原因是步数越少采样器近似越粗、训练-测试差距越大，Slow Flow 问题越突出。

### 消融实验

| 配置 | gFID w/o 引导 / w/ 引导 | 说明 |
|------|------|------|
| $t\sim\mathcal{U}[0,1],\ m_t\sim\mathcal{U}[0,1]$ | 18.27 / 5.07 | 纳入低噪声时刻，比 baseline 还差 |
| $t\sim\mathrm{Beta}(2,1),\ m_t\sim\mathcal{U}[0,1]$ | 18.25 / 5.06 | 同上，低噪声插值有害 |
| $t\sim\mathcal{U}[0,1],\ m_t\sim\mathcal{U}[0,t]$ | 16.57 / 4.25 | 只取高噪声，已超 baseline |
| $t\sim\mathrm{Beta}(2,1),\ m_t\sim\mathcal{U}[0,t]$（Full） | **15.64 / 3.93** | 完整方案 |
| 标准后训练（同步数） | 17.96 / 4.46 | 几乎不变，增益非来自多训 |
| 无后训练 baseline | 17.97 / 4.46 | — |

### 关键发现
- **方向比强度更重要**：只往高噪声一侧混合（$m_t\sim\mathcal{U}[0,t]$）才有效，纳入低噪声时刻会反向掉点——这把 Input Perturbation「无方向加噪需精调」的痛点解释清楚了。
- **增益来自训练方案而非更多训练步**：同步数标准后训练 17.96 ≈ 17.97（无后训练），MixFlow 才把它压到 15.64。
- **越少步、越大 $t$ 增益越大**，与 Slow Flow 现象（$t$ 越大慢化偏差越大、步数越少近似越粗）自洽。
- 普适性强：对线性插值流匹配（SiT）、方差保持扩散（GVP）、表示对齐训练（REPA）、表示自编码器（RAE）、文生图（SD3.5）均有效。

## 亮点与洞察
- **把模糊的「曝光偏差」量化成一个可利用的方向性现象（Slow Flow）**：不再泛泛说训练测试不一致，而是测出「生成数据对应更高噪声时刻 $m_t\le t$ 且偏差随 $t$ 增大」，方法直接照这个规律设计混合区间 $[(1-\gamma)t,t]$，动机和实现严丝合缝。
- **极低改动成本**：不动结构、不动采样、推理零额外开销，接 RAE 仅改 5 行代码就刷到 SOTA 级 FID，工程上几乎白嫖。
- **采样分布的严谨推导**：从「均匀覆盖 $(m_t,t)$ 对」这一目标解析地推出 $t\sim\mathrm{Beta}(2,1)$，而不是拍脑袋调，消融也证明它确实贡献增益——这个「为保证某统计量均匀而反解采样分布」的思路可迁移到其他需要联合采样时刻/噪声的训练方案。

## 局限与展望
- **依赖一个已收敛的预训练模型做后训练**，Slow Flow 现象本身也是基于已训练模型统计出来的；从零开始训练时该现象是否成立、是否同样有效，文中未直接验证。
- $\gamma$ 在不同模型上最优值不同（SiT 用 0.8、RAE 用 0.4），仍需少量调参；作者未给出按模型/数据自动确定 $\gamma$ 的方法。
- 增益在大模型/多步设置下变小（如 SiT-XL/2 w/ 引导仅 2.15→1.99、RAE 仅 1.51→1.43），更适合少步、训练-测试差距大的场景。
- 慢化插值的「最近邻对应更高噪声时刻」是统计观测，文中以直觉+toy example 解释，缺乏更严格的理论刻画；GVP 下换 velocity 目标能加速收敛的机理也只是 speculate。

## 相关工作与启发
- **vs Input Perturbation**：都改训练，但 Input Perturbation 给真值无方向加高斯噪声、对强度极敏感（过大过小都更差），MixFlow 借 Slow Flow 给出明确方向（只往高噪声混合），消融显示纳入低噪声反而有害，等于解释了前者为何难调。
- **vs Self-Forcing**：Self-Forcing 直接用生成数据当输入、对标准多步扩散计算代价过高（只适合少步/自回归视频），MixFlow 用真值的慢化插值近似生成态，开销几乎为零、适配标准多步训练。
- **vs Epsilon Scaling / Time-Shift Sampler**：二者是推理期启发式（缩放噪声 / 平移采样时刻）校正轨迹，校正不准；MixFlow 改训练、从源头对齐输入分布，且与采样过程解耦，实测优于两者。

## 评分
- 新颖性: ⭐⭐⭐⭐ Slow Flow 现象的量化观测和「慢化插值混合」切入点清晰且新，但本质仍属曝光偏差大家族的一个训练侧改良。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 SiT/GVP/REPA/RAE/SD3.5、多分辨率多步数、与三类曝光偏差方法横比，消融解释到位。
- 写作质量: ⭐⭐⭐⭐ 现象→方法→推导→实验逻辑顺，但部分公式排版（缓存里）较乱，需对照原文。
- 价值: ⭐⭐⭐⭐⭐ 即插即用、几乎零成本就能把已有强模型再推一截（RAE 做到 1.10 FID），实用性高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Linear Image Generation by Synthesizing Exposure Brackets](linear_image_generation_by_synthesizing_exposure_brackets.md)
- [\[CVPR 2026\] Elucidating the SNR-t Bias of Diffusion Probabilistic Models](dcw_snr_t_bias_diffusion.md)
- [\[CVPR 2026\] Mixture of Style Experts for Diverse Image Stylization](mixture_of_style_experts_for_diverse_image_stylization.md)
- [\[ICLR 2026\] MOLM: Mixture of LoRA Markers](../../ICLR2026/image_generation/molm_mixture_of_lora_markers.md)
- [\[CVPR 2026\] TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts](tag-moe_task-aware_gating_for_unified_generative_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
