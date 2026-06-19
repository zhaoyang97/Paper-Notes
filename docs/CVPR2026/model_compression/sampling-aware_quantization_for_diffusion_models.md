---
title: >-
  [论文解读] Sampling-Aware Quantization for Diffusion Models
description: >-
  [CVPR 2026][模型压缩][扩散模型量化] 本文指出扩散模型的「快采样器」和「网络量化」两条加速路线一旦合用就会互相打架——量化噪声会扰动高阶采样器每一步的方向估计、把本应平滑的概率流 ODE 退化成方差爆炸的 SDE，于是提出「采样感知量化」，用一个混合阶轨迹对齐（Mixed-Order Trajectory Alignment）目标把量化后的一阶方向轨迹对齐到全精度高阶方向轨迹，让概率流更线性，从而在稀疏步数下同时拿到「采样加速 + 模型压缩」的双重加速而几乎不掉质。
tags:
  - "CVPR 2026"
  - "模型压缩"
  - "扩散模型量化"
  - "高阶采样器"
  - "概率流ODE"
  - "轨迹对齐"
  - "PTQ"
  - "QLoRA"
---

# Sampling-Aware Quantization for Diffusion Models

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zeng_Sampling-Aware_Quantization_for_Diffusion_Models_CVPR_2026_paper.html)  
**代码**: https://github.com/TaylorJocelyn/Sampling-aware-Quantization  
**领域**: 模型压缩 / 扩散模型 / 量化  
**关键词**: 扩散模型量化, 高阶采样器, 概率流ODE, 轨迹对齐, PTQ, QLoRA

## 一句话总结
本文指出扩散模型的「快采样器」和「网络量化」两条加速路线一旦合用就会互相打架——量化噪声会扰动高阶采样器每一步的方向估计、把本应平滑的概率流 ODE 退化成方差爆炸的 SDE，于是提出「采样感知量化」，用一个混合阶轨迹对齐（Mixed-Order Trajectory Alignment）目标把量化后的一阶方向轨迹对齐到全精度高阶方向轨迹，让概率流更线性，从而在稀疏步数下同时拿到「采样加速 + 模型压缩」的双重加速而几乎不掉质。

## 研究背景与动机

**领域现状**：扩散模型生成质量高但速度慢，瓶颈有二——去噪链太长、噪声估计网络太重。学界分两条线加速：一是设计高阶**快采样器**（DPM-Solver、DDIM、PLMS 等），用数值方法在更大步长下精确逼近反向 SDE/ODE；二是用**量化**压缩噪声估计网络，把 FP32 权重/激活转成低比特定点，降显存和单步算力。

**现有痛点**：这两条线一直被当成互相独立的模块各自优化。但把它们直接叠加时，量化引入的噪声 $\Delta\epsilon_\theta$ 会扰动采样器的方向估计。问题在高阶采样器上尤其严重：$k$ 阶采样器要在区间 $(t_{i-1},t_i)$ 内额外评估 $k-1$ 个中间点 $\{s_j\}$ 来联合估计方向，量化噪声会同时扰动每个中间点的**位置**和该点的**方向估计**，让中间点随时间漂移、最终污染整段联合方向。

**核心矛盾**：高阶采样器的「多中间步联合方向估计」本是为压低截断误差、支撑稀疏步快采样而设计的；但量化噪声不仅破坏它的快速收敛潜力，更可能把原本确定、平滑的概率流 ODE 变成一个**方差爆炸的 SDE**，诱发轨迹弥散（trajectory diffusion），在 W4A4 这类低比特下甚至直接生成失败。

**本文目标**：做到「高保真双重加速」——既要量化压缩网络，又要保住高阶快采样器的稀疏步收敛能力，让两种加速叠加而不互相抵消。

**切入角度**：作者把量化误差放到「采样加速原理」的视角下分析。通过对数值积分的误差上界拆解，发现总误差 $L_\Delta=L_{\Delta_{\text{quant}}}+L_{\Delta_{\text{disc}}}$ 中，量化累积误差项由方向偏差 $\delta$ 直接主导、且被模型非线性放大，远超离散截断误差项。结论是：要把 $\delta$ 压到与离散步长 $O(\lambda_t-\lambda_s)$ 同阶，就能约束误差上界、恢复收敛。

**核心 idea**：重新设计量化方案去**学一个更线性的概率流**——不再让量化只去最小化「量化前后张量的 MSE」，而是让量化后的低阶采样方向轨迹去对齐全精度高阶采样方向轨迹（混合阶轨迹对齐），把方向偏差 $\delta$ 压住、避免误差快速累积导致采样弥散。

## 方法详解

### 整体框架
方法的输入是一个预训练 FP32 扩散网络 $\epsilon_\theta$ 和目标采样器，输出是一个低比特量化网络，使其在稀疏步高阶采样下仍保真。核心是把量化目标从「逐张量 MSE 对齐」改成「混合阶轨迹对齐」：让量化后的一阶采样方向 $\hat\epsilon_\theta(x_{\lambda_t},\lambda_t)$ 去对齐全精度高阶采样在中间节点处的方向 $\epsilon_\theta(x_{\lambda_s},\lambda_s)$，从而把概率流线性化。围绕这个目标，作者给两个比特档位各实例化一套：8 比特用后训练量化变体 SA-PTQ（双阶轨迹采样 + 模块级重建校准），4 比特等低位用 SA-QLoRA（加一个方向余弦约束的 QLoRA 微调）；并把对齐思路推广到 DDIM/PLMS 等通用采样器。这不是一条多模块串行的流水线，而是一个量化目标的重设计 + 两种部署实例，因此用公式说明、不画框架图。

### 关键设计

**1. 混合阶轨迹对齐（Mixed-Order Trajectory Alignment）：让量化后的一阶方向去学全精度的高阶方向**

这是全文的核心。先厘清「轨迹」到底对齐什么：采样轨迹 $\{x_t\}$ 由采样器、初始点 $x_T$、时间表三者唯一决定，而每步方向 $\epsilon_\theta(x_t,t)$ 与样本轨迹一一对应，所以**对齐轨迹本质就是对齐方向序列** $\{\epsilon_\theta(x_t,t)\}$。传统量化目标只是逐点最小化 $\mathbb{E}\lVert\epsilon_\theta-\hat\epsilon_\theta\rVert^2$。本文改成：把量化后的一阶方向轨迹对齐到全精度高阶方向轨迹在中间节点 $s$ 处的值

$$\arg\min_{s,z}\ \mathbb{E}_{(x_t,t)\sim D,(x_s,s)\sim S}\lVert\hat\epsilon_\theta(x_{\lambda_s},\lambda_s)-\epsilon_\theta(x_{\lambda_t},\lambda_t)\rVert^2$$

直觉来自高阶采样器的工作机制：$k$ 阶采样器在区间内评估 $k-1$ 个中间点、每个 $\epsilon_\theta(x_{\lambda_{s_i}},\lambda_{s_i})$ 都编码了高阶导数信息来精化方向。作者借此让量化的一阶方向去逼近这些携带高阶信息的中间方向，等于在量化过程中「内化」了高阶精度，把方向偏差 $\delta$ 压到 $O(\lambda_t-\lambda_s)$ 同阶，使概率流更线性、误差不再快速累积。以 DPM-Solver 为例：DPM-Solver-2 会先在 $\lambda$ 区间中点 $s_i$ 算 $u_i$、再用 $\epsilon_\theta(u_i,s_i)$ 更新 $\tilde x_{t_i}$；本文就把量化方向项 $\hat\epsilon_\theta(\tilde x_{t_{i-1}},t_{i-1})$ 对齐到全精度的 $\epsilon_\theta(u_i,s_i)$，从而线性化高阶轨迹。

**2. SA-PTQ：双阶轨迹采样 + 模块级重建校准，服务 8 比特后训练量化**

针对 W8A8/W4A8，作者把对齐落到后训练量化（无需大规模重训，只校准少量参数）。以 BRECQ 为基线、用 Adaround 做权重量化、只训极少参数 $\alpha$。校准分两步：① **双阶轨迹采样**——同一个初始 $x_T$，先用一阶采样器在每个去噪步收集输入样本得一阶轨迹，再用二阶采样器在每个区间中间点 $s_i$ 收集样本得二阶轨迹；② **混合阶对齐校准**——对每个待重建模块 $f_i$ 及其量化版 $\hat f_i$，逐模块独立最小化跨阶对齐误差

$$\arg\min_\alpha\ \mathbb{E}_{(t_j,s_j)}\lVert f_i(x_{t_j},t_j,\text{cond})-\hat f_i(x_{s_j},s_j,\text{cond})\rVert^2$$

即把量化模块在中间步 $s_j$ 的输出对齐到全精度模块在一阶步 $t_j$ 的输出。这样在不重训主干的前提下，让 8 比特量化也带上高阶方向信息。

**3. SA-QLoRA：加方向余弦约束的双感知 LoRA，服务 4 比特等低比特**

低比特（如 W4A4）下 PTQ 噪声太大，作者把混合阶对齐与 QLoRA 结合。基础 QLoRA 联合优化 LoRA 权重 $w$ 和量化参数 $s,z$。因为方向 $\epsilon_\theta$ 直接决定采样走向，作者在 L2 形式的对齐损失外**再加一个方向余弦约束**，强调方向（而非仅幅度）的一致性：

$$L_{\text{COS}}=1-\frac{\langle\epsilon_\theta(x_{t_i},t_i),\hat\epsilon_\theta(x_{s_i},s_i)\rangle}{\lVert\epsilon_\theta(x_{t_i},t_i)\rVert\,\lVert\hat\epsilon_\theta(x_{t_i},t_i)\rVert}$$

$$L_{\text{MOTA}}=\mathbb{E}_{(t_i,s_i)}\lVert\hat\epsilon_\theta(x_{s_i},s_i)-\epsilon_\theta(x_{t_i},t_i)\rVert^2,\quad \arg\min_{w,s,z}\ L_{\text{COS}}+L_{\text{MOTA}}$$

其中 $\{t_i\}$ 是一阶采样点、$\{s_i\}$ 是二阶的额外中间评估点。余弦项保方向、L2 项保幅度，二者合力在极低比特下也能维持稳定收敛，避免概率流 ODE 退化成方差爆炸 SDE。

**4. 推广到通用采样器：DDIM 与 PLMS 的对齐适配**

为提升通用性，作者把对齐思路扩到经典数值解法采样器。DDIM 被证明与 DPM-Solver-1 更新等价，可视为特定噪声表下的一阶轨迹实例；PLMS（基于 PNDM）把采样方程拆成梯度部分与传递部分、用线性多步法估计梯度项 $\epsilon_\theta^{(t)}$，不同阶的数值方法对应不同阶轨迹，于是同样可通过对齐「不同阶数值方法导出的 $\epsilon_\theta^{(t)}$」实现混合阶轨迹对齐。这说明该框架不绑定单一采样器。

### 损失函数 / 训练策略
SA-PTQ 用 Adaround 仅训 $\alpha$，逐模块最小化混合阶对齐 MSE；SA-QLoRA 联合训 LoRA 权重 $w$ 与量化参数 $s,z$，目标为 $L_{\text{COS}}+L_{\text{MOTA}}$。类条件生成用 DPM-Solver-1/2 作低/高阶采样器，steps=20、scale=1.5；无条件生成 steps=50；文生图对齐原生 PLMS 与其降一阶版本，steps=50、scale=7.5。

## 实验关键数据

### 主实验
在 ImageNet 256×256（LDM-4 类条件，DPM-Solver-2，20 步）、LSUN-Bedroom/Church 256×256（无条件）、MS-COCO 512×512（SD-v1.4 文生图）上评测，指标含 FID/sFID/IS/Precision/Recall，效率用 BOPs。

| 配置 (W/A) | 方法 | BOPs(T) | IS↑ | FID↓ | sFID↓ |
|------|------|------|------|------|------|
| 32/32 | FP（全精度） | 102.20 | 174.33 | 9.45 | 8.08 |
| 8/8 | PTQD | 8.76 | 122.46 | 10.76 | 10.58 |
| 8/8 | **SA-PTQ (ours)** | 8.76 | 120.71 | **10.16** | **9.89** |
| 4/8 | EfficientDM | 4.38 | 132.70 | 9.91 | 8.76 |
| 4/8 | **SA-QLoRA (ours)** | 4.38 | **140.56** | **8.55** | **8.51** |
| 4/4 | EfficientDM | 2.19 | 225.20 | 17.28 | 13.78 |
| 4/4 | **SA-QLoRA (ours)** | 2.19 | **242.03** | **13.73** | **12.45** |

W8A8/W4A8/W4A4 分别达 3.99×/7.95×/7.95× 的比特压缩与 11.47×/23.33×/46.67× 的位运算加速。在 W4A8 下 SA-QLoRA 的 FID 8.55 甚至比全精度模型还低 0.9；W4A4 下其它方法因量化噪声把概率流 ODE 变成方差爆炸 SDE 而**生成失败**（PTQ4DM/Q-diffusion/PTQD 无有效结果），SA-QLoRA 仍能收敛、sFID 仅比全精度高 4.37。

### 消融 / 跨设置一致性

| 设置 | 现象 | 说明 |
|------|------|------|
| W8A8 (SA-PTQ) | FID 10.16 < PTQD 10.76 | 后训练档位即超此前 PTQ SOTA |
| W4A8 (SA-QLoRA) | FID 8.55，IS 140.56 | 优于全精度 FID，量化反带正则效应 |
| W4A4 baseline | 多数方法生成失败 | 量化噪声诱发轨迹弥散（ODE→VE-SDE） |
| W4A4 (SA-QLoRA) | FID 13.73，正常收敛 | 混合阶对齐压住误差累积 |

### 关键发现
- **混合阶轨迹对齐确实让概率流更线性**：各比特档位的稳定指标共同印证它有效抑制了高阶采样器在量化下的误差快速累积，这是稀疏步仍保真的根因。
- **低比特才是真考验**：W4A4 下其它方法因 ODE 退化成方差爆炸 SDE 直接崩，本文方法是唯一能正常收敛的，凸显「采样感知」相对「逐张量 MSE」量化的本质优势。
- **量化偶尔带来轻微正则**：W4A8 下 FID 反低于全精度，说明对齐高阶方向轨迹在压缩同时也平滑了采样过程。

## 亮点与洞察
- **把「采样加速」和「网络量化」统一到一个误差视角**：通过误差上界拆解证明量化累积误差项主导、且由方向偏差 $\delta$ 控制，从而把量化目标从「最小化张量 MSE」重新表述为「学更线性的概率流」——这个问题重述是全文最关键的「啊哈」。
- **用高阶采样器的中间方向当监督信号**很巧：让量化的一阶方向去对齐携带高阶导数信息的中间方向，相当于把高阶精度「蒸馏」进量化网络，思路可迁移到任何「低阶逼近高阶轨迹」的加速场景。
- **方向余弦约束** $L_{\text{COS}}$ 单独拎出方向（而非幅度）一致性，对扩散这种「方向决定走向」的过程是恰当的归纳偏置，在极低比特下尤其救命。
- 同一框架按比特档位拆成 SA-PTQ / SA-QLoRA 两个实例，工程上覆盖了从 8 比特免训到 4 比特微调的实际部署需求。

## 局限与展望
- 分析与主要实例以 DPM-Solver 为案例，虽推广到 DDIM/PLMS，但对更多样的随机采样器（SDE 系）和更大规模文生图模型的系统验证有限。
- 需要全精度高阶轨迹作为对齐目标，意味着校准/微调阶段要额外跑全精度采样收集中间点，校准成本相对纯逐张量 PTQ 更高。
- W4A4 虽能收敛，但 sFID 仍比全精度高约 4.37，极低比特下的画质差距尚未完全抹平。
- 混合阶对齐对采样器阶数、步数、中间点选择的敏感性，以及与不同噪声表的交互，论文主文讨论有限（部分推到附录）。

## 相关工作与启发
- **vs PTQ4DM / Q-Diffusion / PTQD**：它们把传统量化适配到扩散的多时间步框架（时间步校准、误差校正），但都忽略了量化噪声对高速采样的影响；本文从采样加速原理出发做联合优化，W4A4 下别人崩它不崩。
- **vs EfficientDM**：同走 QLoRA 低比特路线，但 EfficientDM 仍是逐步 MSE 监督；SA-QLoRA 加了混合阶对齐 + 方向余弦约束，W4A8/W4A4 的 FID/IS 全面更优。
- **vs 高阶采样器（DPM-Solver/DDIM/PLMS）**：以往这些工作只在全精度下追求快采样，默认与量化正交；本文揭示二者叠加会互相破坏，并给出弥合方案，把「采样器 × 量化」从相加变成协同。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从采样加速原理审视量化误差、提出混合阶轨迹对齐统一两条加速线，视角新。
- 实验充分度: ⭐⭐⭐⭐ 覆盖类条件/无条件/文生图与多比特档位，W4A4 对比突出，但采样器多样性和大模型验证有限。
- 写作质量: ⭐⭐⭐⭐ 误差分析推导清晰、动机有力；公式记号偏密、部分细节推到附录。
- 价值: ⭐⭐⭐⭐⭐ 解决了「快采样 + 量化」叠加失效这一实际部署痛点，低比特下尤为实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DeltaQuant: 4-bit Video Diffusion Models with Spatiotemporal Delta Smoothing](deltaquant_4-bit_video_diffusion_models_with_spatiotemporal_delta_smoothing.md)
- [\[ECCV 2024\] Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling](../../ECCV2024/model_compression/adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)
- [\[ICML 2025\] Diffusion Sampling Correction via Approximately 10 Parameters](../../ICML2025/model_compression/diffusion_sampling_correction_via_approximately_10_parameters.md)
- [\[CVPR 2026\] DMGD: Train-Free Dataset Distillation with Semantic-Distribution Matching in Diffusion Models](dmgd_train-free_dataset_distillation_with_semantic-distribution_matching_in_diff.md)
- [\[ICML 2026\] FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for Post-Training Quantization of Diffusion Large Language Models](../../ICML2026/model_compression/fair-calib_frontier-aware_instability-reweighted_calibration_for_post-training_q.md)

</div>

<!-- RELATED:END -->
