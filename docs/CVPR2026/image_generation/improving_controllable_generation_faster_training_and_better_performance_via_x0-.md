---
title: >-
  [论文解读] Improving Controllable Generation: Faster Training and Better Performance via x0-Supervision
description: >-
  [CVPR 2026][图像生成][可控生成] 本文指出 ControlNet 这类可控生成方法沿用底模的 $\epsilon$-监督损失其实是次优的——因为 $\epsilon$-损失等价于按信噪比加权的 $x_0$-损失，会把决定全局布局的早期去噪步几乎压成零权重；改成直接监督干净图像 $x_0$（即去掉这个加权），在 ControlNet / T2I-Adapter / GLIGEN / OminiControl 上把收敛速度最高加快约 2×（用作者新提的 mAUCC 指标衡量），同时画质和控制保真度也一起提升。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "可控生成"
  - "ControlNet"
  - "x0监督"
  - "扩散训练"
  - "收敛加速"
---

# Improving Controllable Generation: Faster Training and Better Performance via x0-Supervision

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Sangare_Improving_Controllable_Generation_Faster_Training_and_Better_Performance_via_x0-Supervision_CVPR_2026_paper.html)  
**代码**: https://github.com/CEA-LIST/x0-supervision  
**领域**: 图像生成  
**关键词**: 可控生成, ControlNet, x0监督, 扩散训练, 收敛加速

## 一句话总结
本文指出 ControlNet 这类可控生成方法沿用底模的 $\epsilon$-监督损失其实是次优的——因为 $\epsilon$-损失等价于按信噪比加权的 $x_0$-损失，会把决定全局布局的早期去噪步几乎压成零权重；改成直接监督干净图像 $x_0$（即去掉这个加权），在 ControlNet / T2I-Adapter / GLIGEN / OminiControl 上把收敛速度最高加快约 2×（用作者新提的 mAUCC 指标衡量），同时画质和控制保真度也一起提升。

## 研究背景与动机
**领域现状**：文生图（T2I）扩散/流模型在视觉质量和语义对齐上已经很强，但单靠文本难以精确指定布局（哪个物体放哪、什么姿态）。可控生成的主流做法是给冻结的预训练 T2I 模型挂一个 adapter（如 ControlNet、T2I-Adapter、GLIGEN、OminiControl），喂入分割图/深度图/边缘/姿态/bounding box 等额外控制信号，再训练这个 adapter 去服从这些信号。

**现有痛点**：几乎所有这些方法在训练 adapter 时，都直接沿用底模原来的训练损失——Stable Diffusion 1.4/1.5 是 $\epsilon$-预测器，就继续用 $\epsilon$-监督损失；FLUX.1 是速度 $u$-预测器，就继续用 $u$-监督损失。这"看起来很自然"，但在某些任务上收敛极慢，尤其是控制信号和目标图像不在空间上对齐的情况（如 GLIGEN 用 bounding box 坐标控布局），动辄要几十万步、超大 batch、数百 GB 显存才能收敛。

**核心矛盾**：作者从去噪动力学切入，发现两个被忽略的事实。其一，扩散采样是**由粗到细**的：早期步（$t$ 接近 $T$）决定整张图的布局，后期步只是在固定布局上加细节，而布局一旦在早期定错，后期无法结构性纠正——对空间控制任务而言早期步至关重要。其二，$\epsilon$-监督损失**隐式地**就是一个 $x_0$-监督损失乘以信噪比 $\text{SNR}=\alpha_t^2/\sigma_t^2$ 的加权；而 SNR 在早期步（低信噪比）迅速趋于 0，于是恰恰把"最该学的早期布局步"的学习信号几乎压没了。

**本文目标**：找一种几乎零成本、又能跨范式（扩散/流匹配）跨架构（UNet/DiT）通用的训练目标改造，让可控生成既收敛更快、又最终性能更好。

**切入角度**：既然 $\epsilon$-损失 = SNR 加权的 $x_0$-损失，而这个加权恰好抑制了对可控生成最关键的早期步，那就**把任意预测器先换算成 $x_0$-预测，再直接用干净图像 $x_0$ 监督**——等价于去掉那个有害的 SNR 加权。又因为控制信号和目标图强相关，早期步预测 $x_0$ 的任务本来就比纯 T2I 容易，去掉加权后模型能更快学会"一开始就把布局画对"。

**核心 idea**：用 $x_0$-监督（监督干净图像，等价于给原 $\epsilon$-损失乘 $1/\text{SNR}$）替换沿用至今的 $\epsilon$-监督，恢复早期去噪步的学习信号，从而加速可控生成训练。

## 方法详解

### 整体框架
方法本质是一个**训练目标的改造**，不改网络结构、不改采样器、不引入任何推理开销，因此不画 pipeline 图。给定一个在 T2I 上预训练好的扩散模型（这里是 $\epsilon$-预测器 Stable Diffusion 1.4/1.5），常规做法是冻结底模、挂上 adapter 接收新控制信号 $c_{\text{novel}}$（分割/深度/边缘/姿态/box），然后继续用 $\epsilon$-损失训练 adapter：
$$\mathcal{L}^{\epsilon}_\theta = \mathbb{E}_{t,\epsilon,x_0}\big[\lVert\epsilon-\epsilon_\theta(x_t,c_{\text{text}},c_{\text{novel}},t)\rVert_2^2\big]$$
本文把这一步改成：先用网络输出的 $\epsilon$ 换算出 $x_0$ 的估计，再用真实干净图像 $x_0$ 去监督它。换算公式直接来自前向过程 $x_t=\alpha_t x_0+\sigma_t\epsilon$：
$$x_\theta(x_t,\cdots,t)=\frac{x_t-\sigma_t\,\epsilon_\theta(x_t,\cdots,t)}{\alpha_t}$$
然后用 $x_0$-监督损失训练：
$$\mathcal{L}^{\epsilon\to x_0}_\theta=\mathbb{E}_{t,\epsilon,x_0}\big[\lVert x_0-x_\theta(x_t,c_{\text{text}},c_{\text{novel}},t)\rVert_2^2\big]$$
作者强调这与 EDM 的网络预条件、一致性训练里的 $x_\theta=c_{\text{skip}}(t)x_t+c_{\text{out}}(t)\epsilon_\theta$ 是同一类思想（这里 $c_{\text{skip}}=1/\alpha_t,\ c_{\text{out}}=-\sigma_t/\alpha_t$）。注意：若底模本来就是 $x_0$-预测器，本方法不适用（已经在监督 $x_0$ 了）。

### 关键设计

**1. x0-监督：把任意预测器换算成 x0-预测再监督干净图，唤回早期去噪步的学习信号**

这一点针对"早期布局步学不到东西"的痛点。可控生成里控制信号与目标图强相关，即便在极低信噪比（$t\to T$）下，从控制信号反推 $x_0$ 也比纯 T2I 容易得多——论文 Fig. 2 直观展示：同样在 $t=199$，纯 SD 预测的 $x_0$ 还很模糊，而分割 ControlNet 预测的 $x_0$ 已经清晰。既然如此，就该**让模型在早期就被强监督去把整张干净图（布局）画对**。做法就是上面的换算+ $x_0$-损失：把预测的 $\epsilon$ 转成 $x_\theta$，用真值 $x_0$ 算 L2。它逼模型在去噪一开始就选对布局，而不是把误差拖到后期再也救不回来。妙处在于完全不动结构、不动采样，只换损失的"监督目标"。

**2. SNR 反加权的等价实现：用一行损失加权解释并复现 x0-监督**

这一点回答"为什么原来的 $\epsilon$-监督是次优的、本方法到底改了什么"。作者形式化地证明 $\epsilon$-损失就是 SNR 加权的 $x_0$-损失：把 $\epsilon=\frac{1}{\sigma_t}(x_t-\alpha_t x_0)$ 代入展开可得
$$\mathcal{L}^{\epsilon}_\theta=\frac{\alpha_t^2}{\sigma_t^2}\,\lVert x_0-x_\theta\rVert_2^2=\frac{\alpha_t^2}{\sigma_t^2}\,\mathcal{L}^{x_0}_\theta$$
即权重就是信噪比 $\text{SNR}=\alpha_t^2/\sigma_t^2$。而 SD 的 SNR（Fig. 6）在低信噪比/早期步几乎是 0、随后才急升，于是早期步几乎拿不到梯度。反过来，对 $x_0$-损失做同样展开会得到 $\mathcal{L}^{\epsilon\to x_0}_\theta=\frac{\sigma_t^2}{\alpha_t^2}\mathcal{L}^{\epsilon}_\theta=\mathcal{L}^{x_0}_\theta$——也就是说，**直接给原 $\epsilon$-损失乘上 $1/\text{SNR}=\sigma_t^2/\alpha_t^2$ 就完全等价于 $x_0$-监督**：
$$\mathcal{L}'_\theta=\mathbb{E}_{t,\epsilon,x_0}\Big[\tfrac{\sigma_t^2}{\alpha_t^2}\,\lVert\epsilon-\epsilon_\theta(x_t,c_{\text{text}},c_{\text{novel}},t)\rVert_2^2\Big]$$
这给出方法的第二种落地方式（不必改预测目标，只重加权），并且实验（Fig. 7）证实它与显式 $x_0$-监督收敛曲线几乎重合，反过来验证了"早期步被压权"这个根因诊断。补充材料还给出各种参数化/范式（$\epsilon$、$v$、$u$、流匹配）到 $x_0$ 的换算公式，使方法可推广到 FLUX.1 等。

**3. mAUCC：对训练时长不敏感的收敛速度度量**

为了量化"收敛更快"，作者借鉴主动学习里的曲线下面积思想，提出 **mAUCC（mean Area Under the Convergence Curve，收敛曲线下面积均值）**。给定训练结束时的收敛曲线（横轴训练步、纵轴某评测指标），先把指标和步数都归一化到 $[0,1]$，再对不同训练时长 horizon 求曲线下平均面积：
$$\text{AUCC}@t_i=\frac{1}{\lceil t_i T_{\max}\rceil}\int_0^{\lceil t_i T_{\max}\rceil} m_s\,ds,\qquad \text{mAUCC}=\frac{1}{N_{th}}\sum_{i=1}^{N_{th}}\text{AUCC}@t_i$$
其中 $m_s$ 是第 $s$ 步的归一化指标，$t_i$ 取 25%~100%（步长 5%）的训练 horizon。单个 AUCC 衡量"累计性能/曲线涨得多快"，对多个 horizon 取平均使 mAUCC 比朴素 AUC 对训练总步数不敏感。若底层指标是 score（如 mIoU/mAP/F1）则越高越好，是误差/距离（如 RMSE/FID）则越低越好。这个指标本身可迁移到任意训练收敛速度的评估场景。

## 实验关键数据

主干与设置：扩散侧用 SD1.4/1.5（$\epsilon$-预测器），流匹配侧用 FLUX.1（OminiControl，$u$-预测器）。代表方法——空间对齐控制用 ControlNet、T2I-Adapter；非空间对齐控制用 GLIGEN（box+text / box+text+image）。除 OminiControl 训 40k 步外其余 200k 步。数据集：MultiGen-20M（深度）、ADE20K（分割）、由分割导出的 Canny 边缘、MS-COCO（姿态/box）。评测：FID 测画质；深度用 Midas+RMSE，分割用 MaskFormer+mIoU，边缘用 F1，姿态/box 用 YOLO+mAP；收敛速度用 mAUCC（按各自指标方向取高/低更好）。

> ⚠️ 下表数值来自 OCR 缓存，个别小数位可能有识别误差，以原文 Table 1/2/3 为准。

### 主实验：空间对齐控制（ControlNet，$\epsilon$ vs $x_0$，Table 1 节选）

| 任务 | 监督 | FID↓ | 控制保真度 | mAUCC |
|------|------|------|-----------|-------|
| Depth | $\epsilon$ | 17.68 | RMSE 35.79↓ | 17.70↓ |
| Depth | **$x_0$** | **17.50** | **35.42** | **15.98** |
| Semantic Seg | $\epsilon$ | 30.05 | mIoU 35.84↑ | 25.19↑ |
| Semantic Seg | **$x_0$** | **29.55** | **39.54** | **31.52** |
| Pose | $\epsilon$ | 44.09 | mAP 58.00↑ | 35.86↑ |
| Pose | **$x_0$** | 44.09 | **59.18** | **42.19** |

ControlNet 上 $x_0$-监督在分割、姿态的 mAUCC 分别提升约 25%、17.65%，FID 与控制保真度也同步改善；论文还观察到 ControlNet 著名的"突然收敛"现象在 $x_0$-监督下更早出现。T2I-Adapter 上提升更猛——姿态控制 mAUCC 提升 65.25%，深度/分割/姿态的最终控制保真度分别提升 16.84%/11.63%/27.97%。OminiControl（流匹配）在深度/分割/边缘上由于底模 FLUX.1 本就收敛极快、差异不明显（mAUCC 变化 2%/1.12%/0.07%），但姿态 mAUCC 仍 +21.80%、分割 mIoU +8.34%，至少持平、最差不掉点。

### 非空间对齐控制（GLIGEN，Table 2）

| 控制 | 监督 | FID↓ | mAP↑ | mAUCC↑ |
|------|------|------|------|--------|
| Box+Text | $\epsilon$ | 32.58 | 30.70 | 8.28 |
| Box+Text | **$x_0$** | **28.38** | **33.30** | **18.38** |
| Box+Text+Image | $\epsilon$ | 21.40 | 21.31 | 6.15 |
| Box+Text+Image | **$x_0$** | 24.23 | 20.76 | **8.07** |

非空间对齐控制本来收敛最慢、最吃资源，这里收益最大：box+text 的 mAUCC 暴涨 121.98%、mAP +8.47%；box+text+image 的 mAUCC +31.22%、mAP 基本持平（变化 2.58%）。

### batch size 效率（Table 3，GLIGEN，指标为 mAUCC）

| 监督 | Box+Text bs16 | bs32 | bs64 | Box+Text+Image bs16 | bs32 | bs64 |
|------|------|------|------|------|------|------|
| $\epsilon$-GLIGEN | 1.41 | 2.58 | 8.28 | 1.77 | 3.26 | 6.15 |
| **$x_0$-GLIGEN** | **1.71** | **7.72** | **18.38** | **1.82** | **4.50** | **8.07** |

非空间对齐控制对 batch size 极敏感（GLIGEN 原文用 bs64、约需 320GB 显存）。$x_0$-监督在 bs32 下取得的 mAUCC 就能媲美 $\epsilon$-监督 bs64，等于把显存需求砍半；从 bs16→bs64，$\epsilon$ 监督 mAUCC 提升 487.23%，$x_0$ 监督提升 974.85%，可扩展性更好。

### 关键发现
- **早期步被 SNR 压权是根因**：给 $\epsilon$-损失乘 $1/\text{SNR}$（去掉加权）后收敛曲线与显式 $x_0$-监督几乎重合（Fig. 7），直接印证"问题出在早期布局步拿不到学习信号"。
- **控制信号越"不空间对齐"，收益越大**：spatially-aligned（深度/分割/边缘）本就好学、提升相对小；non-spatially-aligned（box+text）本就难学、mAUCC 提升上百个百分点。
- **跨范式跨架构通用**：对扩散（UNet 的 ControlNet/T2I-Adapter/GLIGEN）和流匹配（DiT 的 OminiControl）都有效或至少不掉点，只要底模不是 $x_0$-预测器即可套用。
- **$v$/$\epsilon$ 错配监督会变差**：扩散用 $v$-监督、流匹配用 $\epsilon$-监督普遍弱于各自原生监督，说明并非"换任何目标都行"，而是 $x_0$-监督这个特定选择契合可控生成的早期布局需求。

## 亮点与洞察
- **一行损失加权换来约 2× 加速**：核心改动只是把预测目标换算成 $x_0$ 或等价地给 $\epsilon$-损失乘 $1/\text{SNR}$，零结构改动、零推理开销，却能显著提速并提升画质/保真度——典型的"诊断清楚根因后用最小改动解决"。
- **把"$\epsilon$-损失=SNR 加权 $x_0$-损失"讲透并落到可控生成**：这个等价关系本身在扩散参数化文献里早有（Salimans & Ho），但作者第一次把它与"可控生成的早期布局步至关重要"联系起来，给出了"为什么沿用 $\epsilon$-监督次优"的清晰机理解释。
- **mAUCC 是可复用的收敛速度指标**：对训练 horizon 不敏感、可归一化、可用于任意收敛曲线，适合做训练效率类工作的统一评测工具。
- **省显存的实际意义**：bs32 顶 bs64 直接把非空间对齐控制的显存需求砍半，对算力受限的研究者是实打实的可及性收益。

## 局限与展望
- **不适用于已是 $x_0$-预测器的底模**：方法本质是"去掉 $\epsilon$/$v$/$u$ 监督隐含的 SNR 加权"，底模本来就监督 $x_0$ 时无改造空间。
- **空间对齐 + 强底模时增益有限**：OminiControl（FLUX.1）在深度/分割/边缘上几乎看不到提速，作者归因于底模太强、在观测 horizon 内已快速收敛。
- **评估范围**：实验限于少数有可用训练代码的代表方法（ControlNet/T2I-Adapter/GLIGEN/OminiControl）和有限模态，是否对所有可控生成范式都稳定受益仍需更广验证。
- **结论依赖 mAUCC**："加速 2×"等结论以作者自定义的 mAUCC 衡量，跨工作横向比较时需注意指标口径一致。

## 相关工作与启发
- **vs 常规可控生成训练（ControlNet / T2I-Adapter / GLIGEN / OminiControl 原版）**：它们都沿用底模原生监督（$\epsilon$/$u$）训练 adapter；本文指出这隐含 SNR 加权、压制早期布局步，改用 $x_0$-监督即可在不动结构的前提下加速并提点。
- **vs 扩散参数化研究（$\epsilon$- / $x_0$- / $v$-prediction，Salimans & Ho）**：本文复用了"$\epsilon$-损失等价于 SNR 加权 $x_0$-损失"的结论，但把它专门用到可控生成的早期布局诉求上，给出新的应用动机而非新的参数化。
- **vs EDM 预条件 / 一致性训练**：$x_\theta=c_{\text{skip}}x_t+c_{\text{out}}\epsilon_\theta$ 的换算形式与 EDM/一致性训练同源，本文把这套预条件思想迁到 adapter 训练损失上。

## 评分
- 新颖性: ⭐⭐⭐⭐ 改动极简，但把已知等价关系与可控生成早期布局诉求结合的洞察新且有用。
- 实验充分度: ⭐⭐⭐⭐ 覆盖扩散/流匹配、UNet/DiT、空间/非空间对齐多模态，含 batch size 与等价实现验证。
- 写作质量: ⭐⭐⭐⭐ 动机—推导—验证链条清晰，图示直观；部分增益依赖自定义 mAUCC 需读者注意口径。
- 价值: ⭐⭐⭐⭐ 零成本即插即用、省显存、提速明显，对训练可控生成的实践者很实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Understanding, Accelerating, and Improving MeanFlow Training](understanding_accelerating_and_improving_meanflow_training.md)
- [\[CVPR 2026\] PhyCo: Learning Controllable Physical Priors for Generative Motion](phyco_learning_controllable_physical_priors_for_generative_motion.md)
- [\[CVPR 2026\] MoCoDiff: A Controllable Autoregressive Diffusion Model for Expressive Motion Generation](mocodiff_a_controllable_autoregressive_diffusion_model_for_expressive_motion_gen.md)
- [\[CVPR 2026\] SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](solace_self_confidence_rewards_t2i.md)
- [\[AAAI 2026\] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](../../AAAI2026/image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)

</div>

<!-- RELATED:END -->
