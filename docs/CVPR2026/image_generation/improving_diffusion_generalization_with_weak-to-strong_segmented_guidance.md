---
title: >-
  [论文解读] Improving Diffusion Generalization with Weak-to-Strong Segmented Guidance
description: >-
  [CVPR 2026][图像生成][扩散引导] 把扩散采样里的引导方法统一在"弱到强（weak-to-strong, W2S）"视角下分成"条件相关引导（CDG，如 CFG）"和"条件无关引导（CAG，如 AG/SLG）"两类，用合成实验刻画各自的有效区间，进而提出按噪声水平切换两类引导的 **SGG（Segmented Guidance）**，并把这一原则进一步迁移进训练目标，让无引导模型本身的泛化能力变强。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散引导"
  - "弱到强"
  - "CFG"
  - "AutoGuidance"
  - "分段引导"
---

# Improving Diffusion Generalization with Weak-to-Strong Segmented Guidance

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Improving_Diffusion_Generalization_with_Weak-to-Strong_Segmented_Guidance_CVPR_2026_paper.html)  
**代码**: https://github.com/Westlake-AGI-Lab/SGG  
**领域**: 图像生成 / 扩散模型  
**关键词**: 扩散引导, 弱到强, CFG, AutoGuidance, 分段引导

## 一句话总结
把扩散采样里的引导方法统一在"弱到强（weak-to-strong, W2S）"视角下分成"条件相关引导（CDG，如 CFG）"和"条件无关引导（CAG，如 AG/SLG）"两类，用合成实验刻画各自的有效区间，进而提出按噪声水平切换两类引导的 **SGG（Segmented Guidance）**，并把这一原则进一步迁移进训练目标，让无引导模型本身的泛化能力变强。

## 研究背景与动机

**领域现状**：扩散 / 流匹配模型靠多步迭代去噪生成图像，推理时几乎都要叠加"引导"来提升生成质量与可控性。最常用的是 Classifier-Free Guidance（CFG）——训练时随机丢弃条件、推理时在无条件预测和条件预测之间外推；近期 AutoGuidance（AG）则换了个思路，用一个"条件对齐但更弱"的劣质模型来引导主模型。

**现有痛点**：这些引导方法的"适用边界"一直很模糊。AG 这类用弱模型引导的方法在 ImageNet 类条件生成上能超过 CFG，但在大规模文生图（T2I）里单独用往往不如 CFG 稳健，常常只能当 CFG 的补充。实践者面对一个新任务时，无从判断到底该选哪种引导。

**核心矛盾**：引导的本质是用一个"弱信号"去外推出"强信号"，$\mathbf{v}_w = \mathbf{v}_{\text{weak}} + w(\mathbf{v}_{\text{strong}} - \mathbf{v}_{\text{weak}})$。不同方法的差别只在于"弱信号怎么造"——CFG 靠丢条件造弱信号，AG 靠弱化模型造弱信号。而这两种造法在不同条件粒度、不同模型拟合程度下，效果会此消彼长，没有哪个绝对更好。

**本文目标**：(1) 说清 CDG 与 CAG 各自在什么场景下有效、什么场景下失效；(2) 设计一个能同时吃到两者好处的混合引导；(3) 把这套原则从"推理时的外挂"搬进"训练目标"，减少推理时额外的引导前向开销。

**切入角度**：作者用一个可控的递归高斯混合 toy 数据集，精确调节"类别数（条件粒度）"和"递归深度（类内复杂度）"，把两类引导的失效模式隔离出来观察；再在 ImageNet 上量化两类引导在不同时间步对"最优速度场"的纠偏能力。

**核心 idea**：CDG 擅长在高噪声阶段做"类间分离 / 寻找正确流形"，CAG 擅长在低噪声阶段做"类内细节精修"——既然两者在时间轴上各管一段，那就**按采样时间 $\tau$ 把引导分段**：高噪声用 CDG、低噪声用 CAG，并把这一思想固化进训练目标。

## 方法详解

### 整体框架
本文不是提一个新网络，而是给"引导该怎么用"建立一套统一的 W2S 框架，并在推理与训练两端落地。统一外推公式为 $\mathbf{v}_w(\mathbf{x}_t,t,\mathbf{c}) = \mathbf{v}_{\text{weak}} + w(\mathbf{v}_{\text{strong}} - \mathbf{v}_{\text{weak}})$，其中强信号是条件模型输出 $\mathbf{v}(\mathbf{x}_t,t,\mathbf{c})$，弱信号 $\tilde{\mathbf{v}}(\mathbf{x}_t,t,\tilde{\mathbf{c}})$ 的构造方式区分出两大类：**CDG**（条件相关，改条件、模型不变，$\tilde{\mathbf{v}}=\mathbf{v},\tilde{\mathbf{c}}=\varnothing$，代表 CFG）与 **CAG**（条件无关，保条件、弱化模型，$\tilde{\mathbf{v}}=\mathbf{v}_{\text{inferior}},\tilde{\mathbf{c}}=\mathbf{c}$，代表 AG/SLG）。作者先用合成实验和 ImageNet 量化分析两类引导的有效区间，发现它们在时间轴上互补；据此推理端提出按时间分段的 SGG，训练端把同一原则写进回归目标。整套方法是"引导机制 + 训练目标"的改进，不涉及多模块串行 pipeline，因此不配框架图。

### 关键设计

**1. CDG / CAG 二分法与有效区间分析：先搞清楚两类引导各自管哪一段**

作者把五花八门的引导统一到弱信号构造这一个轴上：CDG 通过"操纵条件"造弱信号（典型是 CFG 丢掉条件 $\mathbf{c}\to\varnothing$），CAG 通过"操纵模型"造弱信号（用更小/欠训的网络如 AG，或扰动主模型如 SLG）。为了把两者的失效模式分离出来，作者构造递归高斯混合 toy 数据集，独立控制类别数（条件粒度）和递归深度（类内复杂度）。结论很清晰：在"类少、类内复杂、模型已拟合好"时（CLS=4, Depth=3），CDG 出现 mode-seeking，把样本挤向高密度区、丢失类内多样性，CAG 反而能保住类内覆盖；在"类多、类内简单、模型欠拟合"时（CLS=24, Depth=1），CAG 会产生 off-manifold 的离群样本，CDG 靠强制条件把跑偏的样本拉回正确类别。进一步在 ImageNet 上用 SiT-B/2 度量引导速度与最优速度 $\dot{\mathbf{v}}$ 的 Inception 距离 $\Delta_e = \mathbb{E}_{\mathbf{x}_t}[d(\dot{\mathbf{v}}, \mathbf{v}_w)]$，发现 CDG 的纠偏集中在高噪声步、CAG 集中在低噪声步——这与"语义/类间信息在采样早期定型、细粒度感知细节在采样末期定型"的认知一致。这一分析是后面所有设计的依据。

**2. SGG 分段引导：高噪声用 CDG 寻流形、低噪声用 CAG 修细节**

既然两类引导在时间轴上各管一段，就没必要二选一。SGG 把引导方向 $\mathbf{g}$ 按时间阈值 $\tau$ 分段：

$$\mathbf{g}(\mathbf{x}_t, t, \mathbf{c}) = \begin{cases} \mathbf{v}(\mathbf{x}_t, t, \mathbf{c}) - \mathbf{v}(\mathbf{x}_t, t, \varnothing) & t > \tau \;(\text{CDG}) \\ \mathbf{v}(\mathbf{x}_t, t, \mathbf{c}) - \tilde{\mathbf{v}}(\mathbf{x}_t, t, \mathbf{c}) & t \le \tau \;(\text{CAG}) \end{cases}$$

最终引导速度为 $\mathbf{v}_w(\mathbf{x}_t,t,\mathbf{c}) = \mathbf{v}(\mathbf{x}_t,t,\mathbf{c}) + (w-1)\cdot\mathbf{g}(\mathbf{x}_t,t,\mathbf{c})$。直白说就是：高噪声阶段（$t>\tau$）先用 CFG 式的条件相关引导把样本送到正确的条件流形上，低噪声阶段（$t\le\tau$）切换到条件无关引导精修类内细节。这样既拿到 CFG 的 prompt 贴合度（对应 HPSv2.1），又拿到 CAG 的美学质量（对应 Aesthetic），避开了单用 CFG"美学低"或单用 SLG"贴合度掉"的偏科问题。推理实现里作者用 CFG 当 CDG、SLG 当 CAG。

**3. W2S 训练目标迁移：把分段引导写进回归目标，减少推理引导开销**

引导本来是推理时的外挂，每步要额外前向。作者把 W2S 原则直接搬进训练目标：在标准速度匹配目标 $\mathbf{u}=\epsilon-\mathbf{x}_0$ 上加一项引导差，$\mathbf{u}_{\text{w2s}} = \mathbf{u} + w\cdot\mathbf{g}(\mathbf{x}_t,t,\mathbf{c})$，训练损失为 $\mathcal{L}_s = \mathbb{E}\big[\|\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{c}) - (\mathbf{u} + w\cdot\text{sg}[\mathbf{g}])\|_2^2\big]$，其中 $\text{sg}[\cdot]$ 是 stop-gradient 用于稳定训练。这鼓励强模型在训练时就学到引导的外推能力，推理时无需再额外前向（NFE/s 降到 1）。弱信号的训练时构造给了一套选择：CDG 用 CFG/MG（把无条件项迁进训练），CAG 用 AG（维护一个更小、欠训的弱网络）或 **BR（Branch）**——从中间层引出一个辅助输出分支当弱信号，条件无关且训练时不需要额外引导前向，开销最低。训练版 SGG 同样按 $\tau$ 分段：高噪声用 CFG 信号、低噪声用 BR 信号。⚠️ 文中 SLG 这类层扰动方法被作者发现"整进训练会掉点"而排除，细节以原文附录为准。

## 实验关键数据

### 主实验：推理时引导对比（SD3 / SD3.5）
在 SD3-Medium 与 SD3.5-Medium 上、用 MS-COCO-1K 与 LAION-1K 提示评测，指标为 HPSv2.1（与 prompt 贴合度正相关）和 Aesthetic（美学分）。可见 CFG 偏向贴合度、SLG 偏向美学，SGG 在两者上都取得有竞争力的分数（下表为 SD3.5 / MS-COCO-1K 一列）：

| 方法 | NFE/s | HPSv2.1 ↑ | Aesthetic ↑ |
|------|-------|-----------|-------------|
| Conditional（无引导） | 1 | 21.204 | 4.978 |
| CFG | 2 | 29.199 | 5.279 |
| SLG | 2 | 27.295 | 5.714 |
| S2-Guidance | 3 | 29.614 | 5.342 |
| **SGG（本文）** | 2 | **29.736** | **5.717** |

CFG 美学偏低（5.279），SLG 贴合度偏低（27.295），SGG 同时拿到高 HPSv2.1（29.736）和高 Aesthetic（5.717），印证了"分段吃两者好处"的设计意图。

### 消融 / 训练时集成（ImageNet 256×256，SiT-B/2）
训练时把 W2S 写进目标，条件设定下各弱信号构造（MG / AG / BR / SGG）都优于 baseline，SGG 最好且把推理 NFE/s 压到 1：

| 配置 | NFE/s | FID ↓ | sFID ↓ | IS ↑ |
|------|-------|-------|--------|------|
| SiT-B/2（baseline） | 1 | 31.22 | 6.41 | 49.59 |
| + CFG（推理引导） | 2 | 6.02 | 5.47 | 183.83 |
| MG | 1 | 5.88 | 6.19 | 253.74 |
| BR | 1 | 16.02 | 5.13 | 76.21 |
| **SGG** | 1 | **4.58** | **4.95** | 264.06 |
| SGG + REPA | 1 | **3.07** | 4.88 | 242.15 |

### 关键发现
- SGG 在单次前向（NFE/s=1）下 FID 4.58，优于需要两次前向的推理时 CFG（6.02），说明把引导迁进训练既省推理开销又提质量。
- 叠加 REPA 表征对齐后 FID 进一步降到 3.07，W2S 训练与表征加速方法正交可叠加。
- 无条件设定下 CDG 天然不适用，但 CAG（AG/BR）仍能把 FID 从 61.27 降到 43–46，说明条件无关引导在无条件生成里仍有价值。
- AG 作为弱信号需要额外维护一个弱网络（time/it 1.27），BR 几乎零额外训练开销（time/it 1.02），是更实用的 CAG 构造。

## 亮点与洞察
- **把"选哪种引导"从经验玄学变成区间问题**：用 toy 实验把 CDG/CAG 的失效模式（mode-seeking vs off-manifold 离群）干净地隔离出来，并对应到采样时间轴的高/低噪声段，这个"时间分段"的洞察是全文最有解释力的地方。
- **分段引导几乎零成本**：SGG 只是在采样过程中按阈值 $\tau$ 切换引导项，不改网络、不加训练，却能同时拿到贴合度和美学，是可以直接迁移到任意 CFG 流程的 trick。
- **引导可以"内化"进训练**：W2S 训练目标 + stop-gradient 的写法，把推理外挂变成模型自带能力，对追求低 NFE 的部署场景很有启发；BR 这种"中间层引出辅助分支当弱信号"的造法尤其轻量。

## 局限与展望
- 训练时集成的验证主要在 SiT-B/2 / ImageNet 这种受控规模上（受算力约束），大规模文生图的训练时集成是否同样有效尚未直接验证；推理时 SGG 才在 SD3/SD3.5 上验证。
- 分段阈值 $\tau$ 是一个需要选的超参，文中未充分展开它对不同任务/模型的敏感性，实际使用可能需要调。
- SLG 这类层扰动弱信号"整进训练会掉点"被直接排除，背后机理只在附录讨论，说明训练时弱信号的构造仍较脆弱、并非任意 CAG 都能迁移。⚠️ 相关细节以原文附录为准。
- 评测指标偏感知质量（HPSv2.1 / Aesthetic / FID），对"多样性"这一 CDG 的核心短板缺少直接定量刻画。

## 相关工作与启发
- **vs CFG**：CFG 是纯 CDG，全程靠丢条件外推，贴合度稳但美学/多样性偏弱、且高引导尺度下过饱和；SGG 在低噪声段换成 CAG 精修细节，补上 CFG 的短板。
- **vs AutoGuidance (AG)**：AG 是纯 CAG，靠弱模型引导，在类条件任务上强但大规模 T2I 里单独用不稳；SGG 把 AG 只用在它擅长的低噪声段，并在高噪声段交给 CDG。
- **vs MG / GFT（训练时修改目标）**：这些方法把 CFG 的无条件项加进训练目标；本文 W2S 训练在此基础上引入分段（高噪声 CFG、低噪声 BR）并给出更轻量的 BR 弱信号构造，FID 更低。

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一视角 + 时间分段 + 训练迁移三连，分段引导的洞察清晰，但单个组件多基于已有方法重组。
- 实验充分度: ⭐⭐⭐⭐ 推理（SD3/SD3.5）+ 训练（SiT/ImageNet）双线验证，但训练集成受算力限制规模偏小。
- 写作质量: ⭐⭐⭐⭐ 从 toy 分析到方法落地逻辑链完整，公式与区间结论讲得清楚。
- 价值: ⭐⭐⭐⭐ SGG 是可直接套用的引导 trick，W2S 训练对低 NFE 部署有实际意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance](../../ICML2026/image_generation/weak_diffusion_priors_can_still_achieve_strong_inverse-problem_performance.md)
- [\[CVPR 2026\] Smoothing the Score Function to Enhance Generalization in Diffusion Models](smoothing_the_score_function_to_enhance_generalization_in_diffusion_models.md)
- [\[CVPR 2026\] Meta-CoT: Enhancing Granularity and Generalization in Image Editing](meta-cot_enhancing_granularity_and_generalization_in_image_editing.md)
- [\[CVPR 2026\] Understanding, Accelerating, and Improving MeanFlow Training](understanding_accelerating_and_improving_meanflow_training.md)
- [\[ICML 2026\] GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance](../../ICML2026/image_generation/guidedbridge_training-freely_improving_bridge_models_with_prior_guidance.md)

</div>

<!-- RELATED:END -->
