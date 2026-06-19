---
title: >-
  [论文解读] Smoothing the Score Function to Enhance Generalization in Diffusion Models
description: >-
  [CVPR 2026][图像生成][记忆化] 本文从理论上证明扩散模型的记忆化（生成样本逐字复制训练样本）源于经验 score function 是一个"尖锐 softmax 加权"的高斯分量和，单个训练点会主导采样导致塌缩；据此提出两种让权重变平滑的方法——噪声去条件化（Noise Unconditioning）和温度平滑（Temperature Smoothing），在几乎不损失生成质量的前提下显著提升泛化、缓解记忆化。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "记忆化"
  - "泛化"
  - "score function"
  - "高斯混合"
  - "温度平滑"
---

# Smoothing the Score Function to Enhance Generalization in Diffusion Models

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Smoothing_the_Score_Function_to_Enhance_Generalization_in_Diffusion_Models_CVPR_2026_paper.html)  
**代码**: https://github.com/XinyuZhou2001/smoothing_the_score  
**领域**: 扩散模型  
**关键词**: 记忆化, 泛化, score function, 高斯混合, 温度平滑

## 一句话总结
本文从理论上证明扩散模型的记忆化（生成样本逐字复制训练样本）源于经验 score function 是一个"尖锐 softmax 加权"的高斯分量和，单个训练点会主导采样导致塌缩；据此提出两种让权重变平滑的方法——噪声去条件化（Noise Unconditioning）和温度平滑（Temperature Smoothing），在几乎不损失生成质量的前提下显著提升泛化、缓解记忆化。

## 研究背景与动机

**领域现状**：扩散模型通过对数据逐步加高斯噪声、再学习一个 score function $\nabla_x\log p_t(x)$ 来反向去噪生成样本。在高维下，数据落在低维流形上（流形假设），直接 score matching 在低密度区梯度估计不准，所以主流做法是对一系列加噪边缘分布分别估计 score，沿整条采样路径都能学准。

**现有痛点**：越来越多研究发现，扩散模型生成的样本里有一部分和训练样本**一模一样**——这就是记忆化（memorization），直接带来隐私与版权风险。理论上这甚至是"必然"的：Fokker–Planck 方程表明，如果用**经验 score function**（对应经验分布的 score）精确求解反向过程，每个采样到的边缘分布都会等于其前向对应分布，最终一定回到训练分布，生成不出任何新样本。

**核心矛盾**：理论说"完美学到经验 score 就只会复制"，可实践中神经网络偏偏能生成新样本。这个理论与实践的矛盾，正是本文要回答的根本问题——**神经网络究竟是怎么部分化解记忆化的？**

**本文目标**：拆成两个子问题：(1) 经验 score function 到底有什么结构，使它导致记忆化？(2) 神经网络学到的 score 和经验 score 差在哪，差异如何带来泛化？回答清楚后，进一步问能否**主动**制造这种差异来增强泛化。

**切入角度**：作者注意到神经网络学到的 $s_\theta(x,\sigma)$ 必然与经验 score $s^*(x,\sigma)$ 略有不同，于是把经验 score 显式写成"高斯分量的 softmax 加权和"，从代数+几何（高维高斯质量集中在薄壳上）两个角度分析这个加权结构。

**核心 idea**：记忆化的病根是 softmax 权重太尖锐、单个训练点独占 score；神经网络因容量/正则限制学到的是**更平滑**的权重，让采样后期由"附近几个样本张成的局部流形"而非"单点"决定——于是只要主动把权重**抹平**（去掉噪声条件 + 加温度），就能换来泛化。

## 方法详解

### 整体框架

本文的主体是一套"诊断 → 开方"的理论框架，而非多模块串行的工程 pipeline：先把经验 score function 解剖成 softmax 加权结构、定位记忆化的数学根源（权重尖锐、单点主导），再据此给出两个让权重平滑的训练侧改动，最后用一个"采样扩张比" $\gamma_{ex}$ 作为泛化的定量代理来统一比较各种 score。

设 $\{\mu_j\}_{j=1}^M$ 是从真实分布 $p_{\text{data}}$ 采的 i.i.d. 样本，经验分布 $p^*(x)=\frac1M\sum_j\delta(x-\mu_j)$。加噪后第 $i$ 个噪声级 $\sigma_i$ 的边缘分布是高斯混合 $p_i^*(x)=\frac1M\sum_{j=1}^M\mathcal{N}(x;\mu_j,\sigma_i^2 I)$，其经验 score 可显式写成

$$\nabla_x\log p_i^*(x) = -\sum_{j=1}^M w_{ij}(x)\,\frac{x-\mu_j}{\sigma_i^2},\qquad w_{ij}(x)=\frac{\mathcal{N}(x;\mu_j,\sigma_i^2 I)}{\sum_{l}\mathcal{N}(x;\mu_l,\sigma_i^2 I)}.$$

即 score 是"指向各训练中心 $\mu_j$ 的方向"的加权平均，权重 $w_{ij}$ 恰是一个 softmax，也代表"$x$ 由以 $\mu_j$ 为心的高斯分量生成"的概率。整套方法围绕这个 softmax 权重的尖锐程度做文章。三个关键设计依次是：**① 用 softmax 加权结构定位记忆化根源**（诊断）、**② 噪声去条件化**、**③ 温度平滑**（两味药）。

### 关键设计

**1. 经验 score 的 softmax 加权结构 = 记忆化根源（诊断）**

要治记忆化，先得说清它从哪来。作者用高维几何给出直观：在高维 $\mathbb{R}^d$ 中，单个高斯 $\mathcal{N}(\mu_j,\sigma_i^2 I)$ 的概率质量集中在一层薄壳上——壳心是 $\mu_j$、半径约 $\sigma_i\sqrt d$、厚度约 $3\sqrt2\,\sigma_i$。采样点 $x$ 可看作落在某噪声级的某个壳里。把权重写成 softmax：$w_{ij}(x)=\mathrm{Softmax}_{j,i}\big(f(x,\mu_j,\sigma_i)\big)$，其中 $f(x,\mu_j,\sigma_i)=-(d-2)\ln\sigma_i-\frac{\|x-\mu_j\|^2}{2\sigma_i^2}$。

作者证明两条主导性质：**σ 主导**——对固定中心 $\mu_j$ 和位置 $x$，存在一个"最优噪声级" $\sigma_j^*$，其对应分量在所有噪声级里独占权重，且 $f(\sigma)$ 对偏离 $\sigma_j^*$ 极其敏感（相邻噪声级权重比可达 $\gamma_\sigma\approx e^{18(1-2\alpha)}$，例如 $\alpha=\tfrac13$ 时 $\gamma_\sigma\approx e^6\approx403$）；**μ 主导**——权重随中心到 $x$ 的距离指数衰减，$\gamma_\mu\approx\exp\!\big(\delta\|\mu_j-\mu_l\|^2/\sigma_j^{*2}\big)$。采样后期 $\sigma$ 变小，$\gamma_\mu$ 急剧增大，**单个最近训练点 $\mu_j$ 独占权重**，后续去噪步骤就一路塌缩到 $\mu_j$——这正是记忆化。而且权重在壳交叠区极其尖锐，$x$ 的微小扰动就能让主导权从一个中心跳到另一个。

为量化"泛化 vs 记忆"，作者定义**采样扩张比** $\gamma_{ex}:=\frac{\|y'-x'\|}{\|y-x\|}$：取 $x$ 附近两个最近训练点 $\mu_0,\mu_1$，沿 $(\mu_0-\mu_1)$ 方向轻扰得 $y$，各走一步去噪到 $x',y'$。记忆化会让该比值无界（局部映射强扩张），而**有界**比值意味着采样映射局部非扩张、保持局部连通性，可作泛化的定量代理。理论给出经验 score 下 $\gamma_{ex}\approx\frac{\|\eta(\mu_0-\mu_1)\|}{\|y-x\|}\big|1-\frac{2}{a+1}\big|$，其中 $a>1$ 是主导权与次主导权之比；当 $d\to\infty$ 时 $a\to\infty$，比值无界——经验 score 泛化极差。神经网络学到的 score 实测 $\gamma_{ex}$ 小得多，说明它**隐式抹平了权重**。后两个设计就是把这种抹平变成"显式可控"。

**2. 噪声去条件化（Noise Unconditioning）：让每个训练点自己挑最优壳**

针对"σ 主导"带来的次优权重：标准扩散在每个采样阶段把噪声级 $\sigma$ 固定，可采样点 $x$ 未必落在大多数训练点的"最优壳"上，导致这些点权重被压得极小、单点更容易独占。作者的做法是**直接去掉 score 网络对噪声的条件输入**，$s_\theta(x,t)\to s_\theta(x)$。在方差爆炸 SDE（$\mu(x,t)=0,\ \sigma(t)=\sqrt{2t}$）下，所有噪声级的边缘分布被统一进一个 $M\times N$ 高斯混合

$$p_{MN}(x)=\frac{1}{Z}\sum_{j=1}^M\sum_{i=1}^N \lambda(\sigma_i)\,\mathcal{N}(x;\mu_j,\sigma_i^2 I),\qquad \lambda(\sigma_i)=\sigma_i^2.$$

这样每个训练点 $\mu_j$ 都能在给定位置 $x$ 处自适应地找到自己的最优壳 $\sigma_j^*$ 来贡献 score，其他点不再被固定噪声级压没——单点主导被削弱，塌缩被推迟（即便最终仍塌向某 $\mu_j$，"塌缩时刻"也因其他中心持续贡献而延后）。更妙的是，采样可重写为对一个**时间无关**目标的梯度上升流 $\frac{dx}{dt}=\frac{\eta}{2}\nabla_x\log p_{MN}(x)$，把生成变成在固定目标 $\log p_{MN}$ 上做优化（训练点即最优解），从而能用优化视角设计采样器。对应训练损失只是把标准 NCSN 损失里的条件输入去掉：$\mathcal{L}_u=\mathbb{E}_{\sigma_i,\mu,x}\big[\frac{\sigma_i^2}{2}\|s_\theta(x)+\frac{x-\mu}{\sigma_i^2}\|^2\big]$，作者证明它等价于对 $p_{MN}$ 做显式 score matching。

⚠️ 一个实践坑：去条件后的 score 在采样后期会自适应到真实隐式噪声级 $\sigma_{n*}=\|x_n-\mu_*\|/\sqrt d$，但采样器仍用预设 $\sigma_n$。ODE 采样器因 $\sigma_n\gg\sigma_{n*}$ 步长 $\propto\sigma_n^2/\sigma_{n*}^2$ 会灾难性过冲；SDE 采样器靠随机项自校正而稳定。解法是像 Predictor-Corrector 那样固定隐式噪声级，或直接用 $x_{n+1}=x_n+\alpha\sigma_{n*}^2 s_\theta(x_n)$ 把预设 $\sigma_n$ 换成 $\sigma_{n*}$。

**3. 温度平滑（Temperature Smoothing）：给 softmax 权重加可调温度**

既然权重本就是 softmax，最自然的抹平手段就是引入温度。作者给每个壳 $i$ 配温度 $T_i$，定义 $T_j^*$ 为 $\sigma_i=\sigma_j^*$ 时对应的温度，温度化权重为

$$w_j^*(x;T)=\frac{\exp\!\big(f(x,\mu_j,\sigma_j^*)/T_j^*\big)}{\sum_{l=1}^M \exp\!\big(f(x,\mu_l,\sigma_l^*)/T_l^*\big)}.$$

$T_i=1$ 时退化为原权重；$T_i$ 增大时主导比 $a$ 下降、$\gamma_{ex}$ 变小、泛化更好。直觉是：在小噪声级用更高温度，让采样持续"探索"由邻近样本张成的局部流形、而不是一头扎进单点；温度不够高最终仍会塌缩，但配合 early stopping 可在保留探索收益的同时避免完全塌缩。温度也不能太高——否则会把过大邻域内的训练点都拉进来，生成点偏离图像流形。训练时仅在噪声级 $\sigma_i\le\sigma_{\text{collapse}}$（接近塌缩）时才启用温度：用 top-$K$ 最近训练样本近似 score $\nabla_x\log p_{MN}(x;T)\approx\sum_{j=1}^K w_{(j)}^*(x;T)\big(-\frac{x-\mu_{(j)}}{\sigma_{(j)}^{*2}}\big)$，并按噪声级自适应组合两种损失：

$$\mathcal{L}=\mathbb{E}_{\sigma_i\sim p_\sigma}\Big[\,\mathcal{L}_u\ \text{if}\ \sigma_i>\sigma_{\text{collapse}};\quad \mathcal{L}_T\ \text{if}\ \sigma_i\le\sigma_{\text{collapse}}\,\Big],$$

实验里取 $T_i=\max\!\big(\sigma_{\text{collapse}}/\sigma_i,\,1\big)$。注意温度版并非严格学 $p_{MN}$ 的 score，而是其平滑代理。它是**即插即用**的，只在小噪声级加少量开销。

### 损失函数 / 训练策略
- 噪声去条件损失 $\mathcal{L}_u$：标准 NCSN 损失去掉噪声输入，等价于对统一高斯混合 $p_{MN}$ 做显式 score matching。
- 温度损失 $\mathcal{L}_T$：用 top-$K$ 近邻样本近似平滑后的 score，仅在 $\sigma_i\le\sigma_{\text{collapse}}$ 时启用，与 $\mathcal{L}_u$ 按噪声级自适应拼接。
- 训练沿用 VE-SDE 噪声调度，$\sigma$ 从 $[\sigma_{\min},\sigma_{\max}]$ 上近似 log-uniform 的 $N$ 个离散点等概率采样；去条件版与 baseline 用同一网络，仅去掉时间嵌入层。

## 实验关键数据

实验以 VE-SDE 为 baseline，在 CIFAR-10、CelebA 64×64、ImageNet 64×64、CelebA-HQ 256×256 上验证，并自建一个 1000 只家猫 + 200 只狞猫（caracal，长耳凶脸）的 64×64 小数据集专门看泛化。质量指标除常规 $\text{FID}(G,T_{tr})$（生成 vs 训练集）外，还算 $\text{FID}(G,T_{test})$（生成 vs 测试集）以反映泛化。

### 主实验：FID 与质量-泛化权衡（Tab. 1，节选）

| 数据集 / 采样器 | 方法 | FID(G,T_tr) | FID(G,T_test) |
|---|---|---|---|
| CIFAR-10, SDE 1K NFE | Conditioning（baseline） | 6.49 | 6.56 |
| CIFAR-10, SDE 1K NFE | Unconditioning | 7.33 | 7.34 |
| CIFAR-10, SDE 1K NFE | $T_i{=}5/\sigma_i,K{=}100$（pixel→feature KNN） | 13.25 → **8.35** | 13.41 → 8.30 |
| CIFAR-10, SDE 1K NFE | $T_i{=}7/\sigma_i,K{=}100$（pixel→feature KNN） | 50.81 → **7.96** | 51.08 → 7.98 |
| CelebA 64×64, SDE 1K NFE | Conditioning（baseline） | 7.25 | 7.81 |
| CelebA 64×64, SDE 1K NFE | Unconditioning | 7.07 | 7.34 |
| CelebA 64×64, SDE 1K NFE | $T_i{=}10/\sigma_i,K{=}100$（pixel→feature KNN） | 61.97 → **8.40** | 60.91 → 8.19 |

要点：去条件化在 CelebA 上 FID 甚至略优于 baseline；温度平滑会让 FID 略升，但换来肉眼可见的泛化（生成出训练集里不存在的属性组合）。"→" 是从像素空间 KNN 换到特征空间 KNN 的变化——在像素空间高温会塌（FID 飙到 50+），换到曲率更小的 ResNet-18 特征空间算 KNN 后稳定回落到 ~8，说明更平的局部流形允许更激进的平滑而不脱离流形。

### 采样扩张比 γ_ex（Fig. 4，验证理论）

| 噪声级 $\sigma^*$ | 经验 Conditioning | 经验 Unconditioning(T=1) | 温度 T=100 | NN 近似 |
|---|---|---|---|---|
| 较大（如 1.0） | 适中 | 更小 | 更小 | 小 |
| 较小（如 0.05） | ≈ 1333（强扩张） | 数百级，明显更小 | 个位~十位 | 个位（最小） |

⚠️ 具体数值以原文 Fig. 4 为准（OCR 数值对位不完全可靠）。趋势清晰：噪声级越小经验 conditioning score 越扩张（趋向记忆化），而去条件、温度平滑、NN 近似的 $\gamma_{ex}$ 都小得多；且 **NN 近似的行为与温度平滑后的经验 score 高度相似**，直接印证"神经网络靠隐式平滑权重实现泛化"这一核心论断。

### 关键发现
- **记忆化 ≈ score 权重过尖锐**：小噪声级下最近训练点独占 softmax 权重，采样塌向单点；两种方法都是把这个权重分布抹平。
- **特征空间 KNN 一致优于像素空间**：相同 $T,K$ 下特征空间生成质量明显更好，因为图像流形在特征空间曲率更小，允许更强平滑——这反过来又验证了"靠局部流形平滑实现泛化"的解释。
- **去条件让步长更大**：去条件 score 在 $p_{MN}$ 整个支撑上良定义，避免大步长的误差累积，NFE 更少，可部分抵消找最近点的开销。
- **温度需适中**：太低仍会塌缩，太高会把过大邻域的点拉进来使样本偏离流形。

## 亮点与洞察
- **把"记忆化"还原成一个 softmax 温度问题**：一旦看出经验 score 是 $\mathrm{Softmax}(f)$ 加权的中心方向和，"权重太尖→单点主导→塌缩"的因果链就一目了然，治法（去条件/加温度）也水到渠成——这种"先解剖再开方"的叙事非常干净。
- **统一了两类先验**：去条件化把多噪声级边缘分布合并成一个固定高斯混合 $p_{MN}$，使采样=对 $\log p_{MN}$ 的梯度上升，从优化视角统一了已有的 unconditioning 方法。
- **可迁移的诊断工具**：扩张比 $\gamma_{ex}$ 作为"采样映射局部是否非扩张"的泛化代理，可推广到分析其他生成模型/采样器的记忆化倾向。
- **特征空间平滑**这一招（在低曲率空间做 KNN/平滑）对任何"基于局部邻域改 score"的方法都有借鉴价值。

## 局限与展望
- 温度版**不是严格学 $p_{MN}$ 的 score**，而是平滑代理，理论与实现间留有缺口。
- 去条件 score 与 ODE 采样器存在噪声级失配，需 PC 校正或显式替换 $\sigma_{n*}$，而求最近训练点对大规模数据集开销大。
- 温度/$K$ 是需要调的敏感超参，过大即塌；实验主要在中小分辨率数据集，高分辨率仅在附录给出。
- 作者展望把方法扩到 latent diffusion——在隐空间做平滑既可能进一步增强泛化，又能省算力。

## 相关工作与启发
- **vs 经验 score / 记忆化-泛化二分法（[26] 等）**：[26] 把泛化解释为"模型未能记忆"，本文更进一步给出**机制**——泛化来自 softmax 权重被抹平、采样由局部流形而非单点决定，并指出大模型若能拟合权重的尖锐性（孤立点/重复样本）记忆仍会发生。
- **vs 标准 NCSN / VE-SDE（[19,21]）**：沿用其噪声调度与网络，但去掉噪声条件输入并把多噪声级统一进一个高斯混合，把采样重述为固定目标上的梯度上升。
- **vs 训练数据提取/去重缓解（[2,18]）**：那些工作从数据侧（去重、caption 多样化）缓解记忆，本文从 **score function 权重平滑**这一更底层的机制侧入手。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把记忆化-泛化矛盾还原成 softmax 权重尖锐度问题，并给出两味直接对症的平滑药，理论框架自洽且解释力强。
- 实验充分度: ⭐⭐⭐⭐ 多数据集 + 自建猫/狞狫集 + γ_ex 定量验证齐全，但高分辨率结果多在附录，FID 在部分设置下有损失。
- 写作质量: ⭐⭐⭐⭐⭐ "诊断→开方"叙事清晰，几何直觉（薄壳）与代数推导配合得当。
- 价值: ⭐⭐⭐⭐⭐ 既给扩散记忆化提供理论基础（含隐私/版权意义），又给出即插即用的泛化增强手段。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Efficient Weighted Sampling via Score-based Generative Models](efficient_weighted_sampling_via_score-based_generative_models.md)
- [\[CVPR 2026\] Improving Diffusion Generalization with Weak-to-Strong Segmented Guidance](improving_diffusion_generalization_with_weak-to-strong_segmented_guidance.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](../../ICLR2026/image_generation/generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[CVPR 2026\] Imagine Before Concentration: Diffusion-Guided Registers Enhance Partially Relevant Video Retrieval](imagine_before_concentration_diffusion-guided_registers_enhance_partially_releva.md)
- [\[ICML 2026\] Efficient Learning of Deep State Space Models via Importance Smoothing](../../ICML2026/image_generation/efficient_learning_of_deep_state_space_models_via_importance_smoothing.md)

</div>

<!-- RELATED:END -->
