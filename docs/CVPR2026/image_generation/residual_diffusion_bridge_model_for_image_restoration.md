---
title: >-
  [论文解读] Residual Diffusion Bridge Model for Image Restoration
description: >-
  [CVPR 2026][图像生成][扩散桥] 本文把扩散桥重新推导成由「均值回归 OU 过程 + Doob h-变换」统一刻画的随机插值，并用**配对图像的残差 $\boldsymbol{\pi}=\mathbf{x}_0-\boldsymbol{\mu}$ 去调制噪声的注入与去除**，让模型只在退化区域施加扰动、保护干净区域不被反复重建，从而在去雨/低光/去雪/去雾/去模糊五类通用恢复任务上平均涨 1.55 dB PSNR，同时把现有各种桥模型证明为本框架的特例。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散桥"
  - "残差调制"
  - "通用图像恢复"
  - "Doob h-变换"
  - "自适应噪声扰动"
---

# Residual Diffusion Bridge Model for Image Restoration

**会议**: CVPR 2026  
**arXiv**: [2510.23116](https://arxiv.org/abs/2510.23116)  
**代码**: https://github.com/MiliLab/RDBM (有)  
**领域**: 扩散模型 / 图像恢复  
**关键词**: 扩散桥, 残差调制, 通用图像恢复, Doob h-变换, 自适应噪声扰动

## 一句话总结
本文把扩散桥重新推导成由「均值回归 OU 过程 + Doob h-变换」统一刻画的随机插值，并用**配对图像的残差 $\boldsymbol{\pi}=\mathbf{x}_0-\boldsymbol{\mu}$ 去调制噪声的注入与去除**，让模型只在退化区域施加扰动、保护干净区域不被反复重建，从而在去雨/低光/去雪/去雾/去模糊五类通用恢复任务上平均涨 1.55 dB PSNR，同时把现有各种桥模型证明为本框架的特例。

## 研究背景与动机

**领域现状**：通用图像恢复（universal image restoration）希望用一个统一模型应对去噪、去雨、去雾、超分等多种退化。扩散模型是当前主力，路线大致三条：(a) 标准扩散把高质量图映射到高斯噪声、再从纯噪声反推；(b) 均值回归扩散（IRSDE 等）让前向终点聚集在退化观测附近，保留任务线索；(c) 扩散桥（DDBM/BBDM/GOUB）直接在「退化分布↔干净分布」两个已知端点之间建立点对点的概率路径，关联更强、保真更高。

**现有痛点**：这三类方法都靠**全局噪声扰动**来构造概率轨迹——前向把整张图打成噪声、反向再整体去噪。这带来两个硬伤：(i) 它**不区分退化程度不同的区域**，对本就干净的区域也强行注噪再重建，而反向过程不可避免有累积误差，于是干净区域被「画蛇添足」地破坏；(ii) 已有桥模型彼此公式各异，缺一个统一的解析视角来说清它们之间到底什么关系、谁是谁的特例。

**核心矛盾**：扩散桥的「全局注噪」与图像恢复的「区域退化不均」之间存在根本错配。恢复任务里退化往往是局部的（雨滴只在部分像素、雾的浓度逐空间变化），但桥模型却用一个全局噪声强度一刀切地处理所有像素。

**本文目标**：(1) 给扩散桥一个统一的 SDE 解析框架，把前向/反向过程都写成闭式；(2) 让噪声扰动**空间自适应**——退化重的地方多注噪多重建，干净的地方少动甚至不动。

**切入角度**：作者注意到，桥模型 SDE 里那个扩散项系数 $\boldsymbol{\pi}$ 一直被默认设成常数 1（全局均匀注噪）。如果把它换成**逐像素的残差** $\mathbf{x}_0-\boldsymbol{\mu}$（干净图减退化图），那么残差大的像素（退化重）噪声强、残差为 0 的像素（本就干净）根本不注噪——区域自适应就天然产生了。

**核心 idea**：用「给定分布之间的残差」来调制扩散桥的噪声注入与去除，把通用的全局桥 SDE 收敛成一个残差到噪声比（RNR）逐像素一致、且随时间平滑衰减的特例，实现退化区域自适应恢复、干净区域保真不变。

## 方法详解

### 整体框架
RDBM 的输入是一对配对图像：干净图 $\mathbf{x}_0\sim p_{HQ}$ 与退化图 $\boldsymbol{\mu}\sim p_{LQ}$（推理时只有 $\boldsymbol{\mu}$）；输出是恢复后的 $\mathbf{x}_0$。整篇方法不是搭一个多模块网络，而是**重写扩散桥背后的随机微分方程**：作者先把标准 OU（Ornstein–Uhlenbeck）过程的扩散项推广为带系数 $\boldsymbol{\pi}$ 的形式（式 9），再施加 Doob h-变换把前向终点钉死在退化图 $\mathbf{x}_T=\boldsymbol{\mu}$ 上（去掉了平稳噪声 $\lambda\epsilon$），得到一个由 $\lambda$、$\theta_t$、$\boldsymbol{\pi}$ 三个量统一参数化的广义扩散桥（式 10），并解出它在任意时刻 $\mathbf{x}_t$ 的闭式解（式 11–13）。关键一步是把 $\boldsymbol{\pi}$ 取成残差 $\mathbf{x}_0-\boldsymbol{\mu}$，使噪声扰动逐像素自适应。反向过程用贝叶斯定理推出确定性采样公式（式 18–19），训练时用一个 U-Net 一次性预测「残差×噪声」的乘积 $\boldsymbol{\pi}\epsilon$。最后作者把流匹配、VE/VP 桥、布朗桥、OU 桥都证明成本框架在特定 $(\theta_t,\lambda,\boldsymbol{\pi})$ 配置下的特例。

整张图描述的是一个标准的「前向加噪—反向去噪」扩散循环（训练时用闭式 $\mathbf{x}_t$ 直接采样、反向时 U-Net 迭代），创新点集中在 SDE 系数 $\boldsymbol{\pi}$ 的取法与统一推导上，因此不另画 pipeline 框架图，下面用公式把每个设计讲清。

### 关键设计

**1. 广义前向过程：用 OU + Doob h-变换把扩散桥写成可控的闭式 SDE**

针对「现有桥模型公式各异、缺统一解析视角」这一痛点，作者从带可调扩散系数的 OU 过程出发
$$d\mathbf{x}_t=\theta_t(\boldsymbol{\mu}-\mathbf{x}_t)dt+\boldsymbol{\pi}\sigma_t d\omega_t,$$
其均值回归特性保证轨迹会朝着 $\boldsymbol{\mu}$ 收敛。再对它施加 Doob h-变换（h 是从 $t$ 到 $T$ 的对数转移核梯度），把终点强制锚定在 $\mathbf{x}_T=\boldsymbol{\mu}$，从而消除桥端点的平稳噪声。在固定漂移-扩散系数比 $\lambda=\sigma_t^2/(2\theta_t)$ 下，得到广义桥 SDE（式 10）
$$d\mathbf{x}_t=\theta_t\coth(\overline{\theta}_{t:T})(\boldsymbol{\mu}-\mathbf{x}_t)dt+\sqrt{2\boldsymbol{\pi}^2\lambda\theta_t}\,d\omega_t,$$
其中 $\overline{\theta}_{s:t}=\int_s^t\theta_z dz$。它有解析解：任意时刻状态服从高斯分布，均值 $\mathbb{E}[\mathbf{x}_t]=\boldsymbol{\mu}+(\mathbf{x}_0-\boldsymbol{\mu})\Theta_t$、方差 $\mathrm{Var}[\mathbf{x}_t]=\boldsymbol{\pi}^2\Sigma_t^2$，其中 $\Theta_t=\sinh(\overline{\theta}_{t:T})/\sinh(\overline{\theta}_{0:T})$、$\Sigma_t^2=2\lambda\sinh(\overline{\theta}_{0:t})\sinh(\overline{\theta}_{t:T})/\sinh(\overline{\theta}_{0:T})$。有了闭式 $\mathbf{x}_t$，训练时就能在任意时间步直接一步采样、无需逐步前向，这也是它能跑得动多任务训练的前提。

**2. 残差调制噪声：把扩散系数 $\boldsymbol{\pi}$ 取成残差 $\mathbf{x}_0-\boldsymbol{\mu}$，实现像素自适应扰动**

这是全文的核心。式 11 揭示概率轨迹其实是「残差项」和「高斯噪声项」的加权混合。为刻画其时间动态，作者定义**残差到噪声比（RNR，residual-to-noise ratio）** ——逐像素 $(i,j)$ 在时刻 $t$ 的残差能量与噪声能量之比：
$$R(i,j,t)=\frac{[x_0(i,j)-\boldsymbol{\mu}(i,j)]^2}{2[\boldsymbol{\pi}(i,j)]^2\lambda}\cdot\frac{\sinh(\overline{\theta}_{t:T})}{\sinh(\overline{\theta}_{0:t})\sinh(\overline{\theta}_{0:T})}.$$
它由两部分组成：第一项只跟残差和常数 $\lambda$ 有关，第二项完全由 $\theta_t$ 决定且随时间单调下降（$t\to0$ 趋于无穷、$t\to T$ 趋于无穷小）。以往工作默认 $\boldsymbol{\pi}=1$（全局均匀注噪），会导致两个病态：(i) 退化程度不同的区域被一视同仁，干净区域因反向累积误差被冗余、不完美地重建；(ii) 分子里逐像素的 $[x_0(i,j)-\boldsymbol{\mu}(i,j)]^2$ 可能出现不连续跳变，破坏 RNR 本应平滑单调的衰减。作者的对策是令 $\boldsymbol{\pi}=\mathbf{x}_0-\boldsymbol{\mu}$，此时分子分母里的残差项**直接约掉**，RNR 退化为与像素无关、仅随时间平滑衰减的 $R(t)\propto\sinh(\overline{\theta}_{t:T})/[\sinh(\overline{\theta}_{0:t})\sinh(\overline{\theta}_{0:T})]$。直观效果是：残差大（退化重）的像素方差 $\boldsymbol{\pi}^2\Sigma_t^2$ 大、被多注噪多重建；残差为 0（本就干净）的像素方差为 0、几乎不被扰动——这就把「全局注噪」改成了「按退化程度自适应注噪」，从机制上保护了干净区域。

**3. 残差桥得分匹配：确定性反向采样 + 一次预测「残差×噪声」**

有了前向闭式分布，反向过程需要从 $\mathbf{x}_t$ 推回 $\mathbf{x}_{t-1}$。作者用贝叶斯定理推出确定性采样公式（式 19）
$$\mathbf{x}_{t-1}=\boldsymbol{\mu}+\frac{\Theta_{t-1}}{\Theta_t}(\mathbf{x}_t-\boldsymbol{\mu})-\Big(\frac{\Theta_{t-1}}{\Theta_t}\Sigma_t-\Sigma_{t-1}\Big)\boldsymbol{\pi}\epsilon_t,$$
其中含两个未知量：残差 $\boldsymbol{\pi}$ 和噪声 $\epsilon_t$。由于二者总以乘积形式出现，作者不分别估计，而是让一个 U-Net $\boldsymbol{\pi}_\epsilon^{\dot\theta}(\mathbf{x}_t,t,\boldsymbol{\mu})$ **一次性预测乘积 $\boldsymbol{\pi}\epsilon$**。训练目标从对齐各时间步分布的 KL 散度（式 20）化简为均值匹配，最终落到一个简洁的 L1 损失（见下节）。采样时从退化图 $\boldsymbol{\mu}$ 出发、迭代式 19 即可恢复 $\mathbf{x}_0$，且只需 10 步（NFE=10）就达到最优。

**4. 统一现有桥模型：把它们都纳为本框架的特例**

作者证明主流扩散过程都是 RDBM 在特定 $(\theta_t,\lambda,\boldsymbol{\pi})$ 下的特例：$\boldsymbol{\pi}=0$ 时退化为流匹配（Flow Matching）；$\theta_t\to0$ 且 $\lambda$ 取不同极限时分别对应 VE 桥、VP 桥；$\boldsymbol{\pi}=1$ 时是布朗桥 / OU 桥（如 GOUB）。这个统一不只是形式上的归纳——它实证性地说明了「为什么 $\boldsymbol{\pi}=\mathbf{x}_0-\boldsymbol{\mu}$ 比 $\boldsymbol{\pi}=1$ 更优」：消融里 $\boldsymbol{\pi}=1$（等价已有桥）平均 30.15 dB，而残差版到 31.04 dB，相当于在同一框架下用更优的系数取法击败了它的所有特例。

### 损失函数 / 训练策略
训练目标由各时间步分布对齐的 KL 散度化简为残差桥得分匹配，实现为 L1 损失：
$$\nabla_\theta\big\|\boldsymbol{\pi}\epsilon-\boldsymbol{\pi}_\epsilon^{\dot\theta}(\mathbf{x}_t,t,\boldsymbol{\mu})\big\|_1.$$
训练用 8×A800、PyTorch，500k 迭代、batch 20（均分到各任务）、Adam、学习率 1e-4，随机裁 256×256 patch，骨干为 U-Net。通过改隐藏层通道数 $C$ 与通道倍率得到 T/S/B/L 四个规模（0.45M–7.73M 参数）。噪声调度用 cosine、平稳方差 $\lambda=10/255$，测试用 10 个时间步做全分辨率推理。

## 实验关键数据

### 主实验
五类恢复任务（去雨 / 低光增强 / 去雪 / 去雾 / 去模糊）在混合数据集上重训对比，下表为各任务平均与复杂度：

| 方法 | 年份 | 平均 PSNR↑ | 平均 SSIM↑ | Params(M) | FLOPs(G) |
|------|------|-----------|-----------|-----------|----------|
| GOUB | 2024 | 27.60 | 0.895 | 137.13 | 379.34 |
| ConvIR | 2024 | 29.49 | 0.903 | 14.82 | 128.93 |
| MaIR | 2025 | 29.51 | 0.904 | 20.71 | 110.44 |
| **RDBM-B** | - | **30.24** | **0.904** | **3.65** | **23.97** |
| **RDBM-L** | - | **31.04** | **0.917** | 7.73 | 32.93 |

RDBM-L 在全部任务上大幅领先，平均较此前 SOTA（MaIR）涨 **1.55 dB PSNR / 0.013 SSIM**；尤为关键的是 **RDBM-B 仅 3.65M 参数 / 23.97 GFLOPs** 就以 30.24 dB 超过 20.71M 的 MaIR，参数量约为其 1/6、FLOPs 约 1/5，体现了残差调制带来的效率优势。模型从 T 到 L 单调涨点，可扩展性良好。

### 消融实验

| 配置维度 | 最优取值 | 平均 PSNR / SSIM | 说明 |
|---------|---------|-----------------|------|
| 噪声调度 | Cosine | 31.04 / 0.917 | 优于 Linear(30.99) 与 Sigmoid(30.84) |
| 平稳方差 $\lambda$ | 10/255 | 31.04 / 0.917 | 1/255 为 30.36，100/255 跌到 29.08 |
| 采样步数 NFE | 10 | 31.04 / 0.917 | 2 步仅 22.81；20/50/100 步反而缓降 |
| 扩散系数 $\boldsymbol{\pi}$ | $\mathbf{x}_0-\boldsymbol{\mu}$ | 31.04 / 0.917 | $\boldsymbol{\pi}{=}0$(流匹配)28.21；$\boldsymbol{\pi}{=}1$(已有桥)30.15 |

### 关键发现
- **残差调制是涨点主力**：$\boldsymbol{\pi}$ 从 0（流匹配）→1（已有桥）→残差，平均 PSNR 从 28.21→30.15→31.04，残差版较 $\boldsymbol{\pi}{=}1$ 再涨近 0.9 dB，直接验证「用残差做空间自适应注噪」优于全局注噪。取绝对值 $|\mathbf{x}_0-\boldsymbol{\mu}|$ 结果相近（30.94），说明方法对残差符号不敏感。
- **NFE=10 是甜点，更多步反而掉点**：步数从 2→10 涨点，超过 10 后缓降。作者解释为统一模型在多重退化样本上倾向先去主退化再去次退化，步数过多会让输出偏离单一参考图——这是「通用模型 + 单退化评测」框架下的特有现象。
- **$\lambda$ 过大全面崩坏**：$\lambda=100/255$ 时去雾从 33.45 跌到 27.03，说明全局噪声强度过高会淹没残差调制的自适应信号。

## 亮点与洞察
- **「换一个系数」式的优雅创新**：不改网络结构、不加模块，只把扩散桥 SDE 里一直被默认为常数 1 的扩散系数 $\boldsymbol{\pi}$ 换成残差 $\mathbf{x}_0-\boldsymbol{\mu}$，就让 RNR 从「逐像素跳变」变成「全局平滑衰减」，并天然实现区域自适应。这种「在已有框架的自由度里找最优取法」的思路很值得借鉴。
- **统一性既是理论贡献也是说服力来源**：把流匹配 / VE/VP 桥 / 布朗桥 / OU 桥都证明成特例，不仅梳理了扩散桥家族的谱系，还让「残差版击败 $\boldsymbol{\pi}{=}1$」这个消融具备了「在同一框架内打败所有前辈」的解释力。
- **效率红利可迁移**：RDBM-B 用 1/6 参数超过 SOTA，说明把退化先验（残差）显式注入扩散系数，能省去网络去学「哪里该改、哪里别动」的负担，这一思路可推广到其他带配对监督的生成式恢复 / 转换任务。

## 局限性 / 可改进方向
- **依赖配对的残差**：$\boldsymbol{\pi}=\mathbf{x}_0-\boldsymbol{\mu}$ 训练时需要干净图与退化图严格配对、像素对齐；对无配对或弱对齐数据（真实世界退化往往无 GT）该如何构造残差，论文未深入。
- **统一模型对多重退化的「偏科」**：作者自己发现，当样本含多种退化时模型会先去主退化、输出偏离单参考图，导致步数过多反而掉点。这意味着在真正复合退化场景下，单参考 PSNR 评测可能低估其视觉质量，也提示该框架对「多退化共存」尚无显式建模。
- **理论以高斯/线性 SDE 为前提**：闭式解建立在 OU 过程与高斯转移核之上，对非高斯、强非线性退化（如严重压缩伪影、复杂运动模糊）是否仍成立，缺乏验证。
- **改进思路**：可探索无配对设定下用估计残差或自监督残差替代真实残差；或对复合退化引入逐退化的 $\boldsymbol{\pi}$ 分解。

## 相关工作与启发
- **vs IRSDE（均值回归 SDE）**: IRSDE 让前向轨迹回归到带平稳噪声的退化图，仍是全局扰动；RDBM 用 Doob h-变换去掉终点平稳噪声并用残差调制注噪，空间自适应。实验上 IRSDE 平均仅 19.55 dB，远低于 RDBM。
- **vs GOUB / DDBM / BBDM（扩散桥）**: 这些方法默认 $\boldsymbol{\pi}=1$ 做全局注噪，是 RDBM 的特例；RDBM 把系数换成残差后在同框架内超过它们（GOUB 27.60 vs RDBM-L 31.04），且参数量小一个量级。
- **vs Flow Matching**: 流匹配丢弃随机噪声、构造确定性传输路径，对应 RDBM 中 $\boldsymbol{\pi}=0$；消融显示该设定仅 28.21 dB，说明适度的残差调制噪声比纯确定性路径更利于恢复。
- **vs PromptIR / MaIR 等判别式通用恢复**: 它们靠 prompt / 状态空间建模一次前向出结果；RDBM 是生成式迭代恢复，用更少参数取得更高保真，但需多步采样。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把扩散桥扩散系数取成残差，既是简洁创新又统一了整个桥模型家族
- 实验充分度: ⭐⭐⭐⭐ 五任务 + 四规模 + 四维消融充分，但复合退化与无配对设定未覆盖
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨、特例表清晰，但公式密集、对读者数学门槛较高
- 价值: ⭐⭐⭐⭐⭐ 小参数超 SOTA + 统一框架，对生成式图像恢复有较强方法论与实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EMR-Diff: Edge-aware Multimodal Residual Diffusion Model for Hyperspectral Image Super-resolution](emr-diff_edge-aware_multimodal_residual_diffusion_model_for_hyperspectral_image_.md)
- [\[CVPR 2026\] Low-Rank Residual Diffusion Models](low-rank_residual_diffusion_models.md)
- [\[CVPR 2026\] ResCa: Residual Caching for Diffusion Transformers Acceleration](resca_residual_caching_for_diffusion_transformers_acceleration.md)
- [\[CVPR 2026\] Decoupled Residual Denoising Diffusion Models for Unified and Data Efficient Image-to-Image Translation](decoupled_residual_denoising_diffusion_models_for_unified_and_data_efficient_ima.md)
- [\[CVPR 2026\] CARD: Correlation Aware Restoration with Diffusion](card_correlation_aware_restoration_with_diffusion.md)

</div>

<!-- RELATED:END -->
