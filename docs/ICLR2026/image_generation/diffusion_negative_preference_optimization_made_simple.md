---
title: >-
  [论文解读] Diffusion Negative Preference Optimization Made Simple
description: >-
  [ICLR2026][图像生成][扩散模型对齐] 针对扩散模型偏好对齐里"显式建模负偏好需要训两个模型 + 权重合并"的笨重做法，本文提出 Diff-SNPO：把 CFG 天然的条件/无条件双分支当作正/负偏好的两个出口，在**一个网络**里同时学正偏好和负偏好，并用改造自 Bounded DPO 的有界目标解决朴素做法的"训练越久图越糊"问题，最终在 Pick-a-Pic v2 上以一半算力超过双模型的 Diff-NPO。
tags:
  - "ICLR2026"
  - "图像生成"
  - "扩散模型对齐"
  - "负偏好优化"
  - "Classifier-Free Guidance"
  - "Bounded DPO"
  - "单网络"
---

# Diffusion Negative Preference Optimization Made Simple

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=CU5EHe1KUt](https://openreview.net/forum?id=CU5EHe1KUt)  
**代码**: https://github.com/JoshuaTTJ/DiffSNPO  
**领域**: 扩散模型 / 偏好对齐  
**关键词**: 扩散模型对齐, 负偏好优化, Classifier-Free Guidance, Bounded DPO, 单网络

## 一句话总结
针对扩散模型偏好对齐里"显式建模负偏好需要训两个模型 + 权重合并"的笨重做法，本文提出 Diff-SNPO：把 CFG 天然的条件/无条件双分支当作正/负偏好的两个出口，在**一个网络**里同时学正偏好和负偏好，并用改造自 Bounded DPO 的有界目标解决朴素做法的"训练越久图越糊"问题，最终在 Pick-a-Pic v2 上以一半算力超过双模型的 Diff-NPO。

## 研究背景与动机
**领域现状**：扩散模型靠人类反馈做偏好对齐已成标配，主流是 Diffusion-DPO（Diff-DPO）这类直接偏好优化——不训显式奖励模型，直接用"赢/输"样本对去拉高优样本、压低劣样本的相对似然。同时，扩散采样普遍依赖 Classifier-Free Guidance（CFG）：它放大"条件预测 − 无条件预测"的差，把生成朝提示词对齐的方向推、远离无条件分布产生的平庸输出。

**现有痛点**：Diff-DPO 把同一个优化目标同时加到条件和无条件两个分支上，等于"整体分布往好的方向平移"，但并没有专门去**拉大 CFG 赖以工作的对比度**，也没有显式去"压制"不想要的输出——负反馈只是隐式地体现在"相对偏好"里，对"主动远离坏样本"几乎没有控制力。为补这一点，CHATS 和 Diff-NPO 引入了显式负偏好：额外训一个在**反转标签**（赢/输互换）上学习的"负模型"，采样时拿它替换 CFG 的无条件分支。

**核心矛盾**：这套双模型方案有两个绕不开的代价。其一，训练和采样都要维护两个独立网络，显存、算力、时间直接翻倍，扩展性差。其二，两个模型是**各自独立训**出来的，参数相关性差，直接拿负模型当无条件分支会导致输出失配；Diff-NPO 为此做**权重合并**（在 reference、正模型、负模型之间插值），虽然提升了画质，却把结果严重偏向正模型，**稀释了负对齐信号**——这恰恰削弱了 NPO 本想强化的偏好对比。

**本文目标**：能不能既拿到"显式建模负偏好"的好处，又不付双模型的算力与失配代价？

**切入角度**：CFG-enabled 的扩散模型本身就有条件分支和无条件分支两条通路。与其再养一个模型，不如**直接复用这个内建的双分支**——条件分支喂正偏好，无条件（空条件）分支喂负（反转）偏好，对比天然就在一个网络里。

**核心 idea**：把正负偏好塞进同一个网络的两条 CFG 分支，省掉第二个模型和权重合并；再用一个**有界偏好目标**防止赢样本似然崩塌，让"简单、稳定、高效"的负偏好建模成立。

## 方法详解

### 整体框架
Diff-SNPO 的目标是：在单个扩散网络内同时完成正偏好对齐与负偏好对齐，复刻 Diff-NPO"显式压制坏输出"的能力，但去掉双模型与权重合并。整体只改训练目标、不改网络结构与采样管线——输入是人类偏好对 $(x^w_0, x^l_0, c)$（赢图、输图、提示词），输出是一个对齐后的扩散去噪网络 $\epsilon_\theta$，推理时照常用标准 CFG 采样。

转法分三步：(1) 借用 CFG 的分支标签 $Y\in\{+1,-1\}$——$Y=+1$ 走条件分支（有效条件 $\tilde c=c$），$Y=-1$ 走无条件分支（$\tilde c=\varnothing$），由 CFG dropout 概率 $p$ 控制走哪条；(2) 在条件分支上加正偏好、在无条件分支上加**反转**偏好，构成"朴素 Diff-SNPO"；(3) 发现朴素做法会把图练糊，于是把 Bounded DPO 改造进扩散（Diff-BDPO-UB），给目标里的 loser 项加界，得到最终 Diff-SNPO 目标。

这是一个纯目标函数（loss）层面的方法，没有多模块 pipeline，故不画框架图，用公式讲清即可。

### 关键设计

**1. 用 CFG 双分支替代双模型：把负偏好塞进无条件分支**

针对"双模型翻倍算力 + 权重合并稀释负对齐"这个核心痛点，本文不再额外训负模型，而是直接利用扩散网络内建的条件/无条件双分支。形式上引入分支标签 $Y\in\{+1,-1\}$，$\Pr(Y=+1)=1-p$、$\Pr(Y=-1)=p$（$p$ 是 CFG dropout 概率），有效条件为

$$\tilde c(Y)=\begin{cases}c,& Y=+1\ (\text{条件分支})\\ \varnothing,& Y=-1\ (\text{无条件/空分支}).\end{cases}$$

条件分支照常学正偏好，无条件分支则学**反转**偏好（赢输互换）。这样正负偏好的对比被保留在同一套参数里：推理时 CFG 的修正项 $\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t)$ 天然就是"被正偏好拉高的方向 − 被负偏好压低的方向"，不再需要 Diff-NPO 那样事后把三个模型权重插值合并（$\hat\theta^-=\theta_{\text{ref}}+\alpha(\theta^+-\theta_{\text{ref}})+\beta(\theta^--\theta_{\text{ref}})$），从根上消除了"合并偏向正模型、负对齐被稀释"的失配。

**2. 朴素 Diff-SNPO 的崩塌诊断：对抗梯度导致赢样本似然下降、图越练越糊**

直接把 Diff-DPO 目标套到双分支上得到朴素版本：

$$\mathcal L_{\text{Naive}}(\theta)=-\mathbb E\big[\log\sigma\big(Y\,T\,\omega(t)\,\beta\,(\Delta^w_t(\tilde c(Y))-\Delta^l_t(\tilde c(Y)))\big)\big],$$

其中 $\Delta^w_t,\Delta^l_t$ 是赢/输样本相对 reference 的噪声预测误差差值。但训练越久图越糊、高频细节越少。作者把根因归到 DPO 的一个已知毛病：成对 margin 的改善常常来自**同时压低赢和输的似然**（对输惩罚更狠），而非稳定拉高赢样本。本文用赢样本似然比 $\frac{\pi_\theta(x^w)}{\pi_{\text{ref}}(x^w)}\approx \mathbb E[e^{\Delta^w_t(c)}]$ 跟踪，发现朴素版的该比值随训练**持续下降**。更糟的是：两条分支共享参数，却被施加**对称但相反**的更新（一条要抬某样本似然、另一条要压它），梯度互相打架，模型只能"取平均"来调和——在生成任务里这种平均会削弱对比、抹掉细节，于是产生模糊。结论是要**打破这种破坏性的对称**，让学习偏向"抬高赢概率"。

**3. 把 Bounded DPO 搬进扩散：给 loser 项加界，防止它主导损失**

朴素崩塌的本质是 loser 项失控：当模型不断压低输样本概率时，目标里的 $\log\pi_\theta(y^l\mid x)$ 会**不成比例地变大**，让损失被 loser 支配，甚至反过来把赢样本的似然也拉低。BDPO（Cho et al. 2025）的解法是把 loser 项换成一个**含 reference 非零贡献的混合分布**

$$\pi_{\text{mix}}(y\mid x)=\lambda\,\pi_\theta(y\mid x)+(1-\lambda)\,\pi_{\text{ref}}(y\mid x),\quad \lambda\in(0,1),$$

这样 loser 的贡献被 reference 兜住、不会无限膨胀，从而**保住"促进赢样本"这一本意**，且 BDPO 与 DPO 有相同的全局最优、同时对赢似然加了下界，训练更稳。本文把它适配到扩散：先定义轨迹级 reward 与 Diff-BDPO 目标，再沿 Diff-DPO 的思路用 ELBO 上界 + Jensen 不等式得到可优化的逐步上界 $\mathcal L_{\text{Diff-BDPO-UB}}$，其中关键的两项是

$$m(x_t,c)=d_\theta-d_{\text{ref}},\qquad m_{\text{mix}}(x_t,c)=-\log\!\big(\lambda e^{-d_\theta}+(1-\lambda)e^{-d_{\text{ref}}}\big)-d_{\text{ref}},$$

$d_\theta=T\omega(t)\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2$、$d_{\text{ref}}$ 同理（把 $\theta$ 换成 reference）。直观上 $m_{\text{mix}}$ 就是把 loser 的误差项用混合分布做了"软封顶"。

**4. 最终 Diff-SNPO 目标：用分支标签把正负偏好统一进一个有界目标**

把第 3 点的 Diff-BDPO-UB 套进第 1 点的单模型负偏好框架，得到最终目标：

$$\mathcal L_{\text{SNPO}}(\theta)=-\mathbb E_{(x^w,x^l,c),t,Y}\Big[\log\sigma\big(\beta\,(m(\tilde x^w(Y),\tilde c(Y))-m_{\text{mix}}(\tilde x^l(Y),\tilde c(Y)))\big)\Big],$$

其中赢/输样本随分支标签互换：

$$\tilde x^{w/l}(Y)=\begin{cases}x^{w/l},& Y=+1\\ x^{l/w},& Y=-1.\end{cases}$$

也就是说，走条件分支（$Y=+1$）时按原始正偏好算，走无条件分支（$Y=-1$）时把赢输对调成负偏好，再统一用有界的 $m,m_{\text{mix}}$ 计分。这条目标同时做到三件事：单网络、显式负偏好、有界稳定。实测它让赢样本对数似然随训练**稳步上升**，并消除了朴素版的渐进模糊。

### 损失函数 / 训练策略
训练即最小化上面的 $\mathcal L_{\text{SNPO}}$。关键超参：正则系数 $\beta=2000$（SD1.5）/ $5000$（SDXL，沿用 Diff-DPO 配置），混合系数默认 $\lambda=0.9$；AdamW，学习率 $2.048\times10^{-8}$；SD1.5 batch 512 训 3000 步、SDXL batch 2048 训 625 步，8×A6000。注意 $\lambda=1.0$ 即退化为朴素 Diff-SNPO（无 reference 兜底），会触发模糊崩塌。

## 实验关键数据

### 主实验
Pick-a-Pic v2，DDIM 50 步、CFG=7.5，4 个种子均值（HPSv2 / PickScore / Aesthetic / ImageReward）：

| 骨干 | 方法 | HPSv2 | PickScore | Aesthetic | ImageReward |
|------|------|-------|-----------|-----------|-------------|
| SD1.5 | Baseline | 26.24 | 20.64 | 5.285 | 0.122 |
| SD1.5 | Diff-DPO | 26.55 | 21.01 | 5.382 | 0.297 |
| SD1.5 | Diff-NPO（双模型）| 26.92 | 21.46 | 5.538 | 0.379 |
| SD1.5 | CHATS（双模型）| 27.20 | 21.05 | **5.685** | 0.300 |
| SD1.5 | Diff-BDPO（本文）| 26.64 | 21.15 | 5.446 | 0.317 |
| SD1.5 | **Diff-SNPO（本文）**| **27.23** | **22.24** | 5.626 | **0.694** |
| SDXL | Diff-NPO | 28.30 | 22.67 | **5.945** | 0.985 |
| SDXL | CHATS | 28.25 | 22.34 | 5.879 | **1.054** |
| SDXL | **Diff-SNPO** | **28.33** | **22.69** | 5.813 | 1.010 |

SD1.5 上 Diff-SNPO 在 HPSv2/PickScore/ImageReward 上全面领先，ImageReward 几乎是 Diff-NPO 的近两倍（0.694 vs 0.379），且只用一半算力。SDXL 上整体持平 SOTA，但 Aesthetic 略低于 DPO/Diff-NPO——作者归因于 Pick-a-Pic 的"偏好赢样本"在强骨干 SDXL 上反而不如 base 输出美观（数据集问题，CHATS/Diff-BDPO 同样如此），并非方法缺陷。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Diff-SNPO（$\lambda=0.9$）| HPSv2 27.23 | 完整方法，训练稳定无模糊 |
| Naive-SNPO（$\lambda=1.0$）| 两项 reward 明显下降 | 去掉 reference 兜底 → 赢似然下降 → 模糊 |
| $\beta=1000/2000/3000$ | HPSv2 27.30/27.23/27.23 | 对 $\beta$ 不敏感，沿用 Diff-DPO 默认即可 |

负对齐质量（Table 5，越高越好）：Diff-DPO 31.86%、Diff-NPO 合并后 52.34%（合并前 63.80%）、**Diff-SNPO 57.45%**——Diff-NPO 权重合并掉了 10+ 个点，而单模型设计完整保住了负对齐信号。

效率（Table 4，8×A6000）：Diff-NPO 显存 44.2GB×2、相对速度 1.00×；CHATS 46.3GB、1.86×；**Diff-SNPO 44.2GB、2.00×**（条件/无条件分支并行算，CHATS 是串行）。

### 关键发现
- 收益不只来自 BDPO 的稳定化：Diff-BDPO（只稳定、无负偏好）远弱于 Diff-SNPO，说明**负偏好建模本身**才是涨点主力。
- $\lambda=1.0$（朴素版）是分水岭：少了 reference 兜底，赢似然单调下降、图渐糊，证实"对抗梯度取平均"的崩塌诊断。
- Diff-NPO 的"权重合并"是双刃剑：提画质却把负对齐准确率从 63.8% 砍到 52.3%，暴露训练/推理失配；单模型从根上避免。

## 亮点与洞察
- **把 CFG 的双分支当作"现成的正负两个模型"**：这是最妙的一笔——无须额外参数，CFG 的条件/无条件通路天然就是对比结构，等于免费拿到了 Diff-NPO 的负对齐位置。这个视角可迁移到任何带 CFG dropout 的条件生成模型。
- **诊断 + 对症下药的闭环**：先用"赢似然比随训练下降"定量坐实模糊根因（共享参数 + 对抗梯度取平均），再精准地用 BDPO 的混合分布给 loser 加界，逻辑非常干净。
- **"显式负偏好可以又简单又稳又快"**：在双模型几乎成为负偏好标配的当下，证明单模型即可，且省一半算力、采样更快——对实际落地很有价值。

## 局限与展望
- **强骨干上美学增益缩水**：SDXL 的 Aesthetic 略低于 base，源于 Pick-a-Pic 的赢样本本身不够美；作者承认这是数据集偏置而非方法本身，但也意味着方法增益依赖偏好数据质量。
- **隐式准确率非金标准**：负对齐用的 implicit accuracy 会受奖励模型 artifact 影响，作者已注明它只用于揭示训练/推理失配，不能当作总体性能的定论。
- **只验证了图像扩散（SD1.5/SDXL）**：未涉及视频、3D 等其他扩散场景，分支复用思路在更复杂条件结构下是否一样稳定有待检验。
- **上界近似**：最终目标是 Diff-BDPO 的 ELBO + Jensen 上界，与精确目标存在 gap，理论与实际最优之间的差距未深入分析。

## 相关工作与启发
- **vs Diff-DPO**：Diff-DPO 对条件/无条件分支施加**相同**目标，只平移分布、不强化 CFG 对比；本文给两分支施加**相反**偏好，专门拉大对比度，并补上 BDPO 防崩塌，因此涨点更明显。
- **vs Diff-NPO**：Diff-NPO 训独立负模型 + 推理时权重合并，翻倍算力且合并稀释负对齐（准确率掉 10+ 点）；本文单模型、无合并，负对齐准确率反而更高、算力减半。
- **vs CHATS**：CHATS 也是双模型，通过扰动条件嵌入融合正负信号，显存与本文相当但**串行**处理两分支故更慢（1.86× vs 2.00×），且美学优先、其他指标落后。
- **vs Bounded DPO（BDPO）**：BDPO 原为 LLM 的 DPO 稳定化（混合分布给 loser 加界）；本文把它经 ELBO 上界**适配到扩散轨迹**，并嵌入单模型负偏好框架，是 BDPO 在扩散对齐里的首次落地。

## 评分
- 新颖性: ⭐⭐⭐⭐ "用 CFG 双分支替代双模型"视角清爽，BDPO 适配是合理但增量的组合
- 实验充分度: ⭐⭐⭐⭐ SD1.5/SDXL 双骨干 + 5 指标 + 算力/负对齐消融充分，但仅限图像域
- 写作质量: ⭐⭐⭐⭐⭐ 痛点→诊断→对症的叙事非常清楚，公式与动机衔接好
- 价值: ⭐⭐⭐⭐ 一半算力做到更好的负偏好对齐，对扩散对齐落地实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Towards Better Optimization for Listwise Preference in Diffusion Models](towards_better_optimization_for_listwise_preference_in_diffusion_models.md)
- [\[ICLR 2026\] Reinforcing Diffusion Models by Direct Group Preference Optimization](reinforcing_diffusion_models_by_direct_group_preference_optimization.md)
- [\[ICLR 2026\] ViPO: Visual Preference Optimization at Scale](vipo_visual_preference_optimization_at_scale.md)
- [\[AAAI 2026\] Rethinking Direct Preference Optimization in Diffusion Models](../../AAAI2026/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[ICLR 2026\] Consis-GCPO: Consistency-Preserving Group Causal Preference Optimization for Vision Customization](consis-gcpo_consistency-preserving_group_causal_preference_optimization_for_visi.md)

</div>

<!-- RELATED:END -->
