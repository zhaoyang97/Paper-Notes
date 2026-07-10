---
title: >-
  [论文解读] NoiseTilt: Noise-Tilted Reverse Kernels for Diffusion Reward Alignment
description: >-
  [ECCV 2026][图像生成][噪声倾斜反向核] 本文提出 NTRK(Noise-Tilted Reverse Kernel):在扩散推理时**不改反向核的均值**,而是把奖励梯度经一个「白化算子」变成一个合法的高斯噪声方向后注入到噪声项里,从而在每步只采一个样本的前提下同时拿到「梯度引导」和「噪声兼容性」;在美学图像生成上仅用 25 NFE 就超过了最强基线在 500 NFE 下的奖励,算力降低 20 倍。
tags:
  - "ECCV 2026"
  - "图像生成"
  - "噪声倾斜反向核"
  - "白化算子"
  - "推理时奖励对齐"
  - "置信区间投影"
  - "无训练引导"
---

# NoiseTilt: Noise-Tilted Reverse Kernels for Diffusion Reward Alignment

**会议**: ECCV 2026  
**arXiv**: [2606.18066](https://arxiv.org/abs/2606.18066)  
**代码**: 无（原文未给出 NTRK 的官方代码仓库，仅引用了 FLUX / Wan2.1 等 base 模型链接）  
**领域**: 扩散模型 / 对齐RLHF  
**关键词**: 噪声倾斜反向核、白化算子、推理时奖励对齐、置信区间投影、无训练引导

## 一句话总结
本文提出 NTRK(Noise-Tilted Reverse Kernel):在扩散推理时**不改反向核的均值**,而是把奖励梯度经一个「白化算子」变成一个合法的高斯噪声方向后注入到噪声项里,从而在每步只采一个样本的前提下同时拿到「梯度引导」和「噪声兼容性」;在美学图像生成上仅用 25 NFE 就超过了最强基线在 500 NFE 下的奖励,算力降低 20 倍。

## 研究背景与动机

**领域现状**:推理时(inference-time)奖励对齐已经成为扩展预训练扩散模型能力最有效的手段之一——不用重训模型,只要把用户偏好写成奖励函数 $r(\cdot)$,就能在迭代去噪的每一步把采样分布往高奖励区域推,已经被广泛用于去模糊、超分、美学生成、文图对齐乃至可控视频生成。

**现有痛点**:所有这些方法本质都在回答同一个问题——如何修改或利用反向高斯核来引入奖励信息,而它们分成两大互斥流派,各有硬伤。**均值平移(mean-shifted / 梯度引导)**类以 DPS 为代表,把奖励梯度直接加到反向核的均值上(见公式 $\tilde{\bm{\mu}}_\theta=\bm{\mu}_\theta+\lambda_t\nabla_{\bm{x}_t}r(\hat{\bm{x}}_{0|t})$),能拿到梯度方向但会把中间状态推出「噪声兼容区(noise-compatible regime)」——即预训练模型真正训练过的那条窄带,梯度越大偏离越远,导致样本走向 OOD、画质下降,还容易 reward hacking。**搜索(search-based)**类如 SVDD 则从原始反向核里采 $K$ 个候选、按奖励 argmax 挑一个,完全保留了原生核和画质,但**没有任何梯度信号**,只靠随机采样撞高奖励方向,当高奖励样本落在低密度区时需要海量采样。

**核心矛盾**:梯度引导 与 噪声兼容性 是这条 trade-off 的两端,据作者所知**此前没有任何方法能同时占住两者**(见 Table 1)。

**本文目标 / 切入角度**:作者的目标就是补上这个空白。关键观察是:搜索类方法之所以能保画质,是因为它**不动均值,只在噪声项里做文章**(挑一个好噪声本质是一种隐式的「噪声倾斜」);那能不能不采 $K$ 个、而是直接**构造**出那个好噪声?

**核心 idea**:保持预训练均值 $\bm{\mu}_\theta$ 完全不动,把奖励信息**全部**从噪声项注入。难点在于:反向核是高斯核,注进去的东西必须仍然是「典型高斯噪声」(不只是范数匹配,还要空间不相关、整体服从高斯统计),而原始奖励梯度是结构化、确定性的,直接当噪声注入会产生「非典型噪声(atypical noise)」同样把状态推出分布(Figure 2)。于是本文引入统计学习里的**白化(whitening)**思想,设计一个白化算子 $\mathcal{W}$ 先把梯度处理成噪声兼容的方向再注入——白化不是辅助步骤,而是 NTRK 的核心机制,对齐效果直接随白化质量提升。

## 方法详解

### 整体框架

NTRK 要解决的是「如何把奖励梯度注入反向核而不破坏噪声兼容区」。整体思路只有两步:(1) 用白化算子 $\mathcal{W}$ 把结构化的奖励梯度 $\nabla_{\bm{x}_t}r(\hat{\bm{x}}_{0|t})$ 变成一个合法的白高斯噪声方向 $\bm{w}_t$;(2) 用「高斯混合恒等式」把 $\bm{w}_t$ 和一份独立随机噪声按 $\sqrt{\rho_t}$ / $\sqrt{1-\rho_t}$ 混合成 $\tilde{\bm{\epsilon}}_t$,替换掉基础采样里的 $\bm{\epsilon}_t$,而均值 $\bm{\mu}_\theta$ 和名义噪声尺度 $\sigma_t$ 一字不改。核心是纯采样核 + kernel 推导 + 一个投影式白化算子,因此下面用公式讲清而不画 pipeline 图。

先把问题形式化:奖励对齐目标是找一个目标分布 $p_0^*=\arg\max_q \mathbb{E}_{\bm{x}_0\sim q}[r(\bm{x}_0)]-\beta\,\mathcal{D}_{\text{KL}}[q\|p_0]$($\beta$ 越小奖励倾斜越强)。对应的最优反向核可近似为 $p_\theta^*(\bm{x}_{t-1}|\bm{x}_t)\propto p_\theta(\bm{x}_{t-1}|\bm{x}_t)\exp(V_{t-1}(\bm{x}_{t-1})/\beta)$,其中难解的价值函数用 Tweedie 后验均值近似 $V_t(\bm{x}_t)\approx r(\hat{\bm{x}}_{0|t})$,$\hat{\bm{x}}_{0|t}:=\mathbb{E}[\bm{x}_0\mid\bm{x}_t]$。**四种反向核就是这个近似的四种实现方式**,判据是「标准化扰动 $\bm{\eta}_t:=(\bm{x}_{t-1}-\bm{\mu}_\theta)/\sigma_t$ 还是不是标准高斯噪声」:

- **Base**:$\bm{x}_{t-1}=\bm{\mu}_\theta+\sigma_t\bm{\epsilon}_t$,此时 $\bm{\eta}_t=\bm{\epsilon}_t$ 天然是高斯,但没有奖励信息。
- **Mean-Shifted (DPS)**:$\bm{\eta}_t^{\text{mean}}=\bm{\epsilon}_t+\frac{\lambda_t}{\sigma_t}\nabla_{\bm{x}_t}r$,多出一个确定性梯度项,**不再是高斯噪声**——这正是画质下降的根因,也是本文点出的关键 mismatch。
- **Search-Based (SVDD)**:采 $K$ 个候选选奖励最高的 $\bm{\epsilon}_t^{(i^\star)}$,$\bm{\eta}_t$ 仍是高斯(选出来的还是某个真高斯采样),但代价是每步 $K$ 次评估。
- **Noise-Tilted (NTRK, 本文)**:$\bm{\eta}_t^{\text{noise}}=\tilde{\bm{\epsilon}}_t$,单次采样、既含梯度又是合法高斯。

### 关键设计

**1. 噪声倾斜反向核:把奖励从均值搬到噪声,靠高斯混合保持合法性**

痛点是均值平移会污染 $\bm{\eta}_t$。NTRK 的机制是保留 $\bm{\mu}_\theta$ 不动,只替换噪声项。核心用到一条基本恒等式:若 $\bm{\epsilon}_1,\bm{\epsilon}_2\sim\mathcal{N}(\bm{0},\bm{I})$ 独立,则对任意 $\rho\in[0,1]$,$\sqrt{\rho}\,\bm{\epsilon}_1+\sqrt{1-\rho}\,\bm{\epsilon}_2\sim\mathcal{N}(\bm{0},\bm{I})$。于是定义引导噪声

$$\tilde{\bm{\epsilon}}_t=\sqrt{\rho_t}\,\bm{w}_t+\sqrt{1-\rho_t}\,\bm{\epsilon}_t,\qquad \bm{x}_{t-1}=\bm{\mu}_\theta(\bm{x}_t,t)+\sigma_t\tilde{\bm{\epsilon}}_t$$

其中 $\bm{w}_t:=\mathcal{W}(\nabla_{\bm{x}_t}r(\hat{\bm{x}}_{0|t}))$ 是白化后的奖励方向,$\bm{\epsilon}_t$ 是无偏随机项,$\rho_t\in[0,1]$ 控制引导强度。**这一步之所以有效,是因为只要 $\bm{w}_t$ 本身是合法白噪声,混合结果就严格保持 $\mathcal{N}(\bm{0},\bm{I})$**——即 $\bm{\eta}_t^{\text{noise}}=\tilde{\bm{\epsilon}}_t$ 依旧是标准高斯,反向核的均值和 $\sigma_t$ 都和 base 一致,每步天然停留在噪声兼容区。它相当于把 SVDD 的「采 $K$ 个挑最好」压缩成「直接构造一个最好」,把每步成本从 $K$ 降到 $1$(附录给了 $\rho_t$ 在局部线性假设下相对搜索引导的启发式解释)。

**2. 白化算子 $\mathcal{W}$:把结构化梯度投影到「典型高斯集」而不丢方向**

上一条的全部合法性都押在「$\bm{w}_t$ 是合法白噪声」上,但原始梯度是结构化确定量,直接注会变成非典型噪声(Figure 2:非典型噪声会诱发伪影)。难点在于高维下 $\mathcal{N}(\bm{0},\bm{I})$ 虽然处处有密度,但几乎全部概率质量集中在一个极窄的**典型集(typical set)**壳层上,而典型集没有闭式刻画。$\mathcal{W}$ 的做法是**用一族由标准正态已知统计量导出的高置信约束去近似典型集**,并把 $\mathcal{W}$ 定义成一串到对应置信集上的欧氏投影(全部取 $99.99\%$ 置信界,$\alpha=10^{-4}$)。

其核心积木是**两级序统计投影(2OS, Two-level Order Statistics)**。把待白化向量 $\bm{x}\in\mathbb{R}^N$ reshape 成 tile 矩阵 $\bm{Y}\in\mathbb{R}^{M\times D}$($N=MD$),先对每行排序再对每列排序:$\bm{Z}=\mathrm{sort}_0(\mathrm{sort}_1(\bm{Y}))$。直觉上 $Z_{r,j}$ 是「所有 tile 里第 $j$ 个序统计量中第 $r$ 小的那个」,是一种「秩的秩」摘要;对标准高斯而言每个 $Z_{r,j}$ 都会尖锐集中,因此每个 $(r,j)$ 都能算出很紧的分位数区间 $(L_{r,j},U_{r,j})=(\Phi^{-1}(q_{r,j}^{\text{lo}}),\Phi^{-1}(q_{r,j}^{\text{hi}}))$(由嵌套序统计量的 Beta 分位构造)。投影即 **sort–clip–unsort**:排序 → 把每个 $Z_{r,j}$ clip 到区间 → 逆排列还原;作者证明这恰好等于到置信集 $\mathcal{C}_{\text{2os}}=\{\bm{Y}:L_{r,j}\le Z_{r,j}\le U_{r,j}\}$ 的欧氏投影。**为什么有效**:2OS 阻止极端值扎堆在少数 tile 里,逼每个 tile 都有一份从小到大均衡分布的值,像真高斯;把 2OS 再作用到 tile 级均值/能量统计上可约束块级矩,再在多个正交变换域(如傅里叶域)上重复就能捕捉纯值域投影够不到的结构化相关性。完整 $\mathcal{W}$ 就是这些 2OS 投影在多域上的复合。

**3. 置信区间投影 vs 硬约束:只压非典型结构、几乎不动真噪声**

这是 $\mathcal{W}$ 相对已有噪声兼容方法的关键改进。此前 WGNC 首次把噪声兼容性 reframe 成投影问题,但用的是**傅里叶域上的硬等式范数约束**,副作用是连本来就合法的噪声也会被扭曲。NTRK 用**置信区间投影**替代硬等式:结构化输入会被大幅白化,而典型高斯噪声因为本就落在 $99.99\%$ 区间内,几乎原样通过(Figure 4 最右两列 cosine similarity $>0.99999$)。这条性质很重要——它保证了公式中 $\sqrt{1-\rho_t}\,\bm{\epsilon}_t$ 那份随机项不被算子破坏,混合后整体仍是干净高斯。⚠️ 2OS 的完整多域构造与 Beta 分位推导在附录 B,以原文为准。

### 损失函数 / 训练策略

NTRK 是**完全无训练(training-free)、推理时**的采样方法,不引入任何 loss、不更新任何参数,是对现有采样循环的 drop-in 替换。关键超参:引导强度 $\rho_t\in[0,1]$、KL 温度 $\beta$、置信水平固定 $1-\alpha=99.99\%$。所有实验统一固定采样步数为 **25 步**。NTRK 可叠加多粒子策略 Best-of-N(BoN):跑多条独立采样轨迹选奖励最高者,以匹配总 NFE 预算;正文中带 BoN 的方法标记为 †。此外因为它只动采样、不动权重,与微调(如 MixGRPO)**正交**,可以直接叠在微调模型之上。

## 实验关键数据

设置:图像用 FLUX、视频用 Wan2.1 作为 base flow 模型,全程 25 采样步。单粒子方法(DPS/FreeDoM)统一用 BoN 补齐到相同 NFE 做公平对比。区分 **target reward**(优化时用的)与 **held-out reward**(优化时没见过的)。

### 主实验

美学 + 文图对齐图像生成(Table 2 节选,↑ 越高越好,粗体为该列最优):

| 方法 | NFE | 美学-Aesthetic ↑ | 美学-HPSv2 ↑ | 美学-ImageReward ↑ | 文图-PickScore ↑ | 文图-Aesthetic ↑ |
|---|---|---|---|---|---|---|
| Base [FLUX] | 25 | 6.0282 | 0.2759 | 1.0538 | 0.2054 | 5.4664 |
| BoN | 500 | 6.7310 | 0.2890 | 1.1419 | 0.2146 | 5.8582 |
| DPS† | 500 | 6.7647 | 0.2861 | 1.0639 | 0.2147 | 5.8073 |
| FreeDoM† | 533 | 6.8406 | 0.2853 | 0.9941 | 0.2133 | 5.8492 |
| SVDD | 500 | 7.1363 | 0.2814 | 1.0256 | 0.2204 | 5.8743 |
| RBF | 500 | 6.9900 | 0.2826 | 1.0761 | 0.2202 | 5.8618 |
| DAS | 500 | 6.9384 | 0.2860 | 1.0568 | 0.2139 | 5.8385 |
| Ψ-Sampler | 500 | 7.0116 | 0.2847 | 1.1235 | 0.2120 | 5.7329 |
| **NTRK (Ours)** | **25** | **7.4510** | 0.2928 | **1.2565** | **0.2224** | 5.7720 |
| **NTRK† (Ours)** | 500 | **7.9656** | **0.2932** | 1.1669 | **0.2327** | **5.9020** |

关键点:**NTRK 仅 25 NFE(1/20 算力)的美学分 7.4510 就已超过所有 500 NFE 基线的最高分(SVDD 7.1363)**;加 BoN 到 500 NFE 更冲到 7.9656。held-out 的 ImageReward / HPSv2 也拿最优,说明不是靠 reward hacking 牺牲画质换的。

视频生成(Table 3,target = VideoReward = MQ+VQ+TA 之和):

| 方法 | NFE | VideoReward ↑ | Dynamic ↑ | Aesthetic ↑ | Subject ↑ |
|---|---|---|---|---|---|
| Base [Wan2.1] | 25 | -0.399 | 0.9300 | 0.6104 | 0.9589 |
| DPS | 25 | -0.130 | 0.9350 | 0.5867 | 0.9398 |
| FreeDoM | 25 | -0.211 | 0.6900 | 0.5880 | 0.9586 |
| **NTRK (Ours)** | 25 | **3.465** | **0.9500** | **0.6120** | **0.9591** |

VideoReward 从基线的 -0.399 / DPS 的 -0.130 直接拉到 **3.465**,量级差距悬殊,且在多个 VBench held-out 指标上也最优或次优。

叠加微调模型(Table 4,base = FLUX + MixGRPO 微调,target = PickScore):

| 方法 | NFE | PickScore ↑ | Aesthetic ↑ | HPSv2 ↑ | ImageReward ↑ |
|---|---|---|---|---|---|
| Base | 25 | 0.2054 | 5.4664 | 0.2316 | 0.1710 |
| MixGRPO | 25 | 0.2166 | 6.5245 | 0.2679 | 0.7605 |
| └ DPS† | 500 | 0.2235 | 6.6966 | 0.2840 | 1.0501 |
| └ **NTRK† (Ours)** | 500 | **0.2545** | **6.7281** | **0.3224** | **1.2648** |

在微调模型之上,NTRK 在 target 和全部 held-out 上都稳压 DPS,验证「推理时对齐 ⊥ 微调,可叠加」。⚠️ 上表 NTRK† 的 Aesthetic 原文数字为 6.7281,此处照录,以原文为准。

### 消融实验

论文没有单列一张传统「逐模块开关」的消融表,其核心「消融」体现在 **Table 2 的 NFE-效率维度**:NTRK 在 25 NFE 与 500 NFE 两档都给出结果,以隔离「方法本身」与「BoN 多粒子」的贡献——25 NFE 的 NTRK(纯方法,无 BoN)已超所有 500 NFE 基线,说明增益主要来自噪声倾斜核而非堆采样。另外 Figure 4 是对白化算子 $\mathcal{W}$ 的**逐组件定性消融**:从左到右依次叠加 2OS 投影 → tile 级统计 → 多域,latent 越来越像典型高斯、生成样本越来越真实;而对本就是典型高斯的输入 $\mathcal{W}$ 几乎不改变(cosine $>0.99999$),侧证了置信区间投影相对 WGNC 硬约束「只压结构、不伤真噪声」的设计目标。

### 关键发现
- **贡献最大的模块是白化算子 $\mathcal{W}$**:作者明确指出「对齐效果直接随白化质量提升」,$\mathcal{W}$ 是把「噪声倾斜」这个想法变得可行的核心;去掉白化直接注梯度就会退化成非典型噪声、诱发伪影(Figure 2)。
- **效率发现极其抢眼**:美学生成上 25 NFE 打平/超过 500 NFE 最强基线,20× 算力缩减——因为它把「搜索 $K$ 个好噪声」换成「构造 1 个好噪声」,把随机撞运气变成确定性引导。
- **held-out 奖励不塌**:很多梯度引导方法涨 target 时 held-out 画质会崩(reward hacking),NTRK 因为始终待在噪声兼容区,held-out 指标同样领先,说明它是「真对齐」而非「刷分」。
- **视频任务上量级碾压**:VideoReward 从负值跳到 3.465,是全文最戏剧化的提升,暗示在预训练分布更脆弱的视频域,「不出分布」这一性质收益更大。

## 亮点与洞察
- **把奖励从均值搬到噪声,是个视角上的「阿哈」**:所有前作都在纠结怎么改均值,本文反其道——均值一字不改,把全部奖励信息塞进噪声项。这一步让「梯度引导」和「保持在训练分布内」第一次不再互斥(Table 1 里 NTRK 是唯一三项全 ✓)。
- **高斯混合恒等式用得极巧**:$\sqrt{\rho}\bm{\epsilon}_1+\sqrt{1-\rho}\bm{\epsilon}_2$ 保持 $\mathcal{N}(\bm{0},\bm{I})$ 是教科书事实,但用它来「安全地把一个引导方向掺进噪声而不破坏高斯性」是很聪明的复用,$\rho_t$ 天然成了引导强度旋钮,且理论上零分布偏移。
- **2OS 的 sort–clip–unsort = 欧氏投影**:把一个看似 heuristic 的排序-裁剪操作证明成对置信集的严格投影,既优雅又可复用——这套「用高置信区间近似高维典型集」的思路可迁移到任何需要「让一个向量看起来像白噪声」的场景(噪声优化、扩散引导稳定化等)。
- **置信区间投影 > 硬约束**这个 trick 很实用:WGNC 的硬等式会误伤合法噪声,改成 $99.99\%$ 区间后「结构化才压、干净就放过」,cosine $>0.99999$ 的近乎恒等性质是它能安全嵌入采样循环的前提。
- **与微调正交、可叠加**:一句「参数微调 ⊥ 采样引导」被 Table 4 坐实,意味着 NTRK 可以作为任何已对齐模型的免费增益层。

## 局限与展望
- **白化算子的计算/工程成本未充分讨论**:每步都要在多个变换域上做多次「排序-裁剪-逆排序」,原文正文没给 $\mathcal{W}$ 相对朴素梯度引导的 wall-clock 或显存开销对比,只统计了 NFE;在高分辨率/长视频 latent 上排序成本可能不可忽略。⚠️ 附录或有更多细节,以原文为准。
- **超参 $\rho_t$ 的选择依赖启发式**:$\rho_t$ 对搜索引导的对应关系是在「局部线性假设」下的启发式解释(附录 A.5),并非严格最优;不同任务/时间步如何调 $\rho_t$ 缺少系统性指导。
- **典型集近似是「约束近似」而非精确刻画**:$\mathcal{W}$ 用一族置信约束逼近典型集,理论上无法保证覆盖典型集的全部性质,残留的高阶结构化相关可能漏过(作者也承认典型集难闭式刻画)。
- **缺传统模块消融**:没有「去掉 2OS / 去掉多域 / 去掉 tile 统计各掉多少分」的定量表,只有 Figure 4 的定性递进,读者较难量化每个组件的边际贡献。
- **展望**:把置信区间投影推广到更多变换域或可学习的白化;把 $\rho_t$ 做成随时间步/奖励自适应;探索在离散扩散或更多模态上的噪声倾斜。

## 相关工作与启发
- **vs DPS(均值平移)**:DPS 把奖励梯度加到反向核均值上拿梯度引导,但标准化扰动 $\bm{\eta}_t$ 里多了确定性项、跌出噪声兼容区致画质下降;NTRK 均值不动、把梯度白化后注入噪声项,$\bm{\eta}_t$ 仍是合法高斯,区别在于「奖励走均值 vs 走噪声」。
- **vs SVDD(搜索)**:SVDD 每步采 $K$ 个候选选奖励最高的,保画质但无梯度、成本 $K$×;NTRK 用白化把梯度**直接构造**成那个「好噪声」,单次采样即得,把隐式噪声倾斜显式化。
- **vs DAS / Ψ-Sampler(混合/SMC)**:它们把梯度引导塞进 SMC 粒子框架(Ψ-Sampler 还重塑初始粒子分布),但仍然靠**移动均值**注入梯度,因此继承了均值平移的噪声兼容性问题;NTRK 从机制上避开了移均值。
- **vs FreeDoM**:FreeDoM 在均值平移基础上加额外 MCMC 采样提精度,计算更贵(实验里视频任务只能跑 13 步以匹配 NFE);NTRK 单步单采样,效率显著更高。
- **vs WGNC(噪声兼容投影)**:WGNC 首个把噪声兼容性 reframe 成傅里叶域硬等式范数投影,但硬约束会扭曲本就合法的噪声;NTRK 的 $\mathcal{W}$ 用置信区间投影升级它,「结构化才压、干净就放过」。
- **vs DNO / MPGR / StressDream(正则化)**:这些工作用软约束(范数/高阶谱统计)鼓励高斯典型性,但**无法保证**;NTRK 用投影给出单遍的、可证明的典型性 enforcement。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 「奖励走噪声不走均值」+「sort-clip-unsort 即欧氏投影」的组合是真正意义上补上了梯度引导×噪声兼容这块空白,视角与机制都新。
- 实验充分度: ⭐⭐⭐⭐ 覆盖美学/文图/视频/微调叠加多任务、target 与 held-out 双评,但缺白化算子各组件的定量消融和计算开销对比。
- 写作质量: ⭐⭐⭐⭐ 四种核用统一的「标准化扰动是否高斯」框架讲得非常清晰,Table 1 / Figure 3 一图胜千言;白化算子细节压到附录,正文略需回查。
- 价值: ⭐⭐⭐⭐⭐ drop-in、无训练、与微调正交、20× 提效,几乎可无痛叠到任何预训练/已对齐扩散模型上,实用价值很高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment](../../AAAI2026/image_generation/flowing_backwards_improving_normalizing_flows_via_reverse_representation_alignme.md)
- [\[CVPR 2026\] Resolving Endpoint Underfitting in Diffusion Bridges via Noise Alignment](../../CVPR2026/image_generation/resolving_endpoint_underfitting_in_diffusion_bridges_via_noise_alignment.md)
- [\[ICML 2026\] Pareto-Guided Optimal Transport for Multi-Reward Alignment](../../ICML2026/image_generation/pareto-guided_optimal_transport_for_multi-reward_alignment.md)
- [\[ICLR 2026\] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](../../ICLR2026/image_generation/glass_flows_reward_alignment_diffusion.md)
- [\[ECCV 2026\] SONIC: Spectral Optimization of Noise for Inpainting with Consistency](sonic_spectral_optimization_of_noise_for_inpainting_with_consistency.md)

</div>

<!-- RELATED:END -->
