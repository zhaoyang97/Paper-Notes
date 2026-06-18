---
title: >-
  [论文解读] Bias at the End of the Score
description: >-
  [CVPR 2026][奖励模型] 本文对文本到图像（T2I）系统中广泛使用的五个奖励模型（PickScore、ImageReward、HPS、VQAScore、CLIP）做了一次大规模偏置审计，证明这些被当作"图像质量"代理的打分函数其实编码了系统性的人口统计偏置——在用作噪声优化器时会不成比例地对女性主体超性化、把非白人主体"洗"成白人，并且打分本身与现实世界的人口分布（如各职业的性别比例）高度相关，而非真正衡量质量。
tags:
  - "CVPR 2026"
  - "奖励模型"
  - "人口统计偏置"
  - "T2I"
  - "反事实评估"
  - "超性化"
---

# Bias at the End of the Score

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Magid_Bias_at_the_End_of_the_Score_CVPR_2026_paper.html)  
**代码**: 无  
**领域**: AI安全 / 公平性 / 文本生成图像  
**关键词**: 奖励模型, 人口统计偏置, T2I, 反事实评估, 超性化

## 一句话总结
本文对文本到图像（T2I）系统中广泛使用的五个奖励模型（PickScore、ImageReward、HPS、VQAScore、CLIP）做了一次大规模偏置审计，证明这些被当作"图像质量"代理的打分函数其实编码了系统性的人口统计偏置——在用作噪声优化器时会不成比例地对女性主体超性化、把非白人主体"洗"成白人，并且打分本身与现实世界的人口分布（如各职业的性别比例）高度相关，而非真正衡量质量。

## 研究背景与动机
**领域现状**：奖励模型（Reward Model, RM）是 T2I 流水线的核心部件，被用在多个环节——做数据集过滤、做评测指标、做微调时的监督信号、做生成后的安全/质量筛选。它们把"对齐度、保真度、美学、人类偏好"这些复杂标准蒸馏成一个标量分数 $s_{I,p}=R(I,p)$，社区默认"分数高 = 图更好"。

**现有痛点**：人们已经研究过 RM 集成进 T2I 时的一些已知失败模式——reward hacking（高分但忽略 prompt）、mode collapse（不同初始噪声塌缩成同一张高分图）、catastrophic forgetting。但 RM 作为"打分函数"本身的鲁棒性和公平性，几乎没人系统检验过；尤其是基于人类偏好训练的 PickScore/ImageReward/HPS 这一类，从未经过像 T2I 模型那样的公平性/安全性专项测试。

**核心矛盾**：RM 的训练数据本身就带偏——生成模型与 prompt 数据集的分布偏差、人类标注者的偏好、模型架构与训练过程的归纳偏置，都会注入 RM。一旦合成图像被大规模用于下游，RM 的偏置会被指数级放大。问题的根本在于：**RM 名义上是"图像质量"的中性度量，实际上学到的是"对主流人口特征的服从度"**，但这种偏离一直被当作质量提升来使用。

**本文目标**：从两个层面量化 RM 偏离"质量度量"这一隐含规格的程度——(1) 当 RM 被当作优化器时，它会怎样系统性地改写图像里未被 prompt 指定的人口属性；(2) 在打分层面，RM 的分数能否被种族/性别预测，这种打分差异是否镜像现实人口分布。

**核心 idea**：把审计拆成"优化探针 + 反事实打分"两部分——先用 ReNO 框架让 RM 去优化噪声、观测它对人口属性造成的实际扭曲（Part I），再用反事实数据集 + 回归/排序分析揭示扭曲背后的打分级机制（Part II），从而把"行为偏置"和"打分偏置"因果地串起来。

## 方法详解

### 整体框架
这是一篇**机制分析 / 审计型**论文，没有提出新的生成或训练方法，核心是一套两段式的实证审计设计。形式化地，一个奖励模型是函数 $R$，给图像-prompt 对赋一个标量 $s_{I,p}=R(I,p)$；被评测的有五个主流 RM（PickScore、ImageReward、HPS、VQAScore、CLIP）外加 Aesthetic score，以及一个不依赖任何数据分布的 **Incompression** 基线（最大化 DCT 高频系数）作中性参照。

整个审计分两部分，第一部分看"RM 当优化器时干了什么"，第二部分挖"打分层面为什么会这样"：

- **Part I（优化探针）**：在冻结生成器参数的前提下，用 RM 的梯度去优化初始噪声向量，观测被优化图像在两类人口维度上的系统性扭曲——超性化（hypersexualization）和人口收敛（demographic convergence）。
- **Part II（反事实打分）**：直接评测 RM 对"只在人口属性上有差异"的反事实图像组的打分，用线性回归 + 排序分析定位打分差异，并与现实劳工统计相关性对照。

（纯实证分析，无多模块串行 pipeline，故不画框架图。）

### 关键设计

**1. 优化探针：把 RM 从"评估器"变成"优化器"来暴露其梯度偏置**

要看 RM 的偏置如何影响生成内容，最直接的方式是让它去"驱动"生成。本文采用 ReNO 框架：给定冻结的一步式生成模型 $G_\theta(\varepsilon, p)$ 和奖励函数 $R$，优化目标是只动初始噪声 $\varepsilon$ 而不动模型参数，

$$\varepsilon^{\star}=\arg\max_{\varepsilon} R\big(G_\theta(\varepsilon, p), p\big),$$

用迭代梯度上升求解：

$$\varepsilon_{t+1}=\varepsilon_{t}+\eta\,\nabla_{\varepsilon_t}\Big[K(\varepsilon_t)+\lambda\,R\big(G_\theta(\varepsilon_t, p), p\big)\Big],$$

其中 $\eta$ 是学习率，$K$ 是正则项，$\lambda$ 控制奖励优化的方向与幅度（$\lambda=+1$ 为最大化奖励，$\lambda=-1$ 为最小化）。基座用 SDXL-Turbo、PixArt-α DMD、SD-Turbo 三种。这个设计的巧妙在于：当 prompt 没指定人口信息时，理想的 RM 应当是人口中性的、只改善质量；任何在优化轨迹中出现的、与人口相关的方向性漂移，都直接暴露了 RM 梯度里隐藏的偏置——它把"评估偏置"放大成了"可观测的生成扭曲"。

**2. 超性化度量：用 NSFW 二值 + 皮肤暴露比双信号刻画"被优化出来的色情化"**

光看 NSFW 分类不够，很多色情化不会触发二值分类器。本文用两个互补信号：(1) 用预训练 NSFW 分类器把图分成 neutral/low/medium/high 四类，再坍缩成二值指标——$\mathrm{nsfw}(x)=0$ 当 $\arg\max_c p(c\mid x)=\text{neutral}$，否则为 1；(2) 皮肤暴露比（Skin Exposure, SE）——身体皮肤像素占可见人体总区域（身体皮肤+脸部皮肤+衣物+头发）的比例，能捕捉不触发二值分类器的更细微色情化。对每次实验度量优化前后的变化：

$$\Delta_{\text{nsfw}}=\mathrm{nsfw}(x^{\star})-\mathrm{nsfw}(x_0),\qquad \Delta_{\text{skin}}=\mathrm{skin}(x^{\star})-\mathrm{skin}(x_0).$$

按性别分层后能直接读出"超性化是否性别不均"——结论是显著不均，女性主体被不成比例地超性化。

**3. 人口收敛追踪：SeedSelect 初始化特定人群 + CFD anchor 分类 + 双向 λ 测方向性**

要看优化是否会改写主体的种族/性别，得先稳定地造出"某个特定人群"的初始图。直接采样很低效（基座对欠指定 prompt 难以生成多样人群），本文借鉴 SeedSelect：用过度指定 prompt（如"a photo of an Asian female doctor"）的少量参考图，在噪声输入空间做梯度搜索，找到只用"a photo of a doctor"也能生成该人群的噪声 $\varepsilon_0$。然后从这个初始化跑 ReNO，追踪优化前后人口属性的迁移。为判定感知到的种族/性别，采用 **anchor-based** 方法降低伤害——用 Chicago Face Database（CFD，含真人自报种族/性别）为八个种族-性别组各构造一个锚点嵌入：

$$a_{rg}=\frac{\bar e_{rg}}{\lVert \bar e_{rg}\rVert_2},\qquad \bar e_{rg}=\frac{1}{|D_{rg}|}\sum_{x\in D_{rg}} f(x),$$

$f$ 是冻结的 CLIP ViT-L/14 图像编码器；新图取与各锚点余弦相似度最高者作分类。再配合 $\lambda=\pm1$ 双向优化探测偏置的方向性——正向优化把非白人推向白人，负向优化则反过来保住非白人身份，证明这是结构性的人口先验而非随机噪声。

**4. 打分级机制：反事实数据集 + OLS 回归 + 排序分析，把"行为"归因到"打分"**

Part I 看到的是行为扭曲，要解释根因得回到打分本身。本文用三个反事实数据集（CausalFace、SocialCounterfactuals、PAIRS），每组图像只在受保护人口属性上变化、其余语义与上下文固定，搭配四类语义不同的 prompt（SCM 暖度-能力、ABC agency/belief、DALL-Eval Descriptor、Occupation 职业集）。两种互补分析：(1) **线性回归**——对每个模型/数据集/prompt 拟合

$$s^{R}_{I,p}\approx\beta_0+\beta_1\rho_I+\beta_2\gamma_I+\beta_3(\rho_I\times\gamma_I)+\epsilon_I,$$

$\rho$ 为种族、$\gamma$ 为性别，所有分数在各数据集/模型内标准化为零均值单位方差以可比；用 $p<0.05$ 找出 RM 对"仅人口属性不同"的图给出系统性不同分数的系数，再比效应量 $\beta_x$ 定位偏置最大处。(2) **排序分析**——对每个 RM/数据集/prompt 类别，先在反事实组内按分数排名，再跨性别、跨反事实集平均得到某种族组的平均排名 $r_a$（$r_a=1$ 为最高分组），即便原始分差很小也能捕捉相对偏好顺序。关键发现是回归效应与现实世界相关：职业 prompt 产生最强的人口效应，且性别效应量与美国劳工统计局的各职业女性就业占比相关，说明 RM 打的不只是质量、更是"对主流人口分布的服从度"。

## 实验关键数据

### Part I：优化诱导的扭曲

| 现象 | 关键数字 | 说明 |
|------|---------|------|
| 超性化（NSFW 率增幅，PickScore） | 女性 +19% vs 男性 +7%（约 2.7×） | 跨所有基座，PickScore 增幅最高；女性被不成比例影响 |
| 皮肤暴露增幅（PickScore） | 女性约为男性 2.3× | SE 同样女性更高 |
| 最大单组合 NSFW 增幅 | 25%（PickScore + PixArt-α DMD） | 基座×RM 组合里最严重 |
| 非白人→白人迁移率（正向优化） | ImageReward 76.1% / HPS 89.2% / CLIP 36.2% | 初始非白人图被优化后大量被分类为白人；初始白人则几乎保持白人 |
| 性别翻转（正向优化） | ImageReward >39%、CLIP >26% 的女性图被重分类为男性 | 优化把女性推向男性 |

### Part II：打分级偏置

| 分析 | 关键发现 | 说明 |
|------|---------|------|
| 回归（职业 prompt） | top30 最大性别分差里平均 27.4 个、race×gender 里 23.4 个为职业 prompt | 职业语境触发最强人口效应，CausalFace 上尤甚 |
| 与劳工统计相关 | RM 给男性主导职业的男性脸打分更高、女性主导职业的女性脸更高 | 打分镜像现实就业性别比；但仍有大量职业即使女性主导也给男性脸更高，反向不成立 |
| 排序（种族） | HPS、ImageReward 把白人排最高（白男>白女）、亚裔最低 | 即使负面 prompt（"dishonest"）也保持，证明是结构性人口先验 |
| VQAScore 例外 | 正面 prompt 白人最高，负面 prompt 白人显著靠后 | 跨所有反事实数据集一致 |

### 关键发现
- **打分偏置 → 优化漂移的因果链清晰**：因为白人主体无论 prompt 正负都拿更高分，梯度上升就持续把噪声推向"白人输出"的隐空间区域，这直接解释了 Part I 的种族收敛。
- **PickScore 在超性化上最糟，ImageReward/HPS 在种族收敛上最强**：不同 RM 的偏置维度不同，没有一个是"安全"的。
- **SocialCounterfactuals 上人口效应被削弱**：更丰富的上下文线索（制服、背景）能部分抑制人口打分影响，提示数据构造方式会影响偏置显现。
- **Incompression 基线确认**：用不依赖数据分布的 DCT 高频基线做参照，说明上述漂移来自 RM 的数据偏置，而非生成器优化本身的副作用。

## 亮点与洞察
- **"把评估器当优化器"是个漂亮的放大镜**：评估偏置平时藏在标量分数里看不见，一旦用它的梯度去优化噪声，偏置就被放大成肉眼可见、可量化的生成扭曲——这个探针思路可迁移到任何"标量奖励 + 可微生成器"的审计场景（包括 LLM 的 RLHF 奖励模型）。
- **行为 + 机制双段闭环**：Part I 给"它做了坏事"的证据，Part II 给"为什么"的打分级解释，并用同一批 RM 把两者因果对上，比单纯报告"有偏置"更有说服力。
- **anchor-based 人口分类降低伤害**：不直接训练种族分类器，而是用真人自报标签（CFD）构造 CLIP 锚点嵌入做最近邻，这种做法在做敏感属性测量时值得借鉴。
- **"质量度量其实是服从度度量"**：把 RM 分数与劳工统计相关性摆出来，把"RM 编码现实人口频率先验"这个抽象论点变成了可证伪的实证，是全文最"啊哈"的地方。

## 局限性 / 可改进方向
- **"反事实"并不严格**：作者自己承认这些图并非真正的反事实，尤其种族无法定义严格反事实，图像里常夹带非人口变化（耳环、胡须、背景差异），可能混入打分。
- **感知属性二元/离散化**：用"female image subject"等离散类别指代感知到的人口信息，简化了复杂身份，可能漏掉真实经历的细微差别（作者在脚注里强调过）。
- **只用了 ReNO 一种优化框架**：Part I 的结论依赖 ReNO，换别的噪声优化/选择技术结论是否一致留待未来工作。⚠️ 这意味着"优化即扭曲"的强度可能与具体优化器耦合。
- **OOD 风险**：部分 prompt（SCM/ABC）可能落在某些 RM 训练分布之外，处于潜在 OOD 区，虽然作者强调跨模型/数据集结论一致，但绝对效应量需谨慎解读。
- **改进方向**：把审计延伸到更多优化器与生成后筛选（best-of-N）、设计去偏的 RM 训练/数据采集流程、把"服从主流分布"这一更普遍的多样性塌缩问题纳入更广义的鲁棒性度量。

## 相关工作与启发
- **vs Concept2Concept**：他们审查 Pick-a-Pic 数据集的概念关联（发现含 CSAM），关注的是训练数据内容；本文直接审计 RM 作为打分/优化函数的行为偏置，从"数据有什么"推进到"模型打分会怎样扭曲生成"。
- **vs 分析 HPS/PickScore 数据集的工作**：那项工作发现数据集有非中性的性别/种族偏好，但主要看偏好对齐微调如何影响视频生成的偏置；本文聚焦这些 RM 原本被训练的图像生成任务，并把偏置定位到打分级机制。
- **vs 已知 RM 失败模式（reward hacking / mode collapse / catastrophic forgetting）**：以往关注"忽略 prompt""多样性塌缩""能力遗忘"等通用失效；本文专门切入人口统计公平性这一高危害维度，并指出它可能是更广义鲁棒性问题的一个窗口。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次对 T2I 奖励模型做系统的人口偏置审计，"评估器当优化器"探针 + 打分级回归的组合设计有新意。
- 实验充分度: ⭐⭐⭐⭐⭐ 五个 RM × 三个基座 × 三个反事实数据集 × 四类 prompt，行为与机制双段闭环，证据扎实。
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰、因果链讲得明白，公式与度量定义齐全；个别图表依赖正文交代。
- 价值: ⭐⭐⭐⭐⭐ 直接质疑了被广泛默认可靠的 RM，对 T2I 数据筛选/微调/评测全链路都有警示意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] End-to-End Hyper-Relational Information Extraction for Engineering Diagrams via Dynamically Tokenized Relation Transformer](end-to-end_hyper-relational_information_extraction_for_engineering_diagrams_via_.md)
- [\[CVPR 2026\] Bias In, Bias Out? Finding Unbiased Subnetworks in Vanilla Models](bias_in_bias_out_finding_unbiased_subnetworks_in_vanilla_models.md)
- [\[ICML 2026\] CyberGym-E2E: Scalable Real-World Benchmark for AI Agents' End-to-End Cybersecurity Capabilities](../../ICML2026/others/cybergym-e2e_scalable_real-world_benchmark_for_ai_agents_end-to-end_cybersecurit.md)
- [\[CVPR 2026\] Rank-Guided Pseudo-Bias Learning for Robust Black-Box Adaptation](rank-guided_pseudo-bias_learning_for_robust_black-box_adaptation.md)
- [\[ACL 2025\] Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey](../../ACL2025/others/behavioural_vs_representational_systematicity_in_end-to-end_models_an_opinionate.md)

</div>

<!-- RELATED:END -->
