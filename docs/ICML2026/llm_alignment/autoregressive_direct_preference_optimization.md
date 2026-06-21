---
title: >-
  [论文解读] Autoregressive Direct Preference Optimization
description: >-
  [ICML2026][LLM对齐][DPO] 作者发现 DPO 在推导目标函数时其实是"先按整条回答建 Bradley-Terry 偏好模型、事后才假设模型自回归",顺序反了;ADPO 把自回归假设**提前**到建 BT 模型之前——在输出空间的前缀闭包上定义能量函数,推出一个极简的新损失:把 DPO 里的求和符号从 log-sigmoid 内部**挪到外部**,并由此首次区分出"token 长度 $\mu$"与"反馈长度 $\mu'$"两个独立的长度度量,打通从整条回答到单 token 的任意粒度训练。
tags:
  - "ICML2026"
  - "LLM对齐"
  - "DPO"
  - "自回归"
  - "Bradley-Terry"
  - "前缀闭包"
  - "粒度"
---

# Autoregressive Direct Preference Optimization

**会议**: ICML2026  
**arXiv**: [2602.09533](https://arxiv.org/abs/2602.09533)  
**代码**: [项目页](https://stjohn2007.github.io/ADPO_project/)  
**领域**: 对齐RLHF / 偏好优化  
**关键词**: DPO, 自回归, Bradley-Terry, 前缀闭包, 粒度  

## 一句话总结
作者发现 DPO 在推导目标函数时其实是"先按整条回答建 Bradley-Terry 偏好模型、事后才假设模型自回归",顺序反了;ADPO 把自回归假设**提前**到建 BT 模型之前——在输出空间的前缀闭包上定义能量函数,推出一个极简的新损失:把 DPO 里的求和符号从 log-sigmoid 内部**挪到外部**,并由此首次区分出"token 长度 $\mu$"与"反馈长度 $\mu'$"两个独立的长度度量,打通从整条回答到单 token 的任意粒度训练。

## 研究背景与动机
**领域现状**:DPO 已成为对齐 LLM 的主流——它绕开显式奖励模型,直接用偏好对优化策略,既高效又可扩展,衍生出 SimPO、TDPO、TGDPO、cDPO 等一大批变体。

**现有痛点**:这些变体(包括号称 token 级的)几乎都**根本上依赖整条回答级别的 BT 模型**——BT 假设奖励 $r(x,y)$ 定义在完整回答 $y\in\mathcal{Y}$ 上。可 LLM 明明是自回归地一个 token 一个 token 生成的,这种"回答级建模"和"自回归生成"之间存在结构性错配。

**核心矛盾**:为什么大家都默认回答级?因为奖励模型通常学的是评价完整 $(x,y)$ 对,让人去评一个不完整前缀 $(x,y_{\le i})$ 不现实。但 DPO 的特殊之处是**它压根不需要显式奖励模型**——这就给了一个机会:可以引入一个更贴合自回归结构的隐式奖励函数,而不必受"人类只能评完整回答"的约束。

**本文目标**:能不能定义一组能量函数,让 DPO 推导里那个本该自回归的玻尔兹曼分布 $p_2$ **显式地**就是自回归分布,而不是事后才硬塞自回归假设?

**切入角度**:回看 DPO 推导会发现一个裂缝——可学习模型 $\pi_\theta$ 是自回归的,但式 (4) 里定义的 $p_2$ 并不是按自回归形式写出来的,自回归是"导完目标才假设"的。

**核心 idea**:把能量函数的定义域从输出空间 $\mathcal{Y}$ 扩到它的**前缀闭包 $\mathcal{Y}^*$**(所有不完整前缀的集合),在前缀上建 BT 模型,自回归假设就自然内生于推导之中,得到的损失把求和挪出 log-sigmoid。

## 方法详解

### 整体框架
ADPO 是对 DPO 推导的一次"重新奠基",不改架构、不加模块,核心是三步换地基:① 不在完整回答上、而在**前缀闭包 $\mathcal{Y}^*$** 上定义两个能量函数(前缀似然能量 $E_1^*$ 与前缀后验能量 $E_2^*$),并在 $E_2^*$ 里直接假设参考模型自回归;② 用前缀上的能量构造**前缀级 BT 模型**(对每个前缀长度连乘 BT 偏好概率);③ 最小化负对数似然,reparameterize 后得到 ADPO 损失。最终损失与 DPO 几乎一模一样,只差一个位置:DPO 是 $-\log\sigma\big(\beta\sum_i(\cdots)\big)$(求和在 log-sigmoid 里),ADPO 是 $-\sum_i\log\sigma\big(\beta(\cdots)\big)$(求和在 log-sigmoid 外)。在这之上,理论分析进一步揭示出两个独立的长度度量,从而把 DPO 与 token 级方法统一进一个"粒度家族"。这是一篇纯理论/损失推导的工作,没有 pipeline 可画,故不配框架图。

### 关键设计

**1. 在前缀闭包上定义能量与前缀级 BT 模型,推出"求和挪出 log-sigmoid"的损失**

DPO 的能量是 $E_1(x,y)=-r(x,y)$、$E_2(x,y)=-\frac1\beta r(x,y)-\log\pi_{\text{ref}}(y|x)$,都定义在完整回答上,所以对应的 $p_2$ 是整条回答的一个分布,自回归性是事后强加。ADPO 把定义域换成前缀闭包 $\mathcal{Y}^*=\bigcup_{y}\{y_{\le i}:0\le i\le T'\}$,定义前缀能量 $E_1^*(x,y_{\le i})=-r^*(x,y_{\le i})$、$E_2^*(x,y_{\le i})=-\frac1\beta r^*(x,y_{\le i})-\log\pi_{\text{ref}}(y_i|y_{<i},x)$——注意 $E_2^*$ 的参考项是**逐 token 条件概率** $\pi_{\text{ref}}(y_i|y_{<i},x)$,自回归被写进了能量本身。于是 $p_2$ 天然分解为自回归形式 $p_2(y|x)=\prod_i p_2(y_i|y_{<i},x)$,消除了与 $\pi_\theta$ 的错配。前缀级 BT 模型则把偏好概率写成对所有前缀长度的连乘 $p_1(y^w\succ y^l|x)=\prod_{i=1}^{T'}\frac{\exp(-E_1^*(x,y^w_{\le i}))}{\sum_{y_{\le i}\in Y_i}\exp(-E_1^*(x,y_{\le i}))}$。最小化 $-\log p_1$ 并 reparameterize $p_2=\pi_\theta$ 后,得到

$$\mathcal{L}_{\text{ADPO}}=-\mathbb{E}_{(x,Y)\sim\mathcal{D}}\Big[\sum_{i=1}^{T'}\log\sigma\big(\beta\log\tfrac{\pi_\theta(y^w_i|y^w_{<i},x)}{\pi_{\text{ref}}(y^w_i|y^w_{<i},x)}-\beta\log\tfrac{\pi_\theta(y^l_i|y^l_{<i},x)}{\pi_{\text{ref}}(y^l_i|y^l_{<i},x)}\big)\Big].$$

和 DPO 比,求和从 log-sigmoid **内部**移到了**外部**。直觉上,DPO 是"先把整条回答的对数比加总、再过一次 sigmoid 比胜负",ADPO 是"在每个前缀位置上各比一次胜负、再加总"——后者让偏好信号在每个 token 步都生效,粒度更细,且作者强调这没有违背 DPO 的理论根基,差异只来自能量函数定义域的不同。

**2. 自回归重参数化完备性(Theorem 1):任何奖励都能被自回归模型表示**

光有漂亮损失还不够,得证明这套前缀奖励不是凭空臆造、和经典奖励是相通的。作者先证 **Proposition 1(前缀级重参数化完备性)**:对任意前缀奖励 $r^*$,在其 reward-shift 等价类 $[r^*]$ 里存在唯一代表 $r^*_\circ$,使得 $r^*_\circ(x,y_{\le i})\equiv\beta\log\frac{\pi(y_i|y_{<i},x)}{\pi_{\text{ref}}(y_i|y_{<i},x)}$。再用 **Definition 3(可加分解)** 把普通奖励 $r(x,y)=\sum_i r^*(x,y_{\le i})$ 拆成前缀奖励之和,**Lemma 1** 保证每个奖励都有可加分解。两者一拼即得 **Theorem 1**:所有与前缀级 BT 模型一致的奖励类,都能写成 $r(x,y)=\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$,且 $\pi$ 是自回归模型。这比原 DPO 理论更进一步——DPO 只说存在某个奖励重参数化,ADPO 显式证明可用**自回归**模型实现,把 DPO 理论和自回归 LLM 范式真正对齐。

**3. 区分 token 长度 $\mu$ 与反馈长度 $\mu'$ 两个独立度量**

理论分析的副产品是一个反直觉但深刻的洞察。**Corollary 1**:当 $\mu'(y)\equiv1$ 时 ADPO 退化回 DPO。但作者强调这不是说"输出空间被限成单 token",而是说**原版 DPO 隐含了一个反馈长度度量 $\mu'$,它给每条序列都赋长度 1**——也就是 DPO 默认"把整条回答当一个不可分的反馈单元来评"。这个隐含约束正是过去 DPO 变体都把求和塞进 log-sigmoid 内部的根源。作者据此把两个长度度量分开:$\mu$ 是 LLM 分词带来的 **token 长度**,$\mu'$ 是评价场景带来的 **反馈长度**(把回答经评价度量 $\nu:\mathcal{Y}\to\mathbb{R}$ 映成一维、故 $\mu'=1$)。两者来源不同(一个来自 tokenization、一个来自前缀闭包),因此**理论上独立**:DPO 是 $\mu'\equiv1$(整条回答),token 级 ADPO 是 $\mu'=\mu$(逐 token),而独立地选这两个度量就能在任意粒度训练。

**4. 静态/自适应两族粒度,在 DPO 与 token 级之间自由插值**

把 $\mu'$ 解放出来后,ADPO 允许 $1\le\mu'(y)\le\mu(y)$ 的中间粒度:把序列 $y$ 切成若干子序列 $\{z_i\}$,每个前缀 $z_{\le i}$ 当一个可获隐式反馈的单元,损失写成 $\mathcal{L}_{\text{ADPO}}=-\mathbb{E}\big[\sum_i\log\sigma(\beta S^w_\theta(i)-\beta S^l_\theta(i))\big]$,其中 $S_\theta(i)$ 是子序列内逐 token 对数比的累加。切法由"强复合"$\xi$ 决定,给出两族:**静态族**固定窗口 $k$,每序列切成 $\lceil T/k\rceil$ 段($k=1$ 即 token 级,$k$ 越大越粗);**自适应族**固定段数 $m$、尽量均匀切($m=1$ 即 DPO,$m>1$ 自然加细)。这让 DPO 和各种 token 级方法成为同一连续谱上的两端。

### 损失函数 / 训练策略
ADPO 与 DPO 训练流程一致,只换损失,不增显式奖励模型,KL 约束下的最优解被保持(附录 B)。论文还把 ADPO 思想叠加到 cDPO(关键 token 加权)上得到 cADPO,验证这套地基可与现有变体正交组合。

## 实验关键数据

### 主实验
四个基座(Llama-3-8B / Gemma-3-12B / Qwen-3-8B / DeepSeek-Math-7B),两个数学推理基准(GSM8K / MATH):

| 方法 | Llama-3-8B GSM/MATH | Gemma-3-12B GSM/MATH | Qwen-3-8B GSM/MATH | DS-Math-7B GSM/MATH |
|------|------|------|------|------|
| DPO | 64.37 / 18.00 | 77.03 / 39.80 | 86.96 / 53.80 | 67.78 / 32.00 |
| **ADPO(本文)** | **68.08 / 21.00** | **78.32 / 41.20** | **88.10 / 55.40** | **69.98 / 33.40** |
| cDPO | 67.90 / 16.80 | 77.18 / 38.60 | 90.98 / 56.80 | 72.90 / 33.40 |
| **cADPO(本文)** | **68.76 / 20.20** | **78.85 / 40.40** | **91.74 / 57.20** | **73.54 / 35.40** |

ADPO 在全部 4×2=8 个设置上都超过对应的 DPO;把思想叠到 cDPO 上的 cADPO 也几乎全面优于 cDPO,说明前缀级地基与已有 token 加权技巧正交可叠加。

### 粒度家族消融(静态族)

| 方法 | $\mu'(y)$ | 复合 | 粒度 | Llama GSM/MATH | Gemma GSM/MATH | Qwen GSM |
|------|-----------|------|------|------|------|------|
| ADPO | $\mu(y)/8$ | $\xi_{\text{static}}(k{=}8)$ | 八 token | 64.97 / 18.20 | 76.88 / 40.00 | 87.79 |
| ADPO | $\mu(y)/4$ | $\xi_{\text{static}}(k{=}4)$ | 四 token | 66.26 / 17.40 | 77.41 / 40.80 | 88.48 |

不同窗口 $k$ 给出不同粒度,效果随基座/数据集而异,印证"粒度可调且影响学习行为"。

### 关键发现
- **求和挪位带来稳定提点**:仅把求和移出 log-sigmoid 这一个改动,在 8 个设置上稳定优于 DPO,且理论最优解不变。
- **与现有变体正交**:cADPO 普遍优于 cDPO,说明 ADPO 不是替代而是可作为底座叠加。
- **粒度可调是真自由度**:$\mu'$ 与 $\mu$ 独立,$k$/$m$ 给出 DPO 到 token 级之间的连续谱,为后续设计留出空间。

## 亮点与洞察
- **"求和在 log-sigmoid 内 vs 外"这一行字背后是地基之差**:本文最漂亮处是把一个看似工程式的损失微调,追溯到"自回归假设该在建 BT 模型之前还是之后"这个根本顺序问题,理论自洽且优雅。
- **揭示 DPO 的隐含约束 $\mu'\equiv1$**:点破"DPO 默认把整条回答当单一反馈单元"是限制此前所有变体把求和塞进 sigmoid 内的根源,这个 framing 很有解释力。
- **两个长度度量解耦可迁移**:token 长度与反馈长度独立这一抽象,给"任意粒度偏好优化"提供了统一语言,可指导设计新的粒度感知损失。

## 局限与展望
- 实验只在数学推理(GSM8K/MATH)上验证,对话、安全、长文本等对齐场景未测,泛化性证据有限。
- 静态/自适应族的最优粒度($k$、$m$)需逐任务调,论文未给选择粒度的原则性指引,实践成本待评估。
- 提升幅度多在 1–3 个点级别,虽稳定但不算大;"理论更对齐"的优雅性是否在更大规模、更难任务上转化为更大实益,尚需验证。
- 前缀级 BT 的隐式奖励是否真对应有意义的"过程级偏好",还是仅是数学上等价的重写,缺乏直接的过程质量分析。

## 相关工作与启发
- **vs DPO**:同一最优解、同样无需显式奖励模型,但 ADPO 把自回归假设提前到建 BT 之前,损失求和移出 log-sigmoid,且把 DPO 揭示为 $\mu'\equiv1$ 的特例。
- **vs TDPO / TGDPO / cDPO 等 token 级方法**:它们在 token 位置上加 KL 约束、奖励引导或关键 token 加权,但仍**根本依赖回答级 BT 模型**;ADPO 从建模地基上就改成前缀级 BT,因此能与 cDPO 正交叠加成 cADPO。
- **vs Rafailov et al. (2024) 的 token 级 soft-Q 解读**:那是对原回答级 BT 公式做事后的 token 级解释,**不改 BT 模型本身**;ADPO 在应用 BT 之前就把能量域扩到前缀闭包,得到一个**不同的**目标函数。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 DPO 推导的顺序裂缝补上,推出前缀级 BT 与两个独立长度度量,理论角度新颖。
- 实验充分度: ⭐⭐⭐ 4 基座 2 基准 + 粒度消融较扎实,但仅限数学推理、提升幅度有限。
- 写作质量: ⭐⭐⭐⭐⭐ 从裂缝观察到能量重定义再到定理与粒度家族,推导清晰、Table 1 对比一目了然。
- 价值: ⭐⭐⭐⭐ 给偏好优化提供更贴合自回归的理论地基与任意粒度训练框架,可作他法底座。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Boosting Direct Preference Optimization with Penalization](boosting_direct_preference_optimization_with_penalization.md)
- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](../../ICLR2026/llm_alignment/token-importance_guided_direct_preference_optimization.md)
- [\[ICLR 2026\] SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](../../ICLR2026/llm_alignment/safedpo_preference_optimization_safety.md)
- [\[ICLR 2026\] ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment](../../ICLR2026/llm_alignment/activedpo_active_direct_preference_optimization_for_sample-efficient_alignment.md)
- [\[ACL 2025\] DiffPO: Diffusion Alignment with Direct Preference Optimization](../../ACL2025/llm_alignment/diffpo_diffusion_alignment.md)

</div>

<!-- RELATED:END -->
