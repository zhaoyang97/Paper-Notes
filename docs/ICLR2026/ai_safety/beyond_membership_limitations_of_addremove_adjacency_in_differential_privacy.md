---
title: >-
  [论文解读] Beyond Membership: Limitations of Add/Remove Adjacency in Differential Privacy
description: >-
  [ICLR 2026][AI安全][差分隐私] 论文指出主流 DP 库默认的 **add/remove 邻接**只保护"成员是否在训练集里"，对"已知在训练集里、想推断其属性/标签"的攻击其实只能提供 **substitute 邻接**下弱得多的保护；作者设计了一套面向 substitute 邻接的 canary 审计工具，实证出经验隐私泄漏可以**突破 add/remove 报出的 $\varepsilon_{AR}$ 上界**，却紧贴 substitute 账本预测的 $\varepsilon_S$。
tags:
  - "ICLR 2026"
  - "AI安全"
  - "差分隐私"
  - "add/remove 邻接"
  - "substitute 邻接"
  - "属性推断"
  - "DP-SGD"
  - "隐私审计"
  - "canary"
---

# Beyond Membership: Limitations of Add/Remove Adjacency in Differential Privacy

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=C4jAhm8L1V](https://openreview.net/forum?id=C4jAhm8L1V)  
**代码**: 随论文补充材料提供（prv-accountant for substitute adjacency）  
**领域**: 差分隐私 / 隐私审计 / AI 安全  
**关键词**: 差分隐私, add/remove 邻接, substitute 邻接, 属性推断, DP-SGD, 隐私审计, canary  

## 一句话总结
论文指出主流 DP 库默认的 **add/remove 邻接**只保护"成员是否在训练集里"，对"已知在训练集里、想推断其属性/标签"的攻击其实只能提供 **substitute 邻接**下弱得多的保护；作者设计了一套面向 substitute 邻接的 canary 审计工具，实证出经验隐私泄漏可以**突破 add/remove 报出的 $\varepsilon_{AR}$ 上界**，却紧贴 substitute 账本预测的 $\varepsilon_S$。

## 研究背景与动机
**领域现状** —— 差分隐私（DP）把"对手区分两个相邻数据集的能力"约束成一个 $(\varepsilon,\delta)$ 上界，而"相邻"的定义（邻接关系）直接决定了保护语义。深度学习里事实标准是 DP-SGD + **add/remove 邻接**（$D'$ 由 $D$ 增删一条记录得到），Opacus、prv-accountant 等几乎所有主流库都默认它，其设计初衷是抵御**成员推断**（某人是否参与训练）。

**现有痛点** —— 很多真实场景保护的目标根本不是"是否在训练集里"，而是**已知在训练集里的某条记录的属性/标签**：监督微调里的标签隐私、用户敏感属性推断等。这类攻击的威胁模型对应的是 **substitute 邻接**（$D'$ 把 $D$ 里某条 $z$ 替换成 $z'$），而 add/remove 的上界对它只是"间接、宽松"的保护。

**核心矛盾** —— add/remove 的 $\varepsilon_{AR}$ 能推出一个 substitute 的 $\varepsilon_S$，但代价是参数大幅放宽（群体隐私定理给出 $\varepsilon_S = 2\varepsilon_{AR}$）。可实践者拿着库里报的 $\varepsilon_{AR}$ 数字，会**误以为属性隐私也有这么强**——这是一种系统性的"隐私高估"。但缺一套能把这个 gap 量化、审计出来的工具。

**本文目标** —— 证明并实测：在保护目标是 per-record 属性而非成员时，add/remove 账本会**高估保护强度**；给出 substitute 邻接下的紧致审计方法。

**核心 idea**：**「换记录而非增删记录」的最坏情况 canary** —— 不再用"加/删一条 canary"做审计，而是构造一对方向相反、范数都顶到裁剪界 $C$ 的 canary $(z, z')$ 做"替换"，让对手在 hidden-state 威胁模型下也能逼出 substitute 邻接的真实泄漏。

## 方法详解

### 整体框架
方法分两块：**(1) 审计协议**——把"challenger 训练、对手造 canary、对手区分"组织成一个可重复 $R$ 次的成员博弈（Algorithm 1），用 $\mu$-GDP 把区分成功率转成经验 $\varepsilon$ 下界；**(2) canary 构造**——按对手能力从强到弱给出 5 个场景 S1–S5（梯度空间 vs 输入空间 × 不同先验知识），每个场景配一套造 canary 的算法。两块拼起来回答"在 substitute 邻接下，DP-SGD 究竟泄漏多少"。

```mermaid
flowchart TD
    A["目标记录 z"] --> B{对手能改梯度?}
    B -->|能 · 梯度空间| C["S1 最坏数据集 canary<br/>S2 最坏梯度 canary (hidden-state)"]
    B -->|否 · 输入空间| D["S3 互补输入 / S4 误标 / S5 自然对抗样本"]
    C --> E["训练 R 次 DP-SGD<br/>随机用 z 或 z'"]
    D --> E
    E --> F["区分分数<br/>logit(z)-logit(z') 或 (gz/C)·(θT-θ0)"]
    F --> G["Clopper-Pearson 估 FPR/FNR<br/>→ μ-GDP → 经验 εS 下界"]
    G --> H{εS(审计) vs εAR(账本)?}
    H --> I["超过 εAR、紧贴 εS<br/>⇒ add/remove 高估属性保护"]
```

### 关键设计

**1. 替换式最坏情况 canary：把"增删"变成"对冲"。** add/remove 审计里 canary 在两数据集之一**存在/缺席**；substitute 审计则要让 canary 在两边**都在、但相反**。作者构造 $z$ 使其梯度 $\|g_z\|=C$ 全程顶到裁剪界，再把 $z$ 替换成 $z'$ 使 $\|g_{z'}\|=C$ 且**方向恰好相反**，其余样本贡献 0 梯度，从而把可区分性拉满。一个关键的诚实之处是：与 Nasr et al. (2021) 不同，作者**不假设无 canary 的步学习率为 0**，因此把"子采样把 canary 稀释掉"这件事如实算进了噪声，使审计反映 DP-SGD 真实动态。在子采样率 $q$ 下，$T$ 步内 canary 被采到 $k$ 次服从 $B\sim\text{Binomial}(T,q)$，累积梯度条件分布为 $\Pr(g_T\mid B=k)\sim\mathcal{N}(\pm kC,\,T\sigma^2C^2)$，对 $k$ 求和得到边缘分布

$$\Pr(g_T\mid D \text{ or } D') = \sum_{k=0}^{T}\binom{T}{k}q^k(1-q)^{T-k}\,\mathcal{N}\!\big(g_T;\pm kC,\,T\sigma^2C^2\big),$$

对手用 $\log\Pr(g_T\mid D)-\log\Pr(g_T\mid D')$ 当区分分数，逼出 substitute 邻接下最紧的经验 $\varepsilon_S$ 下界。

**2. Hidden-state 下的 canary 家族：从梯度空间退到输入空间。** 真实对手通常碰不到中间模型，只能看最终模型，且往往只能改输入而非梯度。作者据此把 canary 排成一个"能力递减"谱系：**S2 最坏梯度 canary**（Algorithm 2）——挑训练中幅度变化最小的那个参数维度 $j^\*=\arg\min_j S_j$（$S_j$ 累积 $|\theta^j_{t+1}-\theta^j_t|$），只在该维赋 $\pm C$、其余置 0，保证 $\|g_z\|=\|g_{z'}\|=C$ 且方向相反，用 $\theta_T-\theta_0$ 当分数（适配联邦学习审计）；**S3 互补输入 canary**（Algorithm 3）——用无 DP 参考模型，优化 $x'$ 使 $g_{z'}$ 与 $g_z$ 的余弦相似度最小、同时 MSE 约束让 $g_{z'}$ 与 $g_z$ 尺度相近，损失 $L_{\text{cosim}}+L_{\text{MSE}}$ 梯度下降而得；**S4 误标 canary**（Algorithm 4）——固定输入 $x$、在标签空间里挑使梯度最"对冲"的 $y'$；**S5 自然对抗 canary**（Algorithm 5）——从辅助集 $D_{aux}$ 里选与 $z$ 梯度余弦最小的真实样本。输入空间一律用 $\text{logit}(z;\theta_T)-\text{logit}(z';\theta_T)$ 做分数，对应数据投毒/标签推断这类真实威胁。

**3. 群体隐私换算只是宽松上界，应改用 substitute 账本。** 实践中常把 substitute 视作"一次 add + 一次 remove"的复合，由群体隐私（Dwork & Roth, 2014）得 Theorem 4.1：满足 $(\varepsilon_{AR},\delta_{AR},\sim_{AR})$-DP 的算法是 $(\varepsilon_S,\delta_S,\sim_S)$-DP，其中 $\varepsilon_S=2\varepsilon_{AR},\ \delta_S=(1+e^{\varepsilon_{AR}})\delta_{AR}$。但这个换算与算法无关、过于保守。论文主张：对可由隐私损失随机变量（PRV）/隐私损失分布（PLD）刻画的算法（如 Poisson 子采样 DP-SGD），直接用**数值账本**算 substitute 邻接的隐私曲线会紧得多。作者据此把 Microsoft 的 prv-accountant 适配到 substitute 邻接，得到的 $\varepsilon_S$(Accounting) 明显比 $\varepsilon_S$(Group Privacy) 紧——这条紧的曲线正是审计要去对齐的"真值"。论文还提醒：Theorem 4.1 是在按比例缩放 $\delta$ 的前提下成立，若固定 $\delta$，$\varepsilon_S$ 甚至可能超过 $\varepsilon_{AR}$。

**评分转换（$\mu$-GDP）：** 审计统一走 Gaussian DP。由 $R$ 次博弈的混淆矩阵，用 Clopper–Pearson（$\alpha=0.05$）估 FPR/FNR，再算 $\mu_{\text{lower}}=\Phi^{-1}(1-\text{FPR})-\Phi^{-1}(\text{FNR})$，最后用 Dong et al. (2019) 的 $\delta(\varepsilon)=\Phi(-\frac{\varepsilon}{\mu}+\frac{\mu}{2})-e^{\varepsilon}\Phi(-\frac{\varepsilon}{\mu}-\frac{\mu}{2})$ 把 $\mu$ 下界翻成 $\varepsilon$ 下界。

## 实验关键数据

### 主实验设置与结论

| 设置 | 模型 / 数据 | 关键观察 |
|------|-------------|----------|
| 最坏数据集 canary (S1) | 合成最坏情况, $\delta=10^{-5},C=1,T=500$, $R=25\text{K}$ | $\varepsilon$(审计) **超过** $\varepsilon_{AR}$，紧贴 $\varepsilon_S$(Accounting)；$\varepsilon_S$(Accounting) 比 $\varepsilon_S$(Group Privacy) 更紧 |
| 自然数据微调 | ViT-B-16(IN21K) 末层 + CIFAR10(500) | 大 $q$ 下 S2–S5 全部 **突破** $\varepsilon_{AR}$；梯度 canary (S2) 最紧 |
| 文本微调 | Sentence-BERT 线性头 + SST-2(5K) | 梯度 canary 审计紧；输入 canary 也能超过 $\varepsilon_{AR}$ |
| 从零训练 | 3 层 MLP + Purchase100(50K), DP-Adam | 输入 canary 审计弱、**不**超 $\varepsilon_{AR}$；但梯度 canary 仍紧贴 $\varepsilon_S$ |

### 消融 / 敏感性

| 因素 | 影响 |
|------|------|
| 子采样率 $q$（1.0 / 0.25 / 0.0625） | $q$ 越小 canary 越被"稀释"，审计越松；输入空间 canary 受影响最大 |
| 裁剪界 $C$ / 训练步数 $T$ / 学习率 $\eta$（附录 A2–A4） | 输入空间 canary 对这三者敏感，后期训练步审计效力下降 |
| canary 对模型效用 | 几乎无影响（附录 A1），不损 utility |

### 关键发现
- **梯度 canary 是"金标准"**：无论微调还是从零训练，$\varepsilon$(审计) 都紧贴 substitute 账本 $\varepsilon_S$，稳定突破 add/remove 上界。
- **微调比从零训练更危险**：微调模型对输入空间 canary 更脆弱——只需一次投毒微调数据，就能让泄漏超过 $\varepsilon_{AR}$，尤其在常用的大 $q$ 区间。
- **从零训练更"皮实"**：非凸优化 + DP-Adam + 子采样让输入 canary 失效，此时 add/remove 确实够用——说明风险高度依赖训练范式。

## 亮点与洞察
- **概念层面的"皇帝新衣"**：揭示一个被广泛默认却被忽视的语义错配——库里报 $\varepsilon_{AR}$，但用户真正想要的属性/标签隐私归 substitute 邻接管，二者差一大截。
- **审计而非只证明**：不停留在"add/remove ⇒ substitute 上界更松"的理论事实，而是造出能**实证逼破** $\varepsilon_{AR}$ 的攻击，把抽象 gap 变成可测的经验泄漏。
- **威胁模型谱系完整**：S1–S5 覆盖从"能改梯度的最强对手"到"只能投一条自然样本的现实对手"，并标注各场景所需先验（Table 2），实用性强。
- **诚实的审计建模**：明确不沿用"无 canary 步学习率为 0"的简化，把子采样稀释如实计入，使结论更可信。

## 局限与展望
- **局限性**：实验集中在小规模微调（末层/线性头）与中小数据集（CIFAR10 500 样本、SST-2 5K、Purchase100），未覆盖大模型全量微调或 LLM SFT 的真实规模；输入空间 canary 在从零训练 + DP-Adam 下失效，说明结论对范式敏感、不能无条件外推。
- **可改进方向**：把 substitute 审计推广到更现实的黑盒/查询访问；研究在大 $q$ 下兼顾 utility 又能堵住属性泄漏的会计或机制；将 substitute 账本作为默认选项集成进主流 DP 库。
- **展望**：呼吁实践者在"保护目标是属性/标签"时直接报告 substitute 邻接的 $\varepsilon_S$，而非沿用 add/remove，避免系统性高估隐私。

## 相关工作与启发
- **DP 审计谱系**：Jayaraman & Evans (2019) 首揭经验泄漏与理论界的 gap，Nasr et al. (2021/2023)、Steinke et al. (2023)、Cebere et al. (2025) 推进最坏 canary 与现实威胁模型——本文把这条线从 add/remove 迁到 substitute，并借用 Cebere 的最少更新参数思路造梯度 canary。
- **邻接关系与群体隐私**：Kairouz et al. (2021) 的 zero-out、Dwork & Roth (2014) 的群体隐私定理是本文 Theorem 4.1 的来源，论文指出其作为换算工具过松、应让位于 PRV/PLD 数值账本（Gopi et al., 2021）。
- **$\mu$-GDP 审计**：Dong et al. (2019) 的 Gaussian DP 与 Nasr et al. (2023) 的 $\mu$-GDP 审计法是经验 $\varepsilon$ 估计的工具基座。
- **启发**：邻接关系的选择不是纯理论细节，而是直接影响"报出来的隐私数字是否对应你真正想防的攻击"——做隐私系统时应先对齐"保护目标 ↔ 邻接定义 ↔ 账本"。

## 评分
- **新颖性**: ⭐⭐⭐⭐ —— 不是新机制，而是点破并量化了一个被默认忽视的语义错配，配套 5 场景 substitute canary 审计工具，视角新颖。
- **实验充分度**: ⭐⭐⭐⭐ —— 覆盖图像/文本/表格三模态、微调与从零训练，含 $q/C/T/\eta$ 敏感性与 utility 检验；但规模偏小、未触及大模型。
- **写作质量**: ⭐⭐⭐⭐ —— 定义、定理、算法、图表层次清晰，威胁模型先验表（Table 2）很贴心。
- **价值**: ⭐⭐⭐⭐ —— 对所有用 DP 库做属性/标签隐私的实践者是直接、可操作的警示，可能推动库默认报告 substitute 邻接。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Privacy Beyond Pixels: Latent Anonymization for Privacy-Preserving Video Understanding](privacy_beyond_pixels_latent_anonymization_for_privacy-preserving_video_understa.md)
- [\[ICLR 2026\] Membership Privacy Risks of Sharpness Aware Minimization](sam_membership_privacy_risks.md)
- [\[ICLR 2026\] INO-SGD: Addressing Utility Imbalance under Individualized Differential Privacy](ino-sgd_addressing_utility_imbalance_under_individualized_differential_privacy.md)
- [\[ICLR 2026\] Exponential-Wrapped Mechanisms: Differential Privacy on Hadamard Manifolds Made Practical](exponential-wrapped_mechanisms_differential_privacy_on_hadamard_manifolds_made_p.md)
- [\[ICLR 2026\] Optimizing Canaries for Privacy Auditing with Metagradient Descent](optimizing_canaries_for_privacy_auditing_with_metagradient_descent.md)

</div>

<!-- RELATED:END -->
