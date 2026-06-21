---
title: >-
  [论文解读] Convergent Differential Privacy Analysis for General Federated Learning
description: >-
  [ICLR2026][AI安全][差分隐私] 本文用 f-DP 框架 + shifted interpolation 技术，首次证明了非凸光滑目标下两类经典联邦学习方法（Noisy-FedAvg / Noisy-FedProx）的"最坏隐私"在通信轮数 $T\to\infty$ 时**收敛到常数下界**而非发散，从理论上推翻了"FL-DP 长期训练必然耗尽隐私预算"的旧认知。
tags:
  - "ICLR2026"
  - "AI安全"
  - "差分隐私"
  - "联邦学习"
  - "f-DP"
  - "收敛隐私"
  - "shifted interpolation"
---

# Convergent Differential Privacy Analysis for General Federated Learning

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=7Zbe5ad3eX](https://openreview.net/forum?id=7Zbe5ad3eX)  
**代码**: 待确认  
**领域**: 联邦学习 / 差分隐私 / 学习理论  
**关键词**: 差分隐私, 联邦学习, f-DP, 收敛隐私, shifted interpolation

## 一句话总结
本文用 f-DP 框架 + shifted interpolation 技术，首次证明了非凸光滑目标下两类经典联邦学习方法（Noisy-FedAvg / Noisy-FedProx）的"最坏隐私"在通信轮数 $T\to\infty$ 时**收敛到常数下界**而非发散，从理论上推翻了"FL-DP 长期训练必然耗尽隐私预算"的旧认知。

## 研究背景与动机
**领域现状**：联邦学习（FL）让多个客户端在不上传原始数据的前提下协同训练，但模型/梯度仍可能被反演攻击或成员推断攻击间接泄露隐私。把差分隐私（DP）嵌进 FL、在上传参数前加高斯噪声，就得到 FL-DP 这一主流隐私保护范式。

**现有痛点**：几乎所有 FL-DP 的隐私分析都建立在**组合定理（composition theorem）**和"逐次迭代的隐私放大（privacy amplification by iteration）"之上。组合定理的本质是隐私预算线性累加——把每一轮 $(\epsilon_t,\delta_t)$ 直接相加，于是 $(\epsilon,\delta)$-DP 要求噪声方差 $\sigma^2$ 必须正比于 $T$（甚至 $TK$）。这导致一个反直觉的结论：当通信轮数 $T\to\infty$ 时隐私界**任意发散**，意味着 FL-DP 在长期训练下会彻底失去隐私保护能力。

**核心矛盾**：这个理论结论和实验现象严重不符——实践中用**常数级噪声**做很多轮训练，模型仍然有可观的隐私保护。问题根源在于组合分析过于宽松：它把每一轮当成独立的隐私损失叠加，完全没利用"局部—全局优化动态会让敏感度自然收缩"这件事。在凸/强凸的噪声梯度下降里，已有工作（Chourasia 2021、Altschuler & Talwar 2022 等）用 RDP 证出了**收敛隐私**，但 FL 的多步局部更新 + 异构数据导致局部模型有偏，分析技术一直没能迁移过来。

**本文目标**：(1) 为非凸光滑目标下的 FL-DP 建立**紧（tight）且收敛**的最坏隐私界；(2) 搞清 $K$（局部步数）、$m$（客户端规模）、$V$（裁剪范数）、$\alpha$（FedProx 正则系数）等关键超参对隐私的真实影响。

**切入角度**：作者放弃组合定理，改用 Dong 等 2022 提出的 **f-DP**——一种基于假设检验 Type I/II 错误权衡曲线的、信息论无损的 DP 定义；再结合 Bok 等 2024 的 **shifted interpolation** 技术，把"逐轮放大 $T$ 次"换成"沿一条插值路径只放大 $T-t_0$ 次"，从而把发散的界压成收敛的界。

**核心 idea**：用 f-DP 的权衡函数 + 一条可优化的插值序列，把全局敏感度拆成"数据敏感度 + 模型敏感度"并用插值系数 $\lambda_t$ 逐项缩放，证明加权敏感度和 $H_0$ 收敛到常数，于是隐私下界收敛。

## 方法详解

### 整体框架
这是一篇纯理论分析论文，没有提出新算法，而是给两类既有方法做更紧的隐私证明。被分析的两类方法本身很标准：

- **Noisy-FedAvg**：每个客户端做 $K$ 步裁剪梯度下降 $w_{i,k+1,t}=w_{i,k,t}-\eta_{k,t}g_{i,k,t}$（$g$ 是被裁剪到范数 $V$ 的梯度），上传前加各向同性高斯噪声 $n_i\sim\mathcal{N}(0,\sigma^2 I_d)$，服务器聚合得到全局模型。
- **Noisy-FedProx**：局部目标多一个近端正则项 $\frac{\alpha}{2}\|w-w_t\|^2$，迭代形式为 $w_{i,k+1,t}=w_{i,k,t}-\eta_{k,t}[g_{i,k,t}+\alpha(w_{i,k,t}-w_t)]$。

分析针对的是**全局模型隐私**：客户端上传时的本地隐私由加噪直接保证，难点在于服务器把聚合参数发回客户端后、攻击者能否从全局模型反推某个样本是否参与了训练。整条证明链路是：把相邻数据集 $C,C'$ 上的全局更新写成 $w_{t+1}=\phi(w_t)+n_t$（$\phi$ 是 $K$ 步局部更新的累积，$n_t\sim\mathcal{N}(0,\sigma^2 I_d/m)$ 是平均噪声）→ 构造 shifted interpolation 序列把权衡函数 $T(w_T;w_T')$ 下界化为加权全局敏感度之和（Theorem 1）→ 把全局敏感度拆成数据/模型两部分并给出递推系数（Theorem 2）→ 求解关于 $\lambda_t,t_0$ 的最小化问题、用 $t_0=0$ 松弛得到可解形式（Sec 4.3）→ 解出四种学习率策略下的收敛界（Theorem 3、4）。

### 关键设计

**1. 用 f-DP 权衡函数替代组合定理：从"宽松相加"到"无损刻画"**

旧分析用 $(\epsilon,\delta)$-DP 或 RDP，组合时隐私预算直接线性相加（$(\epsilon_1,\delta_1)+(\epsilon_2,\delta_2)\to(\epsilon_1+\epsilon_2,\delta_1+\delta_2)$），这是发散的根源。本文改用 f-DP：把"攻击者能否区分相邻数据集 $C$（$H_0$）与 $C'$（$H_1$）"建模为假设检验，用 Type I 错误 $E_I=\mathbb{E}_{M(C)}[\chi]$ 和 Type II 错误 $E_{II}=1-\mathbb{E}_{M(C')}[\chi]$ 的最优权衡曲线刻画隐私。权衡函数定义为 $T(P;Q)(\gamma)=\inf\{1-\mathbb{E}_Q[\chi]\mid \mathbb{E}_P[\chi]\le\gamma\}$，它凸、连续、非增。当两个分布是高斯时退化为 Gaussian-DP：$T_G(\mu)(\gamma)=\Phi(\Phi^{-1}(1-\gamma)-\mu)$，$\mu$ 越大隐私越弱。f-DP 是信息论无损的，避免了组合分析里那种"为了能相加而过度放松"的损失，且结论可无损转换回 $(\epsilon,\delta)$-DP 和 RDP（Table 3）。

**2. Shifted interpolation：把"放大 $T$ 次"压成"放大 $T-t_0$ 次"**

传统隐私放大要在每一轮都基于 $w$ 与 $w'$ 的关系放大一次，累积 $T$ 次必然发散。作者借鉴 Bok 等 2024，构造一条插值序列把放大次数从 $T$ 降到 $T-t_0$：

$$\tilde{w}_{t+1}=\lambda_{t+1}\phi(w_t)+(1-\lambda_{t+1})\phi'(\tilde{w}_t)+n_t,\quad t=t_0,\dots,T-1$$

设 $\lambda_T=1$ 使 $\tilde{w}_T=w_T$，并以 $\tilde{w}_{t_0}=w'_{t_0}$ 作为插值起点，$0\le\lambda_t\le1$ 是待优化系数。Theorem 1 据此把权衡函数下界化为：

$$T(w_T;w_T')\ge T_G\!\left(\frac{\sqrt{m}}{\sigma}\sqrt{\textstyle\sum_{t=t_0}^{T-1}\lambda_{t+1}^2\|\phi(w_t)-\phi'(\tilde{w}_t)\|^2}\right)$$

核心是：发散的全局敏感度 $\|\phi(w_t)-\phi'(\tilde{w}_t)\|$ 可以被插值系数 $\lambda_t\le1$ 逐项**缩放压制**，只要 $\lambda_t$ 选得好，加权和就能收敛——这就是收敛隐私的来源。

**3. 全局敏感度二分解：数据敏感度 + 模型敏感度**

全局敏感度 $\|\phi(w_t)-\phi'(\tilde{w}_t)\|$ 同时受"数据不同"和"初始状态不同"两种因素影响，直接分析极难。作者引入辅助序列 $\phi'(w_t)$，把敏感度拆开（Theorem 2）：

$$\|\phi(w_t)-\phi'(\tilde{w}_t)\|\le \underbrace{\rho_t\|w_t-\tilde{w}_t\|}_{\text{模型敏感度}}+\underbrace{\gamma_t}_{\text{数据敏感度}}$$

**数据敏感度** $\gamma_t$ 度量从同一初始化出发、在不同数据集上训练几步产生的误差（纯由数据差异引起）；**模型敏感度** $\rho_t\|w_t-\tilde{w}_t\|$ 度量从不同初始化、在同一数据集上训练产生的误差（由两个初始状态的相似度决定）。$\rho_t,\gamma_t$ 随学习率策略不同而不同（如 Noisy-FedAvg 常数学习率下 $\rho_t=(1+\mu L)^K$、$\gamma_t=\frac{2\mu V}{m}K$）。Remark 2.1 点明：$\rho_t$ 恒大于 1（非凸的典型特征），因此敏感度上界本身会随 $t$ 发散——但配合 $\lambda_t\le1$ 的缩放，最终仍能收敛。

**4. $t_0=0$ 松弛 + 求解最小化问题，证出收敛常数界**

Theorem 1 的界依赖待优化参数 $\lambda_t$ 和 $t_0$。作者把加权敏感度记为 $H(\lambda_t,t_0)=\sum_{t=t_0}^{T-1}\lambda_{t+1}^2(\rho_t\|w_t-\tilde{w}_t\|+\gamma_t)^2$，想最小化它得到最紧的界。这里有个两难：$t_0$ 太小则引入的稳定性间隙小、但 $T-t_0$ 轮累积让敏感度暴涨；$t_0$ 太大则累积误差小、却仍因全局敏感度无界而发散。作者退而求其次，取 $t_0=0$ 做松弛——此时稳定性误差恰为 0（因为 $\tilde{w}_0=w'_0$ 起点对齐），避免了发散，且 $H_0\ge H^\star$，于是 $T(w_T;w_T')\ge T_G(\sqrt{m}H^\star/\sigma)\ge T_G(\sqrt{m}H_0/\sigma)$。关键在于：即便是松弛后的 $H_0$，求解出来仍收敛为常数形式，所以隐私下界收敛。

最终的两个核心定理：

- **Theorem 3（Noisy-FedAvg）**：四种学习率策略（常数 C / 周期衰减 CD / 阶梯衰减 SD / 逐次衰减 ID）下均给出收敛界。以阶梯衰减 $\eta_{k,t}=\frac{\mu}{t+1}$ 为例：$T(w_T;w_T')>T_G\!\left(\frac{2\mu V K}{\sqrt{m}\sigma}\sqrt{2-\frac{1}{T}}\right)$，括号内随 $T\to\infty$ 趋于常数 $\frac{2\mu V K}{\sqrt{m}\sigma}\sqrt{2}$ 而非发散。这是**首个非凸函数 FL-DP 的收敛隐私分析**。Remark 3.1 总结：隐私主要由裁剪范数 $V$、局部步数 $K$、规模 $m$、噪声 $\sigma$ 决定，$V$ 越大间隙越大，$m$ 越大隐私越强（敏感度 $O(1/\sqrt{m})$），常数级噪声即可实现收敛隐私；Remark 3.2 指出该分析也适用于部分参与（把 $m$ 换成参与数 $n$，$m=n=1$ 即退化为标准 DP-SGD）。

- **Theorem 4（Noisy-FedProx）**：当 $\alpha>L$、$\eta<\frac{1}{\alpha-L}$ 时，$T(w_T;w_T')\ge T_G\!\left(\frac{2V}{\sqrt{m\alpha}\sigma}\sqrt{\frac{2\alpha-L}{L}\left(1-\frac{2}{(\frac{\alpha}{\alpha-L})^T+1}\right)}\right)$。近端正则项带来一个关键优势：隐私**不再依赖局部步数 $K$**，即使常数学习率也成立。Remark 4.1 指出 $\alpha$ 越大隐私越强（敏感度 $O(1/\sqrt{\alpha})$），但 $\alpha$ 过大会拖慢训练——所以 $\alpha$ 是优化与隐私之间的精细 trade-off。这给出一个"双赢"洞见：**设计良好的局部正则项能同时改善优化和隐私**。

### 损失函数 / 训练策略
无新增训练目标，分析建立在 Assumption 1（每个局部目标 $f_i$ 满足 $L$-光滑：$\|\nabla f_i(w_1)-\nabla f_i(w_2)\|\le L\|w_1-w_2\|$）这一标准非凸光滑假设下。

## 实验关键数据

实验目的不是刷 SOTA，而是**验证理论**：全局敏感度确实有界、且随 $m/K/V/\alpha$ 的变化与理论预测一致。

### 主实验
Noisy-FedAvg 在 MNIST(LeNet-5)/CIFAR-10(ResNet-18) 上的精度（Dir-0.1 高异构，固定 $TK=30000$，阶梯衰减学习率）：

| 设置 | 噪声 $\sigma$ | $m=50,K=50$ | $m=100,K=50$ |
|------|------|------|------|
| MNIST | $10^{-1}$ | 95.40 | 97.32 |
| MNIST | $10^{-3}$ | 98.41 | 98.94 |
| CIFAR-10 | $10^{-1}$ | 53.76 | 62.02 |
| CIFAR-10 | $10^{-3}$ | 70.98 | 75.38 |

$\sigma=1.0$ 时训练直接发散（表中标"-"）。客户端越多噪声影响越小（更多噪声参与平均→趋近噪声均值→近似无噪），与 $O(1/\sqrt{m})$ 一致；$\sigma$ 从 $10^{-3}$ 升到 $10^{-1}$，MNIST 上 $m=20/100$ 分别掉 5.57%/1.62%，CIFAR-10 掉 14.19%/11%。

### 消融实验
Noisy-FedAvg vs Noisy-FedProx（$T=600$，性能与敏感度对比，敏感度越小隐私越强）：

| 配置 | 精度 | 全局敏感度 |
|------|------|---------|
| Noisy-FedAvg | 60.67 | 31.33 |
| Noisy-FedProx $\alpha=0.01$ | 60.69 | 30.97 |
| Noisy-FedProx $\alpha=0.1$ | 60.94 | 18.52 |
| Noisy-FedProx $\alpha=1$ | 56.33 | 6.34 |

### 关键发现
- **敏感度有界且不随 $T,K$ 发散**：Fig.2 显示，增大 $K$ 虽会在过程中抬高敏感度，但优化收敛后的上界不变——这正是"隐私下界存在且不受 $T,K$ 影响"的实验证据，直接戳穿了组合分析的发散预测。
- **$m,V$ 的影响与理论吻合**：敏感度随 $m$ 增大而降（$O(1/\sqrt{m})$），随 $V$ 增大而升（$O(V)$）。
- **FedProx 的近端项是隐私"免费午餐"**：$\alpha=0.01$ 时敏感度与 FedAvg 几乎持平、精度还略高；$\alpha=0.1$ 时敏感度近乎砍半（31.33→18.52）而精度反升到 60.94；$\alpha=1$ 时敏感度暴降到 6.34（隐私大幅增强）但精度掉到 56.33。说明合适的 $\alpha$ 能在几乎不损精度的前提下显著降低敏感度、增强隐私。

## 亮点与洞察
- **推翻长期认知**：把"FL-DP 长期训练隐私必然耗尽"这一建立在组合定理上的悲观结论证伪——常数级噪声足以维持收敛的隐私下界。这对实践意义很大：不必为了长训而无限加噪。
- **f-DP + shifted interpolation 的组合拳很巧**：f-DP 提供无损的隐私度量，shifted interpolation 把放大次数从 $T$ 降到 $T-t_0$，两者缺一不可。把这套技术从凸/强凸的噪声梯度下降迁移到非凸 + 多步局部更新 + 异构数据的 FL 场景，是本文最硬的技术贡献。
- **敏感度二分解是可复用的分析工具**：用辅助序列把全局敏感度拆成"数据敏感度 + 模型敏感度"，思路清晰，可迁移到其他多步局部更新算法（SCAFFOLD、FedDyn 等）的隐私分析。
- **"正则即隐私"的洞见**：近端项让隐私摆脱对 $K$ 的依赖，暗示一大批基于局部正则的 FL 优化方法（FedDyn 等）本身就自带隐私优势。

## 局限与展望
- **只覆盖两类经典方法**：分析限于 Noisy-FedAvg / Noisy-FedProx，对带动量校正、方差缩减、primal-dual 的更先进 FL 优化器尚未给出收敛隐私界。
- **依赖 $L$-光滑假设**：非凸但要求光滑，对非光滑目标（如带 ReLU 不可微点的实际网络、$\ell_1$ 正则）严格性存疑；裁剪操作本身引入的偏置对界的影响也未充分讨论。
- **是"最坏隐私"下界而非精确隐私**：$t_0=0$ 的松弛带来 $H_0\ge H^\star$ 的间隙，界偏保守；实际隐私可能更强，紧致性只在附录 F 简单讨论。
- **实验规模偏小**：MNIST/CIFAR-10 + LeNet/ResNet-18，缺大模型、大规模客户端、跨设备真实异构场景的验证；精度本身不是重点，但敏感度度量在真实大模型上是否同样有界值得进一步检验。

## 相关工作与启发
- **vs 组合定理类分析（Wei 2020 / Shi 2021 / Zhang 2021 / Noble 2022）**：他们用 $(\epsilon,\delta)$-DP/RDP + 组合，要求 $\sigma^2\propto T$ 或 $TK$ 才能维持隐私，$T\to\infty$ 时界发散；本文用 f-DP + shifted interpolation 得到 $O(2-1/T)$ 的收敛界，常数噪声即可。Table 3 把本文结果无损转回 $(\epsilon,\delta)$-DP（$\sigma=o(\cdots\sqrt{2-1/T})$）和 RDP（$\epsilon=O(\frac{\zeta V^2}{m\sigma^2}(2-1/T))$）做对比。
- **vs 凸/强凸收敛隐私（Chourasia 2021 / Altschuler & Talwar 2022 / Bastianello 2024）**：他们在凸或 $\beta$-强凸下用 RDP 证出收敛隐私，但限于单机噪声梯度下降或强凸假设；本文首次把收敛隐私推到**非凸 + 联邦多步局部更新**，且不需要强凸。
- **vs Bok 等 2024（shifted interpolation 原始工作）**：Bok 在单机凸/强凸场景提出插值技术，本文将其扩展到 FL 的局部—全局动态，并补上数据/模型敏感度的二分解来处理多步异构更新。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个非凸 FL-DP 收敛隐私分析，推翻长期认知，技术迁移有难度。
- 实验充分度: ⭐⭐⭐ 实验只为验证理论、规模偏小，但对一篇理论论文够用。
- 写作质量: ⭐⭐⭐⭐ 证明链路清晰、定理表格化对比到位，公式密度高但可读。
- 价值: ⭐⭐⭐⭐⭐ 给 FL-DP 提供了更可信的隐私保证，对"正则即隐私"的洞见有实践指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Federated Learning of Quantile Inference under Local Differential Privacy](federated_learning_of_quantile_inference_under_local_differential_privacy.md)
- [\[ICLR 2026\] Fairness via Independence: A General Regularization Framework for Machine Learning](fairness_via_independence_a_general_regularization_framework_for_machine_learnin.md)
- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](../../NeurIPS2025/ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [\[ICLR 2026\] Beyond Membership: Limitations of Add/Remove Adjacency in Differential Privacy](beyond_membership_limitations_of_addremove_adjacency_in_differential_privacy.md)
- [\[ICLR 2026\] RESFL: An Uncertainty-Aware Framework for Responsible Federated Learning by Balancing Privacy, Fairness and Utility](resfl_an_uncertainty-aware_framework_for_responsible_federated_learning_by_balan.md)

</div>

<!-- RELATED:END -->
