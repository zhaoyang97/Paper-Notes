---
title: >-
  [论文解读] Debiased Front-Door Learners for Heterogeneous Effects
description: >-
  [ICLR 2026][因果推理][前门调整] 本文把 back-door 设定下成熟的 DR-Learner 与 R-Learner 移植到**前门(front-door)识别**场景，提出 FD-DR-Learner 和 FD-R-Learner 两个去偏估计器，即使 nuisance 函数只以 $n^{-1/4}$ 的慢速率收敛，条件前门效应 $\tau(C)$ 仍能达到 quasi-oracle 速率。
tags:
  - "ICLR 2026"
  - "因果推理"
  - "前门调整"
  - "异质处理效应"
  - "去偏学习"
  - "quasi-oracle 速率"
  - "DR-Learner"
  - "R-Learner"
---

# Debiased Front-Door Learners for Heterogeneous Effects

**会议**: ICLR 2026  
**代码**: [https://github.com/yonghanjung/FD-CATE](https://github.com/yonghanjung/FD-CATE)  
**领域**: 因果推断 / 异质处理效应  
**关键词**: 前门调整, 异质处理效应, 去偏学习, quasi-oracle 速率, DR-Learner, R-Learner  

## 一句话总结
本文把 back-door 设定下成熟的 DR-Learner 与 R-Learner 移植到**前门(front-door)识别**场景，提出 FD-DR-Learner 和 FD-R-Learner 两个去偏估计器，即使 nuisance 函数只以 $n^{-1/4}$ 的慢速率收敛，条件前门效应 $\tau(C)$ 仍能达到 quasi-oracle 速率。

## 研究背景与动机
**领域现状**：观测数据做因果推断时，最大的麻烦是未观测混杂——处理 $X$ 受到既影响 $X$ 又影响结果 $Y$ 的隐变量 $U$ 干扰，此时 $E[Y|X=1]-E[Y|X=0]$ 是有偏的。Pearl 的前门准则给出一条出路：找一个把 $X$ 影响传递给 $Y$ 的可观测中介 $Z$（如「主动安全带法 $X$ → 安全带使用率 $Z$ → 乘员死亡 $Y$」），只要 $Z$ 本身不被 $U$ 混杂，就能绕开 $X\!-\!Y$ 之间的隐混杂识别因果效应。

**现有痛点**：前门方向的去偏估计虽已发展（Fulcher 2019、Guo 2023、Jung 2024 等），但**几乎都只估计总体平均效应(ATE)**；而平台/政策制定者真正想要的是**个体化的条件前门效应 $\tau(C)$**。另一条线上虽有针对异质效应的深度估计器（Xu & Gretton 2022、Chen 2025 的 LobsterNet），但它们**不具备去偏性(debiasedness)**——一旦 nuisance 拟合不准，估计就跟着崩。换言之，"前门 + 异质 + 去偏"三者从未被同时满足（见原文 Table 1c 的对比）。

**核心矛盾**：back-door 下的 DR/R-Learner 之所以强大，靠的是 Neyman 正交化带来的去偏性——nuisance 慢收敛也不拖累目标。但前门估计量结构更复杂（涉及 $m,e,q$ 三组 nuisance 及其密度比组合），无法直接照搬 back-door 的伪结果(pseudo-outcome)构造。

**本文目标**：构造前门版的伪结果与正交损失，让"任意现成 ML 模型 + 慢速 nuisance"也能快速收敛到 $\tau(C)$。**核心 idea**：(1) **FD 伪结果(FDPO)** 把前门效应写成一个对 nuisance 误差只剩二阶项的可回归量；(2) **偏微分线性重参数化** 把前门效应分解成两个标准 back-door R-Learner 子问题 $b(C)$（$X\!\to\!Z$）和 $g(XC)$（$Z\!\to\!Y$），再用 pseudo-g 把组合项 $\gamma_g$ 的误差从 $\hat e_X$ 解耦出来。

## 方法详解

### 整体框架
两个学习器解决同一个目标 $\tau(C)=\sum_{z,x}\{q(z|1C)-q(z|0C)\}e_x(C)m(zxC)$，但路线不同：**FD-DR-Learner** 走"单一伪结果回归"路线——构造一个条件均值恰等于 $\tau_{\bar x}(C)$ 的伪结果，直接回归到 $C$ 上；**FD-R-Learner** 走"分解-组合"路线——把数据生成过程重写成两段偏微分线性模型，用现成的 back-door R-Learner 分别学通路系数 $b$、$g$，再合成。两者都靠 Neyman 正交结构换来对 nuisance 误差的二阶依赖。

```mermaid
flowchart TD
    A[观测数据 V=(C,X,Z,Y)<br/>前门结构: X→Z→Y, U 混杂 X,Y] --> B{两条去偏路线}
    B --> C[FD-DR-Learner]
    B --> D[FD-R-Learner]
    C --> C1[拟合 nuisance m,e,q]
    C1 --> C2[构造 FD 伪结果 φ_x̄<br/>含密度比 ξ,π + 修正项]
    C2 --> C3[回归 φ_1-φ_0 到 C → τ̂_DR]
    D --> D1[偏微分线性重参数化<br/>X→Z 得 b, Z→Y 得 g]
    D1 --> D2[BD-R-Learner 学 b、g]
    D2 --> D3[pseudo-g 解耦 ê_X 误差 → γ̂]
    D3 --> D4[τ̂_R = b̂·γ̂]
```

### 关键设计

**1. 前门伪结果 FDPO：把效应写成只对 nuisance 误差二阶敏感的可回归量。** FD-DR 的核心是为每个干预值 $\bar x$ 构造伪结果 $\varphi_{\bar x}(V;\eta)$。它由三块拼成——一项以密度比 $\xi_{\bar x}(ZXC)=q(Z|\bar xC)/q(Z|XC)$ 加权的残差 $Y-m(ZXC)$，一项以逆倾向 $\pi_{\bar x}(XC)=\mathbb{I}(X=\bar x)/e(X|C)$ 加权的修正 $r_{me}(ZC)-\nu_{meq}(XC)$，再加一个直接项 $s_{mq\bar x}(XC)$：

$$\varphi_{\bar x}(V;\eta)=\xi_{\bar x}\{Y-m\}+\pi_{\bar x}\{r_{me}-\nu_{meq}\}+s_{mq\bar x}.$$

这个构造的妙处在于 Lemma 2 给出的两个性质：**一致性** $\tau_{\bar x}(C)=E[\varphi_{\bar x}(V;\eta)\mid C]$，意味着只要把 $\varphi_1-\varphi_0$ 回归到 $C$ 上就直接得到 $\tau(C)$；**双稳健性**——当用估计 $\hat\eta$ 替换真值时，偏差 $E[\varphi_{\bar x}(V;\hat\eta)-\varphi_{\bar x}(V;\eta)]$ 完全由 nuisance 误差的**两两乘积**($\{\hat m-m\}\{\xi-\hat\xi\}$ 之类)构成。于是只要 $\hat q$ 准、或者 $(\hat m,\hat e)$ 准，二者有其一就能抵消一阶偏差。Theorem 1 据此给出 $\|\hat\tau_{DR}-\tau\|_2^2\lesssim R_{DR}+\sum\|\hat m-m\|^2\|\hat\xi-\xi\|^2+\dots$，所有 nuisance 都达 $n^{-1/4}$ 时即收敛到 quasi-oracle 速率。

**2. 前门的偏微分线性重参数化：把一个前门问题拆成两个 back-door R-Learner。** FD-R 不直接碰复杂的前门估计量，而是先证明前门结构等价于一组分层偏微分线性模型（Prop. 2）：$Z=a(C)+Xb(C)+\epsilon_Z$ 描述 $X\!\to\!Z$，$Y=f(XC)+Zg(XC)+\epsilon_Y$ 描述 $Z\!\to\!Y$。由于 $C$ 对 $(X,Z)$、$(X,C)$ 对 $(Z,Y)$ 各自满足 back-door 准则，$b(C)$ 和 $g(XC)$ 都可以用**标准的 BD-R-Learner 现成学**——也就直接继承了 R-Learner 的去偏性（慢 nuisance 不拖累）。Theorem 2 进一步证明异质前门效应能写成这两段通路系数的乘积：

$$\tau(C)=b(C)\,\gamma_g(C),\qquad \gamma_g(C)=E[g(XC)\mid C].$$

这一步把"难"的前门估计转译成两个"已被解决"的子问题，附带好处是 $b$、$g$ 本身就是 $X\!\to\!Z$ 和 $Z\!\to\!Y$ 通路强度的可解释中间量，可直接用于诊断。

**3. pseudo-g：把组合项 $\gamma_g$ 的误差从倾向得分 $\hat e_X$ 中解耦出来。** 拿到 $\hat b,\hat g$ 后还要估 $\gamma_g(C)=e_X(C)g(1C)+\{1-e_X(C)\}g(0C)$。最朴素的 plug-in $\hat\gamma_{plug}$ 直接代入 $\hat e_X$，但其误差 $\hat\gamma_{plug}-\gamma_g$ 含一项 $\{g(1C)-g(0C)\}(\hat e_X-e_X)$——被倾向得分的精度卡住瓶颈，$\hat e_X$ 慢则整体慢。本文转而定义 pseudo-g：

$$\zeta_{\tilde\eta_z}(XC)=\{1-e_X\}\tilde g(0C)+e_X\,\tilde g(1C)+\{X-e_X\}\{\tilde g(1C)-\tilde g(0C)\}.$$

Lemma 3 表明它满足 $E[\zeta_{\eta_z}\mid C]=\gamma_g(C)$，且**误差校正项里 $\hat e_X$ 的偏差被消掉**——$E[\zeta_{\hat\eta_z}\mid C]-\gamma_g$ 只剩 $e_X\{\hat g(1C)-g(1C)\}+\{1-e_X\}\{\hat g(0C)-g(0C)\}$，纯由 $\hat g$ 误差决定，而 $\hat g$ 可经 BD-R-Learner 高效学到。最后回归 $\zeta$ 到 $C$ 得 $\hat\gamma$，返回 $\hat\tau_R(C)=\hat b(C)\hat\gamma(C)$。Theorem 3 给出其误差由 quasi-oracle 速率加上 nuisance 误差的二阶/四阶项控制，同样在 $n^{-1/4}$ 下达到 quasi-oracle。

## 实验关键数据

### 合成实验（已知真值 $\tau(C)$，nuisance 用 XGBoost）
对比 plug-in 基线 FD-PI、FD-DR、FD-R 在四种 regime 下的 RMSE（mean ± 95% CI）：

| 实验设定 | FD-PI (plug-in) | FD-DR | FD-R | 结论 |
|---|---|---|---|---|
| (a) 变样本量 $n$，无结构噪声 | 高 | 低 | 低 | 两个去偏器一致碾压 plug-in |
| (b) nuisance 限制在 $n^{-1/4}$ 慢速率 | 收敛极慢 | 可靠收敛 | 可靠收敛 | 验证去偏性 |
| (c) 注入噪声 $\rho\epsilon,\ \rho\in[0,1]$ | $\rho$ 增大急剧恶化 | 稳定 | **更低且更稳** | FD-R 对 nuisance 误差最不敏感 |
| (d) 弱重叠(positivity 趋 0/1) | 严重退化 | 方差膨胀(用了逆权重) | **最稳，全程领先** | FD-R 因避开密度比更耐弱重叠 |

### 真实案例：州安全带法与死亡率(FARS)
州-年面板数据，$X$=是否有主动安全带法、$Z$=安全带使用率、$Y$=乘员死亡、$C$=协变量：

- 两个学习器估出的 $\hat\tau$ 分布均偏负（主动法降低死亡率），总体均值约 $-0.047$(FD-DR)/$-0.046$(FD-R)。
- 集中曲线显示 **>95% 的单位在主动法下死亡率下降**，仅极少数上升，符合"安全带法整体起保护作用"的预期。
- SHAP 归因显示 **年龄、时段、是否驾驶员** 是解释效应异质性的主导特征。

### 关键发现
- **去偏性可实证**：当 nuisance 只达 $n^{-1/4}$ 慢速率时，plug-in 明显落后，而 FD-DR/FD-R 仍快速收敛，印证理论。
- **FD-DR vs FD-R 的实操分工**：nuisance 拟合得准、重叠充分时 FD-DR 占优；重叠弱、nuisance 噪声大时 FD-R 因避开密度比而更稳——与 §4.1 的理论判断完全吻合。

## 亮点与洞察
- **"移植 + 重参数化"的范式很干净**：没有发明全新估计器，而是先把前门问题转译成已被攻克的 back-door R-Learner 子问题，再用伪结果/正交损失补齐去偏性，复用了整套成熟理论工具。
- **pseudo-g 是点睛之笔**：它精准定位并消除了"组合项误差被倾向得分卡瓶颈"这一隐藏的慢收敛源，是 FD-R 能整体达 quasi-oracle 的关键。
- **两个学习器互补而非竞争**：原文 §4.1 直接给出 practitioner guidance（重叠强用 FD-DR、重叠弱用 FD-R），并提出按重叠程度自适应路由的设想，工程友好。
- **模型无关**：nuisance 与目标都可用任意现成 ML（实验用 XGBoost），落地门槛低。

## 局限与展望
- **依赖 positivity(重叠)假设**：$e(X|C)$、$q(Z|XC)$ 接近 0/1 时方差膨胀（FD-DR 尤甚），作者建议重叠诊断、比值稳定化与重叠感知的不确定性，并计划做"弱重叠时自动路由到 FD-R"的自适应机制。
- **仅限二元中介 $Z$**：理论建立在 binary $Z$ 上，而许多实际场景中介是连续或多维的，扩展到一般中介是明确的后续方向。
- **缺与异质前门深度基线的直接对比**：实验主要对 plug-in，未与 LobsterNet 等异质前门方法在同一基准上正面比较去偏收益。

## 相关工作与启发
- **back-door 去偏谱系**：AIPW(Robins 1994)、TMLE(van der Laan)、DR-Learner(Kennedy 2023)、R-Learner(Nie & Wager 2021)、orthogonal statistical learning(Foster & Syrgkanis 2023)、DML(Chernozhukov 2018)——本文正是把其中 DR/R-Learner 移植到前门。
- **前门平均效应去偏**：Fulcher 2019(双稳健 FD-ATE)、Guo 2023(one-step/TMLE)、Jung 2024(统一协变量调整)；深度可扩展但无去偏的 Xu & Gretton 2022 / Xu 2024。
- **异质前门**：Chen 2025 的 LobsterNet(多任务神经网，但无去偏性)是最接近的前作，本文补上了去偏这一缺口。
- **启发**：当一个识别公式结构复杂时，"先重参数化成若干已被解决的标准子问题、再设计正交伪结果补去偏性"是一条可推广到其他识别准则(如 napkin/general ID)的通用配方。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 首次同时满足"前门 + 异质 + 去偏"，pseudo-g 解耦与偏微分线性重参数化是实质性的新构造，虽属"移植成熟工具"路线但补齐了明确空白。
- **实验充分度**: ⭐⭐⭐ 合成实验对四种 regime 系统验证了理论，FARS 案例有说服力；但缺与异质前门深度基线的正面对比，主对比仅为 plug-in。
- **写作质量**: ⭐⭐⭐⭐ 理论推导清晰、Table 1c 的三维定位与 §4.1 的实操指南很贴心，可读性强。
- **价值**: ⭐⭐⭐⭐ 为存在隐混杂但有合规中介的观测场景(政策、医疗、平台)提供了可直接落地、模型无关、带快速收敛保证的个体化因果估计工具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ALM-MTA: Front-Door Causal Multi-Touch Attribution Method for Creator-Ecosystem Optimization](alm-mta_front-door_causal_multi-touch_attribution_method_for_creator-ecosystem_o.md)
- [\[ICLR 2026\] Learning Exposure Mapping Functions for Inferring Heterogeneous Peer Effects](learning_exposure_mapping_functions_for_inferring_heterogeneous_peer_effects.md)
- [\[ICLR 2026\] GDR-learners: Orthogonal Learning of Generative Models for Potential Outcomes](gdr-learners_orthogonal_learning_of_generative_models_for_potential_outcomes.md)
- [\[ICLR 2026\] Topological Causal Effects](topological_causal_effects.md)
- [\[ICLR 2026\] Matching without Group Barrier for Heterogeneous Treatment Effect Estimation](matching_without_group_barrier_for_heterogeneous_treatment_effect_estimation.md)

</div>

<!-- RELATED:END -->
