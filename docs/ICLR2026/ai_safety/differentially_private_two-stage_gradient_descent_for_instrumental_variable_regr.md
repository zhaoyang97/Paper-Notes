---
title: >-
  [论文解读] Differentially Private Two-Stage Gradient Descent for Instrumental Variable Regression
description: >-
  [ICLR2026][AI安全][差分隐私] 本文提出 DP-2S-GD——第一个面向工具变量回归（IVaR）的差分隐私算法：把经典的两阶段最小二乘（2SLS）改写成两阶段梯度下降，在每步梯度更新里做逐样本裁剪并注入校准好的高斯噪声，从而满足 $\rho$-zCDP，并给出了显式刻画"优化—隐私—采样"三方权衡的有限样本收敛率。
tags:
  - "ICLR2026"
  - "AI安全"
  - "差分隐私"
  - "工具变量回归"
  - "两阶段最小二乘"
  - "梯度扰动"
  - "有限样本收敛"
---

# Differentially Private Two-Stage Gradient Descent for Instrumental Variable Regression

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=XQDy4obYLZ](https://openreview.net/forum?id=XQDy4obYLZ)  
**代码**: 待确认  
**领域**: 差分隐私 / 因果推断 / 学习理论  
**关键词**: 差分隐私, 工具变量回归, 两阶段最小二乘, 梯度扰动, 有限样本收敛

## 一句话总结
本文提出 DP-2S-GD——第一个面向工具变量回归（IVaR）的差分隐私算法：把经典的两阶段最小二乘（2SLS）改写成两阶段梯度下降，在每步梯度更新里做逐样本裁剪并注入校准好的高斯噪声，从而满足 $\rho$-zCDP，并给出了显式刻画"优化—隐私—采样"三方权衡的有限样本收敛率。

## 研究背景与动机
**领域现状**：工具变量回归（Instrumental Variable Regression, IVaR）是因果推断的基础工具。当回归变量 $x$ 被未观测混淆因子 $u$ 影响、违反外生性假设时，普通最小二乘（OLS）会得到有偏、不一致的估计；IVaR 引入一个与 $x$ 相关、但只通过 $x$ 影响结果 $y$ 的工具变量 $z$，从而恢复结构参数 $\beta$ 的一致估计。它在推荐系统（曝光被历史偏好混淆）、强化学习（动作与奖励受未观测上下文共同影响）等机器学习场景越来越重要。经典做法是两阶段最小二乘（2SLS），有闭式解。

**现有痛点**：IVaR 的很多应用涉及敏感数据——个人健康记录、金融交易、用户行为。这类场景下，哪怕只是发布模型估计或中间统计量，都可能泄露单个个体的信息。差分隐私（DP）提供了严格的数学框架来保证算法输出不暴露任何单条数据。但据作者所知，**此前没有任何工作处理过"在差分隐私约束下做 IVaR"这个问题**。

**核心矛盾**：DP 在 OLS 上已有成熟方案，其中梯度扰动（clip + 注高斯噪声）配合现代隐私会计能给出最锐的统计率。但 IVaR 不像 OLS 那样是单个矩条件——它的闭式 2SLS 估计量依赖**嵌套的矩阵乘法与求逆**，敏感度极难直接刻画；再加上弱工具下的病态性（ill-posedness），充分统计量扰动、共识类机制都难以直接搬过来。简言之：闭式解结构让"该往哪儿注噪、注多少"无从下手。

**本文目标**：能否为线性 IVaR 模型设计差分隐私算法，并同时达到统计高效的收敛率？这需要同时解决三个子问题——(1) 找到一个能"在迭代层注噪"的算法结构；(2) 严格控制每步敏感度并校准噪声；(3) 给出非渐近的效用保证。

**切入角度**：作者不去硬碰闭式 2SLS，而是把它换成一个**两阶段梯度下降**（2S-GD）的迭代过程。梯度法的好处是：噪声可以在每次迭代的梯度更新上以受控方式注入，敏感度由裁剪阈值直接界定，并且能用 zCDP 的可加组合规则干净地累加多步隐私损失。

**核心 idea**：把 2SLS 改写成可迭代的两阶段梯度下降，在两个阶段的梯度上分别做逐样本裁剪 + 注高斯噪声，用 zCDP 会计把每步隐私损失加起来，从而第一次在 IVaR 上同时拿到隐私保证和可证收敛率。

## 方法详解

### 整体框架
IVaR 的线性模型是
$$y = \beta^\top x + \epsilon_1, \qquad x = \Theta^\top z + \epsilon_2,$$
其中 $\epsilon_1,\epsilon_2$ 因共同混淆因子 $u$ 而相关。给定数据 $(Z,X,Y)=\{(z_i,x_i,y_i)\}_{i=1}^n$，IVaR 求解一个双层优化问题：外层 $\hat\beta=\arg\min_\beta \frac1n\sum_i (y_i-\beta^\top\hat\Theta^\top z_i)^2$，内层 $\hat\Theta=\arg\min_\Theta \frac1n\sum_j \|x_j-\Theta^\top z_j\|^2$。经典 2SLS 给出闭式解：第一阶段 $\hat\Theta=(Z^\top Z)^{-1}Z^\top X$（把 $x$ 对 $z$ 回归），第二阶段 $\hat\beta_{\text{2SLS}}=(\hat\Theta^\top Z^\top Z\hat\Theta)^{-1}\hat\Theta^\top Z^\top Y$（把 $y$ 对预测出的 $\hat X=Z\hat\Theta$ 回归）。

本文的整体流程是：**先把闭式 2SLS 替换成一个非私有的两阶段梯度下降基线 2S-GD**——每次迭代交替更新第一阶段投影矩阵 $\Theta^{(t)}$ 和第二阶段回归参数 $\beta^{(t)}$，等价于 2SLS 的"梯度版"；**再把它私有化为 DP-2S-GD**——相比 2S-GD 加两处改动：(i) 两个阶段的梯度都做逐样本裁剪以界定敏感度，(ii) 在 $\Theta$ 与 $\beta$ 的更新上每步注入高斯扰动，噪声尺度按目标隐私预算 $\rho_1,\rho_2$ 校准。最后用 zCDP 的可加组合给出整体隐私保证，并配一个有限样本误差界刻画三方权衡。输入是敏感数据 $(Z,X,Y)$ 和隐私预算 $(\rho_1,\rho_2)$，输出是隐私化的参数轨迹 $\{\Theta^{(t)}\},\{\beta^{(t)}\}$。

### 关键设计

**1. 用两阶段梯度下降替代闭式 2SLS：把"该往哪注噪"变成可解的问题**

闭式 2SLS 的麻烦在于 $\hat\beta$ 依赖嵌套的 $Z^\top Z$ 求逆和矩阵连乘，敏感度无法直接界定，强行往 Gram 矩阵 $Z^\top Z$ 上注噪又会因病态（弱工具）放大误差。作者改用一个迭代算法 2S-GD：交替做两步梯度下降，第一步用梯度 $z_i(z_i^\top\Theta^{(t)}-x_i^\top)$ 更新 $\Theta^{(t)}$，第二步用梯度 $\Theta^{(t)\top}z_i(z_i^\top\Theta^{(t)}\beta^{(t)}-y_i)$ 更新 $\beta^{(t)}$。这个改写的价值不在于算得更快，而在于它**把隐私注入点从"无从下手的闭式解"转移到了"每一步清晰的梯度"上**——梯度是逐样本求和的结构，单样本的影响一眼可见，于是裁剪和注噪都有了明确的着力点；同时梯度法天然支持正则化、minibatch、流式数据、早停，模块化、分阶段，比闭式解更贴合实际训练管线。

**2. 逐样本梯度裁剪界定敏感度**

差分隐私要求"单条数据改变对输出的影响有界"，但 IVaR 的梯度里 $z_i,x_i,y_i$ 都是亚高斯随机量，单样本梯度范数原则上无界。作者在两个阶段分别用裁剪阈值 $\gamma_1,\gamma_2$ 对逐样本梯度做 $\mathrm{CLIP}$：$\Theta$ 更新里裁剪 $z_i(z_i^\top\Theta^{(t)}-x_i^\top)$，$\beta$ 更新里裁剪 $\Theta^{(t)\top}z_i(z_i^\top\Theta^{(t)}\beta^{(t)}-y_i)$。裁剪把每步更新的敏感度硬性压到 $2\gamma_1/n$、$2\gamma_2/n$ 量级，从而决定了为达到目标隐私所需的高斯噪声尺度。一个关键的技术细节是：$\gamma_1,\gamma_2$ 被取成 $\gamma_1=\gamma_2=c_0(\sqrt q+\sqrt{\tau+\log(nT)})^2$ 量级，使得**以高概率裁剪根本不会真的改动梯度**（见 Lemma D.1）——也就是说裁剪只用来界定最坏情况敏感度，正常情况下不引入额外偏差，这让后续把私有算法的收敛率和非私有 2S-GD 对齐成为可能。

**3. 两阶段高斯注噪 + zCDP 预算分配：用可加组合干净地累加多步隐私损失**

在裁剪之后，DP-2S-GD 每步给两个阶段分别注入高斯噪声：$\Theta^{(t+1)}=\Theta^{(t)}-\frac\eta n\sum_i \mathrm{CLIP}_{\gamma_1}(\cdot)+\eta\Xi^{(t)}$，$\beta^{(t+1)}=\beta^{(t)}-\frac\alpha n\sum_i\mathrm{CLIP}_{\gamma_2}(\cdot)+\alpha\nu^{(t)}$，其中 $\mathrm{vec}(\Xi^{(t)})\sim N(0,\lambda_1^2 I)$、$\nu^{(t)}\sim N(0,\lambda_2^2 I)$。隐私分析把两个阶段当作两个独立的高斯机制，敏感度由 $\gamma_1,\gamma_2$ 控制。作者**特意选用零集中差分隐私（zCDP）而非 $(\epsilon,\delta)$-DP**：因为算法要在两个阶段跨 $T$ 次迭代组合大量相同的高斯机制，而 $(\epsilon,\delta)$-DP 的组合会线性累加 $\epsilon,\delta$、公式繁琐，zCDP 则用 Rényi 散度刻画隐私损失、参数直接相加，组合更紧也更干净。具体地（Proposition 3.1），只要取
$$\lambda_1=\frac{2\gamma_1}{n}\sqrt{\frac{T}{\rho_1}},\qquad \lambda_2=\frac{2\gamma_2}{n}\sqrt{\frac{T}{\rho_2}},$$
算法就满足 $\rho$-zCDP，其中 $\rho=\rho_1+\rho_2=\frac{2T}{n^2}\big(\frac{\gamma_1^2}{\lambda_1^2}+\frac{\gamma_2^2}{\lambda_2^2}\big)$。这里 $\rho_1$ 保护第一阶段轨迹 $\{\Theta^{(t)}\}$、$\rho_2$ 保护第二阶段轨迹 $\{\beta^{(t)}\}$，按需分配（有些场景第一阶段 $\Theta$ 是公开/非敏感的，可令 $\rho_1=\infty$ 不注噪），整体 $(\rho_1+\rho_2)$-zCDP 端到端可控。

**4. 优化—隐私—采样三方权衡的有限样本误差界（理论核心）**

本文最实质的贡献是把上面这套私有迭代的非渐近误差界算清楚（Theorem 3.1）。在 Assumption 2（$z$ 各向同性亚高斯、误差亚高斯）下，以高概率有
$$\|\beta^{(T)}-\hat\beta\|\lesssim \underbrace{\kappa(\tau)^{T/2}}_{\text{优化}}+\underbrace{\frac{\sqrt p(\sqrt q+\sqrt\tau)^3}{n\sqrt{\min\{\rho_1,\rho_2\}}}\sqrt T}_{\text{隐私噪声}}+\underbrace{\frac{\sqrt{pq}(\tau+\log(pq))}{\sqrt n}}_{\text{统计误差}},$$
其中 $0<\kappa(\tau)<1$ 是收缩率。三项分别对应：**优化项**随迭代数 $T$ 指数衰减（梯度下降本身的收敛）；**隐私噪声项**随 $\sqrt T$ 增长（为保隐私，噪声尺度 $\lambda$ 里含 $\sqrt{T}$，注噪越多迭代累积越大）；**统计项**随样本量 $n$ 增大而减小（无噪梯度下降的固有统计误差）。这个分解的妙处在于它**直接给出了迭代数 $T$ 的最优区间**——多迭代能压低优化项但放大隐私项，于是最优 $T$ 是关于 $n$ 的"亚线性但超对数"（sub-linear yet super-logarithmic），即 $T\lesssim \rho_1 n^{2-\epsilon}/[p(\sqrt q+\sqrt\tau)^6]$。技术上最难的部分正是控制"隐私噪声"与"梯度动态收缩"在多步迭代中的相互作用。此外 Corollary 3.1 说明令 $\rho_1=\rho_2=\infty$（不注噪）时该界退化为非私有 2S-GD 的收敛率，自洽。

### 损失函数 / 训练策略
没有额外训练损失：算法就是对 IVaR 双层目标做两阶段梯度下降，外加裁剪与注噪。关键超参是两阶段步长 $\eta,\alpha$（需满足式 (3) 的稳定性条件，实验中取 $\eta=\frac1{(1+\delta(\tau))^2}$、$\alpha=\frac2{2\bar\gamma(\tau)+\gamma(\tau)}$）、裁剪阈值 $\gamma_1,\gamma_2$、噪声尺度 $\lambda_1,\lambda_2$（由 $\rho_1,\rho_2,T$ 反解），以及迭代数 $T$。Remark 3.3 还给出近似最优步长，并指出误差在最优步长邻域内不敏感。

## 实验关键数据

### 主实验
作者在合成数据与真实数据上验证理论。隐私强度约定：$\rho=0.1$ 为强隐私、$\rho=1$ 为中等、$\rho=10$ 为弱隐私（对应 $\delta=10^{-5}$ 时 $\epsilon$ 分别约 2.25 / 7.79 / 31.47）。真实数据用经典的 Angrist 数据集，研究"生育对母亲劳动供给的因果效应"，工具变量是"前两个孩子是否同性别"。

| 实验 | 设置 | 关键结果 |
|--------|------|------|
| 合成数据 vs $n$（Fig 3a） | $p=q=r=5$, $T=20$, $\rho=10$ | 各点落在误差界的平台区，误差以 $1/\sqrt n$ 速率下降，符合理论 |
| 合成数据 vs $n$（Fig 3b） | $p=q=r=50$, $T=20$ | $T=20$ 违反式 (5) 的 $T$ 条件，误差显著大于低维情形，印证条件必要性 |
| Angrist 真实数据（Fig 5） | $n=8065$, $T=20$, $\rho_1=\rho_2=1$ | $\beta^{(T)}$ 集中在 $-4.3$（多生一个孩子使母亲年劳动供给减少约 4.3 周），与 2SLS 基准一致；约 15 次迭代后收敛 |

### 消融 / 分析实验

| 配置 | 现象 | 说明 |
|------|---------|------|
| 改变迭代数 $T$、$\rho_1$ 小（Fig 4a） | 超过某临界 $T$ 后误差急剧上升 | 与式 (5) 对 $T$ 的上界吻合：对 $\{\Theta^{(t)}\}$ 要强隐私时只能用中等 $T$ |
| 改变 $T$、只对 $\beta$ 注噪（$\rho_2$ 小，Fig 4b） | 误差走势贴合理论曲线 Fig 2 | 仅保护 $\beta$ 时（$\rho_1=\infty$）不需要式 (5) 的 $T$ 条件 |
| 隐私预算 $\rho_1,\rho_2$ 大小（Fig 5b） | 预算越大，估计越紧贴 2SLS 基准 | 隐私—精度权衡的直接体现 |
| $\rho=\infty$（Corollary 3.1） | 退化为 2S-GD 收敛率 | 但仍比闭式 2SLS 慢一个 $\sqrt p$ 因子（Remark 3.8） |

### 关键发现
- **最优迭代数 $T$ 是关于 $n$ 的"亚线性、超对数"**：太少迭代优化项没收敛，太多迭代隐私噪声累积爆炸，存在一个由式 (5) 刻画的甜点区，实验中临界点位置与理论预测吻合。
- **隐私预算如何在两阶段分配很关键**：当样本量有限、又要对第一阶段 $\Theta$ 强隐私（$\rho_1$ 小）时，误差会在某个 $T$ 之后急剧变差；若只需保护 $\beta$，则没有这个限制。
- **梯度法相比闭式 2SLS 有固有的 $\sqrt p$ 损失**（Remark 3.8）：即便完全不注噪，两阶段梯度下降逼近 $\hat\beta$ 的速率也比闭式 2SLS 慢一个 $\sqrt p$，根源是梯度迭代只能近似第二阶段的矩条件、残差项 $\frac1n Z^\top r$ 带来额外误差。

## 亮点与洞察
- **"换算法换注噪点"是解决问题的关键一招**：闭式 2SLS 的嵌套求逆让敏感度无从刻画，作者不去硬算闭式解的敏感度，而是退一步把它改写成迭代梯度法——隐私注入从"算不出敏感度的闭式解"挪到"逐样本可见的梯度"上，整个问题就盘活了。这个"为了可分析性而改写算法结构"的思路可迁移到其他闭式估计量的隐私化。
- **三项误差分解给出了可操作的超参指导**：把误差拆成"优化 / 隐私噪声 / 统计"三项，并直接读出最优 $T$ 是 $n$ 的亚线性超对数——这不是事后解释，而是能指导实践中怎么选迭代数的定量结论。
- **用 zCDP 而非 $(\epsilon,\delta)$-DP 的选择很务实**：在跨两阶段、跨 $T$ 步组合大量相同高斯机制的场景下，zCDP 的可加组合让公式干净、组合更紧，是分析多步迭代隐私的恰当工具选择。
- **第一项工作的定位清晰**：在 IVaR + DP 这个空白处第一次同时给出隐私保证和可证收敛率，且诚实地指出了梯度法相对闭式 2SLS 的 $\sqrt p$ 固有差距，把局限摆在明面上。

## 局限与展望
- **梯度法相对闭式 2SLS 慢 $\sqrt p$**：作者承认（Remark 3.8、结论）无论有没有隐私约束，两阶段梯度下降收敛到 $\hat\beta$ 都比闭式 2SLS 慢一个 $\sqrt p$ 因子，这是梯度逼近 2SLS 的固有限制；如何通过算法改造缩小这个差距是公开问题。
- **缺隐私—精度权衡的下界**：本文只给了上界，没有证明 IVaR 在 DP 下的匹配下界，因此无法判断当前率是否最优——这正是作者列出的未来方向。
- **只覆盖线性 IVaR**：方法和分析都建立在线性结构模型与一系列分布假设（亚高斯、各向同性、工具有效性）之上；非线性 IV（如 KernelIV、DeepIV）的差分隐私化未涉及。
- **收敛依赖条件数**：和多数私有一阶方法一样，收敛严重依赖问题的条件数（弱工具下病态会更糟），实验也主要在条件较好的合成与单一真实数据集上验证。

## 相关工作与启发
- **vs OLS 的差分隐私方法（梯度扰动 DP-SGD）**: 已有工作（Bassily et al. 2014; Abadi et al. 2016）表明梯度扰动 + 裁剪在 OLS 上给出最锐的统计率，且组合紧、可扩展。本文承接这一路线，但 OLS 只有单个矩条件、IVaR 是嵌套的两阶段矩条件，直接的难点是闭式 2SLS 敏感度难刻画——本文用两阶段梯度下降把注噪点搬到迭代上加以解决。
- **vs 充分统计量扰动 / 共识机制**: 这两类（对 $Z^\top Z$、$X^\top y$ 注噪，或用 propose-test-release / 指数机制直接私有化估计量）在 OLS 上可用，但在 IVaR 上因弱工具病态与矩方程的高敏感度难以适配，且纯充分统计量管线在高维下需要更大样本量（与条件数多项式相关）。本文选择梯度扰动正是为了避开 Gram 矩阵注噪的谱依赖放大。
- **vs 优化视角的 IVaR（2S-GD、双层梯度下降等）**: Della Vecchia & Basu (2023)、Liang et al. (2025) 等把 IVaR 做成可扩展/在线的随机优化或双层梯度下降并给收敛保证，但都假设对数据无限制访问、不提供端到端差分隐私。本文是在这类优化框架上**第一个加上 DP 并给出兼顾隐私与收敛率的有限样本分析**。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ IVaR + 差分隐私的首个工作，填补明确空白，"改写算法以便注噪"的思路干净有力
- 实验充分度: ⭐⭐⭐⭐ 合成 + Angrist 真实数据印证理论各项，但数据集规模与多样性有限，主要服务于验证而非大规模实证
- 写作质量: ⭐⭐⭐⭐⭐ 动机、算法、隐私与效用分析层层递进，误差三项分解讲得清晰，局限交代诚实
- 价值: ⭐⭐⭐⭐ 为敏感数据下的因果分析提供了有理论保证的隐私化工具，理论结论（最优 $T$、三方权衡）对实践有直接指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Missing Mass for Differentially Private Domain Discovery](differentially_private_domain_discovery.md)
- [\[ICLR 2026\] PE-SGD: Differentially Private Deep Learning via Evolution of Gradient Subspace for Text](pe-sgd_differentially_private_deep_learning_via_evolution_of_gradient_subspace_f.md)
- [\[NeurIPS 2025\] Differentially Private High-dimensional Variable Selection via Integer Programming](../../NeurIPS2025/ai_safety/differentially_private_high-dimensional_variable_selection_via_integer_programmi.md)
- [\[ICLR 2026\] On Optimal Hyperparameters for Differentially Private Deep Transfer Learning](on_optimal_hyperparameters_for_differentially_private_deep_transfer_learning.md)
- [\[ICLR 2026\] Optimizing Canaries for Privacy Auditing with Metagradient Descent](optimizing_canaries_for_privacy_auditing_with_metagradient_descent.md)

</div>

<!-- RELATED:END -->
