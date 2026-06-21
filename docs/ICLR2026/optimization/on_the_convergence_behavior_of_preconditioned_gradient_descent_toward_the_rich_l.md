---
title: >-
  [论文解读] On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime
description: >-
  [ICLR 2026][优化/理论][预条件梯度下降] 本文从神经正切核（NTK）的特征值动力学出发，用理论与实验说明 Gauss-Newton / Levenberg-Marquardt 这类预条件梯度下降（PGD）能把谱偏置导致的"各频率模式收敛速度悬殊"拉平成均匀收敛，从而大幅压缩 grokking 的延迟泛化窗口；但 PGD 会粘在惰性（lazy）NTK 子空间里、最终泛化偏弱，需要在惰性阶段耗尽后切回一阶方法（如 Adam）才能恢复泛化。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "预条件梯度下降"
  - "谱偏置"
  - "Grokking"
  - "神经正切核"
  - "Gauss-Newton"
---

# On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=CXlsqTAf1E](https://openreview.net/forum?id=CXlsqTAf1E)  
**代码**: 待开源（作者承诺发布于 https://github.com/sandialabs）  
**领域**: 优化 / 学习理论  
**关键词**: 预条件梯度下降, 谱偏置, Grokking, 神经正切核, Gauss-Newton

## 一句话总结
本文从神经正切核（NTK）的特征值动力学出发，用理论与实验说明 Gauss-Newton / Levenberg-Marquardt 这类预条件梯度下降（PGD）能把谱偏置导致的"各频率模式收敛速度悬殊"拉平成均匀收敛，从而大幅压缩 grokking 的延迟泛化窗口；但 PGD 会粘在惰性（lazy）NTK 子空间里、最终泛化偏弱，需要在惰性阶段耗尽后切回一阶方法（如 Adam）才能恢复泛化。

## 研究背景与动机

**领域现状**：神经网络存在"谱偏置"（spectral bias / F-Principle）——训练时先学低频分量、后学高频分量。主流解释是 NTK 视角：函数空间的误差演化由 NTK 矩阵 $K_t = J_t J_t^T$ 的特征值谱决定，大特征值对应的模式收敛快、小特征值对应的模式收敛慢。谱偏置在图像分类等任务里相当于一种隐式正则、抑制高频噪声从而帮助泛化。

**现有痛点**：但在科学计算（PINN 求解 PDE、拟合含高频结构的函数）里，必须快速逼近高频解，谱偏置反而成了瓶颈。与此同时，grokking（延迟泛化：模型先记住训练集、很久之后才突然在测试集上泛化）是另一个拖慢训练的现象。已有工作（Wadia 2021、Buffelli 2024）发现高阶/曲率方法虽然收敛快，却往往泛化更差，因此对"是否该用高阶优化器"长期存在争议。

**核心矛盾**：谱偏置的根源是 NTK 条件数极差——只有 $O(1)$ 个特征值"大"，绝大多数模式按 $1 - \lambda_k/\lambda_N \ll 1$ 的速率龟速收敛。而 grokking 被 Kumar (2024)、Zhou (2024) 假设为"从惰性 NTK 主导的 lazy regime 过渡到富特征学习 rich regime"的产物。这两件事其实同源：模型长期困在 lazy 子空间里、被谱偏置卡着只学低频。

**本文目标**：（1）证明 PGD 能从根本上改善 NTK 的条件数、消除谱偏置；（2）用 PGD 把 lazy regime "均匀地"快速探索完，验证 grokking 确实是 lazy→rich 的过渡行为；（3）解释为什么高阶方法泛化差，并给出补救方案。

**切入角度**：把优化当成数值线性代数里的预条件问题——既然谱偏置等价于"病态条件数下的各向异性收敛"，那就用经典预条件子（曲率/Hessian 信息）把损失景观重塑成更各向同性的，让所有频率模式以接近一致的速率收敛。

**核心 idea**：用预条件梯度下降（GN / LM）替代一阶 GD/Adam 去"匀速探索" NTK 子空间，消除谱偏置带来的延迟；再在惰性阶段耗尽后切回一阶方法弥补泛化短板。

## 方法详解

### 整体框架
本文不提出新网络结构，而是对"优化器如何重塑误差模式的收敛动力学"做理论刻画 + 实验验证。骨架是三件事串起来的因果链：① 在 NTK / 无限宽度近似下，把误差按 NTK 特征向量解耦成一组独立的标量 ODE，得到每个模式的收敛速率表达式；② 给 GD、LM、GN 三种更新规则分别推导这个速率，证明 LM/GN 把"速率正比于特征值 $\lambda_i$"改造成"速率被压平"，从而条件数从 $\kappa_{GD}=\lambda_N/\lambda_1$ 改善到 $\kappa_{LM}\ll\kappa_{GD}$（取 $\mu=\lambda_1$ 时 $\kappa_{LM}\approx 2$）；③ 把这套"匀速探索 lazy 子空间"的能力用到 grokking 上，说明 PGD 压缩延迟泛化，但因为停留在 lazy 子空间而泛化不足，需切一阶方法补足。

设定为最小二乘回归 $\min_\theta L(\theta)$，$L(\theta)=\tfrac12\lVert f(\theta)-y\rVert^2$，$f$ 是深度 $L$、等宽 $W$ 的 MLP。取步长 $\eta\to0$ 的连续梯度流，令 $J_t=\nabla_\theta f(\theta(t))$ 为 Jacobian，则函数空间动力学为 $\frac{\partial f}{\partial t} = -J_t J_t^T (f-y) = -K_t\,e$，其中误差 $e(t)=f(\theta(t))-y$、NTK 矩阵 $K_t=J_t J_t^T$ 对称半正定（充分过参时高概率正定），可正交特征分解。

### 关键设计

**1. 预条件消除谱偏置：把"速率∝特征值"压平成均匀收敛**

针对谱偏置的根因——各模式收敛速率被各自 NTK 特征值悬殊地拉开。作者先在无限宽度（$K_t\to K_\infty$ 时不变）下把误差解耦：设 $\Lambda=\mathrm{diag}(\lambda_i)$ 为 $K_t$ 的特征值、$\hat e_i$ 为第 $i$ 个特征向量上的误差分量，则普通梯度流给出 $\frac{\partial}{\partial t}\hat e_i = -\lambda_i\hat e_i$（Lemma 3.1→3.3 的基线）。这就是谱偏置的精确写照：学习率必须小到能稳住最大特征值 $\lambda_N$，于是小特征值模式按 $\sim\lambda_i/\lambda_N$ 龟速衰减。

引入 Levenberg-Marquardt 预条件，更新为 $\theta_{n+1}=\theta_n-\eta(\mu I + J_t^T J_t)^{-1}J_t^T(f-y)$（最小二乘里相当于岭回归），做同样的梯度流推导得到（Lemma 3.2）：

$$\frac{\partial}{\partial t}\hat e_i = -\frac{\lambda_i}{\mu+\lambda_i}\,\hat e_i.$$

映射 $\lambda_i \mapsto \frac{\lambda_i}{\mu+\lambda_i}$ 把不同量级的特征值"压"到接近 1 的范围，条件数从 $\kappa_{GD}=\lambda_N/\lambda_1$ 改善为 $\kappa_{LM}=\frac{\lambda_N}{\lambda_1}\cdot\frac{\lambda_1+\mu}{\lambda_N+\mu}\ll\kappa_{GD}$；特别地取 $\mu=\lambda_1$ 时 $\kappa_{LM}\approx 2$。当 $\mu\to0$，LM 退化为 Gauss-Newton：$\theta_{n+1}=\theta_n-\eta(J_t^T J_t)^{\dagger}J_t^T(f-y)$（$\dagger$ 为带截断 $\varepsilon$ 的伪逆，因 Jacobian 可能病态/奇异），此时（Lemma 3.3）

$$\frac{\partial}{\partial t}\hat e_i = -\mathbf 1_{\lambda_i(e)>\varepsilon}\,\hat e_i,$$

即除被伪逆截断丢弃的最小特征值（通常对应几何最高频）外，**所有模式以一致速率指数衰减**——谱偏置被彻底抹平。LM 因此可看作 GN 的信赖域变体，$\mu$ 是 SGD 与 GN 之间的插值旋钮。

**2. Grokking 是 lazy→rich 的过渡行为：用 PGD 把延迟窗口压扁**

针对争议——grokking 到底是权重衰减、自适应优化器、还是别的原因导致？作者承接 Kumar (2024)、Zhou (2024) 的假设：网络初期被困在 lazy 子空间 $f(x,\theta)\approx f(x,\theta_0)+J_0(\theta-\theta_0)$ 里，谱偏置只让它学到最低频特征，要等"逃出"惰性区、进入特征学习的 rich regime 才突然泛化。既然设计 1 证明了 PGD 能匀速探索 NTK 子空间，那么如果 grokking 真源于"谱偏置 + 长期滞留 lazy regime"，PGD 就应当显著缩短到泛化的时间——因为它同时、均匀地推进所有模式而非先低频后高频。

实验通过对模型输出/初始化乘缩放因子 $\alpha$ 来人为放大 lazy regime（$\alpha\to\infty$ 对应更大的惰性区），在模加法、高维多项式回归、Transformer 模算术、MNIST 等任务上验证：SGD 下随 $\alpha$ 增大，train 与 test 之间的延迟（grokking gap）越拉越长；而 LM 把这个延迟"跨 $\alpha$ 一致地压缩"，且测试动力学对 $\alpha$ 几乎不敏感。这从优化动力学侧给"grokking 是过渡现象、谱偏置是主因而非过拟合/自适应"提供了直接证据。

**3. PGD 泛化短板与混合策略：先 PGD 探惰性区、再切一阶补泛化**

针对反直觉现象——PGD 收敛飞快却最终泛化更弱（Transformer 模算术里 GGN 仅 45% vs Adam 100%）。作者把这归因于：高阶方法把误差快速降到 lazy/NTK 解 $w^*_{NTK}$ 附近后就"粘"在惰性子空间里，难以离开该平面去做真正的非线性特征学习（rich regime），而 NTK 解 $w^*_\mu$ 相对真实目标 $w^*$ 是欠泛化的。补救方案与 PDE 实践里"二阶收尾"恰好相反：**先用 PGD 把线性/惰性区耗尽，再切回一阶方法**（如 MNIST 上先 2000 步 LM、再 20000 步 AdamW），让一阶更新离开子空间、找回甚至超过原有的测试精度。这条"PGD 起手 + 一阶收尾"的训练流程是本文给出的实操结论，也解释了 Wadia (2021)、Buffelli (2024) 观察到的高阶方法泛化差之谜。

> 适用前提：上述理论严格成立于 NTK / 近线性区（无限宽度或特定初始化/缩放使 $K_t$ 随时间近似不变）。一旦进入非线性特征学习主导的 rich regime，线性曲率近似（式 $f\approx f_0+J_0(\theta-\theta_0)$）失效，PGD 的优势随之消失——这既是理论边界，也正是实验里观察到"预条件有效性在 rich regime 戛然而止"的原因。

### 损失函数 / 训练策略
回归任务用 MSE（此时 Fisher 信息矩阵与 Gauss-Newton 一致，仅需 Jacobian）；Transformer 分类任务用交叉熵（须用 generalized Gauss-Newton, GGN）。大矩阵 $(\mu I + J_t^T J_t)$ 的求逆用共轭梯度等迭代法或 Sherman-Morrison-Woodbury 公式近似，常配线搜索定步长。混合策略：LM/GN 起手若干步 → 切 AdamW/Adam 收尾。

## 实验关键数据

### 主实验（收敛与 grokking 的定性结论）

| 任务 | 优化器对比 | 关键观察 |
|------|-----------|---------|
| 拟合多频函数 $u(x)=\tfrac13\sum_{k=1}^3 k\sin((2k{+}1)\pi x-k)$（2 层、宽 80） | SGD vs LM($\mu{=}0.5,0.1$) vs GN | GN 各频率模式近均匀指数衰减；LM 随 $\mu$ 在 SGD↔GN 间插值，印证 Lemma 3.2/3.3 |
| 2D Poisson 方程 PINN（宽 256，解 $\sin(\pi nx)\sin(\pi my)$） | SGD / Adam / LM($\mu{=}0.1$) | SGD/Adam 初期靠消低频快降；LM 随频率升高优势越明显，各频率斜率近似一致 |
| MNIST grokking（初始化×$\alpha$） | AdamW / Adam / LM | LM 缩短泛化延迟，但最终泛化精度低于一阶；grokking 与权重范数增/减/不变无关 |
| Transformer 模算术 | GGN vs Adam | GGN 验证精度上升更快但停滞在 45%，Adam 达 100%；GGN 增加迭代也不促泛化、反而过拟合 |

### 消融 / 分析

| 配置 | 现象 | 说明 |
|------|------|------|
| 仅 SGD（放大 $\alpha$） | grokking 延迟随 $\alpha$ 增大 | lazy regime 被撑大，谱偏置拖慢未见模式收敛 |
| 仅 LM/GN | 延迟被跨 $\alpha$ 一致压缩，但最终泛化弱 | 匀速探索 NTK 子空间，却粘在 lazy 区 |
| LM 2000 步 → AdamW 20000 步（MNIST） | 恢复甚至超过一阶最终精度 | 先耗尽惰性区再用一阶离开子空间 |
| 多项式回归"最大频率"误差（训练/测试 1D 子空间） | SGD 下两子空间不同时收敛；PGD 下近同时收敛 | 直接证明谱偏置导致未见模式慢收敛、PGD 抹平差异 |

### 关键发现
- **GN 给出"全频率均匀指数衰减"**是最强的理论-实验吻合点（Lemma 3.3 的 $\mathbf 1_{\lambda_i>\varepsilon}$ 在 FFT 模式误差图上表现为各频率斜率一致）。
- **grokking 延迟对缩放 $\alpha$ 的依赖在 PGD 下基本消失**，支持"grokking = lazy→rich 过渡 + 谱偏置主因"，而非过拟合或自适应。
- **高阶方法的泛化短板可被"切一阶收尾"修复**——这是与 PDE 领域"二阶收尾"惯例相反的反直觉结论。

## 亮点与洞察
- 把"谱偏置"和"grokking"两个看似不同的现象统一到同一根因（困在病态 NTK 子空间），再用预条件这把数值线性代数的老钥匙一并解释，视角干净有力。
- 三条 Lemma 把 GD→LM→GN 的误差速率写成 $\lambda_i \to \frac{\lambda_i}{\mu+\lambda_i} \to \mathbf 1_{\lambda_i>\varepsilon}$ 的连续谱，$\mu$ 一个旋钮贯穿全程，理论极简且可验证。
- "PGD 起手 + 一阶收尾"是可直接迁移的训练 trick：凡是受谱偏置/慢高频之苦的任务（PINN、隐式神经表示），都可考虑用曲率方法快速探完线性区再切 Adam。

## 局限与展望
- 理论严格只在 NTK / 近线性区成立，rich regime 的收敛分析仍是开放问题，PGD 在那里失效。
- 未处理 grokking 的另一可能诱因——训练/测试集大小（数据侧因素），分析全程聚焦优化动力学。
- 高阶方法每步求逆代价不小（须 CG / SMW 近似 + 线搜索），论文坦言计算开销非平凡；GN 伪逆截断 $\varepsilon$ 会丢掉最高频模式。
- 大多实验是浅 MLP / 小任务 + 人为缩放诱导 grokking，向大规模真实模型的外推性待验证；代码尚未公开。

## 相关工作与启发
- **vs Kumar et al. (2024) / Zhou et al. (2024)**：他们从"lazy→rich 过渡"和"训练/测试谱失配"两个角度提出 grokking 假设，本文不另起炉灶，而是用 PGD 这一可操控的优化器**主动验证**这些假设——通过匀速探索 NTK 子空间直接压缩延迟，给假设补上优化动力学侧的因果证据。
- **vs Wadia (2021) / Buffelli (2024)**：他们观察到高阶/曲率方法泛化更差但未给清晰机理，本文把原因落到"粘在 lazy 子空间"，并给出"切一阶收尾"的修复方案，把负面观察转成可用策略。
- **vs Adam（一阶自适应）**：Adam 给每参数独立学习率、相当于对角预条件，缩短 lazy regime 但忽略跨参数耦合，病态方向仍慢；GN/LM 用完整曲率信息处理跨参数交互，故能在高频上明显胜出。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把谱偏置与 grokking 用预条件统一解释、并给反直觉的混合训练结论，视角新但用的是经典工具
- 实验充分度: ⭐⭐⭐ 任务覆盖回归/PINN/模算术/MNIST 且与理论吻合，但多为浅网络小规模、靠人为缩放诱导 grokking
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰、Lemma 与实验图一一对应，逻辑链完整
- 价值: ⭐⭐⭐⭐ "PGD 起手+一阶收尾"对 PINN/隐式表示等高频任务有实操价值，也深化了对优化动力学与学习相变的理解

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the Convergence Direction of Gradient Descent](on_the_convergence_direction_of_gradient_descent.md)
- [\[ICLR 2026\] Gradient Descent with Large Step Sizes: Chaos and Fractal Convergence Region](gradient_descent_with_large_step_sizes_chaos_and_fractal_convergence_region.md)
- [\[ICLR 2026\] Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking](egalitarian_gradient_descent_a_simple_approach_to_accelerated_grokking.md)
- [\[ICLR 2026\] Fast Convergence of Natural Gradient Descent for Over-parameterized Physics-Informed Neural Networks](fast_convergence_of_natural_gradient_descent_for_over-parameterized_physics-info.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](../../ICML2026/optimization/on_the_convergence_rate_of_lora_gradient_descent.md)

</div>

<!-- RELATED:END -->
