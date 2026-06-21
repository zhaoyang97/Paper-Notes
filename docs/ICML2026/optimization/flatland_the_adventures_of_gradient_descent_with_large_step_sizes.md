---
title: >-
  [论文解读] Flatland: The Adventures of Gradient Descent with Large Step Sizes
description: >-
  [ICML 2026][优化/理论][大学习率] 本文给出一个只需"局部 Lipschitz / Hölder 梯度连续"的统一"大学习率"定义，并用非单调线搜索造出一种一阶自适应步长，使梯度下降从训练一开始就运行在 Edge of Stability（EoS）上、把 sharpness 压到全局最小 $2/K$；同时发现"太早进入全局平坦区反而有害"，再用自稳定约束把失败的训练救回来。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "大学习率"
  - "Edge of Stability"
  - "sharpness"
  - "非单调线搜索"
  - "自稳定"
---

# Flatland: The Adventures of Gradient Descent with Large Step Sizes

**会议**: ICML 2026  
**arXiv**: [2606.06722](https://arxiv.org/abs/2606.06722)  
**代码**: 待确认  
**领域**: 优化理论 / Edge of Stability  
**关键词**: 大学习率, Edge of Stability, sharpness, 非单调线搜索, 自稳定

## 一句话总结
本文给出一个只需"局部 Lipschitz / Hölder 梯度连续"的统一"大学习率"定义，并用非单调线搜索造出一种一阶自适应步长，使梯度下降从训练一开始就运行在 Edge of Stability（EoS）上、把 sharpness 压到全局最小 $2/K$；同时发现"太早进入全局平坦区反而有害"，再用自稳定约束把失败的训练救回来。

## 研究背景与动机
**领域现状**：深度学习里 sharpness（训练损失 Hessian 的最大特征值 $\lambda_1(H)$）被普遍认为和泛化相关，"平坦极小值更好泛化"是个老假设（Hochreiter & Schmidhuber, 1997），SAM 的成功也佐证了这一点。另一条线索来自 Cohen et al. (2021)：固定步长 $\eta$ 的 GD 会经历两个阶段——先是 progressive sharpening（损失单调降、sharpness 升），然后进入 EoS（损失非单调震荡、sharpness 稳定在 $2/\eta$ 附近）。越早撞上 EoS 的步长，损失震荡越剧烈、收敛却越快；再大就发散。

**现有痛点**：既然大步长又快又平，大家都想用，但"大""小"步长在不同论文里定义不一致（Mohtashami 2023 / Wang 2025 各说各话）。更要命的是 Cohen et al. (2021) 自己也指出：**给定步长 $\eta$，根本不知道 GD 会不会、以及多早撞上 EoS**。神经网络目标函数通常不是全局 $L$-smooth 的，经典凸优化里"$\eta < 2/L$ 保收敛"的结论直接失效。

**核心矛盾**：要"尽可能大"的步长来加速、平坦化，又要"保证收敛"——而能不能保证收敛取决于一个全局光滑常数 $L$，可这个 $L$ 在深度网络上根本不存在（或大到没用）。于是 Goodfellow et al. (2016) 把它列为深度学习的长期开放问题：**如何选最大的、仍能保证 GD 收敛的学习率？**

**本文目标**：(1) 给"大学习率"一个只依赖局部光滑性的统一定义；(2) 设计一个**不用二阶信息**就能实际找到这种步长的一阶算法；(3) 刻画这种步长把网络带到了什么样的点，以及对收敛和泛化意味着什么。

**切入角度**：作者注意到 EoS 下 GD 的非单调震荡行为，和上世纪的**非单调线搜索**（Grippo 1986）允许损失暂时上升、却仍能收敛的行为高度相似。既然非单调线搜索有成熟的收敛理论，何不借它来"主动制造"EoS？

**核心 idea**：用"等式型非单调线搜索"自动求步长——它既能被证明 Lipschitz/Hölder-aware（从而隐式 sharpness-aware），又天然把 GD 推到 EoS；再加一个自稳定上界，避免太早跌进有害的全局平坦区。

## 方法详解

### 整体框架
本文的方法不是一条数据 pipeline，而是"一个定义 + 一个步长求解算法 + 一组性质证明 + 一个修正项"。整体逻辑是：先把"全局 $L$-smooth"这一不现实的假设，替换成只沿反梯度方向、只在局部成立的 **segment smoothness**；在这个弱假设下重新定义什么叫"大步长"；然后造一个带回溯与外推的**等式线搜索**去逼近这种步长；证明它满足 Lipschitz-aware、Hölder-aware、进而在驻点附近 sharpness-aware，且全局收敛；最后用实验观察到"太早平坦有害"，补上一个自稳定上界把训练救回来。这是一篇理论分析为主的论文，没有需要图示的多模块串行流程，故不强行画框架图。

记 $w_\eta := w - \eta\nabla f(w)$。局部 segment smoothness 定义为

$$l(w,\eta) := \sup_{\hat\eta\in(0,\eta]}\frac{\|\nabla f(w)-\nabla f(w_{\hat\eta})\|}{\|w-w_{\hat\eta}\|},$$

约定 $0/0=0$。只要每个 $w$ 存在某个 $\eta_w>0$ 使 $l(w,\eta_w)$ 有界，就称 $f$ 是 segment smooth——这比全局光滑弱得多，神经网络目标函数能满足。基于它有局部下降引理：当 $\eta<2/l(w_k,\eta_{w_k})$ 时函数值单调下降；反之容易构造发散例子（如二次函数）。

### 关键设计

**1. Segment smoothness 与统一的"大步长"定义：把全局光滑换成局部、方向性的弱假设**

经典理论卡在"需要全局 $L$"上，本文用上面那个只沿反梯度段成立的 $l(w,\eta)$ 取而代之。关键性质是 $l(w,\eta)$ 关于 $\eta$ 单调，因此总能把 $\eta_w$ 撑大到满足 $\eta_w\ge 2/l(w,\eta_w)$。在此之上，**大步长**被定义为（Definition 2.3）：步长序列 $\{\eta_k\}$ 在迭代 $k$ 处是"大"的，当且仅当 (i) $\eta_k\ge 1/l(w_k,\eta_{w_k})$，且 (ii) 存在 $p\in[0,P]$ 使 $\eta_{k+p}>2/l(w_{k+p},\eta_{w_{k+p}})$。直白说：步长不必每步都越过保单调下降的阈值 $2/l$，但必须**每隔至多 $P$ 步越一次**——这正好制造出 EoS 式的非单调震荡，同时不至于像"永远超阈值"那样直接发散。这个定义第一次把散落在各文献里互相打架的"大/小步长"说法统一在局部光滑量上。

**2. 等式型（非单调）线搜索：不用二阶信息就把步长顶到 EoS**

有了定义还要能"算"出这种步长。作者基于非单调线搜索条件

$$f(w_k-\eta_k\nabla f(w_k))\le f(w_k)-c\eta_k\|\nabla f(w_k)\|^2+\epsilon_k,$$

其中 $\epsilon_k:=R_k-f(w_k)\ge 0$，$R_k$ 是参考值（取 $R_k=f(w_k)$ 退化为单调线搜索；取 $R_k=\max_i f(w_{k-i})$ 即 Grippo 1986 版；本文主用 Zhang & Hager 2004 版）。算法同时带**回溯与外推**：若不等式初始不成立就把步长**减半**，若初始成立就把步长**翻倍**，于是 $\eta_k$ 会逼近让不等式取等号的 $\eta_k^*$（一般可推广为乘/除常数 $\delta\in(0,1)$）。正因为它逼近"等号"，步长被顶到尽可能大，自然踩在稳定边缘上。

**3. 从 Lipschitz/Hölder-aware 推出 sharpness-aware：一阶方法隐式感知曲率**

等式线搜索给出的步长满足 Lipschitz-awareness（Prop 2.5）：$\eta_k\ge \frac{2\delta(1-c)}{l(w_k,\eta_{w_k})}$，于是只要 $2\delta(1-c)\ge 1$ 就保证 Definition 2.3 的 (i)。非单调版还额外满足 Hölder-awareness（Prop 2.8，对任意 $\nu\in[0,1]$ 的 Hölder 连续梯度成立），并能在不要求全局 Lipschitz 的前提下证明全局收敛 $\lim_k\|\nabla f(w_k)\|=0$（Thm 2.9）。最关键的推论 2.10：当 $w_k\to w^*$ 时 $l(w_k,\eta)$ 收敛到 $w^*$ 处的方向 sharpness，于是对充分大的 $k$，$\eta_k\ge \frac{2(1-c)\delta}{|\lambda_1(H(w^*))|+\varepsilon}$——**步长被驻点处 Hessian 最大特征值的倒数下界住**，即一阶方法通过其震荡行为隐式拿到了二阶（sharpness）信息，无需真的算 Hessian。作者还指出 Polyak 初值的回溯线搜索同样能 Lipschitz/sharpness-aware（但非 Hölder-aware），而 CDAT（Roulet 2024，用二阶信息、$\sigma\ge 2$）虽然乘积超过 EoS 阈值，却不保证收敛。

**4. 自稳定上界：拦住"太早进全局平坦区"这条岔路**

实验里作者发现一个反直觉现象：大步长常把 sharpness 压到惊人地小且恒定的值 $2/K$（$K$ 为类别数，MSE 损失），并证明这正是 sharpness 的**全局最小**——最后一层 bias 对应的子 Hessian 是对角阵、$K$ 个元素都等于 $2/K$；而紧邻的下一批特征值两两求和为 0，说明这些点其实是**鞍点**。太早撞上这种全局平坦区，网络学不到有意义特征，损失卡在"随机猜测"水平（如 $K=10$ 时约 0.09），本文称之为"unsuccessfully-flat"。借助自稳定论证（Damian 2022），作者把大步长**限制在小于 $K$**，让 GD 改走略陡一点的山谷，从而把"unsuccessfully-flat"的失败训练扭转成"successfully-flat"——sharpness 略高于 $2/K$、但训练损失很低、测试精度更高。

### 损失函数 / 训练策略
主实验用 MSE 损失（cross-entropy 在附录），全批量 GD，从原数据集子采样 5000 张图、每 100 次迭代算一次 sharpness。线搜索参数固定 $c=10^{-4}$、$\delta=0.5$（全批量 GD 的常用值）；CDAT 的 $\sigma=2.06$，SAM 步长网格搜索选取。

## 实验关键数据

### 主实验
作者在"数据集 × 网络"的笛卡尔积上跑：CIFAR10/100、SVHN、EMNIST，配 MLP、CNN、resnet34、wide resnet50_2、densenet121、vgg11。正文聚焦 3 组不同类别数 $K$ 的组合，对比 5 种方法在四个量上的曲线（sharpness×步长、步长、sharpness、训练损失）。

| 方法 | 是否用二阶信息 | $\eta_k\!\cdot\!\lambda_1(H)$ 行为 | 把 sharpness 压到 | 收敛保证 |
|------|------|------|------|------|
| LS（单调线搜索） | 否 | sharpness 持续升、步长降，不总在阈值上 | 不降反升 | 有 |
| NLS（非单调等式线搜索，本文） | 否 | 恒 $>1-c$、常 $>2$，从头在 EoS | $2/K$（全局最小） | 有（Thm 2.9） |
| PoNLS（Polyak 初值版，本文） | 否 | 同 NLS | $2/K$ | 有 |
| CDAT（Roulet 2024） | 是 | 恒 $>2$ | 弱于 NLS/PoNLS | 不保证 |
| SAM | 是（扰动） | EoS 定义不同、乘积无意义 | 弱于 NLS/PoNLS | — |

核心结论：NLS/PoNLS 的 $\eta_k\cdot\lambda_1(H(w_k))$ 始终大于 $1-c$、且常常大于 2，配合损失的频繁震荡，证实它们**从训练第一步就运行在 EoS**，且在压低 sharpness 上全面优于 SAM 与 CDAT。

### 消融 / 分析实验

| 现象 | 观测 | 含义 |
|------|------|------|
| sharpness 收敛到 $2/K$ | NLS/PoNLS 多数情况把 $\lambda_1(H)$ 压到 $2/K$ 并恒定 | 该值被证明为 sharpness 全局最小（bias 子 Hessian 对角、$K$ 个 $2/K$） |
| 紧邻特征值两两和为 0 | 第一批 $K$ 个之下的特征值成对抵消 | 这些全局平坦点实为鞍点 |
| 太早平坦 | PoNLS 在 EMNIST×CNN 全程卡在初始损失 | "unsuccessfully-flat"：损失停在随机猜测水平（$K=10$ 约 0.09） |
| 加自稳定上界（$<K$） | 失败训练转为 successfully-flat | sharpness 略高于 $2/K$，训练损失低、测试精度升 |

附录进一步验证：SGD、Vision Transformer、warmup 步长、cross-entropy 损失下，"早期全局平坦区"同样损害收敛，结论具普适性。

### 关键发现
- **一阶方法能隐式感知曲率**：NLS 不碰 Hessian，却因震荡被 sharpness 倒数下界住，等价拿到二阶信息——这是全文最反直觉的点。
- **"越平越好"是个误区**：把 sharpness 推到全局最小 $2/K$ 不仅不帮收敛，反而因为那是鞍点而让网络学不到特征。略陡的山谷才训练得更好。
- **自稳定是关键修正**：仅靠一个"步长 $<K$"的上界，就把卡死的训练救活，说明问题不在步长"大"，而在"何时、撞向哪种平坦"。

## 亮点与洞察
- **把上世纪的非单调线搜索接到现代 EoS 上**：用 Grippo 1986 / Zhang-Hager 2004 的成熟收敛理论，给"主动运行在 EoS"提供了既能算又能证的工具，比起靠手调固定步长撞 EoS 优雅得多。
- **统一定义解决文献口径之争**：Definition 2.3 用局部 segment smoothness 把"大/小步长"钉死，结束了各论文各说各话的状态，这条贡献对后续理论工作很有复用价值。
- **"全局最小 sharpness = 鞍点"这一刻画**：用最后一层 bias 子 Hessian 对角化得到 $2/K$，再用特征值成对抵消判定鞍点，既漂亮又解释了为何"太平不好"，可迁移去分析其他过参数化网络的平坦区。

## 局限与展望
- **主实验是全批量 GD + 5000 样本子集 + MSE**：虽附录补了 SGD/ViT/CE，但与真实大规模 mini-batch 训练仍有距离，结论外推需谨慎。
- **没给运行时对比**：作者承认线搜索带来的每步额外开销只在附录做了初步比较（$\delta=0.5$ 时与 CDAT/SAM 相近，$\delta=0.9$ 时更贵），实际加速效益未充分量化。
- **自稳定上界 $<K$ 偏经验**：来自 Damian 2022 的自稳定论证给了直觉，但"恰好取 $K$"在更一般任务上是否最优、如何自适应选取，仍待研究。
- **sharpness↔泛化仍是开放的**：本文给出"略陡更好"的反例，但并未给出何种平坦度最有利泛化的可操作判据。

## 相关工作与启发
- **vs Cohen et al. (2021)**：他们用固定步长**观察**到 EoS 两阶段现象，但无法判断给定 $\eta$ 是否/何时进 EoS；本文用线搜索**主动构造**从头就在 EoS 的步长，并给出收敛证明。
- **vs SAM (Foret 2020)**：SAM 靠梯度扰动显式追平坦极小；本文不加扰动、不算二阶，靠震荡隐式 sharpness-aware，且在压低 sharpness 上更强，还指出"太平=鞍点"的风险——某种意义上是对"平坦即好"叙事的修正。
- **vs CDAT (Roulet 2024)**：CDAT 用二阶信息和 $\sigma$ 把步长顶在 EoS，但不保证收敛；本文纯一阶、有 Thm 2.9 的收敛保证，是其"免二阶 + 有理论"的替代。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把非单调线搜索与 EoS 统一，并回答了深度学习的长期开放问题
- 实验充分度: ⭐⭐⭐⭐ 网络/数据集覆盖广、现象刻画细，但缺大规模与运行时的硬对比
- 写作质量: ⭐⭐⭐⭐ 理论链条完整、动机清晰，但定义与命题密集，门槛偏高
- 价值: ⭐⭐⭐⭐⭐ 给"大学习率为何有效/何时有害"提供了可证明的框架与反直觉洞察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Gradient Descent with Large Step Size Restores Symmetry in Deep Linear Networks with Multi-Pathway](gradient_descent_with_large_step_size_restores_symmetry_in_deep_linear_networks_.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[AAAI 2026\] Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training](../../AAAI2026/optimization/beyond_the_mean_fisher-orthogonal_projection_for_natural_gradient_descent_in_lar.md)
- [\[NeurIPS 2025\] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](../../NeurIPS2025/optimization/large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)

</div>

<!-- RELATED:END -->
