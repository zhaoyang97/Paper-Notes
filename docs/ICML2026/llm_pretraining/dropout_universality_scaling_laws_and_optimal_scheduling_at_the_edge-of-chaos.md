---
title: >-
  [论文解读] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos
description: >-
  [ICML 2026][预训练][平均场理论] 作者把 dropout 看作平均场信号传播理论中破坏 $c^*=1$ 完美对齐不动点的"外场" $h$，推出 Landau 方程、两参数标度坍塌以及 smooth/kinked 激活的两个不同普适类，并由此得到一个"零开销"的实用结论——**前置 dropout（front-loaded schedule）**在同等预算下比常数 dropout 在 MLP 和 ViT 上把测试损失降低 18–35%。
tags:
  - "ICML 2026"
  - "预训练"
  - "平均场理论"
  - "edge-of-chaos"
  - "dropout 调度"
  - "普适类"
  - "标度律"
---

# Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos

**会议**: ICML 2026  
**arXiv**: [2605.21648](https://arxiv.org/abs/2605.21648)  
**代码**: 有 (dropout-universality-experiments, 论文附 commit-pinned 仓库)  
**领域**: 训练理论 / Dropout / 平均场 / 信号传播 / 调度  
**关键词**: 平均场理论, edge-of-chaos, dropout 调度, 普适类, 标度律

## 一句话总结
作者把 dropout 看作平均场信号传播理论中破坏 $c^*=1$ 完美对齐不动点的"外场" $h$，推出 Landau 方程、两参数标度坍塌以及 smooth/kinked 激活的两个不同普适类，并由此得到一个"零开销"的实用结论——**前置 dropout（front-loaded schedule）**在同等预算下比常数 dropout 在 MLP 和 ViT 上把测试损失降低 18–35%。

## 研究背景与动机

**领域现状**：随机初始化深网的平均场理论（Poole et al. 2016；Schoenholz et al. 2017）把网络分为 ordered / chaotic / 临界三个相，"edge-of-chaos" 上相关长度 $\xi_c$ 发散，信号能传得最深。He 初始化的 $\sigma_w^2=2$ 实际上就是 ReLU 的临界条件。

**现有痛点**：dropout 是工业界的默认正则项，但在 MFT 里的处理仅仅是说"它会摧毁 $c^*=1$ 不动点"，并没有给出可用的标度律；调度上业界要么常数 dropout，要么 stochastic depth / curriculum dropout 这类启发式，**为什么**某种调度更好缺乏一阶原理性的解释。

**核心矛盾**：dropout 一方面带来正则（降低过拟合），另一方面会**切断**信号沿深度的相关性传播——这两者是在每一层独立调节的，但目前的理论既不能告诉你给定预算下应该怎么沿深度分配 dropout，也不能告诉你 smooth vs kinked 激活该不该用不同策略。

**本文目标**：(i) 把 dropout 嵌进 MFT 给出标度律级别的描述；(ii) 区分 smooth/kinked 激活的普适类；(iii) 把理论翻成可执行的调度规则。

**切入角度**：把 dropout 看成统计力学里的"外场" $h$，把去对齐量 $m\equiv 1-c^*$ 看成"序参量"，问题瞬间变成 Landau 临界现象的标准范式——已经有现成的 RG / 标度坍塌 / 普适类工具可以套。

**核心 idea**：dropout 在 $c=1$ 处给相关性映射加了一个**常数偏移**，使得 $c^*<1$ 仍然是不动点但相关长度有限；这个偏移就是"外场" $h$；预算 $\sum_\ell h_\ell = L\bar{h}$ 下最大化 $\xi_{\rm eff}$ 是一个凹优化，**饱和阶梯解**最优；regularization reach 进一步选出"前置"那一支。

## 方法详解

### 整体框架

这篇论文要回答一个被业界默认却没人讲清楚的问题：固定 dropout 总量，沿网络深度该怎么分配才最优？作者的整条逻辑是把随机初始化深网的平均场理论（MFT）搬过来——无 dropout 时前向相关性满足递推 $c^l = F(c^{l-1})$，临界（edge-of-chaos）由 $\chi_\perp \equiv F'(1) = 1$ 定义，此时 $c=1$ 是完美对齐不动点、信号能传到最深。引入 keep-probability $\rho$ 的 inverted dropout 后，相关性映射形变为 $\bar{F}_\rho(c)$，$c=1$ 不再是不动点；作者把这个偏移识别为统计力学里的"外场" $h$、把去对齐量 $m\equiv 1-c^*$ 当"序参量"，于是整套 Landau 临界现象的工具（标度律、普适类、标度坍塌）立刻可用，最后把理论翻译成一条可执行的 dropout 调度规则。实验侧只验证这条规则：MLP / ViT 在 CIFAR-10/100 上固定总预算 $\bar{h}$，比较常数、前置、后置、线性等调度。

### 关键设计

**1. 把 dropout 识别为破坏对齐对称性的外场 $h$：让临界现象的整套工具能用上**

之前的工作（Schoenholz et al. 2017）只说 dropout"摧毁 $c=1$ 不动点"就停住了，相关长度失去定义、RG 分析无从下手。作者的关键一步是把独立 mask 作用后的相关性递推 $\bar{F}_\rho$ 在 $c=1$ 处求值，得到 $\bar{F}_\rho(1) = 1 - \frac{1-\rho}{\rho \bar{q}^*}\sigma_w^2 \int Dz\,\phi^2(\sqrt{\bar{q}^*}z) < 1$，由此把那个偏移定义成外场 $h \equiv 1-\bar{F}_\rho(1)$（弱 dropout 下 $h \approx a(1-\rho)$，与 dropout 概率线性相关），并把序参量取为 $m\equiv 1-c^*$。把 $\bar{F}_\rho(1-m)$ 在 $m=0$ 处 Taylor 展开、代入不动点条件 $1-m = \bar{F}_\rho(1-m)$，就得到一个标准的 Landau 方程

$$h = \tfrac{g_\rho}{2}m^2 - tm,\qquad m(t,h) = \frac{t+\sqrt{t^2+2g_\rho h}}{g_\rho},$$

其中 $t\equiv \chi_\rho - 1$ 充当约化温度。这一步之所以是后续一切的前提：它证明被 dropout 形变后的递推**仍然有一个 $c^*<1$ 的不动点**，相关长度重新有定义，标度律才有立足点。

**2. Smooth 与 kinked 两个普适类 + 两参数标度坍塌：激活函数的选择被归入临界指数**

同样是 edge-of-chaos，tanh 和 ReLU 的临界行为为何截然不同（Yang & Schoenholz 2017 在 ResNet MFT 里就观察到）？作者指出答案完全由相关性映射在 $c=1$ 邻域的**解析结构**决定。Smooth 激活（tanh、GELU）满足 Price 定理、可在 $c=1$ 光滑 Taylor 展开，二阶项 $g_\rho m^2$ 主导，给出 $m\sim\sqrt{h}$（$\delta=2$）、$\xi\sim h^{-1/2}$；kinked 激活（ReLU）的 $\phi''$ 含 $\delta$ 函数，在 $c=1$ 出现分支点，方程退化成 $h = \kappa m^{3/2} - tm$，给出 $m\sim h^{2/3}$（$\delta=3/2$）、$\xi\sim h^{-1/3}$。两类各自的临界指数（$\nu_t, \beta, \theta_{\rm rel}, \gamma, \delta, \nu_\rho, \alpha$）成套给出。同一普适类内的所有 $(t,h)$ 曲线还能被两参数标度坍塌到单一普适函数：smooth 类定义 $\tilde{m}\equiv m\sqrt{g_\rho/(2h)}$、$\tilde{t}\equiv -t/\sqrt{2g_\rho h}$，闭式坍塌为 $\tilde{m} = \sqrt{1+\tilde{t}^2}-\tilde{t}$；kinked 类则是 $m = (h/\kappa)^{2/3}\mathcal{F}\big(t/(\kappa^{2/3}h^{1/3})\big)$，crossover scale 为 $|t|\sim \kappa^{2/3}h^{1/3}$。Hermite 谱展开提供了一个独立的二次诊断——smooth 激活的 Hermite 系数指数衰减，kinked 激活幂律衰减。这样一来，"用 tanh 还是 ReLU"这种工程决策被纳入统计力学普适类：同类内的细节无关紧要，跨类必须换标度规律，而这是首次在 dropout 这根轴上把两类区分开。

**3. 前置 dropout 调度：凹预算分配选出阶梯，regularization reach 选出"早层填满"**

有了相关长度的标度律，"dropout 该放哪层"就变成一个干净的优化问题。让 keep probability 随层 $\ell$ 变化，有效逆相关长度近似 $\xi_{\rm eff}^{-1} \approx \frac{1}{L}\sum_\ell \sqrt{t^2+2g_\rho h_\ell}$；在临界点 $t=0$ 上简化为 $\xi_{\rm eff}^{-1} \propto \frac{1}{L}\sum_\ell h_\ell^{1/2}$，约束是预算 $\sum_\ell h_\ell = L\bar{h}$ 与上界 $h_\ell \leq h_{\max}$。由于 $h^{1/2}$ 是凹函数，Jensen 不等式直接给出：把预算押到 $\{0, h_{\max}\}$ 两端的**阶梯**解最优，相对常数 dropout 的增益是 $\xi_{\rm step}/\xi_{\rm const} = \sqrt{h_{\max}/\bar{h}}$。但 MFT 主目标对层的排列不变——阶梯放前放后一样好，简并需要第二条原理打破。作者引入"下游暴露" $\mathcal{D}_\ell \approx h_\ell \xi_c\big(1-e^{-(L-\ell)/\xi_c}\big)$，刻画早层的 mask 被更多下游层"看见"，这是一个关于 $\ell$ 单调递减的权重，于是线性规划解唯一地落到**早层填满**——也就是 front-loaded schedule。同样的论证对 kinked 类的 $\int h^{1/3}$ 照样成立。整个推导把"该把 dropout 放哪"化成"凹函数预算分配 + 单调权重打破简并"两步标准优化，结论是零额外算力、给定预算即可照用的调度规则。

### 损失函数 / 训练策略

实验侧不改训练目标，只改 dropout 沿层的分布：固定平均预算 $\bar{h}$，比较 constant / linear-decreasing / step（前置）/ step（后置）等 schedule。理论侧的 $\bar{F}_\rho$、$\chi_\rho$、$g_\rho$ 都通过 Gaussian 测度积分直接数值求解 MFT 递推，作为实验的对照基准。

## 实验关键数据

### 主实验

| 实验设置 | 调度 | 损失下降 | Δacc (pp) | 相对提升 |
|---------|------|---------|-----------|----------|
| MLP 过拟合（Fig.6） | Step (early) | +17.9% | +0.83 | +2.0% |
| MLP 预算控制（Fig.7） | Big step (1/3) | +22.6% | +1.08 | +2.6% |
| ReLU $\bar{h}=0.1$ sweep | Big step (1/3) | **+35.4%** | +2.04 | +5.0% |
| GELU $\bar{h}=0.1$ sweep | Big step (1/3) | +29.8% | +0.62 | +1.5% |
| ViT CIFAR-100 | Linear (decreasing) | +4.2% | +0.66 | +1.4% |
| ViT CIFAR-10 ablation | Both blocks, step (early) | +6.3% | +0.52 | +0.7% |

ViT CIFAR-100 上 linear-decreasing 达到 49.38% vs constant 48.69%（$p<0.05$）。

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| Smooth (tanh/GELU) MFT 递推 | $m\sim\sqrt{h}$（$\delta=2$），$\xi\sim h^{-1/2}$ | 与 Landau $m^2$ 项一致 |
| Kinked (ReLU) MFT 递推 | $m\sim h^{2/3}$（$\delta=3/2$），$\xi\sim h^{-1/3}$ | 分支点 $m^{3/2}$ 项主导 |
| 两参数标度坍塌（Fig.2） | 所有 $(t,h)$ 曲线坍塌成单一普适函数 | smooth 类闭式 $\tilde{m}=\sqrt{1+\tilde{t}^2}-\tilde{t}$ |
| Width 远大于 depth | front-loading 优势稳定 | 在 MFT 适用域内成立 |
| 高 dropout / 窄网络 | 优势减弱 | 精确就在理论失效处 |

### 关键发现

- ReLU MLP 上获得**最大**收益（+35.4% 损失下降），印证 kinked 类对 $h$ 的低阶非线性允许更激进的预算重分配。
- Smooth 类（GELU）+29.8% 也很显著，说明结论跨激活函数普适。
- ViT 上 schedule 优势缩小到 4–6%，与理论一致：attention/skip connection 改变全局深度动力学但保留局部 Gaussian kernel，理论的"次序"判断（早层有利）依然成立但量级下降。
- 把 dropout 撞到理论失效区（高 $\bar{h}$、窄网络）后优势消失——是支持理论的**反向**证据。

## 亮点与洞察

- 把 dropout 当成统计力学的"外场" $h$、把去对齐量当成"序参量" $m$，整套 Landau / 临界指数 / 标度坍塌的物理工具立刻可用——这是非常优雅的"问题对齐"，给后续把任何超参当作"场"提供了模板。
- 普适类的判定依据是激活的**解析结构**（Taylor 可展开 vs 分支点），不是常见的"是否 scale-invariant"——这个判据更细，能解释为什么 ReLU 系列和 tanh 系列在很多深度行为上分裂。
- "凹预算优化 → 阶梯饱和 → 用 regularization reach 单调权重打破简并 → 前置"——这个**两步分解**很有启发：MFT 主目标常常对置换不变，需要次级原理选具体方案。

## 局限与展望

- **仅前向 MFT**：后向梯度 covariance 同样存在 mask 独立性导致的对角/非对角不对称，作者给了递推 (18) 但没有完整发展 backward critical theory；finite-width gradient susceptibilities、训练时 mask 关联、catapult 后表示变化都未建模。
- **架构限制**：CNN/ResNet 的 dropout-deformed MFT 只是论证（App. A.4），没有真正的实验；ViT 上虽然结论成立但 attention 大-宽限有更细致的处理空间。
- **初始化理论**：所有结论是**初始化时刻**的，没有刻画训练动力学中 representation 学习对 schedule 的反作用——这是这一类工作普遍的局限。
- **mask 关联**：当 batch 内 mask 共享时 $c=1$ 不动点恢复，正则化强度变弱，需要新分析。

可延伸方向：（i）把同样视角应用到 weight decay、warm-up、adaptive dropout；（ii）发展 finite-width gradient critical theory；（iii）attention head dropout 的普适类分析；（iv）训练时间维度的 schedule（结合 curriculum dropout）。

## 相关工作与启发

- **vs Schoenholz et al. (2017)**：他们最早指出 dropout 破坏 $c=1$ 不动点，但说"临界消失"就结束了；本文证明 $c^*<1$ 仍是不动点，于是 RG / 标度律工具继续可用，这是关键的"半步推进"。
- **vs Hayou et al. (2019)**：观察到 smooth/ReLU 在 edge-of-chaos 行为差异，本文给出**临界指数**和普适类的形式化判据。
- **vs Stochastic Depth (Huang et al. 2016) / Curriculum Dropout (Morerio et al. 2017) / LayerDrop (Fan et al. 2020)**：那些是时间维或整层维的调度，本文是**空间深度维**的 dropout 强度调度，相互正交可叠加。
- **vs Roberts et al. (2022) 的 scale-invariant 判据**：本文的 smooth/kinked 划分是基于解析结构而非 scale-invariance，覆盖范围不同且更适合判定 dropout 标度律。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 把统计物理普适类的工具完整搬到 dropout 调度，并第一次给出 smooth/kinked 不同临界指数
- 实验充分度: ⭐⭐⭐⭐ MLP/ViT 在 CIFAR 上有完整 $\bar{h}$-sweep + 激活函数 ablation，缺 CNN/ResNet 实验
- 写作质量: ⭐⭐⭐⭐⭐ Landau 方程 + 标度坍塌 + 调度三段式推导清晰，理论与实验对照精准
- 价值: ⭐⭐⭐⭐ "零开销"的 front-loaded schedule 立刻可用，理论框架对后续超参作为"场"的研究有奠基意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Explaining Data Mixing Scaling Laws](explaining_data_mixing_scaling_laws.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[ICML 2026\] XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge](xtransfer_modality-agnostic_few-shot_model_transfer_for_human_sensing_at_the_edg.md)
- [\[ICML 2026\] Scaling Depth Capacity via Zero/One-Layer Model Expansion](scaling_depth_capacity_via_zeroone-layer_model_expansion.md)

</div>

<!-- RELATED:END -->
