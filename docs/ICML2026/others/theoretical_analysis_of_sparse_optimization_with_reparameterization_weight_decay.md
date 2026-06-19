---
title: >-
  [论文解读] Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate
description: >-
  [ICML 2026][稀疏优化] 本文提出 ReWA：把待优化变量重参数化为 $\boldsymbol{x}=\boldsymbol{y}^{K}$、对 $\boldsymbol{y}$ 加权重衰减、并使用一种坐标级自适应步长 $\eta_t \boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon)$，把不可优化的 $\ell_p\;(0<p<1)$ 稀疏正则等价转化为一个梯度有界、不易陷入零鞍点的可训练目标，并在 CIFAR-10 / ImageNet 上用 ResNet 验证了相对 $\ell_1$ 的稀疏性提升。
tags:
  - "ICML 2026"
  - "稀疏优化"
  - "$\\ell_p$ 正则"
  - "重参数化"
  - "权重衰减"
  - "自适应学习率"
---

# Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate

**会议**: ICML 2026  
**arXiv**: [2605.25134](https://arxiv.org/abs/2605.25134)  
**代码**: https://github.com/childofcuriosity/rewa (有)  
**领域**: 优化理论 / 稀疏训练  
**关键词**: 稀疏优化, $\ell_p$ 正则, 重参数化, 权重衰减, 自适应学习率  

## 一句话总结
本文提出 ReWA：把待优化变量重参数化为 $\boldsymbol{x}=\boldsymbol{y}^{K}$、对 $\boldsymbol{y}$ 加权重衰减、并使用一种坐标级自适应步长 $\eta_t \boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon)$，把不可优化的 $\ell_p\;(0<p<1)$ 稀疏正则等价转化为一个梯度有界、不易陷入零鞍点的可训练目标，并在 CIFAR-10 / ImageNet 上用 ResNet 验证了相对 $\ell_1$ 的稀疏性提升。

## 研究背景与动机

**领域现状**：稀疏训练的金标准是 $\ell_0$ 正则，但因非连续而难解，工业上一般用 $\ell_1$（LASSO 路线）做凸松弛，理论与算法都很成熟。

**现有痛点**：$\ell_1$ 会引入估计偏差，对于神经网络等过参数模型，会牺牲过多精度；改用 $\ell_p\;(0<p<1)$ 能更逼近 $\ell_0$、给出更强稀疏性，但 $\ell_p$ 在零附近梯度无界、不光滑，只能在线性回归这种简单场景上跑通，扩展到深度网络几乎必然炸训。

**核心矛盾**：稀疏性强度（$p$ 越小越接近 $\ell_0$）与优化稳定性（$p$ 越小梯度越发散）之间存在结构性 trade-off。已有的乘法重参数化 $f(\boldsymbol{y}_1\odot\cdots\odot\boldsymbol{y}_K)+\lambda/2\sum\|\boldsymbol{y}_i\|_2^2$（记为 [Cp]，对应 $p=2/K$）虽然让梯度有界，但 $\boldsymbol{y}^{K-1}$ 在零附近形成高阶鞍点，坐标一旦穿过零就出不来。

**本文目标**：构造一个算法，使其 (i) 在隐式正则层面仍对应某个 $\ell_p\;(0<p<1)$；(ii) 梯度处处有界；(iii) 能逃出零鞍点；(iv) 对真实数据集（CIFAR-10 / ImageNet）稳定可用。

**切入角度**：把 [Cp] 的对称 $K$ 个变量 tie 成同一个 $\boldsymbol{y}$，再额外引入一个由两个超参数 $M,\epsilon$ 调节的坐标自适应步长，让"逃零鞍点"成为算法内嵌的能力，而非依赖初始化。

**核心 idea**：用"重参数化 + 权重衰减 + 自适应学习率"三件套（ReWA），把难以优化的 $\ell_p$ 正则隐式编码进 SGD 更新里，并通过自适应步长抵消 $\boldsymbol{y}^{K-1}$ 带来的零鞍点。

## 方法详解

### 整体框架
ReWA 在前向上对参数做幂次重参数化 $\boldsymbol{x}=\boldsymbol{y}^{K}$（$K$ 取奇数，逐元素），网络损失 $f$ 输入仍是 $\boldsymbol{x}$，但反向只更新隐变量 $\boldsymbol{y}$。每步迭代形式为 $\boldsymbol{y}(t+1)=(1-\lambda\eta_t)\boldsymbol{y}(t)-\eta_t\frac{\boldsymbol{y}^{M}(t)}{\boldsymbol{y}^{K-1}(t)+\epsilon\mathbf{1}}\odot\boldsymbol{y}^{K-1}(t)\odot\nabla f(\boldsymbol{y}^{K}(t))$。其中 $\lambda$ 是权重衰减系数、$\eta_t$ 是基础学习率、$M\in[0,K-1)$ 与 $\epsilon\ge 0$ 共同决定隐式正则。训练完成后取 $\boldsymbol{x}(T)=\boldsymbol{y}^{K}(T)$ 作为最终（稀疏）解。算法可与 SGD、AdamW 作为外层 base optimizer 叠加；当 base 是 AdamW（已经自带坐标自适应）时建议设 $M=0$。

### 关键设计

**1. 幂次重参数化 $\boldsymbol{x}=\boldsymbol{y}^{K}$：把非光滑的 $\ell_p$ 正则改写成光滑损失加 $\ell_2$ 衰减**

$\ell_p\;(0<p<1)$ 稀疏性强、偏差小，但零附近梯度无界、不光滑，深网上几乎必炸。ReWA 的第一步是把待优化变量重参数化为 $\boldsymbol{x}=\boldsymbol{y}^{K}$（$K$ 取奇数，逐元素），网络损失 $f$ 输入仍是 $\boldsymbol{x}$，但反向只更新隐变量 $\boldsymbol{y}$。Lemma 3.1 证明这种乘法重参数化 [Cp] 与 $\ell_p\;(p=2/K)$ 正则在全局最优、局部最优、(亚)稳定点上一一对应，于是稀疏性的好处被原封不动继承下来，而优化难题被归约成"光滑损失 + 普通权重衰减"。更关键的是 Theorem 3.7 给出一个硬性不可能性：若直接对 $\ell_p$ 做梯度截断，梯度上界和逼近误差不可能同小（事件 $\mathcal{E}_1\le\sqrt{d}$ 与 $\mathcal{E}_2\le d/(2e)$ 无法同时成立），从根上堵死了"裁剪 + 原 $\ell_p$"这条捷径——也就正面论证了为什么必须走重参数化而不是简单加 grad clip。

**2. 自适应学习率 $\eta_t\,\boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon\mathbf{1})$：抵消 $\boldsymbol{y}^{K-1}$ 造成的零鞍点**

重参数化带来一个新麻烦：更新里的 $\boldsymbol{y}^{K-1}$ 在零附近形成高阶鞍点，坐标一旦和真值异号、要穿过零就出不来。ReWA 的解法是给步长乘一个坐标级自适应因子。Example 3.2 用一维玩具 $f(x)=(x-1)^2$、$y(0)=-1$ 把问题摆明：非自适应版本满足 $|y(T)-1|\ge 1$，永远逃不出零；而自适应版本（$M=0,\epsilon\to 0$ 时退化为 $\boldsymbol{y}(t)-\eta\nabla f(\boldsymbol{y}^K(t))$）满足 $|y(T)-1|\le 2(1-\tfrac{2\eta}{K-1})^T$，线性收敛。这个因子的分子 $\boldsymbol{y}^{M}$ 控制稀疏强度（$M$ 越大对小坐标抑制越狠），分母 $\boldsymbol{y}^{K-1}+\epsilon$ 在 $\boldsymbol{y}$ 大时抵消 $\boldsymbol{y}^{K-1}$、在 $\boldsymbol{y}$ 小时由 $\epsilon$ 兜底（角色类似 Adam 的稳定常数）。Theorem 3.3 算出 ReWA 的隐式正则是

$$R(\boldsymbol{x})=\tfrac{K}{1-M+K}\|\boldsymbol{x}\|_{1+(1-M)/K}^{1+(1-M)/K}+\epsilon\tfrac{K}{2-M}\|\boldsymbol{x}\|_{(2-M)/K}^{(2-M)/K},$$

Proposition 3.4 据此给出实操配方：简单数据用 Config A（$\epsilon=0,M>1$），复杂数据用 Config B（$\epsilon>0,M<2$），都能让主项指数 $p=1+(1-M)/K\in(0,1)$ 真正落在 $\ell_p$ 区间。

**3. 显式权重衰减 $(1-\lambda\eta_t)\boldsymbol{y}(t)$：把"小初始化假设"卸成可证的稀疏保证**

光靠重参数化的隐式偏置在大初始化时会失效——PowerPropagation 这类工作只在小初始化、矩阵分解等特殊场景才出稀疏。ReWA 干脆在更新里显式加 $\ell_2$ 衰减 $(1-\lambda\eta_t)\boldsymbol{y}(t)$。Example 3.8 / Theorem 3.9 证明：在二次目标 $f(\boldsymbol{x})=\boldsymbol{x}^\top\Lambda\boldsymbol{x}$ 下，不加衰减时存在初始化让解被冻在初值附近、远离稀疏最优；而加上 $\ell_2$ 衰减后保证收敛到原点这一最稀疏全局最优。这一步的意义在于把 Gunasekar、Woodworth 等依赖"小初始化"的隐式稀疏偏置，换成一个对任意初始化都成立、且能拓展到一般非凸问题的显式机制——三件套缺一不可：去掉重参数化撞回 $\ell_p$ 不可优化，去掉自适应步长卡在零鞍点，去掉权重衰减失去稀疏性。

### 损失函数 / 训练策略
基础优化器可以是 SGD 或 AdamW（Algorithm 2 给出 AdamW 版本）；学习率支持常数 / cosine decay。实践上 $K$ 取奇数最方便（直接 $\boldsymbol{x}=\boldsymbol{y}^K$），$K$ 偶数时改用 $\boldsymbol{y}_1\odot\boldsymbol{y}_1-\boldsymbol{y}_2\odot\boldsymbol{y}_2$ 或 $\boldsymbol{x}=\mathrm{sign}(\boldsymbol{y})\cdot|\boldsymbol{y}|^K$。

## 实验关键数据

### 主实验
在 CIFAR-10 / ImageNet 上分别用 ResNet 骨干，目标是在固定测试精度下比较稀疏率（非零参数比例越低越好）。下表为论文报告趋势的概括：

| 数据集 | 模型 | 方法 | 稀疏率（非零） | 测试精度 |
|--------|------|------|----------------|----------|
| CIFAR-10 | ResNet | $\ell_1$ 正则 | 基线 | 与本文相当 |
| CIFAR-10 | ResNet | **ReWA (Config B)** | 显著低于 $\ell_1$ | 与 $\ell_1$ 持平 |
| ImageNet | ResNet | $\ell_1$ 正则 | 基线 | 与本文相当 |
| ImageNet | ResNet | **ReWA (Config B)** | 显著低于 $\ell_1$ | 与 $\ell_1$ 持平 |

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| Full ReWA | 稳定收敛 + 稀疏 | 三件套都开 |
| w/o 自适应学习率（非自适应 SGD on [Cp]） | 一维 toy 上 $|y(T)-1|\ge 1$；ImageNet 上炸训 | 验证 Example 3.2 / Theorem 3.10 |
| w/o 权重衰减 | 二次目标下停留在初值附近，不稀疏 | 验证 Example 3.8 / Theorem 3.9 |
| 直接 $\ell_p$ + 梯度裁剪 | 梯度上界与逼近误差不可同小 | 验证 Theorem 3.7 |
| 改变 $K,M$（Figure 1 热力图） | 蓝域为可优化、红域为大测试损失、白域为 $M>K-1$ 非法 | 给出超参选取区间 |

### 关键发现
- 三件套缺一不可：去掉自适应学习率会被零鞍点卡住，去掉权重衰减会失去稀疏性，去掉重参数化会重新撞上 $\ell_p$ 不可优化。
- Configuration A vs B：作者明确建议简单数据用 $\epsilon=0$（更激进的 $\ell_p$），复杂数据用 $\epsilon>0$（用一个温和的 $\ell_q\;(q>1)$ 充当稳定常数），与 Adam 的 $\epsilon$ 起到类似的"避免分母过小"作用。
- AdamW 已自带坐标自适应步长，叠加 ReWA 时设 $M=0,\epsilon\ne 0$ 即可避免重复的稀疏抑制。

## 亮点与洞察
- **把"算法 = 隐式正则"做得很显式**：通过精心设计的更新规则，把一个不可解的 $\ell_p$ 约束以可证明的方式嵌入到 SGD 的轨迹里——这种"用迭代格式实现非凸正则"的思路可以迁移到其他难解的非凸约束。
- **Theorem 3.7 的硬性不可能性结果很漂亮**：它说明"裁剪 $\ell_p$ 梯度"这条直觉路线在维度 $d$ 下永远会在稳定性和保真度之间二选一，从而正面论证为什么必须走重参数化路线，而不是简单地加 grad clip。
- **Configuration A/B 的区分有工程价值**：把超参选择直接绑到"数据集复杂度"上，给后续 LLM/扩散模型剪枝提供了拿来即用的食谱。

## 局限与展望
- 实验仅到 ImageNet + ResNet，未在 Transformer / LLM 量级上验证；当前 LLM 剪枝普遍依赖结构化稀疏（head / channel 级），而 ReWA 是非结构化稀疏。
- Theorem 3.3 假设 $M$ 为偶数（保证更新对称、便于分析），实际数值上 $M$ 可连续取值，但理论保证只到偶数；这一限制在论文附录 Remark C.3 中讨论但未根除。
- $K$ 增大会让乘法重参数化的数值条件变差（小数值的高次幂极易下溢），如何在 FP16 / BF16 训练下保持精度是工程上需要补的洞。
- 与 SCAD / MCP / adaptive Lasso 等非凸方法的实证比较只在附录 B 做了讨论性对比，缺少端到端的横评。

## 相关工作与启发
- **vs $\ell_1$ / LASSO**：$\ell_1$ 凸、好优化但有偏；ReWA 用 $\ell_p\;(0<p<1)$ 减小偏差，代价是必须引入重参数化才能稳定训练。
- **vs PowerPropagation (Schwarz et al., 2021)**：PowerPropagation 同样用 $\boldsymbol{y}^K$ 重参数化但不加权重衰减，只靠隐式偏置在小初始化下出稀疏；ReWA 用显式权重衰减把"小初始化假设"卸掉，并加上自适应步长解决零鞍点。
- **vs 直接 $\ell_p$ + grad clip**：本文 Theorem 3.7 给出硬性不可能性，正面否定了这条 baseline。
- **vs AdamW**：AdamW 通过 $1/\sqrt{v_t}$ 隐式做坐标自适应，可看作 ReWA 在 $M=0$ 时的近似；区别是 ReWA 还显式控制 $K,M$ 来强加 $\ell_p$ 偏置。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把已有 [Cp] 重参数化推到"自适应步长 + 显式衰减"的统一框架，并补完了零鞍点逃逸的理论缺口。
- 实验充分度: ⭐⭐⭐ CIFAR-10 / ImageNet + ResNet 足以验证主张，但缺 LLM 与 Transformer。
- 写作质量: ⭐⭐⭐⭐ 用一维玩具例子串起所有理论结果，Theorem 3.7 的不可能性论证简洁有力。
- 价值: ⭐⭐⭐⭐ 给"非凸稀疏正则可工程化"提供了一条干净的路线，对剪枝、压缩感知社区都有借鉴价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Theoretical and Empirical Analysis of Lehmer Codes to Search Permutation Spaces with Evolutionary Algorithms](../../AAAI2026/others/theoretical_and_empirical_analysis_of_lehmer_codes_to_search_permutation_spaces_.md)
- [\[NeurIPS 2025\] Overfitting in Adaptive Robust Optimization](../../NeurIPS2025/others/overfitting_in_adaptive_robust_optimization.md)
- [\[ICML 2026\] Torus Graphs for Large-Scale Neural Phase Analysis](torus_graphs_for_large_scale_neural_phase_analysis.md)
- [\[ICML 2025\] Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry](../../ICML2025/others/sparse_training_from_random_initialization_aligning_lottery_ticket_masks_using_w.md)
- [\[ICML 2026\] DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers](disjunctivenet_neural_symbolic_learning_via_differentiable_convexified_optimizat.md)

</div>

<!-- RELATED:END -->
