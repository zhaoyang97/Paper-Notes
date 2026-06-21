---
title: >-
  [论文解读] Is Spurious Correlation Removal Always Learnable?
description: >-
  [ICML 2026][学习理论][谬误相关] 这篇论文证明：去除谬误相关（spurious correlation）即使在不变结构"统计上可识别"的理想情形下，也可能"计算上不可学习"——存在一族多环境实例，穷举搜索用多项式样本就能恢复不变方向，但任何多项式时间算法要达到常数精度都会反推出一个被广泛相信困难的稀疏恢复问题；同时论文用一个"环境多样性"参数 $\gamma$ 刻画了可识别性、minimax 率与样本复杂度相变。
tags:
  - "ICML 2026"
  - "学习理论"
  - "不变学习"
  - "计算复杂性"
  - "谬误相关"
  - "计算-统计鸿沟"
  - "环境多样性"
  - "minimax 率"
---

# Is Spurious Correlation Removal Always Learnable?

**会议**: ICML 2026  
**arXiv**: [2606.12930](https://arxiv.org/abs/2606.12930)  
**代码**: 待确认  
**领域**: 学习理论 / 不变学习 / 计算复杂性  
**关键词**: 不变学习, 谬误相关, 计算-统计鸿沟, 环境多样性, minimax 率

## 一句话总结
这篇论文证明：去除谬误相关（spurious correlation）即使在不变结构"统计上可识别"的理想情形下，也可能"计算上不可学习"——存在一族多环境实例，穷举搜索用多项式样本就能恢复不变方向，但任何多项式时间算法要达到常数精度都会反推出一个被广泛相信困难的稀疏恢复问题；同时论文用一个"环境多样性"参数 $\gamma$ 刻画了可识别性、minimax 率与样本复杂度相变。

## 研究背景与动机

**领域现状**：机器学习模型常常依赖只在训练分布里成立、换个分布就崩的谬误相关（比如靠背景颜色而非物体本身分类）。不变学习（Invariant Learning，如 IRM、REx）试图利用"多环境"数据，只保留那些"与标签的预测关系跨环境稳定"的特征，理论上已经证明在一定多样性假设下，这种不变结构是**统计可识别**的。

**现有痛点**：可识别归可识别，实践里不变方法却时灵时不灵——同一套方法在某些数据集上明显有用，换个数据集又退化到和 ERM（经验风险最小化）差不多。社区往往把这种不一致归咎于"优化没调好"或"环境数不够多"。

**核心矛盾**：作者追问的是一个更根本的问题：这种不一致，会不会**即使在不变性完全可识别的理想情形下也无法避免**？也就是说，失败可能不是优化问题，而是**计算本身的硬度**——总体目标函数有唯一正确的最大值点，但没有多项式时间算法能找到它。

**本文目标**：(1) 在一个干净的线性-高斯设定里，证明谬误相关去除存在"计算可行性 ≠ 统计可行性"的分离；(2) 找到一个能解释"什么时候好学、什么时候不好学"的结构量；(3) 给出这个量与可识别性、最优估计率、样本复杂度阈值之间的定量关系。

**切入角度**：把"去谬误相关"形式化为从多环境数据里**恢复不变子空间** $V_{\mathrm{inv}}$。这一步让问题落进了一个有成熟工具的框架：一边可以接平均情形复杂性（planted clique / 稀疏恢复的困难性假设），证下界；另一边可以接 Grassmann 流形上的 minimax 理论，给上界。

**核心 idea**：用一句话概括——**把一个被广泛相信困难的稀疏恢复 primitive"包装"成合法的多环境谬误相关实例**，从而把它的计算困难性"搬运"到不变子空间恢复上；再用一个标量"环境多样性" $\gamma$ 把统计侧的可识别性与样本率串起来。

## 方法详解

### 整体框架

论文不是提一个新算法，而是给"谬误相关去除"画出一张**可行性地图**，由两根支柱撑起。

第一根是**计算下界**（第 3 节）：从一个黑盒的、可采样的"有监督稀疏恢复 primitive"出发（其困难性由 planted clique / 稀疏 CCA 类的平均情形归约启发），构造一族多环境谬误相关实例，使得不变方向**统计上可被穷举搜索恢复**，但**多项式时间恢复会与该 primitive 的困难性矛盾**。关键约束是：构造必须是"可采样的"——能在随机多项式时间里、**不接触隐藏稀疏方向或其支撑**的前提下生成样本，否则困难性无法搬运。

第二根是**统计上界**（第 4 节）：在线性-高斯结构下，刻画 $V_{\mathrm{inv}}$ 何时可识别、估计误差的 minimax 率是多少。核心量是 Definition 2.3 引入的**环境多样性** $\gamma$——它度量不同环境对谬误相关的扰动差异有多大。多样性不够 $V_{\mathrm{inv}}$ 就不可识别（哪怕无限样本），多样性足够则问题良态，且存在锐利的有限样本相变。

第二根支柱里还有一个反直觉的信息：**环境的"数量"不如它们的"多样性"重要**——少数几个差异大的环境，可能比一大堆彼此雷同的环境更有用。

### 关键设计

**1. 把谬误相关去除形式化为线性-高斯不变子空间恢复**

要谈"可不可学"，先得把问题钉成一个有数学骨架的对象。论文把谬误相关（SC）实例（Definition 2.1）定义为：把 $\mathbb{R}^d$ 正交分解成不变子空间 $V_{\mathrm{inv}}$（维度 $k$）与谬误子空间 $V_{\mathrm{sp}}$，输入拆成 $X = X_{\mathrm{inv}} + X_{\mathrm{sp}}$，并满足两条：(C1) 条件机制 $\mathbb{P}_e(Y \mid X_{\mathrm{inv}})$ 跨环境**完全一致**（这才叫"不变"）；(C2) $\mathbb{P}_e(X_{\mathrm{sp}} \mid X_{\mathrm{inv}})$ 至少在某对环境间**不同**（这才叫"谬误"——它随环境漂移）。在统计分析里再加上标准线性-高斯结构 $Y = \langle w^*, X_{\mathrm{inv}} \rangle + \epsilon$。

这个形式化的价值在于：它把"去谬误相关"这件含糊的事，变成了"在 Grassmann 流形 $\mathrm{Gr}(k,d)$ 上找那个唯一让预测机制跨环境不变的 $k$ 维子空间"。评价指标用子空间距离 $\operatorname{dist}(V, V') := \|P_V - P_{V'}\|_F$，对轴对齐情形还报特征选择准确率。有了这个干净对象，下界和上界才能各自接上成熟工具。

**2. 可采样困难实例族构造：把稀疏恢复 primitive 包进 SC 实例，证出计算-统计鸿沟**

这是全文的技术核心。论文假设一个黑盒**可采样有监督稀疏恢复 primitive**（Hypothesis 3.2）：存在一个随机多项式时间采样器 $\mathcal{R}$，给定平均情形输入 $A$，吐出 i.i.d. 样本 $(Z_t, Y_t)$；planted 情形下有一个未知 $s$-稀疏单位方向 $v$，$Y$ 只通过 $v^\top Z$ 依赖 $Z$，且 $v$ 是唯一的稀疏预测方向（总体分数 $\Phi_P$ 在 $v$ 处有常数 margin $c_{\mathrm{prim}}$）；最关键的是 (iv)(v)：任何能从多项式样本里恢复出与 $v$ 有常数重叠的多项式时间算法，都能反解底层平均情形稀疏恢复问题，而后者被假设是困难的。

Construction 3.3 把这个 primitive **包装**成 SC 实例：每个环境独立跑 $\mathcal{R}(A)$ 得到 $(Z_t^{(e)}, Y_t^{(e)})$，然后拼出输入

$$X^{(e,t)} = (Z_t^{(e)},\, W^{(e,t)}), \qquad W^{(e,t)} = \mu^{(e)} Y_t^{(e)} + \eta^{(e,t)},$$

令 $V_{\mathrm{inv}} = \mathrm{span}\{(v, 0)\}$。其中谬误块 $W^{(e)} = \mu^{(e)} Y + \eta^{(e)}$ 用环境相关的系数 $\mu^{(e)}$ 制造跨环境漂移，满足 (C2)；而 $Y$ 只通过 $v^\top Z$ 依赖 $Z$，保证 (C1) 成立（Lemma 3.4）。整个构造**只用对 $\mathcal{R}(A)$ 的独立调用和公开高斯噪声，从不碰隐藏方向**，所以是可采样的。

为了让 $V_{\mathrm{inv}}$ 在所有候选里被唯一识别，论文定义了一个**不变性-预测性联合分数** $S(V) = A(V) - \lambda T(V)$：$A(V)$ 继承自 primitive 的预测分数 $\Phi_P$（衡量"能不能预测 $Y$"），$T(V) = \max_{e \neq e'}(\theta_V^{(e)} - \theta_V^{(e')})^2$ 用环境间协方差差异**惩罚跨环境漂移的谬误方向**（衡量"跨环境稳不稳"）。$\lambda$ 取得足够大，使非不变方向被罚得比它带来的预测增益更多。Lemma 3.5 证明 $V_{\mathrm{inv}}$ 在 $S(V)$ 上有一致的常数 margin $c_{\mathrm{mar}}$。

由此得到鸿沟（Theorem 3.8 / 3.9）：(a) $V_{\mathrm{inv}}$ 是 $S(V)$ 的唯一最大值点，**可识别**；(b) 穷举搜索（在 $\mathrm{Gr}(1,d)$ 的 $\epsilon$-网上极大化经验 $S(V)$）用多项式样本 $N = \tilde{O}(\mathrm{poly}(m)/c_{\mathrm{mar}}^2)$ 就能恢复（Lemma 3.6）；(c) 但任何多项式时间算法若达到 $\operatorname{dist}(\hat V, V_{\mathrm{inv}}) \leq \delta_0$，把恢复的子空间投回 primitive 块就得到 $v$ 的常数重叠恢复（Lemma 3.7），从而反解困难问题，矛盾。特别地，**即便 $k=1$**（一维不变方向）这个最简单的情形，鸿沟依然存在；且 Remark 3.10 指出**只需 2 个环境**就够，困难性不来自环境多。

**3. 环境多样性参数 $\gamma$ 与可识别性**

统计侧的主角是多样性 $\gamma$（Definition 2.3）：对谬误坐标 $j$，记 $\mu_{e,j}(x) = \mathbb{E}_e[X_{\mathrm{sp},j} \mid X_{\mathrm{inv}} = x]$，定义环境间极差 $\Delta_j(x) = \max_e \mu_{e,j}(x) - \min_e \mu_{e,j}(x)$，则

$$\gamma := \big(\mathbb{E}[\max_j \Delta_j(X_{\mathrm{inv}})^2]\big)^{1/2}.$$

在均值漂移设定下它简化为 $\gamma = \max_{e,e'} \|\mu_{\mathrm{sp}}^{(e)} - \mu_{\mathrm{sp}}^{(e')}\|_\infty$——直白说就是"不同环境把谬误相关推得有多开"。$\gamma$ 直接决定可识别性：当 $\gamma = 0$（环境对谬误条件均值不产生可区分变化）时，**$V_{\mathrm{inv}}$ 哪怕用无限数据也不可识别**（Corollary 4.7）；$\gamma > 0$ 且方向足够丰富时恢复才变得可能。论文还给出实证代理 $\hat\gamma$（按各特征与标签相关性的环境间极差取中位数），方便在真实数据/学到的表示上直接诊断。

但论文诚实地强调了 $\gamma$ 的局限（Remark 2.4）：$\gamma$ 是**坐标级**标量，对轴对齐特征好用，却不保证"每个谬误方向都被某个环境差异暴露"。真正的子空间可识别性需要**方向丰富性**，由环境差矩阵 $\mathcal{F} = \sum_{e<e'}(\mu_{\mathrm{sp}}^{(e)} - \mu_{\mathrm{sp}}^{(e')})(\cdot)^\top$ 的最小特征值 $\lambda_{\min}(\mathcal{F})$ 把关。Proposition 4.4 给出充分条件：当 $\mathrm{rank}(\mathcal{F}) = d - k$（即需要 $|\mathcal{E}| \geq d - k + 1$ 个环境）时 $V_{\mathrm{inv}}$ 一般可识别。

**4. minimax 率与相变：把多样性翻译成样本复杂度的 $1/\gamma^2$**

在充分多样性 + 局部高斯正则（Assumption 4.10，含不变性 gap 的局部二次曲率与 LAN 正则）下，论文给出最优估计率（Theorem 4.11）：

$$\mathbb{E}[\operatorname{dist}(\hat V, V_{\mathrm{inv}})^2] = \Theta\!\left(\frac{k(d-k)}{n|\mathcal{E}|}\right),$$

其中 $k(d-k)$ 是 $k$ 维子空间在 $\mathbb{R}^d$ 中的内在自由度，$n|\mathcal{E}|$ 是各环境样本汇总后的有效总量。进一步在标签诱导漂移设定（Assumption 4.14，要求 $\lambda_{\min}(\mathcal{F}) \geq c_{\mathcal{F}} \gamma^2$）下，存在**锐利相变**（Theorem 4.15）：临界样本量

$$n^* \propto \frac{k(d-k)}{|\mathcal{E}|\,\gamma^2},$$

超临界（$n > 2n^*$）时误差 $\lesssim k(d-k)/(n|\mathcal{E}|\gamma^2)$，亚临界（$n < n^*/2$）时任何估计量误差都被常数下界卡住。这个 $1/\gamma^2$ 缩放把"多样性"翻译成了"省多少样本"——多样性翻倍，所需样本约降到四分之一，定量印证了"几个差异大的环境胜过一堆雷同环境"。

此外第 5 节给出四类**可处理性条件**（稀疏不变结构 / 谱分离 / 低阶矩可识别 / 环境内不相关），在这些结构下多项式时间算法就能达到 $\tilde O(k(d-k)/(n|\mathcal{E}|))$，并解释了为什么 ColoredMNIST/Waterbirds 这类"低维强漂移"数据集对矩方法/分组方法友好，而 DomainBed 式"高维细微漂移"会削弱这些条件。

## 实验关键数据

实验目的不是刷 SOTA，而是验证三个理论信号：(i) 合成困难实例上的计算-样本鸿沟；(ii) 相变与多样性阈值；(iii) 标准谬误相关基准上的行为。

### 主实验：合成困难实例上的计算-样本鸿沟

| 现象 | 穷举搜索（指数时间） | 最佳多项式时间方法 | 理论解释 |
|------|---------------------|-------------------|----------|
| 达到相同特征选择准确率所需样本 | 远少 | 远多 | Theorem 3.9 的鸿沟 |
| $\gamma = 0$ | 不可恢复（无限样本也不行） | 同左 | Corollary 4.7 不可识别 |
| 样本越过 $n^*$ | 误差骤降 | — | Theorem 4.15 相变 |

合成实验（Figure 1/2）显示：在 planted 困难实例上，穷举搜索用远少于任何多项式时间方法的样本就达到同等准确率，直观展示了"统计可行、计算不可行"的鸿沟；随样本越过 $n^* \propto k(d-k)/(|\mathcal{E}|\gamma^2)$ 出现准确率的相变跳变；多样性 $\gamma$ 越大阈值越低。

### 真实基准：ColoredMNIST（$\gamma = 0.4$）

| 训练总量 $N$ | 方法 | OOD 准确率 | 与 Oracle 差距 |
|------|------|-----------|----------------|
| 500 | Oracle（灰度） | $.820 \pm .012$ | — |
| 500 | Adv-Color | $.745 \pm .018$ | 0.075 |
| 500 | IRM | $.720 \pm .022$ | 0.100 |
| 500 | ERM | $.690 \pm .025$ | 0.130 |
| 2000 | Oracle（灰度） | $.930 \pm .008$ | — |

ColoredMNIST 上各方法与 Oracle 的差距随样本量增大而收缩，与相变行为一致：样本不足时即便用 IRM 也明显逊于"看不到颜色"的 Oracle，样本充足后差距闭合。

### 关键发现
- **鸿沟在 $k=1$ 就出现**：困难性不需要高维不变子空间，最简单的一维情形已经计算困难——这说明问题的硬度是本质的，不是维度堆出来的。
- **多样性 > 数量**：$\gamma = 0$ 时无限样本也学不出来，$\gamma$ 越大临界样本量按 $1/\gamma^2$ 下降；少数差异大的环境胜过一堆雷同环境。
- **失败可能是计算硬度而非调参**：作者建议在把不变学习的失败归因于优化/算法之前，**先估计环境多样性 $\hat\gamma$**——若多样性本就不足，再怎么调也无济于事。

## 亮点与洞察
- **"可识别 ≠ 可学习"的干净反例**：以往讨论不变学习多停在"统计可识别"，本文用可采样归约造出一族"可识别但多项式时间不可恢复"的实例，把计算-统计鸿沟首次清晰落到谬误相关去除上——这是最让人"啊哈"的地方。
- **可采样性是搬运困难性的关键技巧**：构造刻意只用对 primitive 的黑盒调用 + 公开噪声、绝不碰隐藏方向，才使 planted-clique 类困难性能合法地"搬"到 SC 实例上；这个"包装"模式可迁移到其他"看似可识别"的表示学习/因果发现任务，去检验它们是否也藏着计算硬度。
- **$\gamma$ 既是理论量也是可落地诊断**：从总体定义 $\gamma$ 到实证代理 $\hat\gamma$（相关性环境间极差的中位数），给了从业者一个先验筛子——动手训不变模型前先量一量环境多样性。
- **下界与上界配对成"地图"**：第 3 节说"最坏情形下不可学"，第 5 节立刻补上"这四类结构下可学"，避免读者误以为不变学习全盘悲观。

## 局限与展望
- **困难性是条件性、最坏情形的**：整个下界挂在 Hypothesis 3.2 这个黑盒 primitive 上，论文也明确它"受 planted-clique/稀疏 CCA 归约启发，但不声称直接由现有归约推出"——所以是一个有待进一步坐实的困难性假设，而非无条件定理。
- **线性-高斯设定理想化**：真实数据的谬误相关远比线性-高斯复杂，$\gamma$ 作为坐标级标量在旋转/纠缠特征下需要换成谱量 $\lambda_{\min}(\mathcal{F})$，否则可能高估可识别性（Remark 2.4 已自陈）。
- **上界估计量多为低效/指数时间**：Theorem 4.11 的上界由"低效估计量"达到，真正多项式时间达到最优率要落到第 5 节那四类结构性假设里，普适的高效算法仍缺。
- **改进方向**：把 primitive 假设替换成更被广泛接受的标准困难性、把 $\gamma$ 推广到非线性/学习表示上的可识别性度量、以及设计能自适应利用"可处理性条件"的实用算法。

## 相关工作与启发
- **vs IRM / REx 等不变学习方法**：它们提出具体目标函数去逼近不变结构，本文不提算法而是问"这类目标的总体最优解能否被高效找到"，给出了"有时不能"的硬度回答，从而把"方法失灵"的部分原因从优化转向计算。
- **vs Rosenfeld et al. 的可识别性理论**：前者证明不变结构在多样性假设下统计可识别，本文承接这一线，却指出"统计可识别"与"计算可学习"之间存在分离——可识别只是必要不充分。
- **vs planted clique / 稀疏 CCA 困难性**：本文把这一脉平均情形复杂性工具"搬运"到表示学习场景，是方法论上的迁移；其可采样包装思路对因果发现、OOD 检测等"看似可识别"的问题都有借鉴意义。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把计算-统计鸿沟清晰落到谬误相关去除上，$k=1$/双环境即困难的反例很扎实。
- 实验充分度: ⭐⭐⭐ 理论为主，合成 + ColoredMNIST 验证够用但规模有限，真实基准覆盖偏窄。
- 写作质量: ⭐⭐⭐⭐ 下界/上界配成"可行性地图"，对 $\gamma$ 局限与 primitive 条件性都很诚实。
- 价值: ⭐⭐⭐⭐ 给"不变学习为何时灵时不灵"提供了计算硬度视角，并落出"先量多样性"的可操作诊断。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Estimating Correlation Clustering Cost in Node-Arrival Stream](estimating_correlation_clustering_cost_in_node-arrival_stream.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ICML 2025\] Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions](../../ICML2025/learning_theory/sparse-pivot_dynamic_correlation_clustering_for_node_insertions.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/learning_theory/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)

</div>

<!-- RELATED:END -->
