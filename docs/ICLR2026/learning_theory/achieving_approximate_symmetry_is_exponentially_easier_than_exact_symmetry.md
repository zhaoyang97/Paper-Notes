---
title: >-
  [论文解读] Achieving Approximate Symmetry Is Exponentially Easier than Exact Symmetry
description: >-
  [ICLR2026][学习理论][近似对称性] 这篇论文给"强制模型对称"这件事定义了一个可量化的代价——**平均复杂度（averaging complexity）**，并证明了一个指数级分离：在标准条件下，强制**精确对称**需要线性于群大小 $|G|$ 的查询次数，而强制**近似对称**只需对数级 $O(\log|G|/\varepsilon)$，从理论上首次解释了"为什么实践中近似对称往往比精确对称更划算"。
tags:
  - "ICLR2026"
  - "学习理论"
  - "几何机器学习"
  - "近似对称性"
  - "平均复杂度"
  - "群表示论"
  - "等变学习"
  - "傅里叶分析"
---

# Achieving Approximate Symmetry Is Exponentially Easier than Exact Symmetry

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=ncOJYFcleS](https://openreview.net/forum?id=ncOJYFcleS)  
**代码**: 待确认  
**领域**: 学习理论 / 几何机器学习  
**关键词**: 近似对称性, 平均复杂度, 群表示论, 等变学习, 傅里叶分析

## 一句话总结
这篇论文给"强制模型对称"这件事定义了一个可量化的代价——**平均复杂度（averaging complexity）**，并证明了一个指数级分离：在标准条件下，强制**精确对称**需要线性于群大小 $|G|$ 的查询次数，而强制**近似对称**只需对数级 $O(\log|G|/\varepsilon)$，从理论上首次解释了"为什么实践中近似对称往往比精确对称更划算"。

## 研究背景与动机
**领域现状**：几何机器学习的核心思路是把科学数据里观察到的结构（点云的置换对称、图谱方法的符号翻转对称、机器人任务的旋转对称、分子数据的各种对称）当作强归纳偏置喂给模型。把对称性**精确**编码进模型的手段很多：模型无关的有群平均（group averaging）、数据增强、规范化（canonicalization）、帧平均（frame averaging），模型相关的有群等变 CNN、等变 GNN、Deep Sets 等。这些方法被反复证明能改善样本复杂度和泛化。

**现有痛点**：精确对称在很多真实场景里其实"过头了"。医学影像里期望的反射对称并不完美，结果对这类变换只是**轻微**敏感；做对称性发现（symmetry discovery）时只有部分对称信息已知。此时硬塞精确不变性反而会**限制模型的普适性和表达能力**。于是实践者纷纷转向**近似对称**：允许模型在一定程度上违反对称，换来更好的灵活性、鲁棒性和半监督利用对称的能力——而且经验上确实有效。

**核心矛盾**：尽管近似对称在实验上屡屡奏效，**理论上却几乎是空白**——文献里既没有"达到近似对称到底有多难"的统一刻画，更没有把精确与近似两种做法**直接放在同一把尺子上比较**。直觉上近似对称是不变性的"放松版"，应该更容易达到，但没人把这个直觉变成定理。

**本文目标**：把模糊的"达到近似对称的复杂度"形式化，进而回答一个量化问题——给定一个模型，强制近似对称和强制精确对称的复杂度各是多少？两者之间存不存在本质的分离？

**切入角度**：作者假设学习者只能通过**黑箱方式**访问模型，并且只允许对模型做**线性后处理**——给定被预先选定的函数 $f$ 和群元素 $g$，oracle 返回变换后的函数 $x\mapsto f(gx)$，这一次交互叫一个**动作查询（action query, AQ）**。学习者用这些查询的响应做线性组合，目标是用尽量少的查询达到（近似）对称。查询次数就是它的"预算"，也是天然的复杂度度量。

**核心 idea**：用"达到对称所需的最少动作查询数（即最小平均方案的规模）"作为统一标尺，证明精确对称要 $\Theta(|G|)$、近似对称只要 $O(\log|G|/\varepsilon)$，得到指数级分离。

## 方法详解
这是一篇纯理论论文，"方法"即一套**定义 + 定理 + 证明思路**：先把"通过平均强制对称"的代价抽象成平均复杂度，再用群表示论 + 群上的傅里叶分析，分别给精确对称的下界和近似对称的上界，最后两者一夹得到指数分离。下面按"框架—关键设计"展开，因为是理论而非多模块 pipeline，不画 Mermaid。

### 整体框架
全文的逻辑链是：**①** 定义平均方案 $\omega:G\to\mathbb{R}$ 及其诱导的平均算子 $E_\omega$，把"对称化"统一成"对 $\{f(g^{-1}x)\}_g$ 做加权线性组合"；**②** 定义精确 / 弱近似 / 强近似三种对称，以及对应的平均复杂度 $\mathrm{AC}_{\mathrm{ex}}$、$\mathrm{AC}_{\mathrm{wk}}$、$\mathrm{AC}_{\mathrm{st}}$（达到该对称所需的**最小方案规模**）；**③** 对精确对称证下界 $\mathrm{AC}_{\mathrm{ex}}=|G|$（Thm 13，借助对称张量幂把函数类做大，再用表示论的不可约分解逼出"$\omega$ 必须是均匀分布"）；**④** 对近似对称证上界 $O(\log|G|/\varepsilon)$（Thm 15，用随机采样 + 随机矩阵算子范数的大偏差界）；**⑤** 把同一个函数类同时代入 ③④，得到指数级分离。

平均算子的形式是

$$(E_\omega[f])(x) := \sum_{g\in G}\omega(g)\,f(g^{-1}x),\qquad \|\omega\|_{\ell_1(G)}=\sum_{g}\omega(g)=1,$$

方案规模 $\mathrm{size}(\omega):=\#\{g:\omega(g)\neq 0\}$ 就是非零权重的个数，也就是要发的动作查询数。注意 $\omega$ **不依赖**具体的 $x$，否则就退化成（加权）帧平均、复杂度概念会失去意义——这是把问题框成"普适线性组合"的关键约束。

### 关键设计

**1. 平均复杂度：把"强制对称"这件事变成可数的查询预算**

痛点是文献里根本没有统一的"达到近似对称有多难"的度量，导致精确与近似无法对比。作者的做法是把对称化彻底**算子化 + 计数化**：任何想强制对称的方案都可以写成一组权重 $\omega(g)$，对 oracle 返回的变换函数 $f(g^{-1}x)$ 做线性组合；而"代价"就定义成达到目标所需的**最小非零权重数**。三种复杂度统一写成带约束的最小化问题：

$$\mathrm{AC}_{\mathrm{ex}}(F):=\min_\omega \mathrm{size}(\omega)\quad \text{s.t.}\quad (E_\omega[f])(gx)=(E_\omega[f])(x),\ \forall f,g,x.$$

这一定义之所以有效，是因为它**把对称性的几何/代数问题转成了一个组合优化问题**：要数清楚"最少几个群元素的加权平均能把任意 $f$ 对称化"，而这个最小值恰好可以用群表示论精确刻画。平凡上界是显然的——查询全部 $g\in G$（即均匀平均 $\omega\equiv 1/|G|$）一定能达到精确对称，所以三种复杂度都 $\le|G|$；真正的问题是何时能做到**亚线性**。

**2. 弱 / 强近似对称：用 $L^2$ 范数给"差一点对称"一个可调精度**

精确对称要求 $f(gx)=f(x)$ 处处成立，太刚。作者用 $L^2(X)$ 范数衡量"离对称有多远"，并用一个收缩因子 $\varepsilon>0$ 去压缩函数的"非对称分量"。**弱近似**只要求对称误差在群元素上**平均**意义下被压到 $\varepsilon$ 倍：

$$\mathbb{E}_g\!\!\int_X\!|(E_\omega[f])(x)-(E_\omega[f])(gx)|^2 d\mu \le \varepsilon\, \mathbb{E}_g\!\!\int_X\!|f(x)-f(gx)|^2 d\mu;$$

**强近似**则要求对**每一个** $g$ 都成立（把左边的 $\mathbb{E}_g$ 去掉、改成 $\forall g$）。$\varepsilon=0$ 时两者都退回精确对称。这个 $\varepsilon$-收缩的设计之所以关键，是它让"近似程度"成了一个**连续可调的旋钮**，从而能问"复杂度如何随 $\varepsilon$ 变化"；而且选 $L^2$ 范数不是随意的——它让结论能直接对接近似对称回归的泛化理论。Proposition 10 给出了这些量的基本性质：都随 $\varepsilon$ 非增，且 $\mathrm{AC}_{\mathrm{wk}}\le\mathrm{AC}_{\mathrm{st}}\le\mathrm{AC}_{\mathrm{ex}}\le|G|$，还有 $\mathrm{AC}_{\mathrm{wk}}(F,4\varepsilon)\le\mathrm{AC}_{\mathrm{st}}(F,4\varepsilon)\le\mathrm{AC}_{\mathrm{wk}}(F,\varepsilon)$——后者把强弱两种近似在常数因子内联系起来，于是后续只需主攻弱近似。

**3. 精确对称的线性下界：用对称张量幂 + 不可约分解逼出"必须均匀平均"**

要证精确对称"贵"，得先排除平凡函数类（常函数对任何群都已对称，复杂度为 1）。作者引入两个工具。其一是**忠实群作用假设**（Assumption 11）：对每个非单位元 $g$，存在 $f\in F$、$x\in X$ 使 $f(gx)\neq f(x)$，排除退化情形。其二是**对称张量幂** $\widetilde{\mathrm{Sym}}^{\otimes k}(F)=\bigoplus_{\ell=0}^k\mathrm{Sym}^{\otimes\ell}(F)$，它把基函数类 $F$ 扩成"最多 $k$ 个 $F$ 中函数的逐点乘积"张成的更大类——例如 $F$ 是 $\mathbb{R}^d$ 上线性函数时，$\widetilde{\mathrm{Sym}}^{\otimes k}(F)$ 正是总次数 $\le k$ 的多项式空间。Thm 13 断言：存在一个有显式闭式上界的整数 $K$，使得

$$\mathrm{AC}_{\mathrm{ex}}\!\big(\widetilde{\mathrm{Sym}}^{\otimes k}(F)\big)=|G|,\qquad \forall k\ge K,$$

且 $K\le\min\{|G|,\ \sum_{\lambda\in\Lambda}(M_\lambda-1)\}$，其中 $\Lambda$ 是所有 $\rho(g)$ 的特征值集合、$M_\lambda$ 是 $\lambda$ 作为某个 $\rho(g)$ 特征值的最大重数。证明思路（§5.1）很漂亮：由完全可约性，表示 $\rho$ 分解成不可约表示（irreps）的直和；把平均方案 $\omega$ 看成群上的信号，做傅里叶变换 $\widehat\omega(\pi)=\sum_g\omega(g)\pi(g)^*$。要让 $\widetilde{\mathrm{Sym}}^{\otimes k}(F)$ 上的平均产生精确对称，当且仅当对每个**非平凡**且在某 $\mathrm{Sym}^{\otimes k}(\rho)$ 中出现的 irrep 都有 $\widehat\omega(\pi)=0$。而对 $k=K$，**每个非平凡 irrep 都会在某个张量幂里现身**，于是 $\widehat\omega$ 除了平凡 irrep 处全为零——由傅里叶反演，$\omega$ 只能是均匀测度 $1/|G|$，这就逼出 $|G|$ 个非零权重，即线性复杂度。直觉是：函数类越大，要"杀掉"的非平凡频率越多，最终只剩下唯一选择——全群平均。

**4. 近似对称的对数上界：随机采样 + 零均值随机矩阵的算子范数集中**

要证近似对称"便宜"，作者用同一套傅里叶视角，但目标改成"构造 $\omega$ 使所有非平凡 irrep 的 $\|\widehat\omega(\pi)\|_{\mathrm{op}}\le\varepsilon$ 且 $\mathrm{size}(\omega)$ 小"。Thm 15 给出

$$\mathrm{AC}_{\mathrm{st}}(F,\varepsilon)=O\!\Big(\tfrac{\log|G|}{\varepsilon}\Big),\qquad \mathrm{AC}_{\mathrm{wk}}(F,\varepsilon)=O\!\Big(\tfrac{\log|G|}{\varepsilon}\Big),$$

隐藏常数至多 $8/3\approx 2.67$（弱）或 $32/3\approx 10.67$（强）。证法（§5.2）是**概率构造**：独立均匀采样 $n$ 个群元素，记其经验分布为 $\Omega$，拼成块对角矩阵 $\Xi:=\bigoplus_{\pi\ \text{非平凡}}\widehat\Omega(\pi)$。则 $\mathbb{E}[\Xi]=0$，且每个非平凡 $\pi$ 满足 $\|\widehat\Omega(\pi)\|_{\mathrm{op}}\le\|\Xi\|_{\mathrm{op}}$，于是问题归结为**控制一个零均值随机矩阵的算子范数**。标准大偏差界给出：只要 $n\ge c\,\log\dim(\Xi)/\varepsilon$（$c$ 为普适常数），就以高概率有 $\|\Xi\|_{\mathrm{op}}\le\varepsilon$；又因 $\dim(\Xi)\le|G|$，便得到 $O(\log|G|/\varepsilon)$。这个上界之所以强，是它**对所有函数类一致成立**、不依赖 Assumption 11 也不依赖张量幂——尤其包含 Thm 13 里那个需要线性复杂度的类。两者一夹：同一个函数类，精确要 $|G|$、近似只要 $O(\log|G|)$，**指数级分离**就此成立。Remark 16 进一步指出这个对数上界在常数意义下是紧的（存在实例需要至少 $\Omega_\varepsilon(\log|G|)$ 次查询）。

### 一个例子：置换群上的精确复杂度
取 $X=\mathbb{R}^d$、$G=S_d$ 按坐标置换作用、$F$ 为所有线性函数。此时 $\rho(g)$ 就是置换矩阵；若 $g$ 的循环长度为 $(\ell_1,\dots,\ell_t)$，则 $\rho(g)$ 的特征值是各 $\ell_j$ 次单位根。一个计数论证给出 $\sum_{\lambda\in\Lambda}M_\lambda=\sum_{q=1}^d\varphi(q)\lfloor d/q\rfloor=d(d+1)/2$（$\varphi$ 为欧拉函数）。于是只要多项式特征次数 $K=\tfrac{d(d+1)}{2}-1$，强制精确对称就已经需要满额的 $|G|=d!$ 次平均——而近似对称只要 $O(\log d!)=O(d\log d)$，差距是天文级的。

## 实验关键数据
这是纯理论论文，没有 benchmark 实验，只有一张**示意图**和一组**复杂度对照**支撑结论。

### 核心结论对照（理论界）

| 对称类型 | 平均复杂度 | 量级 | 出处 |
|---------|-----------|------|------|
| 精确对称 $\mathrm{AC}_{\mathrm{ex}}$ | $=|G|$（当 $k\ge K$） | 线性 | Thm 13 |
| 弱近似 $\mathrm{AC}_{\mathrm{wk}}(F,\varepsilon)$ | $O(\log|G|/\varepsilon)$，常数 $\le 8/3$ | 对数 | Thm 15 |
| 强近似 $\mathrm{AC}_{\mathrm{st}}(F,\varepsilon)$ | $O(\log|G|/\varepsilon)$，常数 $\le 32/3$ | 对数 | Thm 15 |
| 近似下界 | $\Omega_\varepsilon(\log|G|)$ | 对数（紧） | Remark 16 |

### 关键发现与性质

| 性质 | 内容 | 说明 |
|------|------|------|
| 单调性 | $\mathrm{AC}_{\mathrm{wk}}\le\mathrm{AC}_{\mathrm{st}}\le\mathrm{AC}_{\mathrm{ex}}\le|G|$ | 越强的对称越贵（Prop 10） |
| 强弱互通 | $\mathrm{AC}_{\mathrm{st}}(F,4\varepsilon)\le\mathrm{AC}_{\mathrm{wk}}(F,\varepsilon)$ | 常数因子内可互相转化，故只需研究弱近似 |
| 类越大越难 | $F_1\subseteq F_2\Rightarrow \mathrm{AC}(F_1)\le\mathrm{AC}(F_2)$ | 精确对称随函数类增长而变难 |

- **图 1 的直观验证**：对 100 个元素的 2D 旋转群，原始各向异性函数 $f(x,y)$ 只用 $|S|=5\approx\log(100)$ 个随机旋转做平均（近似对称），效果已经和用全部 $|S|=100$ 个旋转（精确对称）肉眼难分——经验上印证了"$|S|\approx\log|G|$ 就够"的理论。
- **指数分离的来源**：精确对称要求 $\widehat\omega$ 在所有非平凡频率处**精确归零**，这是个刚性约束，只有均匀平均能满足；近似对称只要把这些频率**压到 $\varepsilon$ 以下**，随机采样对数个元素即可，这是柔性约束。刚 vs 柔的差别正是 $|G|$ vs $\log|G|$ 的根源。
- **普适性**：近似上界不依赖忠实作用假设、不依赖张量幂构造，对任意有限维函数类、甚至复值函数都成立。

## 亮点与洞察
- **把"对称化代价"算子化 + 计数化**：平均复杂度这个度量最巧的地方是把几何/代数味很重的"强制对称"问题，转成"最少几个群查询"的组合量，再用群上傅里叶分析精确求解——度量定义本身就决定了证明能走通。
- **同类函数双向夹逼**：很多分离结果是"A 类难、B 类易"，容易被质疑不公平；这篇是在**完全相同的函数类**上同时给出 $|G|$ 下界和 $\log|G|$ 上界，分离无可辩驳。
- **"频率归零 vs 频率压小"的视角可迁移**：把平均方案看成群信号、用其傅里叶谱的稀疏性/算子范数刻画对称化能力，这套工具对数据增强、规范化等其他对称化手段的理论分析都可能直接复用（作者也明说这是独立兴趣点）。
- **理论解释了一个被广泛观察到的经验现象**：实践者早就发现近似对称又灵活又鲁棒，但说不清为什么"更省"。这篇给了"指数级更省"的硬答案，把工程直觉钉成定理。

## 局限与展望
- **只覆盖有限群**：所有结论建立在 $G$ 是有限群上，傅里叶分析、不可约分解、$|G|$ 计数都依赖有限性。作者承认推广到无限群（如连续旋转 $SO(d)$）既自然又困难，可能需要全新工具。
- **抽象模型与真实训练有距离**：平均复杂度刻画的是"黑箱 + 线性后处理"这种抽象交互下的查询代价，并不直接等于真实神经网络训练里的计算/样本成本；它是一个**下界/上界意义的抽象指标**，落到具体架构上的含义还需进一步桥接。
- **没有实证实验**：除图 1 的示意外，论文没有在真实数据集/模型上验证"用 $\log|G|$ 次平均的近似对称模型在下游任务上确实更好"，理论与端到端性能之间还差一段。
- **可延伸方向**：把这套框架用于分析数据增强（也是一种平均）的复杂度；把 $L^2$ 范数换成其他度量（附录 F 已有讨论）看分离是否依然成立；以及探究近似复杂度对 $\varepsilon$ 的依赖能否从 $1/\varepsilon$ 改善。

## 相关工作与启发
- **vs 帧平均 / 规范化（frame averaging, canonicalization）**：这些是具体的精确等变化手段；本文不提新方法，而是给所有"靠平均强制对称"的手段建了一把统一的复杂度尺子，并指出精确手段在抽象模型下天然要付 $|G|$ 的代价。
- **vs 软等变 / 部分等变 / 近似等变 MDP 等近似对称方法**（Finzi 2021、Kim 2023、Romero & Lohit 2022、Park 2025 等）：这些工作经验性地展示近似对称"更灵活更鲁棒"；本文补上理论侧——首次证明近似比精确**指数级更易达成**，为这一整条经验路线提供了原理性辩护。
- **vs 对称下学习难度的理论**（Kiani 2024 等"hardness of learning under symmetries"）：同属对称性理论，但视角不同——前者关心"在对称约束下学习有多难"，本文关心"把对称强制进模型这件事本身有多难（要多少次平均查询）"，并给出精确/近似的指数分离。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把精确与近似对称放进同一复杂度框架并证明指数分离，平均复杂度这一度量本身是新的
- 实验充分度: ⭐⭐⭐ 纯理论论文，只有一张示意图，无下游实证，但理论自洽且给出紧界
- 写作质量: ⭐⭐⭐⭐ 定义—定理—证明思路层层递进，傅里叶视角讲得清晰，附录支撑充分
- 价值: ⭐⭐⭐⭐⭐ 为"近似对称为何更好用"这一广泛经验现象提供了硬核理论解释，工具可迁移到更广的几何机器学习分析

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking](to_augment_or_not_to_augment_diagnosing_distributional_symmetry_breaking.md)
- [\[ICLR 2026\] Reducing Symmetry Increase in Equivariant Neural Networks](reducing_symmetry_increase_in_equivariant_neural_networks.md)
- [\[ICLR 2026\] Branch and Bound Search for Exact MAP Inference in Credal Networks](branch_and_bound_search_for_exact_map_inference_in_credal_networks.md)
- [\[ICLR 2026\] Expressive Power of Implicit Models: Rich Equilibria and Test-Time Scaling](expressive_power_of_implicit_models_rich_equilibria_and_test-time_scaling.md)
- [\[ICLR 2026\] Better Bounds for the Distributed Experts Problem](better_bounds_for_the_distributed_experts_problem.md)

</div>

<!-- RELATED:END -->
