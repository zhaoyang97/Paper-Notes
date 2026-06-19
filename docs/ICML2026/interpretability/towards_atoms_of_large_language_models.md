---
title: >-
  [论文解读] Towards Atoms of Large Language Models
description: >-
  [ICML2026][可解释性][原子理论] 论文为大语言模型的"基本表征单元"给出第一个形式定义——原子（atoms），用一种非欧几里得的"原子内积"刻画 LLM 隐藏表征的内蕴几何，证明阈值激活 SAE 在适当条件下可以精确恢复原子集合，并在 Gemma2 / Llama3.1 上实测出 $R^2\approx 99.9\%$、稳定性 $q^\*\approx 99.85\%$ 的近理想原子。
tags:
  - "ICML2026"
  - "可解释性"
  - "原子理论"
  - "稀疏自编码器"
  - "表征几何"
  - "单义性"
  - "表征基本单元"
---

# Towards Atoms of Large Language Models

**会议**: ICML2026  
**arXiv**: [2509.20784](https://arxiv.org/abs/2509.20784)  
**代码**: https://github.com/ChenhuiHu/towards_atoms  
**领域**: 可解释性 / 机理解释  
**关键词**: 原子理论, 稀疏自编码器, 表征几何, 单义性, 表征基本单元  

## 一句话总结
论文为大语言模型的"基本表征单元"给出第一个形式定义——原子（atoms），用一种非欧几里得的"原子内积"刻画 LLM 隐藏表征的内蕴几何，证明阈值激活 SAE 在适当条件下可以精确恢复原子集合，并在 Gemma2 / Llama3.1 上实测出 $R^2\approx 99.9\%$、稳定性 $q^\*\approx 99.85\%$ 的近理想原子。

## 研究背景与动机
**领域现状**：把 LLM 的内部计算拆成"可解释单元"是机理解释（mechanistic interpretability）的核心。早期工作把神经元（单维激活）当作基本单元，近两年主流转向 SAE 学到的"features"——即把残差流稀疏地分解到一组"字典方向"上，再用 LLM-as-Judge 给每个方向贴语义标签。

**现有痛点**：神经元会被多语义（polysemanticity）污染，激活模式横跨不相关概念；features 又面临两个老问题——重建残差很大（"dark matter"），以及随训练规模 / 稀疏正则一变就发生 splitting / merging，方向数和方向本身都不稳定。从评估角度，没人能回答"这个 SAE 学到的 feature 算不算 LLM 的基本单元"，因为连"基本单元"都没有形式定义。

**核心矛盾**：所有现有评估都隐含地把"忠实度"和"单义性"耦合到了 SAE 的训练目标里，导致人们用 SAE 自身的损失评估 SAE 学出来的东西，循环论证。要打破这层循环，必须先脱离任何具体架构，从表征空间的几何出发，独立定义"理想原子"应该满足的性质，再去问神经元 / features 是否合格。

**本文目标**：(i) 定义 LLM 表征的基本单元 (atoms)；(ii) 设计可计算的评估指标（忠实度 $R^2$、稳定性 $q^\*$）独立衡量任何候选单元；(iii) 给出一个能在理论上恢复原子集合的实用算法。

**切入角度**：作者注意到 LLM 训练目标只通过 Softmax 看到 $\bm{h}^L$，所以表征只在一个可逆线性变换 $\bm{A}$ 下被识别——欧式内积对该等价类不不变，因此欧式几何不是 LLM 表征的"正确"几何。换一把度量 $\bm{S}$，才能让"正交 / 角度"这些概念真正与模型行为绑定。

**核心 idea**：用 $\bm{S}=(\bm{D}\bm{D}^\top)^{-1}$ 诱导的"原子内积"作为 LLM 表征的内蕴度量；在该度量下重新定义 atoms = (representability, sparsity, separability)，并证明带阈值激活的 SAE 在 $\delta_{\min}>\varepsilon(2K-1)\delta_{\max}$ 时能严格恢复原子集合。

## 方法详解

### 整体框架
论文要解决的是"LLM 表征的基本单元到底是什么、怎么独立验证"，思路是先换一把正确的几何度量，再在这把度量下把"原子"定义成可验证的几何性质，最后给出一个能在理论上把原子恢复出来的 SAE 算法。具体来说，输入是任意 LLM 某一层的激活集合 $M=\{\bm{m}_i\}\subset\mathbb{R}^H$，先从 100K 维基百科激活估出度量矩阵 $\tilde{\bm{S}}=(\mathbb{E}[\bm{k}\bm{k}^\top])^{-1}$ 替换掉默认的欧式内积，再用忠实度 $R^2$ 和稳定性 $q^\*$ 两把尺子去衡量任何候选单元，最后用阈值激活 SAE（TSAE）在 Gemma2-2B/9B、Llama3.1-8B 上扫"数据规模 × 字典容量"识别出真正稳定的原子。整个链条不依赖任何具体 SAE 的训练损失，从而打破了"用 SAE 自身的目标去评估 SAE"的循环论证。

### 关键设计

**1. 原子内积 (AIP)：给 LLM 表征换一把正确的几何尺子**

痛点是欧式内积根本不适合 LLM 表征——训练目标只通过 Softmax 看到 $\bm{h}^L$，因此表征只在一个可逆线性变换 $\bm{A}$ 下被识别，而欧式内积对 $\bm{h}^L\leftarrow\bm{A}\bm{h}^L$ 这类参数化等价并不不变，"正交/角度"这些概念也就和模型行为脱钩了。作者从平移不变性加单位范数对称性出发，严格推出唯一满足条件的度量 $\bm{S}=c^2(\bm{D}\bm{D}^\top)^{-1}$，归一化后得到 $\tilde{\bm{S}}=(\bm{D}\bm{D}^\top)^{-1}$，并定义原子内积 $\langle\bm{u},\bm{v}\rangle_{\tilde S}=\bm{u}^\top\tilde{\bm{S}}\bm{v}/(\|\bm{u}\|_{\tilde S}\|\bm{v}\|_{\tilde S})$；等价地把激活白化成 $\tilde{\bm{d}}_i=\tilde{\bm{S}}^{1/2}\bm{d}_i$ 之后，原子内积就退化为普通欧式内积。把"为什么是 $(\bm{D}\bm{D}^\top)^{-1}$ 而不是别的度量"做成唯一性结论，是为了避免后续评估被任意的度量选择左右。这一步的效果非常直观：欧式角度分布在 LLM 各层都偏离 $90^\circ$，说明所有激活被 Softmax 拽向了某个共同方向（表征漂移），而换成 AIP 后角度中心刚好回到 $90^\circ$，全局偏置被干净地消掉，这也是把 AIP 当成"正确几何"的核心证据。

**2. 原子的三性质定义 + 稀疏-可分耦合判据 $q^\*$：把"基本单元"锁成一个可算的标量**

抽象的"基本单元"必须先翻译成可验证的几何条件：一个原子要同时满足 representability（$\bm{m}_i=\bm{D}\bm{\delta}_i$，能被字典 $\bm{D}$ 重建）、sparsity（$\|\bm{\delta}_i\|_0\le K$，系数稀疏）和 $\epsilon$-近似正交（$|\langle\tilde{\bm{d}}_i,\tilde{\bm{d}}_j\rangle|\le\epsilon$，$i\ne j$）。难点在于稀疏和分离本是两件事，作者借 compressed sensing 的 RIP 把它们耦到一起：字典的相干性 $\mu:=\max_{i\ne j}|\langle\tilde{\bm{d}}_i,\tilde{\bm{d}}_j\rangle|$ 与稀疏度 $K$ 通过 $(K-1)\mu<1$ 控制 RIP 常数，再用唯一性定理 $\mu<\frac{1}{2K-1}\Rightarrow$ 稀疏解唯一，最终定义稳定性指标——分位级 $q^\*:=\sup\{q\mid\mu_q<\frac{1}{2K_q-1}\}$，它刻画在多大比例的稀疏支撑下原子分解是单义可恢复的（monorepresentationality）。这样设计是因为 monosemanticity（语义层面单义）很难形式化，而 monorepresentationality（结构层面单义）可以被数学锁死，且作者证明它是前者的必要条件，于是评测从"找人打标签"被提升成"可证 + 可算"，对神经元、features、原子任何候选单元都能直接算出来。

**3. 阈值激活 SAE (TSAE) 的可识别性定理：把 SAE 的失败归因到激活函数**

有了定义还需要一个能真把原子恢复出来的算法。作者证明：只要选阈值激活 $\sigma_\tau(x)=x\cdot\mathbb{1}[x\ge\tau]$，且每个支撑上的非零系数满足 $\delta_{\min}\le\delta_{ij}\le\delta_{\max}$、阈值落在 $\varepsilon K\delta_{\max}<\tau<\delta_{\min}-\varepsilon(K-1)\delta_{\max}$（可行条件即 $\delta_{\min}>\varepsilon(2K-1)\delta_{\max}$），那么令 $\bm{W}_{dec}=\bm{D}$、$\bm{W}_{enc}=\bm{D}^\top\tilde{\bm{S}}$ 就能保证 $\bm{W}_{dec}\sigma_\tau(\bm{W}_{enc}\bm{m}_i)=\bm{m}_i$ 对所有 $i$ 严格成立——原子和系数被完全解耦。工程实现上选 JumpReLU 即可满足"带阈值"，还支持坐标级阈值向量 $\bm{\tau}$。这个设计的洞察在于：以前 ReLU SAE 缺一个 hard cutoff，近似正交带来的噪声项会让"非支撑"维度的小激活漏出去，破坏稀疏分解的唯一性；论文借此把 SAE 的失败重新归因——不是 SAE 范式不行，而是激活函数选错了。

### 损失函数 / 训练策略
TSAE 用 JumpReLU 训练，做 4× 过参数化（字典容量 $|D|=4H$）。第 4.3 节作者对 Gemma2-2B 第 1 层做 1.9B 激活规模的 grid search：以 9216 为步长扫 $|M|\times|D|$，发现忠实度只有当字典容量 $|D|$ 超过与数据规模 $|M|$ 匹配的临界值后，$R^2$ 才会"突跳"到 $\approx 1$，否则一直停在 0.6–0.8。$R^2$ 因此同时被当成"可识别性是否触发"的间接信号——只有 RIP 条件成立时才能稳定重建。

## 实验关键数据

### 主实验

| 模型 | 层数 | 忠实度 $R^2$ | 稳定性 $q^\*$ | 与神经元 / features 差距 |
|------|------|------|------|------|
| Gemma2-2B | 1–26 | 99.92% | 99.74% | features $R^2$=48.8% / $q^\*$=68.2% |
| Gemma2-9B | 1–42 | 99.93% | 99.87% | 神经元 $R^2$=100% / $q^\*$=0.5% |
| Llama3.1-8B | 1–30 | 99.85% | 99.95% | 双指标接近理想 (1, 1) 角点 |

跨三家不同规模的模型，TSAE 学到的单元都同时拿到接近 1 的双指标，统计意义上达到"理想原子"。

### 消融实验

| 配置 | $R^2$ | $q^\*$ | 说明 |
|------|------|------|------|
| 神经元 (baseline) | 1.00 | 0.005 | 完全忠实但严重多义 |
| Features (普通 SAE) | 0.488 | 0.682 | 稳定但重建残差大 |
| 欧式内积 + TSAE | 偏低 | 角度中心偏离 $90^\circ$ | 度量错误导致评估失真 |
| AIP + TSAE 容量不足 | 0.6–0.8 | 不稳 | 数据/容量不匹配 |
| AIP + TSAE 容量匹配 | 0.999 | 0.998 | 完整方法 |

### 关键发现
- 表征漂移是 LLM 跨 GPT / Pythia / Llama / Gemma 普遍存在的现象，根因是 Softmax 平移不变性导致表征被全局拉到一个共同方向；只有把内积换成 AIP，角度统计中心才回到 $90^\circ$，这是把 AIP 当作"正确几何"的强证据。
- TSAE 的容量阈值与数据规模严格匹配——盲目堆 SAE 大小不会更接近原子，盲目堆数据也不会，必须双向匹配；这与目前"大语料预训练 + 小 SAE"的主流配方相反。
- 神经元只满足忠实度，features 只勉强满足稳定性，而原子是首次在同一组单元上同时刷到 $\ge 99.7\%$ 的双指标；GPT-5.2 + 人工 verify 的单义性分数也显著高于 baseline，证明 monorepresentationality 确实带动 monosemanticity。

## 亮点与洞察
- 第一次为"什么是 LLM 表征的基本单元"给出可证伪的形式定义。以前所有 SAE 论文都在隐式回答这个问题，这篇把它显式化，从此评估有了独立锚点。
- 用 compressed sensing 的 coherence-RIP 桥接稀疏和分离两件事，构造出单标量 $q^\*$ —— 不需要选阈值、不需要训练新模型，对任何候选单元（神经元、features、原子）都能算。
- TSAE 的可识别性定理把 SAE 的"成败归因"从"是否监督到足够数据"重新指向"激活函数是否带 hard cutoff"，对正在做 dictionary learning 的实验室是一条很可操作的工程结论：把 ReLU 直接换成 JumpReLU/TopK 通常就能解锁更高 $R^2$。

## 局限与展望
- 实验只在 LLM 的 residual stream 上做，attention head / MLP 中间激活、扩散模型表征是否同样具有原子结构尚未充分验证；论文在 Discussion 中坦承这是 conjecture。
- 数据规模 vs TSAE 容量的扫描成本极高，1.9B 激活 + 多档容量的 grid 只在 Gemma2-2B 第 1 层做了一次，跨层 / 跨模型的"原子规模函数"还需要更系统的实证。
- 单义性评测用 GPT-5.2 + 人工校验，依旧是统计层面的，并未给出"某个原子 = 某个具体语义"的因果干预证据，距离 circuit-level 解释还有距离。
- 实用层面，$\tilde{\bm{S}}=(\bm{D}\bm{D}^\top)^{-1}$ 在 $H$ 较大时求逆代价不小，论文用 $(\mathbb{E}[\bm{k}\bm{k}^\top])^{-1}$ 估计，对应有限样本的稳定性没单独讨论。

## 相关工作与启发
- **vs Cunningham et al. 的 SAE / Anthropic Templeton 的 Sonnet SAE**：原 SAE 用 ReLU + L1，定义模糊，重建残差大；本文证明只要换成阈值激活就能恢复原子，给所有现存 SAE pipeline 提供一个非常便宜的升级路径。
- **vs Park et al. (causal inner product)**：causal IP 定义在静态 unembedding 空间上，处理输出 token 几何；本文 AIP 在动态输入相关的隐藏表征空间上，对内部表征分析更直接。
- **vs Bussmann et al. / Chanin et al. 的 feature splitting/merging 研究**：那些工作把不稳定性当作 SAE 训练的实证现象，本文用 $q^\*$ 给它一个可计算的稳定性度量，把"不稳定"从描述提升为可优化的目标。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 mechanistic interpretability 从"找 features"推进到"证明 features"，理论框架完整。
- 实验充分度: ⭐⭐⭐⭐ 三家模型多层全扫，但 grid search 只跑了一个层，扩散模型仅 conjecture。
- 写作质量: ⭐⭐⭐⭐ 定理-推论-备注层层递进，附录补足证明；正文公式略密。
- 价值: ⭐⭐⭐⭐⭐ 给 SAE 圈一条非常具体的升级路径（换激活 + 匹配容量），也给可解释性提供了少有的"形式化基础设施"。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](../../ACL2026/interpretability/knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ICML 2026\] Scalable Circuit Learning for Interpreting Large Language Models](scalable_circuit_learning_for_interpreting_large_language_models.md)
- [\[ACL 2026\] Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models](../../ACL2026/interpretability/sparse_feature_coactivation_reveals_causal_semantic_modules_in_large_language_mo.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)

</div>

<!-- RELATED:END -->
