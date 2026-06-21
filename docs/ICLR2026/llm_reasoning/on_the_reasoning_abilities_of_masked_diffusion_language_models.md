---
title: >-
  [论文解读] On the Reasoning Abilities of Masked Diffusion Language Models
description: >-
  [ICLR 2026][Reasoning][掩码扩散模型] 本文首次给出掩码扩散语言模型（MDM）推理能力的形式化刻画：证明 MDM 在有限精度对数宽度设定下与「填充循环 Transformer（PLT）」严格等价，能模拟思维链（CoT）所能解的全部问题，并且在可并行问题（如正则语言）上比 CoT 严格更高效——揭示了 CoT 的「顺序性瓶颈」。
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "掩码扩散模型"
  - "并行推理"
  - "思维链"
  - "Transformer"
  - "电路复杂度"
---

# On the Reasoning Abilities of Masked Diffusion Language Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=BVnIsh4Nz1](https://openreview.net/forum?id=BVnIsh4Nz1)  
**代码**: 待确认（纯理论论文，无代码）  
**领域**: LLM推理 / 扩散语言模型 / 表达力理论  
**关键词**: 掩码扩散模型, 并行推理, 思维链, 循环 Transformer, 电路复杂度

## 一句话总结
本文首次给出掩码扩散语言模型（MDM）推理能力的形式化刻画：证明 MDM 在有限精度对数宽度设定下与「填充循环 Transformer（PLT）」严格等价，能模拟思维链（CoT）所能解的全部问题，并且在可并行问题（如正则语言）上比 CoT 严格更高效——揭示了 CoT 的「顺序性瓶颈」。

## 研究背景与动机
**领域现状**：自回归语言模型（如 GPT 系）逐 token 顺序生成，思维链（CoT）通过显式写出中间步骤来增强其表达力，是当今「推理模型」的核心机制。与之并行的另一条技术路线是掩码扩散模型（MDM）：从全掩码串出发，反复地「挑选若干位置 → 填入符号」，可以**并行**生成多个 token，在语言建模、代码、分子设计上已展现出与自回归模型抗衡的潜力。

**现有痛点**：尽管 MDM 工程上越来越火，但它「到底能算什么、算得多快」几乎无人从理论上说清。已有的 MDM 理论工作只盯着它**反向去噪过程的收敛性**——结论是即便对很简单的概率正则语言，去噪步数也必须随串长线性增长。但这类结论建立在「均匀去掩码 + 完美近似」的强假设上，且是渐近性的，很难对实际推理能力下具体判断。

**核心矛盾**：自回归 LM 的表达力理论已经很完善（落在 $\mathsf{AC}^0$ / $\mathsf{TC}^0$ 这些电路类里），但因为 MDM 的处理方式是**非顺序、可并行**的，这套理论无法直接搬过来——两个范式的理论研究基本是脱节的。我们缺一个既**形式严格**、又**贴合实际实现**的框架，来刻画 MDM 如何把「并行 + 迭代精化」组合起来做形式推理。

**本文目标**：(1) 给出 MDM 能可证求解的问题类及其效率；(2) 把 MDM 接到两个已被充分理解的推理框架——CoT 和 PLT——上，从而借用经典复杂度理论来定位 MDM。

**切入角度**：作者注意到，已有理论之所以悲观，是因为强行假设 MDM 只能**均匀随机**地选位置去掩码，这等于剥夺了它「先解哪个子问题」的自由。而真实的 MDM 实现（按置信度或学到的策略挑位置）恰恰拥有这种自由。只要允许模型**自己决定先解哪个简单子问题**，就能绕开难解部分，把任务分解成一串可并行求解的确定性步骤。

**核心 idea**：把 MDM 的反向过程抽象成「规划器（planner）决定解掩码哪些位置 + 预测器（predictor）填符号」两件事，放宽均匀去掩码假设，再证明这个理想化 MDM 与 PLT 等价、与 CoT 可互相（低效）模拟——于是 MDM 的能力边界被三类已知对象夹定。

## 方法详解

### 整体框架
这是一篇纯理论论文，"方法"即一套**表达力刻画框架**。整体思路：先把实际 MDM 抽象成一个可分析的理想化模型，再把这个模型分别与 PLT、CoT 两个参照系联系起来，最后借经典电路复杂度类（$\mathsf{AC}^d$、$\mathsf{NC}$、正则语言等）给出 MDM「能解什么、要多少步」的精确边界。

分析的统一设定是**有限精度、对数宽度**的 Transformer 族 $\{T_N\}$：每个参数/激活用固定比特表示，模型宽度随输入长度 $N$ 对数增长（这是表达力文献的标准假设，恰好够模型唯一区分各个位置）。在这个设定下，作者用两条互补的主线刻画 MDM：① 与 PLT 的等价（§3.1，用来接电路复杂度类）；② 与 CoT 的互模拟（§3.2，用来给出直观的效率对比与分离）。记号上，$\mathrm{MDM}[T,P]$ 表示「至多 $T$ 个去噪步、$P$ 个总输出/填充符号」的 MDM 能识别的语言类，PLT、pCoT 同理。

### 关键设计

**1. 理想化 MDM：规划器 + 预测器，松绑均匀去掩码**

已有 MDM 理论的悲观结论根源在于两条强假设——均匀去掩码（假设 2.1：随机均匀选位置）与完美近似（假设 2.2：Transformer 完美建模所有条件分布）。作者证明（定理 2.1）：若这两条同时成立，模型只能计算 $\mathsf{AC}^0$ 内的函数，因为均匀去掩码逼着模型「对任何子问题都一样擅长」，等价于要它一步直接预测最终答案，而单次 Transformer 前向落在 $\mathsf{AC}^0$ 里。

为此本文把反向过程拆成两个组件：**规划器**在每步决定解掩码哪些位置，**预测器**对这些位置采样符号。这松绑了假设 2.1——标准 MDM 不过是「规划器=均匀随机」的特例，而真实实现（按置信度挑、或学一个策略）正好对应一个非平凡规划器。更关键的是，作者还允许规划器**重采样已解掩码的位置**，从而克服 MDM 另一大软肋——无法回退、改正早期错误。理想化 MDM 不纠结训练目标，只问「允许重采样时，生成过程在理论上能做到什么」。附录还证明（Thm. C.1）规划器与预测器可融合成单一「按置信度去掩码」的模型，所以全部结论都适用于这种流行实现。

**2. MDM 与填充循环 Transformer 等价：接上电路复杂度的桥**

PLT 与 MDM 在直觉上高度相似：两者都**并行地迭代精化信息**——MDM 靠解掩码并预测离散符号，PLT 靠更新残差流；MDM 的去噪步天然对应 PLT 的循环，填充 token 对应 MDM 待生成的 token。定理 3.1 把这点形式化（PLT 需配备外部采样噪声以匹配 MDM 的随机性）：
$$\mathrm{MDM}[T,P]\subseteq \mathrm{PLT}[T,P],\qquad \mathrm{PLT}[T,P]\subseteq \mathrm{MDM}[T,(N+P)D].$$
其中 $D$ 是 PLT 的模型宽度。在 $D=O(\log N)$ 的设定下，二者只差填充长度的一个对数因子，于是（推论 3.1）对任意 $K\ge 1$，$\mathrm{MDM}[T,\tilde O(N^K)]=\mathrm{PLT}[T,O(N^K)]$。这座桥的价值在于：PLT 的表达力已被前人用电路类精确刻画，作者可直接搬运。结合 London & Kanade、Svete 等的结果即得——常数步 MDM 等于 $\mathsf{L\text{-}uniform}\,\mathsf{AC}^0$（推论 3.3，即不比普通 Transformer 强），而 $O(\log^d N)$ 步 + 多项式填充的 MDM 等于 $\mathsf{L\text{-}uniform}\,\mathsf{AC}^d$（推论 3.4），当 $d\to\infty$ 收敛到 $\mathsf{NC}$——全体可并行问题。一个更精细的构造（定理 3.2）还把正则语言压进 $\mathrm{MDM}[\log N, N]$，只要线性输出空间。

**3. MDM 与 CoT 可互相（低效）模拟：给出直观的能力夹逼**

PLT 缺乏实用实现，直觉不够强；作者再把 MDM 接到更流行的 CoT 上。直觉很简单：MDM 每步只解掩码**一个**符号，就退化成 CoT 式的顺序生成。更自然的对应物是 pCoT（并行 CoT，每步可并行预测多个符号）。定理 3.3：$\mathrm{pCoT}[T,P]\subseteq \mathrm{MDM}[T,P+(N+P)^2]$，填充上有**平方级**膨胀——这不是扩散过程本身的代价，而是「用非掩码注意力模拟掩码注意力」的代价（引理 D.14，本身是个独立有趣的结果）；若 MDM 的 Transformer 本就因果掩码，膨胀消失。反方向（定理 3.4）：$\mathrm{MDM}[T,P]\subseteq \mathrm{pCoT}[T,LT(P+N)]$，只有**线性**膨胀（$L$ 为层数，$T$ 来自每步要写出所有填充 token）。把两向串起来（推论 3.5），MDM 的能力被 CoT 上下夹定；并由「多项式步 CoT $=\mathsf{P}$」推出多项式步 MDM 仍落在 $\mathsf{P}$ 内。

**4. 顺序性瓶颈：在对数步预算下 MDM 严格强于 CoT**

前三点搭好框架后，最具冲击力的结论是一道**严格分离**。已知对数步 CoT 仍困在 $\mathsf{TC}^0$ 里，而定理 3.2 把正则语言放进了 $\mathrm{MDM}[\log N, N]$。结合「$\mathsf{TC}^0\ne \mathsf{NC}^1$」这一被广泛接受的假设、以及某些正则语言的 $\mathsf{NC}^1$-完全性，得到（推论 3.7）：
$$\mathrm{CoT}[\log N]\subsetneq \mathrm{MDM}[\log N, N].$$
也就是说，在对数次模型调用的预算下，MDM 能解 $\mathsf{NC}^1$-完全的正则语言，而 CoT 不能。作者把 CoT「无法利用并行结构」这一根本短板命名为**顺序性瓶颈**：CoT 逐步生成，无法把一个本可分解为独立子问题的任务并行求解，所以在可并行问题上需要线性步数，而 MDM 只要对数步。一个具体例子是状态追踪基准——MDM 对数步可解，CoT 需线性步。

> 框架↔关键设计一致性说明：上述四点正是分析主线的两条支柱（PLT 桥、CoT 桥）加上各自的「夹逼对象」（理想化 MDM 是被分析的主体，顺序性瓶颈是 CoT 桥得出的核心分离），术语在三处统一。

### 损失函数 / 训练策略
不适用——本文不训练任何模型，所有结论为可证定理，可由附录 D、E 的证明复现。

## 实验关键数据
本文无实证实验，"关键数据"即一组刻画表达力边界的定理。下面两张表汇总核心结果与对照。

### 主结果：MDM 的表达力定位

| 设定（去噪步数 / 输出空间） | MDM 等价 / 归属的类 | 来源 | 含义 |
|------|------|------|------|
| 常数步 $O(1)$，多项式输出 | $\mathsf{L\text{-}uniform}\,\mathsf{AC}^0$ | 推论 3.3 | 不比单次 Transformer 强，连正则语言都解不全 |
| $O(\log^d N)$ 步，多项式输出 | $\mathsf{L\text{-}uniform}\,\mathsf{AC}^d$ | 推论 3.4 | $d\to\infty$ 收敛到 $\mathsf{NC}$（全体可并行问题） |
| $\log N$ 步，线性输出 $N$ | 包含全部正则语言 | 定理 3.2 | 比推论 3.4 的构造更省输出空间 |
| 多项式步，多项式输出 | $\subseteq \mathsf{P}$ | 推论 3.6 | 不超出多项式时间可解问题 |

### 对照：MDM vs CoT / PLT

| 对照对象 | 关系 | 来源 | 解读 |
|------|------|------|------|
| MDM ↔ PLT | $\mathrm{MDM}[T,\tilde O(N^K)]=\mathrm{PLT}[T,O(N^K)]$ | 推论 3.1 | 差一个对数填充因子，本质等价 |
| pCoT → MDM | $\mathrm{pCoT}[T,P]\subseteq\mathrm{MDM}[T,P+(N+P)^2]$ | 定理 3.3 | MDM 能模拟 pCoT，代价平方填充 |
| MDM → pCoT | $\mathrm{MDM}[T,P]\subseteq\mathrm{pCoT}[T,LT(P+N)]$ | 定理 3.4 | pCoT 能模拟 MDM，代价线性 |
| CoT vs MDM（对数步） | $\mathrm{CoT}[\log N]\subsetneq\mathrm{MDM}[\log N,N]$ | 推论 3.7 | **严格分离**：MDM 在并行问题上更强 |

### 关键发现
- **并行不是万能**：在 $\mathsf{NC}\ne\mathsf{P}$ 假设下，电路值问题、线性规划、上下文无关文法成员判定、Horn-SAT 这些 $\mathsf{P}$-完全问题不能从 MDM 的并行中获益，本质上需要「CoT 式」逐步顺序求解——加上 MDM 无法维护 KV-cache 的开销，这类问题反而是自回归 CoT 更高效。
- **离散通信是瓶颈**：去噪步之间靠生成的离散文本传信息，限制了步间可传递的信息量，因此需要额外输出空间存中间计算——这与「固定精度 vs 对数精度 Transformer」、$\mathsf{AC}$ vs $\mathsf{TC}$ 的差异同源。
- **位置编码至关重要**：携带模型自身算不出的信息（如位置/长度的二进制、除法/取模结果）的 PE 是「循环能提升算力」的前提；定理 3.2 里 Transformer 实际是在 PE 里查预计算好的简单信息。

## 亮点与洞察
- **把火热的工程对象接进成熟理论**：MDM 此前缺乏表达力刻画，本文用「理想化 MDM ≡ PLT」一座桥，把整套电路复杂度结果一次性搬过来，是典型的「换个参照系就豁然开朗」。
- **顺序性瓶颈这一命名很有冲击力**：它把「CoT 为什么在可并行任务上吃亏」从直觉升级为可证分离（$\mathsf{TC}^0\ne\mathsf{NC}^1$ 下），并指出状态追踪这类具体基准 MDM 对数步可解、CoT 需线性步，对选模型有实际指导。
- **首个掩码↔非掩码注意力互模拟引理**（引理 D.13/D.14）本身可独立复用：说明用非掩码注意力模拟因果掩码要平方膨胀、反向只需线性，这解释了 MDM「看似更通用却不利于左到右处理」的代价来源。
- **可迁移思路**：「把生成过程拆成 planner + predictor 再单独分析每部分表达力」这套拆解，可迁移到分析其他非自回归/迭代精化生成器（如潜空间扩散 LM、状态空间模型实现的扩散）。

## 局限与展望
- **作者承认是纯理论 + 强理想化**：构造依赖「完美去掩码的 planner + 完美建模条件分布的 unmasker」。如何让**训练**出来的 planner 逼近这种理想行为，是学习性（learnability）而非表达力问题；而已知模算计数的敏感性（Hahn & Rofin 2024）暗示学习这类规划行为对 Transformer 很难。
- **缺乏实证验证**：本文不含实验，且关于「在形式语言上训练 MDM」的文献几乎空白，难以直接落地验证理论收益——作者把「设计针对性基准来实证 MDM 的并行优势」列为重要未来工作。
- **设定带来的可比性 caveat**：CoT 用因果掩码、MDM/PLT 用非掩码，这一不对称是为贴近各自主流实现而刻意保留的；不同范式的步数/输出空间预算不可直接比大小，结论要在同一设定下读。
- **自己发现的局限**：分离只在对数步预算（$\mathrm{CoT}[\log N]$）下严格证明；作者**猜测**多对数步下也有类似分离，但 $\mathrm{CoT}[\log^d N]$ 的表达力尚未形式化，属未证猜想。

## 相关工作与启发
- **vs 已有 MDM 收敛性理论（Feng et al. 2025; Li & Cai 2025）**：他们在「均匀去掩码 + 完美近似」下分析反向过程收敛，结论悲观（去噪步随串长线性增长）；本文指出正是均匀去掩码假设把模型钉死在 $\mathsf{AC}^0$，松绑后 MDM 表达力大增——是对前人结论根因的诊断与超越。
- **vs CoT 表达力理论（Li et al. 2024; Merrill & Sabharwal 2024）**：他们刻画 CoT Transformer 落在 $\mathsf{TC}^0$（对数步）/ $\mathsf{P}$（多项式步）；本文以此为参照系，反衬出 MDM 在可并行问题上的优势与顺序性瓶颈。
- **vs 循环/填充 Transformer（Saunshi et al. 2025; London & Kanade 2025; Svete et al. 2025）**：他们建立了 PLT ↔ $\mathsf{AC}^d$ 的刻画；本文通过 MDM≡PLT 直接复用这些结果，把 PLT 的能力地图整张搬到 MDM 上。
- **启发**：在「该用自回归 CoT 还是扩散」这一实际选型上，本文给了原则性判据——高度可并行、可分解的问题（正则语言、状态追踪）偏向 MDM；本质顺序、$\mathsf{P}$-完全的问题（CFG 成员、Horn-SAT）偏向 CoT；这也为「整块自回归 + 块内非自回归填充」的混合模型提供了理论动机。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个 MDM 表达力的形式化刻画，并提出 MDM≡PLT 与顺序性瓶颈两个全新结果
- 实验充分度: ⭐⭐⭐ 纯理论论文，定理体系完整自洽，但完全无实证验证（作者亦承认）
- 写作质量: ⭐⭐⭐⭐⭐ 用 takeaway + 复杂度类图把抽象结果讲得清晰可读
- 价值: ⭐⭐⭐⭐⭐ 为「扩散 LM vs 自回归 CoT」的范式选型提供了原则性判据

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization](improving_reasoning_for_diffusion_language_models_via_group_diffusion_policy_opt.md)
- [\[ICLR 2026\] Inpainting-Guided Policy Optimization for Diffusion Large Language Models](inpainting-guided_policy_optimization_for_diffusion_large_language_models.md)
- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)
- [\[ICLR 2026\] Variational Reasoning for Language Models](variational_reasoning_for_language_models.md)
- [\[ICLR 2026\] LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](ladir_latent_diffusion_enhances_llms_for_text_reasoning.md)

</div>

<!-- RELATED:END -->
