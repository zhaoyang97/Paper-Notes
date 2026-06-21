---
title: >-
  [论文解读] Softmax Transformers are Turing-Complete
description: >-
  [ICLR2026][学习理论][图灵完备] 本文首次证明带思维链（CoT）的 **softmax** 注意力 Transformer 是图灵完备的，且这种构造还自带长度泛化保证——关键技巧不是去硬模拟图灵机的读写头，而是用 softmax 自带的"计数"能力（经由 C-RASP）去模拟 Minsky 计数机，并辅以一个与任务无关的相对位置编码（RPE）把任意输入编码成数字。
tags:
  - "ICLR2026"
  - "学习理论"
  - "Transformer 表达力"
  - "形式语言"
  - "图灵完备"
  - "softmax 注意力"
  - "思维链"
  - "C-RASP"
  - "计数机"
---

# Softmax Transformers are Turing-Complete

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=FdkPOHlChS](https://openreview.net/forum?id=FdkPOHlChS)  
**代码**: 待确认  
**领域**: 学习理论 / Transformer 表达力 / 形式语言  
**关键词**: 图灵完备, softmax 注意力, 思维链, C-RASP, 计数机

## 一句话总结
本文首次证明带思维链（CoT）的 **softmax** 注意力 Transformer 是图灵完备的，且这种构造还自带长度泛化保证——关键技巧不是去硬模拟图灵机的读写头，而是用 softmax 自带的"计数"能力（经由 C-RASP）去模拟 Minsky 计数机，并辅以一个与任务无关的相对位置编码（RPE）把任意输入编码成数字。

## 研究背景与动机

**领域现状**：用形式语言理论刻画 Transformer "能算什么、不能算什么"是近年的热点。其中最根本的问题是：带 CoT 的 Transformer 是否图灵完备？此前 Pérez et al. (2021)、Merrill & Sabharwal (2024)、Li & Wang (2025) 等一系列工作都给出了肯定回答。

**现有痛点**：但所有这些图灵完备性证明都依赖 **hardmax（硬注意力）**。硬注意力是个相当不现实的假设——它本质上是"挑出某个位置"的离散操作，**没有可训练性保证**（梯度无法良好传播）。而真实 Transformer 用的是 softmax。一个长期开放的问题是：换成 softmax，CoT Transformer 还图灵完备吗？

**核心矛盾**：硬注意力证明之所以好用，是因为它能直接"定位"图灵机读写头的位置（用 $\langle x, y\rangle$ 形式的平均硬注意力，或 layer norm）。但 softmax 是个软的、归一化的加权平均，**没法精确抽取出某个单一位置**——更一般地，softmax 能否模拟平均硬注意力本身就是开放问题。所以"直接用 softmax 模拟图灵机读写头"这条路走不通。

**本文目标**：(i) 首次证明 softmax CoT Transformer 图灵完备；(ii) 同时给出长度泛化保证，即只用有限长度的训练数据，学到的模型能正确外推到任意长度。

**切入角度**：作者绕开"模拟图灵机读写头"这个难点，转而去模拟 **Minsky 计数机**。计数机不需要读写头定位，它的状态就是若干个计数器的取值，而 softmax Transformer 恰好天生擅长"计数"（统计某个符号出现了多少次）。计数机本身是图灵完备的，于是只要能模拟计数机，就能推出图灵完备性。

**核心 idea**：用 softmax 的计数能力（通过声明式语言 **C-RASP** 刻画）去模拟计数机，而不是去模拟图灵机；对于一元/字母有界语言这天然成立，对于任意语言再补一个与任务无关的相对位置编码（RPE）把输入编码成数。

## 方法详解

### 整体框架

本文的论证不是一个"神经网络 pipeline"，而是一条**表达力等价 + 模拟链**：

$$\text{softmax CoT Transformer (SMAT)} \;\supseteq\; \text{CoT C-RASP} \;\xrightarrow{\text{模拟}}\; \text{Minsky 计数机} \;\equiv\; \text{图灵机}$$

具体来说，作者沿用 Huang et al. (2025) 对长度可泛化 softmax Transformer（称为 SMAT）的形式化：注意力权重为 $\bar w = \mathrm{softmax}(\log n \cdot \{v_j^\top K^\top Q v_i\}_{j=1}^{i})$，其中 $\log n$ 缩放是表达稀疏函数所必需的，FFN 用 ReLU/Heaviside 单层网络。在此之上定义 CoT：模型自回归地一个个吐出 CoT token，直到吐出一个"接受"符号才停机；语言 $L(F)$ 就是所有最终被接受的输入。

C-RASP 是一个只含"过去"算子的声明式语言（等价于 $K_{\text{tr}}[\#]$ / `LTL[Count]` 的片段），它的核心是**计数项** $\overleftarrow{\#}[\varphi]$，表示"到当前位置为止满足谓词 $\varphi$ 的位置个数"。一个关键的桥梁命题（Prop 2.1）保证：凡 CoT C-RASP 能接受的语言，CoT SMAT 也能接受——所以只要在 C-RASP 这个干净的符号层面把构造做出来，就自动落到了 softmax Transformer 上。

论证分三步推进：**(1) 一元/字母有界/置换不变语言**——直接用 C-RASP 模拟计数机即可（第 3 节）；**(2) 任意语言不成立**——纯 C-RASP 连回文都识别不了（第 4 节反例）；**(3) 加一个 RPE 后任意语言成立**——RPE 让 C-RASP 能把输入词编码成一个数，再喂给计数机（第 4 节）。

### 关键设计

**1. 用 C-RASP 模拟计数机，而非用 Transformer 模拟图灵机**

这是全文绕开 softmax 困境的核心一招。图灵机的难点在于读写头位置必须被"精确定位"，而 softmax 做不到精确定位；计数机则不同——它的全部状态就是有限控制状态 $q$ 加上若干计数器取值 $x \in \mathbb{Z}^k$，转移只依赖"某个计数器是否为 0 / 大于 0"这类计数测试。而 softmax Transformer 经由 C-RASP 天生会"数数"。

作者用 CoT token 集合 $\Gamma = \Sigma \cup \Delta$（输入字母并上计数机的转移关系 $\Delta$）来记录计算轨迹：每吐出一个转移 $\tau$ 就相当于计数机走了一步，而**当前每个计数器的值可以纯用计数项重建**。对置换不变语言，第 $i$ 个计数器的值由下式给出：

$$t_i = \overleftarrow{\#}[Q_{a_i}] + \sum_{\rho \in \Delta} u_\rho(i)\cdot \overleftarrow{\#}[Q_\rho], \quad i = 1,\dots,n$$

直观地说：$\overleftarrow{\#}[Q_{a_i}]$ 数出输入里字母 $a_i$ 的初始个数（即计数器初值，对应 Parikh 向量 $\Psi(w)$），$\overleftarrow{\#}[Q_\rho]$ 数出转移 $\rho$ 至今被走过多少次，乘以它对第 $i$ 个计数器的增量 $u_\rho(i)$ 再求和，就还原出当前计数器值。规则形如 $O_{\tau'} \leftarrow \varphi_{\tau'}(t_1,\dots,t_{n+3}) \wedge Q_\tau$：当前末符号是转移 $\tau$（说明计数机在某状态）、且计数测试 $\varphi_{\tau'}$ 在重建出的计数器上成立，就输出下一个转移 $\tau'$。由于计数机确定，规则顺序无所谓。再借助 Lemma 3.5（输入摆在计数器上时，$n+3$ 个计数器足以图灵完备），即得一元/字母有界语言的图灵完备性。论文用 PARITY 作了具体示例（Example 1）：用一个 2 计数机，初始规则判 $\overleftarrow{\#}[Q_a]>0$ 还是 $=0$，非初始步用 $\overleftarrow{\#}[Q_a] - 2\cdot\overleftarrow{\#}[Q_{\tau_0}]$ 不断把 $a$ 的计数减 2，到 0 即接受——而 PARITY 恰恰是**不加 CoT 的 C-RASP 无法表达**的语言，凸显 CoT 的威力。

**2. 任意语言的不可能性：C-RASP 的通信复杂度天花板**

作者诚实地指出，上面的构造对任意语言**不**成立。Prop 4.1 证明 CoT C-RASP 在 $\Sigma=\{a,b\}$ 上不是图灵完备的，反例就是回文（PALINDROME）。根因（Lemma 4.2）是：若语言被 CoT C-RASP 识别，则它在长度 $\le n$ 的限制 $L_n$ 能被一个规模**多项式于 $n$** 的自动机识别——这直接来自 Limit Transformer / C-RASP 的**对数通信复杂度**上界（Huang et al. 2025, Thm 12），且即便配上学到的绝对位置编码（APE）也跑不掉。回文需要把前半段和后半段逐位对齐比较，这种"任意位置精确配对"超出了纯计数能力，故无法被攻克。这个负面结果不是 bug，而是说明：要走向任意语言，必须引入"位置信息"这个新维度。

**3. 与任务无关的相对位置编码，把输入词编码成一个数**

为突破上述天花板，作者给 C-RASP 加上**相对位置编码** RPE，形式上就是一个关系 $R \subseteq \mathbb{N}\times\mathbb{N}$，并引入相对计数项 $\overleftarrow{\#}_R[\varphi]$，在位置 $j$ 处统计 $\{i\in[1,j]: (i,j)\in R,\ i\models\varphi\}$ 的基数。对应到 SMAT，注意力 logit 多一个偏置项：$\bar w = \mathrm{softmax}(\log n\cdot\{v_j^\top K^\top Q v_i + \lambda[\![R]\!](i,j)\}_j)$。令人惊讶的是，作者证明**一个固定的、与要模拟的图灵机无关的 RPE 就够了**。

这个 RPE 基于一个偏函数 $\beta:\mathbb{N}\rightharpoonup\{0,1\}^*$：把数 $x$ 写成二进制，取"最高位起第一个 0 之后"的所有位作为它代表的 01 串。RPE 定义为 $(i,j)\in R \iff i\le j,\ i\in[1,|\beta(j)|],$ 且 $\beta(j)$ 在第 $i$ 位为 1。有了它，构造分两阶段（Phase I/II）：

- **Phase I（编码）**：把任意字母表的词 $w\in\Sigma^*$ 编码成向量 $x\in\mathbb{N}^n$，使得解码 $\sigma(x)=w$。做法是不断追加哑符号 $l_i$ 来调整当前词长 $\ell$，直到 $\beta(\ell)$ 恰好等于目标 01 串 $w_i$，此时放下标记符 $\text'_i$。判定 $\beta(\ell)=w_i$ 的规则巧妙地用 $\overleftarrow{\#}_R[Q_{a_i}] = \overleftarrow{\#}[Q_{a_i}]$ 且 $\overleftarrow{\#}_R[\top]=\overleftarrow{\#}[Q_{a_i}]$ 两个等式来表达"被 $R$ 选中的位置恰好就是携带 $a_i$ 的位置"。最终编码可由计数项 $X_i = \overleftarrow{\#}[\overleftarrow{\#}[\text'_i]=0]$ 取出（即 $\text'_i$ 的位置减一，恰为 $x(i)$）。
- **Phase II（模拟）**：拿到编码 $x$ 后，对集合 $S=\{x\in\mathbb{N}^n:\sigma(x)\in L\}$（它递归可枚举，因 $\sigma$ 可计算）用 Lemma 3.5 的计数机，方法与设计 1 完全相同，只是把 $\overleftarrow{\#}[Q_{a_i}]$ 换成 $X_i$。由此得到任意语言的图灵完备性（Thm 4.3）。论文用"以 b 结尾的词"（Example 2）作示例：$\sigma(x)\in L$ 当且仅当 $x(1)$ 为偶数，恰好复用 PARITY 的计数机。

### 损失函数 / 训练策略
理论部分无训练损失；实验部分（见下）从零训练一个 decoder-only LLaMA 架构，AdamW（weight decay 0.01）、batch 64、最多 30k 步，并用 EarlyStopping 监控验证损失（in-distribution 连续三个 epoch 100% 即停）。

## 实验关键数据

实验目的不是刷 SOTA，而是**验证理论预测**：一元表示（unary）下无需位置编码（NoPE）即可长度泛化，二进制表示（binary）下必须有 RPE。模型在长度 $[1,100]$ 上训练，在 in-distribution（test0, $[1,100]$）和两个 OOD 区间（test1 $[101,200]$、test2 $[201,300]$）上评估。任务是若干复杂数论概念：素数、指数、除法、最大公约数、乘法。

### 主实验

| 任务 | Unary test0 | Unary test2 | Binary+RPE test0 | Binary+RPE test2 | Binary 无RPE test0 | Binary 无RPE test2 |
|------|------|------|------|------|------|------|
| Prime | 100 | 100 | 100 | 100 | 95.00 | 0.00 |
| Exponential | 99.95 | 99.96 | 100 | 100 | 82.80 | 0.00 |
| Division | 99.90 | 99.99 | 100 | 100 | 76.40 | 0.00 |
| GCD | 99.99 | 99.70 | 100 | 100 | 70.20 | 0.00 |
| Multiplication | 99.99 | 99.98 | 100 | 100 | 64.40 | 0.00 |

### 消融实验（RPE 的作用）

| 配置 | 代表性表现（test0 → test2） | 说明 |
|------|------|------|
| Unary + NoPE | ~99.9% → ~99.9% | 一元下天然长度泛化，与理论一致 |
| Binary + RPE | 100% → 100% | 近乎完美外推，验证 RPE 使任意（二进制）输入图灵完备 |
| Binary 无 RPE | 64–95% → ~0% | OOD 完全崩溃，证明 RPE 是长度泛化的必要条件 |

### 关键发现
- **RPE 是长度泛化的关键开关**：二进制表示去掉 RPE 后，所有任务在 test2 上几乎都掉到 0%，而加上 RPE 后稳定 100%——精确呼应"任意语言需要 RPE"的理论。
- **一元表示无需位置编码**：unary 任务即使 NoPE 也能稳定外推到 3 倍训练长度，验证置换不变/字母有界语言只靠计数即可。
- **任务难度不影响结论**：从素数到乘法，难度递增，但只要表示+位置编码搭配正确，泛化结论一致，说明这是表达力层面的结构性结论而非任务特例。

## 亮点与洞察
- **换被模拟对象**：不去硬碰 softmax 难以定位读写头的痛点，而是改去模拟计数机——softmax 的计数能力恰好是它的强项。这种"挑一个与你能力匹配的等价计算模型"的思路，可迁移到分析其他受限计算模型的表达力。
- **一个与任务无关的 RPE 就够了**：通常人们以为不同语言/图灵机需要不同的位置编码，本文却给出单一固定 RPE 即可对所有图灵机生效，把"位置信息"压缩成一个可复用的极简结构。
- **长度泛化与图灵完备性同时拿到**：以往图灵完备性证明用 hardmax，牺牲了可训练性；本文在 Huang et al. (2025) 的可学习框架内完成证明，使"表达力强"和"学得到、外推得了"两个目标统一。

## 局限与展望
- **理论与实践的精度鸿沟**：构造用了 $\log n$ 缩放和 Heaviside 激活等理想化假设（Heaviside 在有限长度下用 ReLU 任意逼近），真实有限精度训练能多大程度复现仍待考。
- **未与复杂度类挂钩**：作者承认未像 Merrill & Sabharwal (2024)、Li & Wang (2025) 那样把 CoT 步数与复杂度类精细对应，refine 图灵完备性的"代价"是 future work。
- **实验规模有限**：只在数论任务、长度 $\le 300$、单一 LLaMA 小模型上验证，是否在更大模型/更长序列上同样成立未充分覆盖。
- **RPE 形式较特殊**：所用 RPE 基于 $\beta$ 的二进制编码，虽然形式简单，但与主流可学习 RPE（如 RoPE、ALiBi）的关系还需进一步厘清。

## 相关工作与启发
- **vs Pérez et al. (2021) / Merrill & Sabharwal (2024)**：他们用 hardmax/平均硬注意力直接模拟图灵机读写头，证明图灵完备但无可训练性保证；本文用 softmax + 模拟计数机，首次在现实注意力机制下拿到图灵完备性，且自带长度泛化。
- **vs Huang et al. (2025)**：本文建立在其 Limit Transformer / C-RASP 可学习框架上，但原框架不含 CoT 和 RPE、不足以图灵完备；本文把其证明技术扩展到 CoT + RPE 两个维度。
- **vs Hou et al. (2025)**：同样追求长度泛化的图灵完备构造，但他们只给出 RASP 层面的构造（非 softmax Transformer），且长度泛化依赖"不重复 n-gram"这一在长输入极限下不现实的条件；本文给出真正的 softmax 构造并理论保证完整长度泛化。
- **vs Sälzer et al. (2026)**：他们证明 softmax Transformer "几乎"图灵完备（其投影给出所有递归可枚举语言）；本文直接给出 softmax CoT Transformer 的完整图灵完备性。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在 softmax 注意力下证明图灵完备，且"模拟计数机替代模拟图灵机"的技术路线在 Transformer 表达力分析里是全新视角。
- 实验充分度: ⭐⭐⭐⭐ 实验精准对齐理论预测，RPE 消融极具说服力，但规模与模型多样性有限。
- 写作质量: ⭐⭐⭐⭐ 论证链条清晰、正反例齐全；不过 C-RASP/计数机的形式化对非理论背景读者门槛偏高。
- 价值: ⭐⭐⭐⭐⭐ 解决了一个长期开放问题，把图灵完备性与可训练性/长度泛化统一，对理解 CoT 为何强大有基础意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Efficient Turing Machine Simulation with Transformers](efficient_turing_machine_simulation_with_transformers.md)
- [\[ICLR 2026\] Transformers Are Inherently Succinct](transformers_are_inherently_succinct.md)
- [\[ICLR 2026\] Probability Distributions Computed by Autoregressive Transformers](probability_distributions_computed_by_autoregressive_transformers.md)
- [\[ICLR 2026\] The Softmax Bottleneck Does Not Limit the Probabilities of the Most Likely Tokens](the_softmax_bottleneck_does_not_limit_the_probabilities_of_the_most_likely_token.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)

</div>

<!-- RELATED:END -->
