# Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition

**会议**: ICLR 2026  
**arXiv**: [2507.09875](https://arxiv.org/abs/2507.09875)  
**代码**: [INK-USC/function-induction](https://github.com/INK-USC/function-induction)  
**领域**: llm_nlp  
**关键词**: mechanistic interpretability, in-context learning, induction heads, function vectors, task generalization, path patching  

## 一句话总结

通过 off-by-one addition（如 1+1=3, 2+2=5）这一反事实任务，利用 path patching 发现大语言模型内部存在 **function induction** 机制——一种超越 token 级别 pattern matching、在函数级别进行归纳推理的注意力头电路，并证明该机制可跨任务复用。

## 研究背景与动机

1. **任务级泛化的重要性**：随着 LLM 应用场景持续扩展，在部署前将所有任务纳入训练数据已不现实，因此模型在推理时通过 in-context learning (ICL) 完成未见任务的能力至关重要。
2. **已有理解的局限**：先前工作对 ICL 的机制理解主要集中在 induction heads（token 级别的 copy-paste，即 [A][B]...[A]→[B]）和 function vectors（单步映射任务如国家→首都），对涉及**多步推理**或**新定义概念**的复杂泛化场景理解不足。
3. **Off-by-one addition 的巧妙设计**：该任务由两步组成——标准加法 + 意外的 +1 操作（即 1+1=3），是一个反事实、多步合成任务。模型要么学会 +1 输出 7（泛化成功），要么遵守算术规则输出 6（泛化失败）。
4. **实验发现驱动深入分析**：六个主流 LLM（Llama-2/3、Mistral、Gemma-2、Qwen-2.5、Phi-4）均能有效完成该任务，且准确率随 shot 数增加而单调上升，这一普遍现象激发了对内部机制的深入探究。
5. **从 token induction 到 function induction**：传统 induction heads 归纳的是零阶常函数 $f = \text{output}([B])$，本文希望揭示模型能否归纳一阶函数 $f(x) = x + 1$，从而将理解从 token 级提升到函数级。
6. **跨任务复用的验证需求**：如果 function induction 是一种通用机制，它应该在结构相似但子步骤截然不同的任务上被复用，这对理解模型的组合性和灵活性具有重要意义。

## 方法详解

### 整体框架

本文采用 **mechanistic interpretability + path patching** 方法，以 Gemma-2 (9B) 为主要分析对象，通过对比 base prompt（标准加法 1+1=2）与 contrast prompt（off-by-one 加法 1+1=3）的激活传播，逐层追踪 +1 函数的计算来源，最终发现一个由三组注意力头构成的电路。

### 关键设计 1: Path Patching 电路发现

- **做什么**：分别在 base prompt $x_{base}$ 和 contrast prompt $x_{cont}$ 上做前向传播，将 $M(\cdot|x_{base})$ 的部分激活替换到 $M(\cdot|x_{cont})$ 中，观察输出是否从 "3+3=7" 回退到 "3+3=6"。
- **核心思路**：定义 logit 差值 $F(C, x) = C(y_{base}|x) - C(y_{cont}|x)$，用归一化的 relative logit difference $r = \frac{F(M', x_{cont}) - F(M, x_{cont})}{F(M, x_{cont}) - F(M, x_{base})}$ 量化替换效应，$r$ 越接近 $-100\%$ 表示该组件对 +1 函数贡献越大。
- **设计动机**：path patching 可以精确追踪激活的因果路径，逐步定位从最终输出到上游组件的信息流。

### 关键设计 2: 三组注意力头的发现

通过逐层 path patching，识别出三组注意力头：

| 组别 | 名称 | 功能 | 注意力模式 |
|------|------|------|-----------|
| Group 1 | **Consolidation Heads** | 聚合信息、最终化输出 | 主要关注当前 token 和 `<bos>` |
| Group 2 | **Function Induction (FI) Heads** | 从 ICL 示例中携带 +1 函数到测试样本 | 在 "=" 位置关注前面各示例的答案 token $c_i$ |
| Group 3 | **Previous Token (PT) Heads** | 在答案位置注册"预期与实际的差异" | 在 $c_i$ 位置关注紧邻的前一个 "=" token |

- **核心思路**：FI Heads 类似传统 induction heads 但操作在函数级别——传统 induction heads 复制 token [B]，而 FI heads 归纳函数 $f(x) = x + 1$。PT Heads 类似传统的 previous token heads，在 ICL 示例中检测模型预期答案与实际答案的偏差。
- **设计动机**：这种分层发现过程（Output → Group 1/2 → Group 3）使电路结构自然浮现，不依赖先验假设。

### 关键设计 3: Function Vector 分析

- **做什么**：构建 naive prompt（如 "2=2\n3=?"），将 FI head 的输出加到残差流中，观察模型 logit 变化，生成 $10 \times 10$ 热力图。
- **核心思路**：每个 FI head 写出 +1 函数的不同"碎片"——例如 H39.7 促进 $x+1$，H28.6 抑制 $x-1$，H32.1 促进大于 $x$ 的数字，H24.9 抑制 $x$。多个头的输出聚合后实现完整的 +1 函数。
- **设计动机**：验证 FI heads 确实承载了 +1 函数的因果效应，而非仅仅是统计相关。

### 损失函数 / 评估指标

本文不涉及训练，核心评估指标为：
- **Accuracy**：off-by-one addition 的正确率
- **Relative logit difference** $r$：归一化的 logit 差值，衡量电路组件对 +1 行为的贡献

## 实验关键数据

### 主实验：ICL 性能与 FI Heads 消融

| 模型 | 4-shot Acc | 8-shot Acc | 16-shot Acc | 消融 FI Heads 后 |
|------|-----------|-----------|------------|-----------------|
| Llama-2 (7B) | ~15% | ~35% | ~55% | 回退至标准加法 |
| Mistral-v0.1 (7B) | ~20% | ~50% | ~65% | 回退至标准加法 |
| Gemma-2 (9B) | 33% | ~70% | 86% | 0% (off-by-one), 100% (标准) |
| Llama-3 (8B) | ~60% | ~95% | ~98% | 回退至标准加法 |
| Phi-4 (14B) | ~65% | ~98% | ~99% | 回退至标准加法 |

消融 6 个 FI heads 后模型完全丧失 off-by-one 能力（准确率降至 0%），而随机消融 6 个头几乎无影响，说明 FI heads 是 +1 函数的必要组件。

### 跨任务泛化消融实验

| 任务对 | Base Task | Contrast Task | Contrast Acc (完整模型) | Contrast Acc (消融 FI Heads) |
|--------|-----------|---------------|------------------------|----------------------------|
| Off-by-2 Addition | 标准加法 | +2 加法 | 非平凡 | 大幅下降 |
| Shifted MMLU | 标准 MCQA | 答案 shift +1 | 非平凡 | 大幅下降（残余非零） |
| Caesar Cipher (k=2) | ROT-0 | ROT-2 | 非平凡 | 大幅下降（残余非零） |
| Base-8 Addition | 十进制加法 | 八进制加法 | 非平凡 | 大幅下降 |

关键发现：相同的 FI heads 在所有四个任务对中均被复用，证明 function induction 机制的**灵活性和组合性**。

### Base-8 加法错误分析

| Case | 描述 | 正确行为 | 模型正确率 | 错误类型 |
|------|------|----------|-----------|----------|
| Case 1 | 无进位 | 不调整 | 93% | 7% 过度泛化（不该调却调了） |
| Case 2 | 进位，需调整个位和十位 | 调整两位 | 16% | 84% 欠泛化（该调却没调） |
| Case 3 | 进位，仅需调整个位 | 仅调个位 | 14% | 83% 欠泛化 |

这表明模型虽能归纳简单的 +2 函数，但难以处理条件性触发（仅在特定条件下施加 +2），暴露了当前模型在多步归纳推理上的瓶颈。

### 关键发现

1. **分布式函数编码**：+1 函数不是由单个注意力头实现，而是由 6-9 个 FI heads 协作完成，每个头写出函数的不同"碎片"（促进 $x+1$、抑制 $x$、抑制 $x-1$ 等）。
2. **FI Heads ≠ FV Heads**：与 Todd et al. (2024) 发现的 function vector heads 完全不重叠——FV heads 位于模型早中层（<20 层），FI heads 位于晚期层（29-31 层），说明 FI heads 是多步任务中后续步骤的专属机制。
3. **跨模型普遍性**：在 Gemma-2、Llama-2、Llama-3、Mistral 四个模型中均发现了三组头的结构，证明 function induction 是一种普遍涌现的机制。

## 亮点与洞察

- **概念创新**：将 induction heads 从零阶（复制 token）推广到一阶（归纳函数 $f(x) = x+1$），是对 ICL 机制理解的本质提升。
- **任务设计精巧**：off-by-one addition 巧妙地将反事实推理与算术结合，使多步推理中的各步骤可分别追踪。
- **机制可组合性**：同一 FI 电路在加法偏移、MCQA 偏移、Caesar Cipher、八进制加法等差异巨大的任务中复用，说明模型内部存在通用的"函数偏移"模块。
- **对评估的启示**：base-8 加法的分析揭示模型可能通过非预期的 shortcut 算法（先做十进制再 +2）获得部分正确率，纯准确率评估可能掩盖推理缺陷。

## 局限性 / 可改进方向

1. **电路不完美**：发现的电路未完全满足 faithfulness 和 completeness 标准（这两者与 minimality 往往互相制约）。
2. **仅分析注意力头**：未深入分析 MLP 层的作用，也未拆解注意力头内部的 QK/OV 电路。
3. **函数类型受限**：当前仅验证了"偏移类"函数（$f(x) = x + k$），对更复杂的函数（如非线性变换）是否也存在类似机制未做探索。
4. **任务为合成/算法性**：未在自然文本中验证 function induction 机制的作用。
5. **数字表征的非线性**：LLM 中数字 token 通常映射到正弦（Fourier）特征空间而非线性空间，增加了可解释性分析的难度。
6. **条件性归纳的失败**：base-8 加法中模型无法在正确条件下触发 +2，说明当前模型对"两步归纳 + 三步任务"的能力有限。

## 相关工作与启发

- **Induction Heads (Olsson et al., 2022)**：本文直接推广了 induction heads 的概念，从 token 级到函数级，是对该经典发现的自然延伸。
- **Function Vectors (Todd et al., 2024; Hendel et al., 2023)**：FI heads 与 FV heads 功能相似但层级位置不同，FI heads 可视为 FV 机制在多步任务晚期步骤中的特化实例。
- **Latent Multi-step Reasoning**：本文提供了模型进行隐式多步推理的电路级证据，补充了基于 multi-hop QA 的行为级分析。
- **对 alignment 的启示**：作者推测 sycophancy、agreement bias 等行为可能共享类似结构——模型从上下文中归纳出"信念修改函数"并应用于输出生成。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 将 induction heads 从 token 级推广到函数级是概念突破，function induction 的命名和形式化具有重要理论价值
- **实验充分度**: ⭐⭐⭐⭐ — 跨 4 个模型、4 个任务对验证，辅以消融、因果干预、热力图分析；但电路发现未完美满足 faithfulness/completeness
- **写作质量**: ⭐⭐⭐⭐⭐ — 结构清晰，概念定义精确，图表信息量大，running example 贯穿全文
- **价值**: ⭐⭐⭐⭐ — 深化了对 ICL 和隐式多步推理的机制理解，对模型评估和预训练设计有实际指导意义；但仅限合成任务，自然场景验证待补
