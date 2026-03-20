---
title: "ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity"
authors: "Xiaoyang Liu, Jiacheng Sun, Yuxiang Qiu, Jieyu Zhang, Zhenguo Li"
affiliations: "Shanghai Jiao Tong University"
venue: "ICLR 2026"
arxiv: "2509.22246"
code: "https://github.com/XiaoyangLiu-sjtu/ASSESS"
tags: ["autoformalization", "evaluation metrics", "tree edit distance", "Lean", "formal mathematics"]
rating:
  novelty: 4
  experiments: 4
  writing: 4
  value: 4
---

# ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity

## 一句话总结

提出 TransTED Similarity，一种基于算子树 (Operator Tree) 和语义变换增强的树编辑距离指标，用于评估自动形式化 (autoformalization) 生成的形式化数学命题与参考命题之间的语义相似度，并构建了 EPLA 基准数据集。

## 研究动机

自动形式化 (autoformalization) 是将自然语言数学命题翻译为形式化证明语言（如 Lean）的任务，近年来随着大语言模型的发展受到广泛关注。然而，如何评估自动形式化的质量一直是一个悬而未决的问题：

1. **基于文本的指标（如 BLEU）**：将形式化语句视为普通文本序列进行比较，完全忽略了形式化语言的语义结构。例如，两个语义等价但写法不同的 Lean 表达式可能获得很低的 BLEU 分数。
2. **基于证明的指标（如 BEq）**：试图通过自动证明两个命题的等价性来判断相似度，但这种方法过于严格——许多语义上高度相似但不完全等价的命题会被直接判为不相似；同时依赖定理证明器的能力上限。
3. **LLM-as-Judge**：使用大语言模型进行打分，但存在不可复现、计算成本高、需要 GPU 等问题。

作者指出，现有方法要么过于宽松（BLEU），要么过于严格（BEq），缺乏一种能够捕捉形式化命题之间**细粒度语义相似性**的评估指标。

## 方法详解

### 核心思想

ASSESS 框架的核心洞察是：形式化数学命题可以被解析为**算子树 (Operator Tree, OPT)**，而在 Lean 等证明助手中，许多 tactic 操作实际上对应树上的特定语义变换。因此，可以通过计算两棵 OPT 之间的**变换增强树编辑距离 (Transformation-augmented Tree Edit Distance, TransTED)** 来衡量命题的语义相似度。

### 两阶段流程

#### 阶段一：解析为算子树 (OPT)

- 利用 **Lean Language Server** 将 Lean 形式化命题解析为表达式树
- 每个内部节点对应一个算子（如函数应用、量词、逻辑连接词等）
- 叶节点对应常量、变量或类型
- 这种表示保留了形式化命题的完整结构信息，比纯文本表示更加精确

#### 阶段二：TransTED 计算

标准的树编辑距离 (TED) 只考虑节点的插入、删除和重命名操作，无法捕捉语义等价的变换。TransTED 在 TED 的基础上引入了**语义变换操作**：

- **交换律变换**：如 `a + b → b + a`，交换可交换运算符的子树
- **分配律变换**：如 `a * (b + c) → a * b + a * c`
- **化简变换**：如 `a + 0 → a`
- **类型等价变换**：Lean 中不同但等价的类型表示
- **重写变换**：对应 Lean 中 `rw`、`simp` 等 tactic 的效果

这些变换从 Lean 的 tactic 系统中系统性地提取，每种变换的代价根据其语义保持程度设定权重。

最终，TransTED 通过**启发式搜索**在变换空间中寻找使树编辑距离最小的变换序列，得到的最小距离被归一化为 0-1 之间的相似度分数。

### TransTED Similarity 定义

$$
\text{TransTED}(T_1, T_2) = 1 - \frac{\min_{\sigma \in \Sigma} \text{TED}(\sigma(T_1), T_2)}{\max(|T_1|, |T_2|)}
$$

其中 $\Sigma$ 是所有合法语义变换序列的集合，$|T|$ 表示树的节点数。

## EPLA 基准数据集

作者构建了 **EPLA (Expert-annotated Provability and Likeness Assessment)** 基准数据集：

- **数据来源**：从 miniF2F 和 ProofNet 两个知名形式化数学数据集中采样命题对
- **规模**：共 1,247 对专家标注的形式化命题对
- **标注维度**：
  - **可证明性 (Provability)**：两个命题是否逻辑等价（可互相证明）
  - **结构相似性 (Structural Likeness)**：两个命题在结构上的相似程度（5 级评分）
- **标注者**：具有形式化数学经验的专家，确保标注质量
- **子集划分**：
  - EPLA-miniF2F：来自 miniF2F 的命题对
  - EPLA-ProofNet：来自 ProofNet 的命题对

这是首个专门用于评估形式化命题相似度指标的基准数据集。

## 实验结果

### 主要结果（EPLA-miniF2F）

| 指标 | 准确率 (%) | Cohen's Kappa |
|------|-----------|---------------|
| BLEU | - | 0.26 |
| BEq | - | 0.29 |
| LLM Judge (GPT-4) | - | ~0.30 |
| **TransTED** | **70.16** | **0.35** |

TransTED 在准确率和 Kappa 系数上均显著优于所有基线方法。

### 关键发现

1. **BLEU 的局限性**：BLEU 与人类判断的一致性很低，因为它无法理解形式化语言的结构语义
2. **BEq 的局限性**：虽然基于证明验证，但由于定理证明器能力有限，大量等价命题无法被验证
3. **TransTED 的优势**：通过结合结构信息和语义变换，在保持可解释性的同时大幅提升了与人类判断的一致性
4. **CPU 友好**：TransTED 完全在 CPU 上运行，无需 GPU，具有良好的可复现性

### 消融实验

- 去掉语义变换（退化为标准 TED）：性能显著下降，证实变换组件是关键改进
- 不同变换类型的贡献：交换律和化简变换贡献最大
- 树的解析粒度：细粒度 OPT 优于粗粒度表示

## 优点

1. **问题重要且定义清晰**：自动形式化的评估是形式化数学社区的核心瓶颈之一，本文给出了明确的解决思路
2. **方法设计优雅**：将 Lean tactic 的语义操作嵌入树编辑距离框架，概念上简洁且有数学基础
3. **基准数据集有价值**：EPLA 填补了该领域缺乏标准评估集的空白
4. **实用性强**：纯 CPU 运行、开源代码、可复现，降低了使用门槛
5. **实验全面**：包含多种基线对比、消融实验和案例分析

## 局限性

1. **变换集合有限**：当前的语义变换集合是手动设计的，可能无法覆盖所有等价变换
2. **启发式搜索**：搜索最优变换序列的算法是启发式的，不保证找到全局最优
3. **Lean 特定**：目前仅支持 Lean 语言，推广到其他证明助手（如 Coq、Isabelle）需要额外工作
4. **绝对性能仍有提升空间**：70% 的准确率说明该问题本身的难度，但也意味着还有改进余地
5. **数据集规模**：1,247 对虽已是开创性工作，但相比 NLP 领域的评估基准仍较小

## 思考与启发

- **形式化语言的特殊性**：不同于自然语言，形式化语言有严格的语义，这为设计更精确的评估指标提供了可能
- **结构化表示的力量**：将命题解析为树结构再比较，比直接比较文本字符串合理得多
- **Tactic 作为语义桥梁**：将证明 tactic 理解为语义变换是一个很有启发性的视角
- **评估指标研究的重要性**：好的评估指标能推动整个领域的进步，这篇工作为自动形式化提供了更可靠的评估工具
- **与代码相似度检测的联系**：OPT + TED 的框架与程序分析中的 AST 比较有异曲同工之处，可能存在交叉借鉴的机会
