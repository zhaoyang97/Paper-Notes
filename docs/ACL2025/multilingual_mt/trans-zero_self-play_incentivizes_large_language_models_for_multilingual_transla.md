---
title: >-
  [论文解读] Trans-Zero: Self-Play Incentivizes Large Language Models for Multilingual Translation
description: >-
  [ACL2025][多语言/翻译][多语言翻译] 提出 Trans-Zero 自博弈框架，仅使用单语数据，通过遗传蒙特卡洛树搜索（G-MCTS）在多语言翻译过程中探索语义一致的候选翻译，结合偏好优化实现无平行数据的多语言翻译训练，性能可媲美大规模监督微调方法。 1. 多语言翻译对平行数据的依赖：当前 LLM 多语言翻译仍需大…
tags:
  - "ACL2025"
  - "多语言/翻译"
  - "多语言翻译"
  - "自博弈"
  - "Monte-Carlo Tree Search"
  - "偏好优化"
  - "无平行数据"
---

# Trans-Zero: Self-Play Incentivizes Large Language Models for Multilingual Translation

**会议**: ACL2025  
**arXiv**: [2504.14669](https://arxiv.org/abs/2504.14669)  
**代码**: [NJUNLP/trans0](https://github.com/NJUNLP/trans0)  
**领域**: 多语言翻译  
**关键词**: 多语言翻译, 自博弈, Monte-Carlo Tree Search, 偏好优化, 无平行数据

## 一句话总结
提出 Trans-Zero 自博弈框架，仅使用单语数据，通过遗传蒙特卡洛树搜索（G-MCTS）在多语言翻译过程中探索语义一致的候选翻译，结合偏好优化实现无平行数据的多语言翻译训练，性能可媲美大规模监督微调方法。

## 背景与动机
1. **多语言翻译对平行数据的依赖**：当前 LLM 多语言翻译仍需大量平行语料进行 SFT，而低资源语言的平行数据极度匮乏，限制了翻译覆盖面。
2. **灾难性遗忘问题**：随着多语言 SFT 规模扩大，一对一 MLE 监督会引入偏置，过多的多语言标注反而稀释预训练知识，导致跨语言性能下降。
3. **MoE 方案的扩展性瓶颈**：已有混合专家（MoE）方法使用手工设计的语言模块路由，但路由复杂度和分布式开销随翻译方向数指数增长。
4. **LLM 内在多语言知识的利用不足**：LLM 在预训练中已积累丰富的多语言知识，但现有方法未能有效激发这些内在能力进行自我提升。
5. **跨语言探索的技术挑战**：系统性的跨语言语义空间探索需要超越简单 prompt 工程的规划方法，传统 LLM 推理范式难以直接适用。
6. **多语言质量评估的外部依赖**：现有翻译质量评估依赖数据驱动的 QE 指标或奖励模型训练，增加了系统复杂度和对外部模块的依赖。

## 方法详解

### 整体框架：Trans-Zero 自博弈多语言翻译
- **功能**：构建一个仅需单语数据的自博弈框架，让 LLM 通过多语言翻译过程的搜索与偏好优化来自我提升翻译能力。
- **为什么**：摆脱对平行数据的依赖，利用 LLM 固有的多语言知识实现资源高效的多语言翻译训练。
- **怎么做**：定义多语言翻译过程（MTP）→ 在 MTP 上执行遗传蒙特卡洛树搜索（G-MCTS）探索候选翻译 → 基于跨语言语义一致性评估翻译质量 → 从搜索树中提取偏好对 → 使用 SPPO 进行偏好优化。

### 关键设计 1：多语言翻译过程（MTP）与 G-MCTS
- **功能**：定义迭代多语言翻译过程作为搜索空间，在其上实施结合遗传算法思想的 MCTS 搜索。
- **为什么**：MTP 将翻译扩展到多语言链路（如 EN→IT→ZH→EN），使得语义一致性可通过回译验证。G-MCTS 的遗传扩展（merge + mutate）解决了标准 MCTS 在翻译探索中多样性不足的问题。
- **怎么做**：
    - **初始化**：以源文本为根节点，top-k 采样生成 $b$ 个目标语言候选翻译作为子节点，通过回译初始化奖励。
    - **遗传扩展**：选择最高 UCB 值节点扩展。若 UCB 最大节点 ≠ utility 最大节点，执行 **Merge**（以两者为 few-shot 示例生成新翻译）；若相同，执行 **Mutate**（翻译模拟中最佳重建文本而非原始输入，引入多样性）。
    - **语义一致性模拟**：对候选翻译展开 $b^n$ 条 MTP 轨迹，计算重建文本与原始输入的一致性分数（BLEURT 双向平均），取 literal 与 free 翻译中较优者作为奖励。

### 关键设计 2：Tree-to-Preference 算法与 SPPO 优化
- **功能**：从完成搜索的 G-MCTS 树中系统提取翻译偏好对，用于自博弈偏好优化。
- **为什么**：搜索树中的节点 utility 自然反映翻译质量排序，可无需外部奖励模型或 QE 模块直接构造偏好数据。距根越远的高 utility 节点说明经历更多翻译步骤仍保持语义一致，其翻译质量更值得偏好。
- **怎么做**：对搜索树做层序遍历并合并重复节点，按 utility 降序选择排序，排序中每次交换生成一个偏好对 $(y_w \succ y_l)$。仅保留 utility 高于根节点的偏好选中节点。通过 softmax 将 utility 差异转化为 SPPO 所需的 win rate，最终用 SPPO 对称损失进行偏好优化。

## 实验关键数据

### 实验 1：与 SFT 和专用翻译模型的对比（Flores-200, 6语言）

| 模型 | EN⇒X (BLEURT) | X⇒EN (BLEURT) | X⇒X (BLEURT) | 平均 (BLEURT) |
|------|---------------|---------------|--------------|--------------|
| Mixtral-8x7B-Instruct | 55.42 | 75.41 | 54.49 | 61.77 |
| ALMA-R | 69.38 | 77.52 | 51.03 | 65.98 |
| Tower-Instruct | 76.74 | 78.73 | 72.98 | 76.15 |
| Llama3.1-SFT (5m) | 75.80 | 78.47 | 73.30 | 75.86 |
| **Llama3.1-Trans-Zero** | **73.71** | **77.60** | **73.28** | **74.86** |
| Qwen2.5-SFT (5m) | 75.32 | 78.21 | 72.99 | 75.49 |
| **Qwen2.5-Trans-Zero** | **75.05** | **78.21** | **72.23** | **75.16** |

**发现**：Trans-Zero 仅用单语数据，在非英语方向（X⇒X）上达到甚至超越 5M 平行数据 SFT 的水平，整体性能与大规模监督方法高度可比。在 EN⇒X 方向略低于 5M SFT，但差距很小。

### 实验 2：G-MCTS 单独作为推理增强的效果

| 模型 | EN⇒X (BLEURT) | X⇒X (BLEURT) | 平均 (BLEURT) |
|------|---------------|--------------|--------------|
| Llama3.1-Instruct | 62.57 | 62.52 | 65.72 |
| + G-MCTS | 64.21 (+1.64) | 68.12 (+5.60) | 67.45 (+1.73) |
| Llama3.1-SFT (5k) | 69.33 | 68.51 | 71.61 |
| + G-MCTS | 71.55 (+2.22) | 71.92 (+3.41) | 73.45 (+1.84) |
| Tower-Instruct | 76.74 | 72.98 | 76.15 |
| + G-MCTS | 76.44 (-0.30) | 74.42 (+1.44) | 76.38 (+0.23) |

**发现**：G-MCTS 作为纯推理增强在非英语方向（X⇒X）提升最为显著（最高 +5.60），证明其跨语言探索能力。对已经很强的模型（Tower-Instruct）提升有限，但在 X⇒X 方向仍有增益。基础模型（如 ALMA-R、Llama3.1-Base）由于翻译能力不足导致搜索失败（Failed），说明 G-MCTS 需要基本的翻译能力作为启动条件。

## 亮点
- **突破平行数据依赖**：首个仅用单语数据实现多语言翻译自博弈训练的框架，在低资源场景下意义重大。
- **G-MCTS 设计精巧**：遗传扩展（merge/mutate）与多语言语义一致性模拟的结合，既保证搜索多样性又提供无需外部奖励的评估信号。
- **非英语方向优势突出**：在最具挑战的 X⇒X 翻译方向上表现尤为亮眼，甚至超越大规模平行数据 SFT。
- **Tree-to-Preference 算法简洁有效**：将搜索树的 utility 排序直接转化为 SPPO 偏好对，避免了额外的奖励模型训练。

## 局限与展望
- **需要基本翻译能力启动**：对翻译能力极弱的 Base 模型（如 Llama3.1-Base）G-MCTS 搜索直接失败，仍需 cold-start 阶段用少量指令数据启动。
- **计算开销大**：G-MCTS 在每个句子上需要大量翻译调用（$b^n$ 条模拟轨迹 × 多轮搜索），32 GPU 并行仍需大量计算资源。
- **语言覆盖有限**：仅在 6 种语言上验证，未涉及真正的低资源语言（如非洲语言、东南亚语言）。
- **EN⇒X 方向略弱于大规模 SFT**：在英译其他语言方向上与 5M SFT 仍有约 2 个 BLEURT 的差距，high-resource 场景下的优势不够明显。

## 与相关工作的对比

### vs ALMA / ALMA-R (Xu et al., 2024a/c)
ALMA 和 ALMA-R 依赖大量平行数据和外部 LLM 生成的偏好标注。Trans-Zero 完全摆脱平行数据，通过自博弈搜索自主生成偏好信号。在 EN⇒X 上 Trans-Zero 与 ALMA-R 可比，但在 X⇒X 方向上大幅超越 ALMA-R（73.28 vs 51.03 BLEURT），体现了自博弈框架对非英语方向的优势。

### vs Self-Play Preference Optimization (SPPO, Chen et al., 2024)
SPPO 提供了偏好优化的博弈论框架，但原始 SPPO 需要外部偏好信号。Trans-Zero 创新性地将 G-MCTS 的搜索 utility 作为偏好来源，实现了翻译场景下的端到端自博弈闭环，无需任何外部评估模块。

### vs 跨语言优化方法 (Geng et al., 2024; She et al., 2024)
已有方法使用强势语言辅助弱势语言优化，但局限于双语场景或需要预定义 pivot 语言。Trans-Zero 通过 MTP 在任意多语言间迭代翻译，搜索空间随语言数量扩展（实验证明 6 语言优于 4 语言），具有更好的可扩展性。

## 补充观察
- 增加参与搜索的语言数量（4→6）可显著提升 Trans-Zero 的性能上界，说明多语言交叉验证的信号随语言数增加而更丰富。
- SFT 性能在超过 100k 平行样本后趋于饱和，而 Trans-Zero 在非英语方向可持续从搜索中获益，暗示搜索式学习与数据式学习的互补性。
- 语言检测失败的翻译在排序中被 utility 减半惩罚，这一简单策略有效过滤了低质量翻译对偏好学习的污染。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首次将 MCTS 自博弈应用于无平行数据多语言翻译，框架设计极具创新性
- 实验充分度: ⭐⭐⭐⭐ — 6 语言、多基线对比、消融分析充分，但缺少真正低资源语言验证
- 写作质量: ⭐⭐⭐⭐ — 方法描述清晰，公式推导完整，案例分析直观
- 价值: ⭐⭐⭐⭐ — 为低资源多语言翻译提供了全新范式，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)
- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [\[NeurIPS 2025\] Exploring the Translation Mechanism of Large Language Models](../../NeurIPS2025/multilingual_mt/exploring_the_translation_mechanism_of_large_language_models.md)
- [\[ACL 2025\] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)
- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)

</div>

<!-- RELATED:END -->
