# Literature Meets Data: A Synergistic Approach to Hypothesis Generation

**会议**: ACL 2025  
**arXiv**: [2410.17309](https://arxiv.org/abs/2410.17309)  
**代码**: [https://github.com/ChicagoHAI/hypothesis-generation](https://github.com/ChicagoHAI/hypothesis-generation) (有)  
**领域**: LLM / 科学发现 / 假设生成  
**关键词**: hypothesis generation, LLM, literature-based, data-driven, human evaluation  

## 一句话总结
提出首个将文献驱动和数据驱动假设生成进行协同整合的方法，通过 Refinement 和 Union 两种策略让 LLM 从论文摘要和观测数据中联合生成更具泛化性的假设，在五个社会科学分类任务的 OOD 数据集上比纯数据驱动方法平均提升 3.37%，并首次通过人类实验证明 LLM 生成的假设能显著改善人类决策准确率（+7.44% / +14.19%）。

## 背景与动机
LLM 驱动的假设生成（hypothesis generation）是 AI for Science 的重要方向。现有方法大致分两类：**理论驱动**（从文献中总结规律生成假设）和**数据驱动**（从观测数据中发现模式生成假设）。理论驱动方法依赖高质量文献，难以适应新数据，且缺乏经验支撑；数据驱动方法（如 HypoGeniC）虽然能适配具体数据，但容易过拟合特定数据集，泛化性差。两类方法各有优劣，但此前没有人尝试过将二者结合——就像爱因斯坦说的"理论决定了你能观察到什么"，文献中的先验知识应该能引导对数据的发现。

## 核心问题
**文献知识和数据模式能否互补？** 具体来说：(1) 如何在一个统一框架中融合来自文献的先验假设和来自数据的经验假设？(2) 这种融合是否能生成更具泛化性和实用性的假设？(3) 这些假设是否真的能帮助人类做出更好的决策？

## 方法详解

### 整体框架
给定一个研究问题 $q$（如"什么特征表明一条评论是虚假的"）、相关论文集 $\mathcal{P}$ 和观测数据集 $\mathcal{D}$，目标是用 LLM $\mathcal{M}$ 生成高质量假设集 $\mathcal{H} = f_{\mathcal{M}}(q, \mathcal{P}, \mathcal{D})$。整个 pipeline 包含三个模块：文献假设生成、数据假设生成（HypoGeniC）、以及两种融合策略（HypoRefine / Union）。

### 关键设计

1. **文献假设生成（Literature-Based）**：从 Semantic Scholar / Google Scholar 手动收集 10 篇相关论文 → 用 S2ORC-doc2json 转 JSON → LLM 生成论文摘要 → 基于摘要指导 LLM 生成针对特定任务的假设。对于 GPT-4o-mini 生成的过于简短的假设，还引入了一个 Specificity Booster 来增加具体示例和细节。

2. **数据驱动假设生成（HypoGeniC）**：沿用 Zhou et al. (2024) 的框架。初始化阶段用少量样本让 LLM 生成初始假设集 $\mathcal{H}_\mathcal{D}$；更新阶段对每个新样本用 top-$k$ 高奖励假设做预测，预测错误的样本进入错误样本池 $\mathcal{W}$，当池满后生成新假设并按 UCB 奖励函数更新假设库。奖励函数为 $r_i = \frac{\sum \mathbb{I}(y_j = \hat{y}_j)}{|\mathcal{S}_i|} + \alpha \sqrt{\frac{\log t}{|\mathcal{S}_i|}}$，兼顾准确率和探索。

3. **融合策略一：HypoRefine（迭代精炼）**：在 HypoGeniC 的初始化阶段同时使用数据样本和论文摘要生成初始假设。在更新阶段，每次从错误样本池生成的新假设 $\mathcal{H}_0$ 要经过多轮交替精炼——奇数轮由数据精炼 agent 用错误样本 $\mathcal{W}$ 精炼，偶数轮由文献精炼 agent 用论文信息 $\mathcal{P}$ 精炼，共 `max_refine=6` 轮。

4. **融合策略二：Union（联合+去冗余）**：分别生成文献假设库和数据假设库 → 各自通过 LLM 冗余检测器（两两比较构成 20×20 矩阵）去除重复 → 从数据假设库取 top-10 + 从文献假设库随机取 10 条，组成最终假设库（大小 20）。这种方式避免了文献假设在 HypoGeniC 奖励函数中被低估的问题。

### 训练策略
- 假设库大小固定为 20
- 训练集 200 样本，初始化用 10 样本
- 奖励系数 $\alpha=0.5$，错误池上限 $w_{max}=10$
- 每次更新生成 1 条新假设
- 推理时用 CoT 提示 LLM 先从 20 条假设中选最相关的，再做预测
- 温度 $1 \times 10^{-5}$，max_tokens=4000

## 实验关键数据

五个社会科学分类任务：欺骗性评论检测（Deceptive Reviews）、AI 生成内容检测（LlamaGC / GPTGC）、说服力预测（Persuasive Pairs）、心理压力检测（Dreaddit）。

| 方法 (GPT-4-mini) | Deceptive (OOD) | LlamaGC (OOD) | GPTGC (OOD) | Persuasive (OOD) | Dreaddit (OOD) |
|---|---|---|---|---|---|
| Few-shot k=3 | 65.56 | 51.11 | 64.22 | 83.64 | 75.00 |
| Literature-only | 59.22 | 49.00 | 54.00 | 78.80 | 67.68 |
| HypoGeniC | 75.22 | 81.67 | 68.56 | 82.20 | 76.56 |
| **HypoRefine** | **77.78** | 55.33 | 63.33 | **89.04** | 78.04 |
| **Lit∪HypoGeniC** | 72.41 | **83.00** | **69.22** | **89.88** | **78.20** |

- Literature + Data 在所有任务、模型配置上均优于其他方法
- 比 few-shot 平均提升 8.97%，比 literature-only 提升 15.75%，比 HypoGeniC 提升 3.37%
- 跨模型迁移：用一个模型生成的假设给另一个模型推理，10/20 case 性能变化 < 3%

**人类评估**：
- AIGC 检测：人类准确率从 58.86% → 73.05%（+14.19%，p=0.01）
- 欺骗检测：人类准确率从 57.14% → 64.58%（+7.44%，p=0.04）
- 100% 参与者认为假设有帮助，>40% 认为"非常有帮助"或"极其有帮助"
- 新颖性：84%（欺骗检测）和 80%（AIGC 检测）的文献-数据假设对被人类判定为互相提供了新颖信息

### 消融实验要点
- **HypoRefine 在 AIGC 检测任务上失效**：文献精炼反而比纯 HypoGeniC 降了 13.64%，因为 AIGC 检测的文献缺乏有效的可解释性特征。此时 Union 策略（Lit∪HypoGeniC）效果更好。
- **Union vs Refine 各有优势**：Deceptive/Persuasive/Dreaddit 上 Refine 更好（+3.92%），AIGC 上 Union 更好。
- **IND 上 HypoGeniC 有时更强**：因为 HypoGeniC 专门针对 IND 数据优化，融合方法的泛化性在 OOD 上体现。
- NotebookLM、HyperWrite 等商业工具生成的假设中包含无效/不相关假设，影响推理性能。

## 亮点
- **首创文献+数据假设融合**：填补了一个显而易见但之前没人做的空白，idea 简洁有效
- **两种互补的融合策略**：HypoRefine（深度融合）适合文献质量高的场景，Union（浅层融合）则更鲁棒，两者结合覆盖了不同情况
- **首次人类实验验证假设实用性**：不只是 benchmark 上刷数字，而是真正让人类用假设做决策并证明有效
- **UCB 奖励函数**：借鉴多臂老虎机的探索-利用平衡，用于假设质量评估，巧妙且合理
- **跨模型迁移性**：生成的假设不绑定特定 LLM，具有可迁移性

## 局限性 / 可改进方向
- **文献规模小且靠人工收集**：每个任务仅 10 篇论文，且手动搜索，未来应接入自动化文献检索（RAG pipeline）
- **仅限分类任务**：研究问题被形式化为分类，未覆盖数学、代码生成等非自然语言表征的任务
- **人类实验规模有限**：60 人，无法区分 HypoGeniC 和 HypoRefine 在人类评估中的差异
- **假设选择依赖消融**：给人类的 3 条假设是通过消融+主观判断选出的，缺乏系统化的假设推荐方法
- **超参数未充分搜索**：所有任务共用一套超参数

## 与相关工作的对比
- **vs HypoGeniC (Zhou et al., 2024)**：本工作的直接前身和数据驱动 backbone。HypoGeniC 只用数据，泛化性差；本文通过引入文献在 OOD 上平均提升 3.37%。
- **vs ResearchAgent (Baek et al., 2024) / SciMON (Wang et al., 2024)**：这些工作用知识图谱从文献生成假设，但要么缺少公开实现，要么难以适配新任务。本文自建了更简洁的文献假设生成管线。
- **vs AI Scientist (Lu et al., 2024)**：AI Scientist 追求全自动化科研流程（从 idea 到写论文），本文更聚焦于假设生成这一步，强调人类在科研中的主导作用（human agency）。

## 启发与关联
- **对 AI for Science 的启发**：theory + data 的协同思路可以推广到其他科学发现场景——例如在药物发现、材料科学中，已有文献知识（分子结构-活性关系）可以引导数据挖掘
- **对 LLM Agent 设计的启发**：双 agent 交替精炼（文献 agent + 数据 agent）是一种通用的多源信息融合模式
- **UCB 在假设管理中的应用**：可以借鉴到 idea 池管理、prompt 优化等场景
- **假设作为人类决策辅助**：不是用 AI 替代人类判断，而是用可解释的假设增强人类能力，这种 human-AI collaboration 的范式值得关注

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次融合文献和数据两种假设生成范式，idea 直觉且自然，但技术上主要是对 HypoGeniC 的扩展
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个数据集 × 2 个模型 × OOD/IND × 自动/人类评估，还有跨模型迁移和新颖性分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分，case study 展示直观
- 价值: ⭐⭐⭐⭐ 为假设生成建立了一套完整的评估范式，首次人类实验验证具有里程碑意义
