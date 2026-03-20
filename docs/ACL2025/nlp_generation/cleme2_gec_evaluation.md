# CLEME2.0: Towards Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2407.00934](https://arxiv.org/abs/2407.00934)  
**代码**: [https://github.com/THUKElab/CLEME](https://github.com/THUKElab/CLEME)  
**领域**: 自然语言处理 / 语法纠错评估  
**关键词**: Grammatical Error Correction, 评估指标, 可解释性, Edit Disentangling, Reference-based Metric  

## 一句话总结
本文提出 CLEME2.0，一种可解释的 GEC 参考评估指标，通过将编辑解耦为四类（正确纠正 TP、错误纠正 FPne、欠纠正 FN、过纠正 FPun）并结合编辑加权技术，在 GJG15 和 SEEDA 两个人工评判数据集上达到了与人工判断最高相关性的 SOTA 结果。

## 背景与动机
- **GEC 评估的现状**：当前主流 GEC 评估指标（如 ERRANT、MaxMatch/M2）基于 Precision/Recall/F0.5 分数，虽然被广泛使用，但存在两个核心问题：
  1. **缺乏可解释性**：P/R/F 分数无法揭示 GEC 系统的具体弱点，开发者难以定位系统需要改进的方面
  2. **无法区分不同类型的错误编辑**：传统指标将所有 False Positive (FP) 编辑一视同仁，但实际上"纠正位置正确但修改错误"（wrong-correction）和"在不需要纠正的地方做了修改"（over-correction）是完全不同的错误类型
- **LLM 时代的新挑战**：大语言模型在 GEC 任务中表现出明显的过纠正（over-correction）倾向，会改变原文含义，但现有指标无法量化这一问题
- **语义信息的缺失**：传统指标对所有编辑赋予相同权重，忽略了不同编辑的重要性差异（如标点修改 vs 内容词修改）

## 核心问题
如何设计一个可解释的 GEC 评估指标，能够从多个维度量化 GEC 系统的表现特征（语法正确性 grammaticality 与忠实性 faithfulness），同时在与人工评判的相关性上超越现有指标？

## 方法详解

### 整体框架
CLEME2.0 的流程分为三步：
1. **编辑提取（Edit Extraction）**：使用 CLEME 的 chunk partition 技术，将源句、假设句和参考句同时对齐，切分为等数量的 chunk 序列
2. **解耦评分（Disentangled Scoring）**：将假设编辑分为 TP、FPne、FPun、FN 四类，分别计算 hit-correction、wrong-correction、under-correction、over-correction 四个维度的分数
3. **综合评分（Comprehensive Scoring）**：通过加权求和将四个分数合并为一个综合分数，并可选编辑加权技术（similarity-based 或 LLM-based）

### 关键设计
1. **编辑解耦（Edit Disentangling）**：
   - **TP（True Positive）**：假设 chunk 与参考 chunk 一致的正确纠正
   - **FPne（False Positive - Necessary）**：假设 chunk 与参考 chunk 不同，但参考确实有修改 → 纠正位置正确但内容错误
   - **FPun（False Positive - Unnecessary）**：假设做了修改但参考未修改 → 不必要的过度纠正
   - **FN（False Negative）**：假设未修改但参考有修改 → 遗漏纠正
   - 核心创新在于将传统的 FP 进一步拆分为 FPne 和 FPun，建立了四类编辑与四个系统特征的一一对应关系

2. **四维解耦分数**：
   - $Hit = \frac{TP}{TP + FP_{ne} + FN}$（正确纠正率）
   - $Wrong = \frac{FP_{ne}}{TP + FP_{ne} + FN}$（错误纠正率）
   - $Under = \frac{FN}{TP + FP_{ne} + FN}$（欠纠正率）
   - $Over = \frac{FP_{un}}{TP + FP_{ne} + FP_{un}}$（过纠正率）
   - 综合分数：$Score = \alpha_1 \cdot Hit + \alpha_2 \cdot (1-Wrong) + \alpha_3 \cdot (1-Under) + \alpha_4 \cdot (1-Over)$

3. **编辑加权技术（Edit Weighting）**：
   - **Similarity-based weighting**：基于 PTScore/BERTScore 计算每个编辑的语义重要性权重，通过模拟部分正确句子来衡量编辑对整体质量的影响
   - **LLM-based weighting**：使用 Llama-2-7B 对每个编辑打 1-5 分的重要性分数，利用 LLM 的语义理解能力来区分不同修改的重要程度

4. **权重因子确定**：通过交叉验证搜索最优权重：
   - Corpus-level: $\alpha_1, \alpha_2, \alpha_3, \alpha_4 = 0.45, 0.35, 0.15, 0.05$
   - Sentence-level: $\alpha_1, \alpha_2, \alpha_3, \alpha_4 = 0.35, 0.25, 0.20, 0.20$

## 实验关键数据

### GJG15 数据集（Corpus-level，6个参考集平均相关性）
| 指标 | 平均相关性 |
|------|-----------|
| M2 | 0.616 |
| ERRANT | 0.625 |
| PT-M2 | 0.666 |
| CLEME-dep | 0.633 |
| CLEME-ind | 0.635 |
| **CLEME2.0-dep** | **0.734** |
| **CLEME2.0-ind** | **0.775** |
| **CLEME2.0-sim-dep** | **0.790** |
| **CLEME2.0-sim-ind** | **0.817** |

### SEEDA 数据集（基于 TrueSkill 的平均相关性）
| 指标 | SEEDA-S (γ) | SEEDA-S (ρ) | SEEDA-E (γ) | SEEDA-E (ρ) | Avg. |
|------|------------|------------|------------|------------|------|
| ERRANT | 0.557 | 0.406 | 0.697 | 0.671 | 0.583 |
| CLEME-dep | 0.633 | 0.501 | 0.755 | 0.757 | 0.662 |
| GoToScorer | 0.929 | 0.881 | 0.901 | 0.937 | 0.912 |
| SOME | 0.892 | 0.867 | 0.901 | 0.951 | 0.903 |
| **CLEME2.0-dep** | **0.937** | **0.865** | **0.945** | **0.939** | **0.922** |
| **CLEME2.0-sim-ind** | **0.921** | **0.907** | **0.953** | **0.981** | **0.941** |

### 消融实验要点
- Hit-correction 和 under-correction 分数与人工判断呈中等正相关
- Wrong-correction 分数呈负相关，但其在综合分数中的权重较大，避免了仅偏好高置信度编辑的评估偏差
- Over-correction 在 corpus-level 呈小正相关，sentence-level 呈小负相关
- Similarity-based weighting 显著优于 LLM-based weighting（后者使用 Llama-2-7B，粒度太粗，仅 1-5 分）
- 即使不使用编辑加权，CLEME2.0 也能达到与其他指标相当或更优的性能

## 亮点
- **核心创新清晰有力**：将 FP 解耦为 FPne（necessary）和 FPun（unnecessary）是一个简单但深刻的设计，直接建立了四类编辑与四个系统特征的映射关系
- **实用价值高**：四维分数能精确定位 GEC 系统的弱点（如 CAMB 系统 27.1% 正确纠正、53.4% 欠纠正、47.0% 过纠正），为开发者和用户提供可操作的诊断信息
- **实验充分且鲁棒**：在 2 个人工评判数据集、6 个参考数据集上全面验证，涵盖 corpus/sentence 两个级别
- **兼顾可解释性与性能**：不仅提供了可解释的多维分析能力，综合分数还在与人工判断的相关性上取得了 SOTA
- **编辑加权技术**：引入 similarity-based 和 LLM-based 两种加权方式，解决了传统指标忽视语义重要性的问题

## 局限性 / 可改进方向
- **语言局限**：目前仅在英语数据集上验证，对其他语言的有效性未经测试
- **数据集局限**：实验主要基于 CoNLL-2014 共享任务的参考集，是二语学习者数据，缺乏多领域、多语种的验证
- **可解释性未经人工验证**：虽然声称提供可解释评估，但缺乏专门的人工评估实验来验证其可解释性
- **LLM 加权效果不佳**：使用 Llama-2-7B 的 LLM 加权效果不如 similarity-based，可能需要更大规模或更精细的 LLM 以提升效果
- **权重因子需要调参**：四个 α 因子通过交叉验证搜索，可能对新数据集不够鲁棒

## 与相关工作的对比
- **vs ERRANT/M2**：传统指标基于 P/R/F0.5，无法区分 FPne 和 FPun，且 ERRANT 的编辑提取依赖特定语言工具
- **vs CLEME**：CLEME2.0 继承了 CLEME 的 chunk partition 技术和双假设评估框架，但核心突破在于编辑解耦和加权
- **vs PT-M2**：PT-M2 虽用预训练模型加权，但仍基于 P/R/F 框架，不具备四维解耦的可解释性
- **vs Reference-less 指标（SOME、IMPARA）**：这些指标依赖微调模型，成本高且鲁棒性差（如 Scribendi Score 在不同数据集表现不一致），而 CLEME2.0 作为 reference-based 指标兼具可解释性和高相关性

## 启发与关联
- **评估指标设计的通用思路**：将错误类型精细化分类以提升可解释性的思路，可推广到其他 NLP 生成任务的评估（如机器翻译、文本摘要）
- **LLM 评估中的过纠正问题**：论文特别关注 LLM 的 over-correction 现象，这在当前 LLM 大规模应用于文本编辑/纠错的趋势下非常及时和重要
- **编辑加权的改进空间**：LLM-based weighting 效果不佳暗示了小规模 LLM 在细粒度评分任务上的局限，未来可探索更大规模 LLM 或专门微调的评分模型

## 评分
- 新颖性: ⭐⭐⭐⭐ 编辑解耦（FPne vs FPun）的想法简洁有效，但整体框架是对 CLEME 的增量改进
- 实验充分度: ⭐⭐⭐⭐⭐ 两个人工评判数据集、六个参考集、corpus/sentence 两级、多种加权方式，消融分析和 case study 充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导明确，但部分表格信息密度过高
- 对我的价值: ⭐⭐⭐⭐ 对 GEC 评估领域有重要参考价值，四维解耦分析的思路可迁移到其他 NLP 评估任务
