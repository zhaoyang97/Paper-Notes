# Document Summarization with Conformal Importance Guarantees

**会议**: NeurIPS 2025  
**arXiv**: [2509.20461](https://arxiv.org/abs/2509.20461)  
**代码**: [https://github.com/layer6ai-labs/conformal-importance-summarization](https://github.com/layer6ai-labs/conformal-importance-summarization)  
**领域**: NLP生成 / 可靠AI  
**关键词**: 文档摘要, Conformal Prediction, 重要性覆盖保证, 抽取式摘要, 分布无关  

## 一句话总结
首次将Conformal Prediction应用于文档摘要，通过校准句子重要性分数的阈值，为抽取式摘要提供用户可控的覆盖率($1-\alpha$)和召回率($\beta$)的严格统计保证，方法模型无关且仅需小规模校准集。

## 研究背景与动机

1. **领域现状**：LLM大幅提升了摘要质量，但在医疗、法律、金融等高风险领域，摘要遗漏关键信息可能造成严重后果。现有摘要方法（无论抽取式还是生成式）无法保证关键内容的覆盖。
2. **现有痛点**：（a）直接用LLM做生成式摘要可能产生幻觉且无法控制信息覆盖率；（b）抽取式方法虽然更忠实但缺乏理论保证；（c）用户无法指定"我希望至少保留80%的重要信息"这样的需求。
3. **核心矛盾**：摘要天然需要压缩（shorter is better），但安全关键场景要求不遗漏重要信息（complete is better）——需要在简洁性和完整性之间提供可控的平衡。
4. **本文要解决什么**：如何为摘要提供形式化的统计保证——以 $\geq 1-\alpha$ 的概率保留 $\geq \beta$ 比例的重要句子？
5. **切入角度**：Conformal Prediction已在分类/回归/QA中提供分布无关保证，本文将其从"精度保证"（conformal factuality ensuring retained claims are factual）扩展为"召回保证"（ensuring important sentences are retained）。
6. **核心idea**：在校准集上找到重要性分数阈值 $\hat{q}$，使得按此阈值过滤后的摘要以 $\geq 1-\alpha$ 概率保留 $\geq \beta$ 的重要句子。

## 方法详解

### 整体框架
输入长文档 $x = \{c_1, \ldots, c_p\}$（按句分割），通过重要性评分函数 $R(c;x)$ 为每句打分，然后用Conformal Prediction校准阈值 $\hat{q}$，保留分数 $\geq \hat{q}$ 的句子组成摘要 $y = F_{\hat{q}}(x)$。输出满足 $\mathbb{P}[B(y;y^*) \geq \beta] \geq 1-\alpha$ 的抽取式摘要。

### 关键设计

1. **广义覆盖保证**
   - 做什么：放宽经典Conformal Prediction的"全覆盖"要求，允许用户指定可接受的召回率 $\beta$。
   - 核心思路：定义召回 $B(y;y^*) = |y \cap y^*| / |y^*|$，目标为 $\mathbb{P}[B(y;y^*) \geq \beta] \geq 1-\alpha$。当 $\beta=1$ 退化为完全覆盖。对每个校准样本计算conformal score $S_\beta(x_i, y_i^*) = \max\{q \in \mathbb{R}^+ | B(F_q(x_i); y_i^*) \geq \beta\}$，即保持 $\beta$ 召回的最大阈值。取所有score的 $\lfloor\alpha(n+1)\rfloor/n$ 分位数作为 $\hat{q}$。
   - 设计动机：与conformal factuality的精度保证（$y \subseteq T(x,y^*)$）对称，这里面向召回保证。$\beta$ 参数让用户灵活控制——医疗场景可能需要 $\beta=1$（不漏）而新闻摘要可接受 $\beta=0.8$。

2. **重要性评分函数 $R(c;x)$**
   - 做什么：为文档中每个句子估计重要性分数。
   - 核心思路：提供两类评分方案——(a) LLM评分：用GPT-4o mini/Gemini/Llama等LLM prompt评分0-1；(b) 嵌入相似度：用SBERT计算句子嵌入，通过中心性（Cosine Centrality）、指向性（Sentence Centrality）、GUSUM、LexRank等图算法汇聚为重要性分数。
   - 设计动机：方法是模型无关的——任何能产生分数的方法都可以作为 $R$。LLM评分通常效果最好（AUPRC更高），但图方法不需要API调用。评分质量直接决定摘要在固定覆盖率下的简洁程度。

3. **混合抽取-生成Pipeline**
   - 做什么：先用Conformal Importance提取重要句子（有覆盖保证），再用LLM改写使文本更流畅简洁。
   - 核心思路：将摘要分解为两个子任务——信息筛选（抽取式，有保证）+ 润色合成（生成式，无保证但实际能保留大部分信息）。类似RAG把检索和生成分开。
   - 设计动机：纯抽取式摘要可能不通顺，纯生成式无法控制覆盖率。两步pipeline在实际中比直接LLM摘要有更高的信息保留率。

### 理论保证（Theorem 1）
在可交换性假设下，对 $\alpha \in [1/(n+1), 1]$：
$$1 - \alpha \leq \mathbb{P}[B(F_{\hat{q}}(x_{n+1}); y^*_{n+1}) \geq \beta] < 1 - \alpha + \frac{1}{n+1}$$
保证紧且仅需 $n=100$ 个校准样本即可实现 ~1% 的覆盖误差。

## 实验关键数据

### 主实验（重要性评分质量 AUPRC + 摘要简洁度）

| 评分方法 | ECT AUPRC | CSDS | CNN/DM | 平均简洁度(α=0.2,β=0.8) |
|---------|-----------|------|--------|------------------------|
| Random (正率) | 0.10 | 0.27 | 0.10 | 0% 压缩 |
| Cos. Sim. Centrality | 0.22 | 0.34 | 0.34 | 22%/11%/18% |
| GUSUM | 0.21 | 0.44 | 0.33 | 11%/24%/27% |
| LexRank | 0.22 | 0.43 | 0.32 | 16%/12%/20% |
| GPT-4o mini | **0.30** | **0.49** | 0.34 | 24%/25%/30% |
| Gemini 2.5 Flash | **0.31** | **0.55** | **0.44** | **26%/37%/33%** |
| Llama3-8B | 0.18 | 0.39 | 0.22 | 13%/11%/14% |

### 覆盖保证验证

| 设定 | 理论下界 | 实测覆盖率(400次随机划分) | 理论上界 |
|------|---------|------------------------|---------|
| α=0.1, β=1.0 | 90% | 90.2% | 91% |
| α=0.2, β=0.8 | 80% | 80.4% | 81% |
| α=0.3, β=0.6 | 70% | 70.1% | 71% |

所有实验中实测覆盖率严格落在理论bounds之间，验证Theorem 1。

### 关键发现
- Gemini 2.5 Flash的重要性评分在所有数据集上最优，GPT-4o mini次之，小模型（Llama3-8B、Qwen3-8B）弱于图方法。
- 仅需100个校准样本即可达到稳定保证（$1/(n+1) \approx 1\%$）。
- 混合pipeline实测信息保留率高于直接LLM摘要（86% vs 79% recall on ECT），同时更简洁。
- $\alpha$ 和 $\beta$ 提供了简洁性-完整性的连续控制，直接LLM摘要只能给出固定的单一trade-off点。

## 亮点与洞察
- **Conformal Prediction从精度→召回的创新扩展**：conformal factuality保证精度（保留的claim是对的），本文保证召回（重要的sentence被保留）——巧妙的对称翻转使CP适用于摘要场景。
- **极致简洁的框架**：核心只需要一个评分函数+校准阈值+过滤，可以无缝叠加到任何现有方法上。任何能给句子打分的方法都可以用。
- **$\alpha$-$\beta$双参数控制**：用户可以精细控制"我能接受多少风险（$\alpha$）"和"至少保留多少比例的重要信息（$\beta$）"，这在高风险应用中非常实用。

## 局限性 / 可改进方向
- 需要有ground truth标注的校准集，虽然100个样本不多，但在新领域获取标注仍有成本。
- 重要性完全由评分函数 $R$ 决定——如果 $R$ 质量差，满足覆盖但摘要会非常长（不够简洁）。
- 按句子粒度分割可能不适合所有场景（如对话数据、表格数据）。
- 混合pipeline的生成步骤无法保证维持覆盖率（虽然实际效果不错）。
- 可换性假设在分布漂移（如同一LLM的不同版本）场景下可能不成立。

## 相关工作与启发
- **vs Conformal Factuality (Mohri & Hashimoto)**: 他们保证QA中保留的claim是事实，本文保证摘要中重要sentence被保留——方向相反但框架类似。
- **vs BERTSum/TextRank等抽取方法**: 传统方法只做排序没有覆盖保证，本文在它们之上加了一个校准层即可获得保证。
- **vs 直接LLM摘要**: LLM给出无法控制的固定recall，本文实现连续可控。

## 评分
- 新颖性: ⭐⭐⭐⭐ CP应用于摘要的首次尝试，$\alpha$-$\beta$泛化有贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集、9种评分函数、400次随机划分验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，实验设计系统
- 价值: ⭐⭐⭐⭐ 对高风险摘要应用有直接实用价值
