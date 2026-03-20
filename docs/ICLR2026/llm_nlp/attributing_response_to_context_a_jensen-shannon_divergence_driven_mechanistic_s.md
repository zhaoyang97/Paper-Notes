# Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation

**会议**: ICLR 2026  
**arXiv**: [2505.16415](https://arxiv.org/abs/2505.16415)  
**代码**: [https://github.com/ruizheliUOA/ARC_JSD](https://github.com/ruizheliUOA/ARC_JSD)  
**领域**: RAG / 可解释性 / 机制分析  
**关键词**: 上下文归因, Jensen-Shannon散度, RAG, 机制可解释性, 注意力头, MLP层, 幻觉缓解  

## 一句话总结
提出ARC-JSD方法，通过计算完整上下文与逐句消融上下文下的响应分布的Jensen-Shannon散度，在无需微调、梯度计算或代理模型的情况下实现高效精准的RAG上下文归因，并结合Logit Lens进行机制分析，定位负责上下文归因的注意力头和MLP层，通过门控操作降低约39%的幻觉率。

## 研究背景与动机

1. **领域现状**：RAG通过结合外部上下文提升LLM生成准确性，但如何可靠地将生成内容归因到具体上下文片段（context attribution）仍是公开挑战。
2. **现有方法的痛点**：
   - 人工标注成本高昂（Zeng et al., 2021; Slobodkin et al., 2024）
   - 梯度方法（MIRAGE）需要反向传播，计算量大
   - ContextCite需数百次前向推理来训练线性代理模型
   - DPO微调方法（SelfCite）需要额外训练
3. **核心矛盾**：现有方法在归因准确率和计算效率之间难以兼顾——要么准确但昂贵，要么快速但不够精确。
4. **切入角度**：利用JSD的对称性、有界性（$[0, \log 2]$）、尺度无关等数学性质，直接衡量消融单个上下文句子后响应分布的变化，跳过代理模型训练。
5. **核心idea**：如果移除某个上下文句子后模型输出分布变化最大（JSD最高），则该句子对生成响应最关键。

## 方法详解

### 整体框架
ARC-JSD分两大模块：(1) 基于JSD的上下文归因（定位关键句子）；(2) 基于JSD+Logit Lens的机制分析（定位关键注意力头和MLP层）。

### 关键设计

1. **JSD驱动的上下文归因（§4.1）**
   - 做什么：对上下文中每个句子 $c_i$，计算移除它后响应分布与完整上下文响应分布的JSD差异
   - 核心公式：$\text{JSD}(c_i) = \sum_{j=1}^{|\mathcal{R}|} \text{JSD}(\mathcal{P}_{\text{LM}}(r_j|\mathcal{C},\mathcal{Q}) \| \mathcal{P}_{\text{LM}}(r_j|\mathcal{C}_{\text{ABLATE}}(c_i),\mathcal{Q}))$
   - JSD最高的句子即为最相关的上下文：$c_{\text{Top-1}} = \arg\max_{c_i} \text{JSD}(c_i)$
   - 设计动机：逐响应token累加JSD，既捕获局部敏感token（如实体名），又不被高熵token主导

2. **JSD+Logit Lens机制分析（§5）**
   - 做什么：将JSD分析从模型整体下沉到每个注意力头和MLP层
   - 核心思路：对每个注意力头 $(\ell,h)$ 和每个MLP层 $\ell$，通过Logit Lens将中间表示投影到词汇空间，分别计算全上下文vs消融上下文下的JSD
   - 关键发现：负责上下文归因的注意力头主要集中在**高层**，MLP层在中高层贡献最大，与Wu et al. (2025a)的NIAH设置发现部分吻合

3. **语义增益验证（§6）**
   - 做什么：从另一角度验证JSD定位的组件——衡量注意力/MLP对正确答案的余弦相似度提升
   - 核心思路：定义 $\Delta^{\ell,\text{Attn}}$ 和 $\Delta^{\ell,\text{MLP}}$ 衡量每层注意力和MLP的语义增益
   - 通过Spearman $\rho$ 计算JSD排序与语义增益排序的相关性，Table 3显示显著正相关，互相验证有效性

4. **JSD门控降低幻觉（§7）**
   - 做什么：用JSD分数作为置信度门控，抑制语义增益为负的高JSD注意力头和MLP层
   - 门控公式：$\text{Mask} = 0.7 + 0.3 \times \text{sigmoid}(G)$，当 $G < 0$ 时mask接近0.7，缩减该组件的贡献
   - 效果：Qwen2-7B-IT在HotpotQA上幻觉率从13.4%降至8.2%（↓39%），Factual F1基本不变（76.1→75.9）

### 计算效率
- ARC-JSD的FLOPs为 $2PT|\mathcal{C}|^2$（$P$为参数量，$T$为每句token数，$|\mathcal{C}|$为句子数）
- ContextCite(256次调用) FLOPs为 $2PT \times 256^2$，当 $|\mathcal{C}| < 256$ 时ARC-JSD更便宜
- MIRAGE需要梯度计算，FLOPs为 $4PT|\mathcal{C}|(2|\mathcal{C}|+1)$
- 实际达到约3倍加速

## 实验关键数据

### 数据集与模型
- 三个QA数据集：TyDi QA（440, 单跳）、HotpotQA（1000, 多跳）、MuSiQue（1000, 多跳，平均93.6句上下文）
- 四个指令微调模型：Qwen2-1.5B/7B-IT, Gemma2-2B/9B-IT
- 额外泛化验证：LLaMA-3.1-8B-IT, Qwen3-Next-80B-A3B-IT

### 主实验（上下文归因Top-1准确率）
- ARC-JSD在MuSiQue上的compute-accuracy trade-off上一致优于所有基线（Fig.2a）
- 平均归因准确率提升约**10.7%**
- ContextCite-32虽然在 $|\mathcal{C}|>32$ 时计算更快，但归因准确率始终低于ARC-JSD
- ARC-JSD在Pareto optimal front上，兼顾准确率和效率

### 指标对比消融（§8, Fig.6）
| 指标 | 相对表现 |
|------|---------|
| JSD | **最优**，对称、有界、尺度无关 |
| KL | 当消融分布有零概率时爆炸，无法跨层比较 |
| TV | 有界但过粗糙，无法区分高熵尾部vs关键token的概率转移 |
| Wasserstein | 需要定义152K词汇上的距离度量，$O(V^3)$复杂度 |
| MMD | 需要核函数和token距离定义 |

### 机制分析验证
- Table 3：JSD排序与语义增益排序的Spearman $\rho$ 在所有数据集和模型上显著（$p<0.05$ 或 $p<0.01$）
- Table 5：消融JSD top-10注意力头的JSD变化（2.23±0.12）显著大于随机10个（1.53±0.76）

### 幻觉降低（Table 4）
| 设置 | 幻觉率 | Factual F1 |
|------|--------|-----------|
| Base RAG | 13.4% | 76.1 |
| Gate Top-5 Attn & MLP | **8.2%** | 75.9 |
| Gate Random 5 | 12.7% | 69.4 |

### 泛化性
- LLaMA-3.1-8B-IT和Qwen3-Next-80B-A3B-IT（MoE）上同样保持compute-accuracy优势（Fig.7）

## 亮点与洞察
- **简洁高效**：整个方法概念清晰——逐句消融+JSD比较，无需训练任何辅助模型，可即插即用到任意RAG系统
- **JSD的选择有理论基础**：对称性避免了方向问题，有界性使跨层比较合理，与KL/TV/Wasserstein的对比消融很convincing
- **机制分析闭环**：JSD定位→语义增益验证→因果消融验证→门控应用，形成了完整的验证与应用链条
- **可视化的MLP层语义演变**：通过Logit Lens展示Qwen2如何在高层从中文token逐步转换为英文（"一只→A", "翅膀→wings"），与语言锚定现象一致
- **实用价值**：门控机制无需重训练即可降低39%幻觉率

## 局限性 / 可改进方向
- **计算复杂度与上下文长度平方成正比**：$O(|\mathcal{C}|^2)$，当上下文超长（如数百句）时仍然昂贵，论文未讨论如何规模化
- **仅评估Top-1归因**：现有QA数据集只有句子级gold label，更细粒度（短语级/子句级）的归因能力未被充分验证
- **门控实验规模有限**：仅在200个HotpotQA样本上验证幻觉降低效果，缺乏大规模和多数据集的验证
- **缺少与SelfCite等微调方法的直接准确率对比**——仅在compute-accuracy tradeoff图上比较
- **"all JSD scores small"的阈值选择**（0.02 bits）缺乏系统分析

## 相关工作与比较
- **vs ContextCite (Cohen-Wang et al., 2024)**：ContextCite需数百次前向推理训练代理模型，线性假设可能错失非线性关系；ARC-JSD直接用JSD量化真实分布变化
- **vs MIRAGE (Qi et al., 2024)**：梯度方法计算昂贵且token-level聚合到sentence-level有信息损失
- **vs Wu et al. (2025a)**：其NIAH设置中模型做复制粘贴，本文评估的是更实际的改写+整合场景
- **vs Sun et al. (2025)**：Sun聚焦定位导致幻觉的源头，本文定位导致正确生成的源头，两者互补

## 评分
- 新颖性: ⭐⭐⭐⭐ JSD用于RAG上下文归因是新颖且合理的，理论与实践结合好
- 实验充分度: ⭐⭐⭐⭐ 3数据集4+2模型，多角度消融和验证，结果一致性好
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，公式推导连贯，case study直观
- 价值: ⭐⭐⭐⭐ 即插即用的归因方法+机制分析洞见，对RAG透明化有实际推动
