# A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks

**会议**: CVPR 2025  
**arXiv**: [2603.12998](https://arxiv.org/abs/2603.12998)  
**代码**: https://github.com/Supltz/Debias_VLM  
**领域**: 多模态VLM  
**关键词**: VLM debiasing, fairness, closed-form solution, Pareto-optimal, cross-modal space

## 一句话总结
提出一个 training-free、data-free 的 VLM 去偏方法，通过在 cross-modal 空间中推导闭式解，实现 Pareto-optimal 的公平性与效用保持，在零样本分类、text-to-image 检索和生成三个下游任务中全面超越已有方法。

## 研究背景与动机
1. **领域现状**：CLIP 等 VLM 在大量下游任务中表现优异，但从大规模网络爬取数据中继承了性别、种族等社会偏见（如 "nurse" 与 "female" 相似度异常高）。
2. **现有痛点**：现有去偏方法各有缺陷 — 有的需要训练额外网络（DeAR、FairerCLIP）、有的需要带敏感属性标注的数据（SFID、CLIP-clip）、有的只处理单模态（BiasedPrompt、SANER）、有的只针对单任务（PRISM 仅做分类），而且几乎所有方法都没有效用保持的理论保证。
3. **核心矛盾**：去偏（fairness）和效用保持（utility）是一对天然矛盾 — 去掉偏见信息不可避免地损失语义信息。之前的方法要么牺牲效用，要么需要大量调参来平衡，且无法提供性能损失的理论上界。
4. **本文要解决什么？** 如何在零训练、零数据的条件下，同时对视觉和文本两个模态去偏，且提供可证明的效用损失上界？如何统一处理多个下游任务？
5. **切入角度**：将去偏问题转化为跨模态单位超球面上的优化问题，通过正交分解将 embedding 拆分为"属性泄露"和"中性内容"两个分量，然后在 Pareto 前沿中找最优点。
6. **核心idea一句话**：将 VLM 去偏建模为超球面上的 Chebyshev scalarisation 问题，推导出闭式最优解 $\alpha^\star$，实现 Pareto-optimal 的公平-效用权衡。

## 方法详解

### 整体框架
输入：VLM 的原始 image/text embedding $\vec{e}_I, \vec{e}_T$，输出：去偏后的 embedding $\vec{u}_I, \vec{u}_T$。分两步：(1) 用 LLM 构建 group prototype 来定义属性子空间 $\mathcal{A}$；(2) 在 cross-modal 空间中搜索最优的去偏 embedding。

### 关键设计

1. **LLM-Guided Group Prototype Construction**:
   - 做什么：为每个敏感属性组（如 male/female）构建代表性 embedding
   - 核心思路：用 LLM（GPT-5）将输入 prompt 注入属性词并生成多种措辞变体（如 "male doctor" → "man doctor", "masculine doctor"），然后取球面均值 $\vec{p}_g$ 作为 group prototype。由 prototype 差向量 $\vec{a}_i = \vec{p}_{g_i} - \vec{p}_{g_1}$ 张成属性子空间 $\mathcal{A}$
   - 设计动机：之前方法直接用单一 prompt 定义属性方向，忽略了同一属性的多种语言表达（"man"/"gentleman"/"boy" 都是 male），导致属性子空间不准确

2. **闭式去偏求解**:
   - 做什么：在超球面 $\mathbb{S}^{d-1}$ 上找去偏 embedding $\vec{u}^\star$
   - 核心思路：将 embedding 正交分解为 $\vec{e} = \vec{e}_{\mathcal{A}_\parallel} + \vec{e}_{\mathcal{A}_\perp}$（属性泄露 + 中性内容）。定义双目标：最小化属性泄露 $L(\alpha) = \alpha$ 和效用损失 $V(\alpha) = 1 - \alpha\|\vec{e}_{\mathcal{A}_\parallel}\| - \sqrt{1-\alpha^2}\|\vec{e}_{\mathcal{A}_\perp}\|$。通过 Lemma 1-2 将高维搜索降到 1D，再用 Chebyshev scalarisation 解 minimax 问题，得到闭式解：$\alpha^\star = \frac{E - \|\vec{e}_{\mathcal{A}_\perp}\|\sqrt{E^2 - \|\vec{e}_{\mathcal{A}_\parallel}\|^2}}{E^2 + \|\vec{e}_{\mathcal{A}_\perp}\|^2}$
   - 设计动机：之前方法（Orth-Proj）直接正交投影相当于 $\alpha=0$（完美公平但效用最差），本文找到 Pareto 前沿中间的最优点，在公平和效用间取得最佳平衡

3. **效用上界保证**:
   - 做什么：提供可证明的 cross-utility 损失上界
   - 核心思路：Proposition 1 证明 $\ell_{cross} \leq \sqrt{2\ell_{self}^{(I)}} + \sqrt{2\ell_{self}^{(T)}}$，因此只需约束 self-utility loss 即可同时保证跨模态对齐。Theorem 1 进一步给出精确的上界表达式
   - 设计动机：之前方法声称保持效用但无理论保证，只能靠经验调参

### 损失函数 / 训练策略
本方法是 training-free 的，不需要任何训练。核心优化通过闭式解直接计算，不涉及梯度下降或迭代。推理时只需计算一次属性子空间投影矩阵（可预计算并缓存），之后对每个 embedding 仅需 $O(d)$ 的向量运算。
整体管线的计算瓶颈在 LLM 生成变体的一次性开销，属性子空间一旦构建后即可复用于所有样本。

## 实验关键数据

### 主实验
| 数据集/任务 | 指标 | 本文 | 之前SOTA | 说明 |
|------------|------|------|----------|------|
| CelebA 分类 | F1↑ | **56.5** | 53.1 (FairerCLIP) | 效用大幅提升 +3.4 |
| CelebA 分类 | $\Delta_{EO}^{Max}$ (G×A) ↓ | **40.1** | 40.0 (RoboShot) | 公平性持平 |
| Flickr30K 检索 | R@5↑ | **90.4** | 87.9 (FairerCLIP) | 检索效用领先 |
| Flickr30K 检索 | MS@1000↓ | **11.8** | 11.7 (CLIP-clip) | 公平性持平 |
| T2I 生成 | $\overline{SP}_5$↓ | **28.8** | 28.4 (Orth-Proj) | 公平性接近 |
| T2I 生成 | Acc^G↑ | **74.6** | 67.2 (SFID) | 效用大幅领先 +7.4 |

### 消融实验
| 配置 | MS@1000 (Flickr30K) | $\Delta_{EO}^{Max}$ (CelebA) | 说明 |
|------|-------|---------|------|
| Full model | 11.8 | 40.1 | 完整模型 |
| 只用 anchor embedding | 13.4 | 41.1 | 不用 LLM 变体，效果下降 |
| 只用 mean embedding | 14.1 | 41.8 | 不用 anchor，更差 |
| 只去偏 image | 13.4 | 41.7 | 单模态不够 |
| 只去偏 text | 13.3 | 41.1 | 单模态不够 |
| 换 DeepSeek v3.2 | 12.0 | 40.1 | 对 LLM 不敏感 |
| 换 Gemini 2.5 Pro | 11.8 | 40.4 | 对 LLM 不敏感 |

### 关键发现
- 同时去偏两个模态（I&T）比只去偏单模态效果一致地更好，验证了偏见在跨模态对齐中被编码的假设
- LLM prototype 构造对具体 LLM 选择不敏感（GPT-5/DeepSeek/Gemini 效果接近），因为任务简单（生成同义变体），不需要高级推理能力
- 需要标注数据的方法反而不一定更好 — 它们对数据域敏感（在人脸数据上训练的去偏网络在全身图像上效果差）
- 方法在推理时计算开销极小（仅需一次矩阵投影和闭式公式求值），相比训练型方法的 GPU 时间优势明显

## 亮点与洞察
- **闭式解替代迭代优化**：通过精巧的数学推导（Lemma 1 降维 + Lemma 2 限定搜索范围 + Chebyshev scalarisation 处理多目标），将看似复杂的超球面优化化为一行公式。巧妙之处在于证明了最优解必然在原始 embedding 的两个分量张成的 2D 平面内
- **理论保证 + 实用性兼得**：不像大多数公平性方法只有经验结果或只有理论但不实用，这篇同时提供了严格的上界证明和多任务的实验验证
- **LLM 辅助属性子空间构建**可迁移到其他需要定义语义子空间的任务：比如可以用 LLM 生成同义变体来构建更鲁棒的概念方向
- **方法的模块化设计**值得学习：属性子空间构建和闭式求解两个阶段完全解耦，可以独立替换或改进各自组件

## 局限性 / 可改进方向
- 效用保证是在 embedding 空间（cosine similarity）层面的，不能直接保证下游 task-specific 指标（F1、Recall）；实际部署时需要额外的 task-level 评估来验证端到端性能
- 属性子空间假设偏见可以用线性子空间建模，对非线性偏见可能不够；现实中的偏见往往是多因素交织的（如种族×性别的 intersectional bias），线性子空间难以完全捕捉
- 只在 CLIP 系列模型上验证，未测试更新的 VLM（如 LLaVA、InternVL）；这些模型的 embedding 结构不同，闭式解是否仍然适用需要进一步研究
- 未来可扩展到解码器端的生成模型去偏
- LLM prototype 构造依赖于人工定义的敏感属性类别，对于新兴或细粒度的偏见维度（如年龄、disability）需要额外的领域知识来定义 group

## 相关工作与启发
- **vs PRISM/Orth-Proj**: 它们做正交投影（$\alpha=0$ 极端情况），本文在 Pareto 前沿找最优点，理论上严格优于它们。Orth-Proj 可以看作本文方法的一个特例（边界条件）
- **vs FairerCLIP/DeAR**: 它们需要训练+标注数据，本文 training-free 且效果更好，说明数据驱动不一定是最优路线。FairerCLIP 在分布偏移场景下泛化较差
- **vs SANER**: 都用到属性方向概念，但 SANER 用静态词表，本文用 LLM 动态生成更准确
- **vs BiasedPrompt**: 只处理文本模态的偏见，忽略了视觉编码器中的偏见传播，本文同时处理双模态更全面
- **启发**：Chebyshev scalarisation 处理多目标优化的范式可以迁移到其他需要 fairness-utility tradeoff 的场景（如推荐系统、NLP 分类器去偏）
- **与 optimal transport 的潜在联系**：去偏过程可看作将 biased embedding 分布 transport 到公平分布上，未来可探索 OT 视角下的扩展

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 闭式解处理 VLM 去偏是首创，数学推导完整优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 三个任务、多个数据集、消融和敏感性分析都很充分
- 写作质量: ⭐⭐⭐⭐⭐ 问题形式化清晰，定理推导逻辑连贯
- 价值: ⭐⭐⭐⭐ 公平性领域的重要贡献，但实际部署场景还需验证
