---
title: >-
  [论文解读] How can embedding models bind concepts?
description: >-
  [ICML2026][信息检索/RAG][概念绑定] 本文把 "embedding 模型为什么不会绑定概念" 形式化成 "binding function 的复杂度问题"：通过几何分析证明 CLIP 的场景嵌入可加性分解成对象与概念之和（解释了单模态可探测、跨模态却失败），并在受控 Transformer 上证明当数据覆盖足够时，模型会学到一个由概念间**乘性交互**主导的低复杂度 binding，从而实现对未见对象组合的系统性泛化。
tags:
  - "ICML2026"
  - "信息检索/RAG"
  - "概念绑定"
  - "CLIP"
  - "嵌入几何"
  - "组合泛化"
  - "多模态对齐"
---

# How can embedding models bind concepts?

**会议**: ICML2026  
**arXiv**: [2605.31503](https://arxiv.org/abs/2605.31503)  
**代码**: 有（论文文末公开仓库）  
**领域**: 可解释性  
**关键词**: 概念绑定, CLIP, 嵌入几何, 组合泛化, 多模态对齐

## 一句话总结
本文把 "embedding 模型为什么不会绑定概念" 形式化成 "binding function 的复杂度问题"：通过几何分析证明 CLIP 的场景嵌入可加性分解成对象与概念之和（解释了单模态可探测、跨模态却失败），并在受控 Transformer 上证明当数据覆盖足够时，模型会学到一个由概念间**乘性交互**主导的低复杂度 binding，从而实现对未见对象组合的系统性泛化。

## 研究背景与动机
**领域现状**：CLIP 这类双编码器视觉-语言嵌入模型在跨模态检索上接近 "bag-of-concepts" 行为：能识别 "红色"、"立方体" 这类单个概念，却在多对象场景里分不清 "红立方体 + 蓝球" 与 "蓝立方体 + 红球"，这就是经典的 concept binding 失败。先前工作已经反复测出该现象，并把锅归给编码器细粒度不足、否定/空间推理弱或概念-对象之间存在本质 trade-off。

**现有痛点**：先前的解释停留在 "behavioral level"——观测到 CLIP 答错就推测某种能力缺失，却没解释一个矛盾：在单模态内（只用图像编码器或只用文本编码器训探针），CLIP 居然可以恢复对象级信息。"不会跨模态绑定" 与 "单模态内能解码对象" 这两件事如何在同一向量里共存？

**核心矛盾**：跨模态对齐要求图像编码器 $B_{\text{img}}$ 和文本编码器 $B_{\text{txt}}$ 给同一场景产出可比的嵌入；只要二者各自学到的 binding 规则不同，对未见对象组合就会失配。问题因此从 "能不能识别对象" 转化为 "二者学到的 concept→object 映射是否是同一个简单规则"。

**本文目标**：(1) 刻画 CLIP 的多对象场景嵌入到底长什么样；(2) 测量它隐式实现的 binding 函数的复杂度；(3) 在受控 Transformer 上验证 binding 是否可被学到、并指出其结构性形式。

**切入角度**：把 binding 写成函数 $B:\mathcal{S}\to\mathbb{R}^d$（场景到嵌入），借鉴 MDL / Occam 原则——若两侧编码器学到的 $B$ 都是低复杂度、组合性的规则，则它们更容易收敛到同一规则并在 OOD 上对齐；高复杂度则各自记忆训练分布，OOD 失败。

**核心 idea**："CLIP 不会绑定" 不是结构性缺陷，而是它学到的 binding function 太高复杂度；只要数据覆盖足够，dual-encoder Transformer 自己就能学到由**乘性交互**实现的低复杂度 binding，从而跨模态对齐到未见组合。

## 方法详解
本文不是一个 "新模型"，而是 "一个形式化框架 + 一组几何/容量诊断实验"。整体管线分为三段：先给 binding 一个可证伪的数学定义，再用这个定义解剖 CLIP 的失败模式，最后用受控合成数据训出一个会绑定的对照组并解释其内部结构。

### 整体框架
**形式化层**：定义概念空间 $\mathcal{C}=\mathcal{C}_1\times\cdots\times\mathcal{C}_C$，对象 $\bm{o}=(c_1,\dots,c_C)$ 是概念取值的元组，场景 $\bm{s}=(\bm{o}_1,\dots,\bm{o}_m)$ 是对象元组的集合。一个模型 $(f,q)$ 由场景编码器 $f$ 与查询编码器 $q$ 组成，用余弦相似度评分。"绑定" 被拆成两个可独立测量的能力——概念识别（场景中所有出现的概念值的得分高于不出现的）与对象识别（场景中所有出现的完整对象的得分高于不出现的对象）。两者都满足才叫 binding；只满足第一项就是 bag-of-concepts。binding function 定义为 $B_{\text{img}}(\bm{s}):=f(\bm{x}_{\bm{s}})$ 和 $B_{\text{txt}}(\bm{s}):=q(\bm{y}_{\bm{s}})$。

**几何诊断层**（第 4 节）：对真实 CLIP，通过子集平均估计对象嵌入 $\bm{u}_{\bm{o}}$ 与概念嵌入 $\bm{u}_c$，验证场景嵌入是否近似可加分解为对象之和（Level-I），以及对象是否进一步分解为概念之和（Level-II）。同时做 "对象级编辑" 实验——直接在嵌入空间做 $\tilde{\bm{z}}=f(\bm{x}_{\bm{s}})-\bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_1'}$，看检索/探针是否表现为反事实场景。

**容量诊断层**（第 5 节）：训一个小 MLP 近似器 $g(\bm{o}_1,\bm{o}_2)$，输入离散概念索引、输出预测的场景嵌入，最小化 $\sum_{\bm{s}}\|f(\bm{x}_{\bm{s}})-g(\bm{o}_1,\bm{o}_2)\|^2$，扫宽度 $\{64,256,1024,4096\}$ 与训练对象覆盖率 $\{0.1,\dots,0.9\}$，在持出对象上测概念/对象识别。同时从零训 dual-encoder Transformer（≈20M 参数，输出 $\mathbb{R}^{512}$，AdamW + 对比损失），在合成数据上系统地变 $C$、$V$、覆盖率，观察 binding 何时泛化。

### 关键设计

**1. 可加分解的两级假设（Level-I / Level-II）：把"场景嵌入由什么组成"落成可证伪的几何性质**

先前工作只在 behavioral level 上观察到 CLIP 答错，却解释不了"单模态可绑定、跨模态却失败"这个矛盾，本设计把它转成一个能直接测的几何假设：$f(\bm{x}_{\bm{s}})\approx \bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_2}\approx \sum_{i}\bm{u}_{c_{1,i}}+\sum_{i}\bm{u}_{c_{2,i}}$，即场景嵌入先可加分解为对象之和（Level-I），对象再分解为概念之和（Level-II）。验证时用三种估计器拿到对象嵌入 $\bm{u}_{\bm{o}}$——多对象场景平均 avg、按对象在元组中位置条件平均 avg+pos、单对象场景平均 single-obj——再用 $R^2$、检索准确率、线性探针准确率三套指标交叉验证。最关键的是"靶向去除"消融：从场景嵌入里减去对应概念分量后概念解码崩到接近随机、对象解码几乎不动，而减去对象分量则两者同时崩，这就把"对象信息存在于对象级分量"钉死。之所以单模态探针能恢复对象，正是因为对象级分量 $\bm{u}_{\bm{o}}$ 显式地"打包"在嵌入里、把每个对象的概念组合压成一个不可加的向量；但它并不要求和文本侧的 $\bm{u}_{\bm{o}}^{\text{txt}}$ 由同一函数得到，于是跨模态自然失配。

**2. binding function 的容量诊断：用小近似器的泛化能力当复杂度尺子**

单纯训探针只能说"对象信息在嵌入里"，说不了"concept→object 的映射有没有简单形式"，于是本设计把抽象的"binding 是否简单"操作化为"能否被小近似器在持出对象上拟合"。具体做法是训一族近似器把离散概念索引映到 CLIP 场景嵌入，再用之前训好的线性探针在持出对象的预测嵌入上测概念/对象识别。结果很尖锐：覆盖率超过 0.3 时概念识别稳定到 ≥80%，但对象识别即使把 MLP 宽度推到 4096、覆盖率推到 0.9 也只能停在 ~20%，换成 XGBoost / Random Forest 现象一致——这说明 CLIP 的 binding 不是"近似器太弱"，而是它本身就是组合特异、近似记忆式的高复杂度函数。这里 MLP 充当的是 SGD 偏好简单解的代理：连它都拟不出泛化解，就支撑了"binding function 高复杂度"这一比"信息缺失"更强的结论。

**3. 乘性交互探针（Additive / Per-obj. products / Global product）：给"会泛化的 binding"一个可重用的函数形式**

在受控 Transformer 训出"会泛化的 binding"之后，要说清它为何能跨模态对齐到未见组合，就必须追问其函数形式到底是纯加性、对象内乘性还是全局乘性。本设计用三种结构化探针逐级拟合场景嵌入：Additive 形如 $\sum_{i=1}^{2}\sum_{k=1}^{2}\bm{u}_{k,c_{ik}}$，是 bag-of-concepts 基线；Per-obj. products 加上对象内乘积 $\sum_i\prod_k \bm{v}_{i,k,c_{ik}}$；Global product 进一步加上跨对象乘积 $\prod_i\prod_k \bm{v}_{i,k,c_{ik}}$。乘积项给每个概念组合一个独立向量，正是纯加性结构表达不了的"绑定信号"，且因为只是最小程度偏离加性、结构本身就是组合性的，两侧编码器更容易收敛到同一 binding 规则。Fig. 9 在 ~500 个变超参的模型上证明：OOD 对象识别准确率与 Global product 探针的拟合质量强正相关，而把同一探针套到 CLIP / DINOv2 上则只恢复概念识别、对象识别接近零——这把"低复杂度 = 乘性结构 = 跨模态可对齐"钉在了同一条经验轴上。

### 损失函数 / 训练策略
受控双编码器沿用 CLIP 的对称对比损失，cosine similarity 评分；优化器 AdamW；每个编码器约 20M 参数，输出维度 $d=512$。诊断 MLP 用 MSE（$\ell_2$）回归到目标嵌入。所有训练在合成多对象数据上完成（CLEVR、CLEVR-2D、PUG:SPARE 以及由 Gemini Nano Banana 2 生成的自然图像），通过控制 $(C,V)$ 和训练对象覆盖率 $\rho_{\text{train}}\in[0.1,0.9]$ 来扫泛化曲线。

## 实验关键数据

### 主实验

| 数据集 | 模型 | $R^2$ (avg / avg+pos) | Retrieval | Probing |
|--------|------|------|----------|------|
| Text (合成 caption) | CLIP | 0.90 / 0.92 | 0.97 | 0.99 |
| PUG:SPARE | CLIP | 0.75 / 0.84 | 0.93 | 0.98 |
| PUG:SPARE | DINOv2 | 0.78 / 0.86 | 0.86 | 0.98 |
| CLEVR | CLIP | 0.78 / 0.83 | 0.94 | 0.96 |
| Text | Random-init | 0.47 / 0.69 | 0.42 | 0.82 |

CLIP 场景嵌入可由对象分量之和高质量重构（$R^2$ 0.75–0.92），且重构后的 retrieval / probing 接近原模型，三对象 CLEVR、含遮挡场景以及自然图像（Gemini Nano Banana 2）都保持。这把 Level-I 的可加分解从 toy 数据外推到了更真实的场景。

| Dataset | Model | Probing (avg / avg+pos / single-obj) | Retrieval (avg / avg+pos / single-obj) |
|---------|-------|--------------------------------------|----------------------------------------|
| CLEVR | CLIP | 0.98 / 0.98 / 0.86 | 1.00 / 1.00 / 0.97 |
| CLEVR-2D | CLIP | 0.98 / 0.98 / 0.92 | 0.99 / 0.99 / 0.97 |
| PUG:SPARE | CLIP | 0.94 / 0.95 / – | 0.86 / 0.94 / – |
| PUG:SPARE | DINO | 0.97 / 0.97 / – | 0.48 / 0.76 / – |

直接在嵌入空间做 "对象替换" $\tilde{\bm{z}}=f(\bm{x}_{\bm{s}})-\bm{u}_{\bm{o}_1}+\bm{u}_{\bm{o}_1'}$ 即可得到对应反事实场景表现的嵌入；尤其是用 single-object 场景估出的对象嵌入也能编辑多对象场景（CLEVR / CLEVR-2D 检索仍达 0.97），证明对象分量在嵌入里几乎是 "可拔插" 的几何对象。

### 消融实验

| 配置 | 文本 Conc. / Obj. | 图像 Conc. / Obj. | 说明 |
|------|-------------------|-------------------|------|
| CLIP-B/32 原始 | 1.00 / 1.00 | 0.94 / 0.96 | baseline |
| − concept 分量 | 0.06 / 0.99 | 0.05 / 0.85 | 概念解码崩，对象解码几乎不动 |
| − object 分量 | 0.05 / 0.04 | 0.02 / 0.01 | 概念与对象同时崩 |
| permute concept (control) | 0.92 / 0.99 | 0.99 / 0.97 | 减错分量不掉点 |
| permute object (control) | 0.96 / 1.00 | 0.86 / 0.92 | 减错对象不掉点 |

这张表是全文最关键的因果证据：嵌入里的对象级分量同时承载了 "对象身份" 与 "对象内部的概念组合"，而概念分量只承载概念本身。

### 关键发现
- CLIP 的 binding 不是 "信息缺失" 而是 "函数复杂度过高"：高容量 MLP / XGBoost / RF 在持出对象上把对象识别压在 ≤20%，但概念识别可以稳到 80%+，意味着 concept→object 的映射在不同组合上接近独立，几乎是记忆。
- binding 泛化对数据覆盖率呈现锐相变：在 $|O|=125{,}000$ 上，训练对象覆盖率从 30% 涨到 40% 时，对象识别从接近随机跃到接近完美；对象空间越大，所需相对覆盖率越低（$|O|\ge 2{,}500$ 时约 30% 即可），与 "组合越多越好泛化" 的小覆盖率结论一致。
- 会泛化的模型其 binding function 可被很小的 MLP 拟合，且 Global product 乘性探针的拟合质量与 OOD 对象识别准确率在 ~500 个模型上呈强正相关——这把 "低复杂度 = 乘性结构 = 跨模态可对齐" 三件事钉在了同一条经验轴上。

## 亮点与洞察
- 把 binding 写成一个**函数**而不是一种能力，是本文最关键的概念升级。一旦定义了 $B_{\text{img}}, B_{\text{txt}}$，就可以用 MDL/Occam 的语言谈复杂度，跨模态对齐失败立刻有了机制性解释——不再依赖 "编码器不够细" 这种含糊归因。
- "靶向去除概念/对象分量" 这套消融范式很优雅：减对应分量塌掉对应能力、减错分量不动、permuted-ID 也不动，三层对照把因果链锁死，可直接迁移到任意 dual-encoder 的内部归因分析。
- 用 MLP / XGBoost / RF 的**泛化能力**作 binding 复杂度的代理，本质上是把 "Kolmogorov 复杂度不可算" 这个问题工程化为 "用一族 SGD 偏好简单解的近似器去探"。这种 "用学习器当复杂度尺子" 的思路对一般的表征分析都有借鉴价值。
- 乘性交互的发现解释了一件长期困惑：为什么单纯加大数据和参数就能让组合泛化 "突然" 出现——因为模型在足够大的对象空间里被迫学一个能被 $\prod$ 表达的简单规则，而这个简单规则恰好让两侧编码器在 OOD 上对齐。

## 局限性 / 可改进方向
- 整套结论以合成数据为载体（CLEVR / CLEVR-2D / PUG:SPARE / Gemini 合成图），现实场景中没有 "组合完备" 的数据集来重复这套受控实验；自然图像里 binding 失败的真实分布可能更复杂。
- "复杂度" 的定义依赖近似器族——本文用 MLP / XGBoost / RF 作代理，但严格说这只是 "对这一族学习器的不可压缩性"，并非真正的 Kolmogorov 复杂度，结论的强度受此限制。
- 作者只指出 "乘性结构有效"，但没说清 dual-encoder Transformer 在内部如何实现这种乘性（attention？token-token 内积？），机制层面还留有空白。
- 没有给出可立刻改造 CLIP 的训练方案；对工业模型而言，"加数据覆盖" 与 "强加乘性归纳偏置" 哪条路更现实，仍待后续工作回答。

## 相关工作与启发
- **vs Trager et al. (2023) / Uselis et al. (2025) / Berasi et al. (2025)**：他们在 CLIP 文本或视觉编码器里证明了**单对象**嵌入的可加分解；本文把这件事推到**多对象场景**，并新增 Level-II 概念分解，把单/多对象的几何叙事统一起来。
- **vs Feng & Steinhardt (2024) / Feng et al. (2025)**：在自回归 LLM 内部找到 token 级的 binding ID 机制；本文研究的是 "单向量场景嵌入" 这种没有 token 中介的设置，得到的是另一种几何叙事（可加 + 乘性），二者互补。
- **vs Kang et al. (2025)**："concept 识别与 object 识别在 CLIP 中存在 trade-off"。本文用嵌入分解直接反驳：两者可以共存于同一嵌入，跨模态失败的真正瓶颈是 binding function 的复杂度。
- **vs Yuksekgonul / Ma / Hsieh / Gurung 等的微调路线**：他们试图通过数据或损失让 CLIP 更 "compositional"；本文给出更上位的解释——任何方法只要无法把 binding function 压成低复杂度（乘性）形式，跨模态对齐就难以泛化。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 binding 失败重述为 "函数复杂度过高 + 乘性结构缺失"，是对一个老问题的根本性概念升级
- 实验充分度: ⭐⭐⭐⭐ 多数据集 + 多模型 + 容量/覆盖率扫 + ~500 模型相关性曲线，但全部基于合成数据
- 写作质量: ⭐⭐⭐⭐⭐ 概念分层清晰、消融对照三重锁死、图表叙事节奏稳
- 价值: ⭐⭐⭐⭐ 为 "如何造一个会绑定的 CLIP" 提供了明确的结构性指引（多覆盖率 + 乘性偏置）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Explaining CLIP Zero-shot Predictions Through Concepts](../../CVPR2026/information_retrieval/explaining_clip_zero-shot_predictions_through_concepts.md)
- [\[ACL 2026\] How Large Language Models Balance Internal Knowledge with User and Document Assertions](../../ACL2026/information_retrieval/how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](../../ACL2026/information_retrieval/can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ACL 2025\] Semantic Outlier Removal with Embedding Models and LLMs](../../ACL2025/information_retrieval/semantic_outlier_removal_with_embedding_models_and_llms.md)
- [\[ACL 2025\] Length-Induced Embedding Collapse in PLM-based Models](../../ACL2025/information_retrieval/length-induced_embedding_collapse_in_plm-based_models.md)

</div>

<!-- RELATED:END -->
