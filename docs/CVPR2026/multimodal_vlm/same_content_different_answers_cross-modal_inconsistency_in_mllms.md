---
title: >-
  [论文解读] Same Content, Different Answers: Cross-Modal Inconsistency in MLLMs
description: >-
  [CVPR2026][多模态VLM][跨模态一致性] 作者提出 REST / REST+ 两个 benchmark，把同一道题分别以「纯文本 / 纯图像（渲染成图）/ 图文混合」三种形态喂给 MLLM，并在严格控制 OCR 正确的前提下度量「同样的内容、不同的答案」这一跨模态不一致现象——评测 15 个前沿 MLLM 发现没有一个能在三种模态上稳定一致（不一致率至少 ~10%，最差超 80%），且模型普遍更偏好文本模态，而这种不一致与模型内部图文表征的余弦相似度（模态鸿沟）显著相关。
tags:
  - "CVPR2026"
  - "多模态VLM"
  - "跨模态一致性"
  - "模态鸿沟"
  - "benchmark"
  - "OCR"
  - "render text-as-image"
---

# Same Content, Different Answers: Cross-Modal Inconsistency in MLLMs

**会议**: CVPR2026  
**arXiv**: [2512.08923](https://arxiv.org/abs/2512.08923)  
**代码**: https://github.com/angelavansprang/Same-Content-Different-Answers  
**领域**: 多模态VLM  
**关键词**: 跨模态一致性、模态鸿沟、benchmark、OCR、render text-as-image

## 一句话总结
作者提出 REST / REST+ 两个 benchmark，把同一道题分别以「纯文本 / 纯图像（渲染成图）/ 图文混合」三种形态喂给 MLLM，并在严格控制 OCR 正确的前提下度量「同样的内容、不同的答案」这一跨模态不一致现象——评测 15 个前沿 MLLM 发现没有一个能在三种模态上稳定一致（不一致率至少 ~10%，最差超 80%），且模型普遍更偏好文本模态，而这种不一致与模型内部图文表征的余弦相似度（模态鸿沟）显著相关。

## 研究背景与动机
**领域现状**：MLLM 被训练把视觉和语言投影到同一个共享语义空间，主流叙事是「图文已被无缝融合」，模型在 VQA、文档理解、复杂推理上都表现强劲。与此同时，最近 DeepSeek-OCR 这类工作提出一个很有诱惑力的方向：把文本**渲染成图像**喂给模型，用 1 个视觉 token 压缩 10 个文本 token、还能保持 97% 的 OCR 准确率，从而大幅节省长文本输入的计算成本。

**现有痛点**：但已有研究又一致观察到「模态鸿沟（modality gap）」——文本 embedding 和图像 embedding 在共享空间里占据**不同区域**，鸿沟越小下游性能越好。这就引出一个被忽视的根本问题：当模型成功地从图里**读出**了文字，它对这段信息的**推理**质量，是否和直接收到原生文本时一样好？如果不一样，那「把文本渲染成图省 token」的省钱方案就可能悄悄牺牲推理正确性。

**核心矛盾**：现有评测无法干净地回答这个问题，因为它们把两件事混在了一起——**识别失败**（OCR 读错字）和**推理不一致**（字读对了但答案不同）。已有 benchmark 要么只评一个模型（Zhang et al.），要么不控制可读性（Omni-R / OmnixR），导致结论里「读不出字」和「读对了却答错」纠缠不清。

**本文目标**：在**严格控制 OCR 正确**的条件下，系统度量同一语义内容在不同模态下答案是否一致，并回答四个 RQ——(RQ1) 前沿 MLLM 是否存在跨模态不一致、哪种模态最强；(RQ2) 不一致是否只是 OCR 没做好；(RQ3) 分辨率/字体/颜色等视觉特征是否影响不一致；(RQ4) 内部表征的跨模态相似度是否与不一致程度相关。

**核心 idea**：构造「渲染等价压力测试」——把**语义完全相同**的题目以三种模态呈现，只在**模型 OCR 完全读对**的子集上比较答案一致性，从而把「识别」从「推理」里剥离出来，直指 MLLM 共享空间是否真的支持模态无关的推理。

## 方法详解
这是一篇 benchmark / 实证分析论文，没有提出新模型或新训练方法，"方法"就是 benchmark 的构造方式、一致性度量指标、以及四组受控分析实验的设计。整体逻辑是：先用 OCR 任务把「读不出字」的样本筛掉，再在「读对字」的干净子集上用三个自定义指标量化跨模态不一致，最后把这种行为层面的不一致和模型内部表征的相似度对上号。因为是纯评测/分析、流程不是多模块串行 pipeline，不画框架图。

### 整体框架
benchmark 的一个样本 = 同一道题的三种「渲染等价」形态：**Text**（纯文本提问）、**Image**（整道题渲染成图）、**Mixed**（上下文是图、问题是文本；选择题则题干为图、选项为文本）。每个样本先跑一个 **OCR 任务**（让模型把图里的文字转写出来）验证可读性，再用 Chain-of-Thought 提示分别在三种模态上作答（temperature=0、正则解析答案、格式非法记为错）。所有一致性分析都**只在 OCR 完全正确的样本上**进行，这是把识别和推理解耦的关键闸门。

数据由四个任务组成：复用三个成熟 benchmark（MMLU、AI2-ARC、GSM-Symbolic）并把文本渲染成图，外加作者新造的 **SoEBench**（解线性方程组）。为压低 OCR 难度，过滤掉 >800 字符和含 LaTeX 的题，图像用白底、DPI 200、黑色 DejaVu Sans 字体。**REST+** 是加难版：对每张图做 10 种视觉扰动（3 字体 × 3 分辨率 = 9 种 + 1 种彩色），保持语义不变，专门用来测视觉特征和视觉 token 数量对一致性的影响。

### 关键设计

**1. REST/REST+ 三模态「渲染等价」样本 + SoEBench：把识别从推理里剥离**

benchmark 的核心创新是「同一内容、三种渲染、严格可控」。三种模态（Text / Image / Mixed）承载**完全相同的语义**，理想中一个真正模态无关的模型应该给出三个相同答案，所以任何答案分歧都直接归因于模态本身而非内容差异。为了不让 OCR 失败污染结论，作者一方面在数据层面压低识别难度（限字数、去 LaTeX、高分辨率印刷体），另一方面在分析层面只保留「OCR 完全读对」的样本。

其中 **SoEBench**（system-of-equations benchmark）是专门为对抗两个 confound 而造的：它只用数字 0–9 和字母 A–E 这种极简符号集，OCR 几乎不可能读错；而且是**新生成的**，保证任何现有 MLLM 在预训练阶段都没见过它，从而排除「文本模态强只是因为文本题被背过了（数据污染）」这一解释。每道题有唯一整数解，要求基础代数推理。这一设计让「文本模态更强」这个观察能更可信地归因于**推理机制差异**而非记忆或识别。

**2. 三个一致性/能力指标（RER / CFR / MMC）：从不同角度量化「同内容不同答案」**

作者用三个指标刻画不一致。**Render-Equivalence Rate（RER）** 度量三种模态答案完全一致的题目占比：

$$\text{RER}=\frac{|\{x \mid f(x_t)=f(x_i,z_i)=f(x_m,z_m)\}|}{N}$$

其中 $f$ 是模型、$N$ 是样本量，$x/z$ 分别是文本/图像输入，RER=1 表示完美一致（注意它只看「答案是否相同」，不看对错）。

**Cross-Modality Failure Rate（CFR）** 度量「能在至少一种模态答对、却没法在全部模态都答对」的题占比：

$$\text{CFR}=\frac{|\{q \mid 1\le \sum_{m\in M} C(q,m) < |M|\}|}{N_c}$$

$C(q,m)\in\{0,1\}$ 是题 $q$ 在模态 $m$ 上的对错，$M=\{\text{text, image, mixed}\}$，$N_c$ 是「至少一种模态答对」的题数。它故意**排除三种模态全错的题**（那是能力不足而非不一致），CFR=0 才是完美一致。CFR 揭示的是「答对与否取决于你用什么格式问」这种最令人担忧的模式。

**Max Modal Coverage（MMC）** 度量「至少一种模态能答对」的题占比，代表如果每题都用最优模态去问、模型潜在能达到的上限。MMC 与「三模态全对率」之间的差，就是被跨模态不一致**白白浪费掉的可解题目**——例如 Phi-4 的 MMC 高达 85.9%，但只有 36.7% 的题三模态都能答对，意味着 ~49% 的题如果没用对模态就会答错。

OCR 质量本身用 **Character Error Rate（CER）** 度量（字符增删改 / 参考长度），只在 CER 对应「完全读对」时纳入一致性统计。

**3. OCR-first 反事实 + 视觉扰动（REST+）：排除替代解释、定位真正的影响因素**

为了证明不一致**不是** OCR 的锅，作者做了一组反事实：让模型先把图里文字转写出来、再解题（"OCR first, then solve"，对应 Chen et al. 的做法）。如果不一致纯由识别引起，这招应该普遍涨点；但结果是有的模型涨、有的模型反而大幅掉点（如 DeepSeek-Small 在 MMLU 上 −13.1），证明「先 OCR 再解」并不能稳定修复不一致，OCR 不是主因。

REST+ 则系统拆解视觉特征的影响：分辨率（DPI 50/100/200）、字体（DejaVu Sans / Courier New / Cursive）、颜色（红绿蓝等）。结论很反直觉——**字体几乎无影响**（多数模型 <2% 差异）；**分辨率有影响**（高 DPI 更一致，且 DPI@50 下掉点往往是 OCR 跟着变差，再次说明要控 OCR）；**彩色文字反而普遍涨点**（红/黄字相对黑字多个模型 >5% 相对提升）。此外通过对比文本 token 数 vs 视觉 token 数发现：除 Qwen2.5-VL-32B 外，几乎所有模型都需要**更多视觉 token 才能达到文本同等准确率**，即"render text-as-image 省 token"在当前模型上多半以推理质量为代价。

**4. 表征相似度↔行为不一致的机制关联（RQ4）：给现象找内部解释**

作者把「行为层面的不一致」和「表征层面的模态鸿沟」对上号。借用 Shukor & Cord 的隐式对齐分数，对每个样本取图像 token 均值 $\bar{\mathbf i}=\frac1A\sum_k \mathbf i_k$ 和文本 token 均值 $\bar{\mathbf t}=\frac1B\sum_k \mathbf t_k$，算余弦相似度 $\mathrm{sim}(\mathbf I,\mathbf T)=\frac{\bar{\mathbf i}\cdot\bar{\mathbf t}}{\|\bar{\mathbf i}\|\|\bar{\mathbf t}\|}$；并进一步用**双向检索准确率**（每个图像样本能否在文本侧检索回它配对的那条，取各层最大值）刻画对齐质量。在 ImageNet（自然图 + 文字标签 + "写下来的标签图"）1000 样本上做实验，发现这个相似度/检索分数与 RER **显著正相关**——一致性高的模型，其匹配样本的图文表征也更接近。这给"同内容不同答案"提供了一个机制性解释：**不一致源于文本和图像表征落在了共享空间的不同区域**，方向不对齐，推理自然分叉。

### 损失函数 / 训练策略
本文不训练模型、无损失函数。评测协议：四种输入提示（OCR + text + mixed + image），后三者用 CoT，输出上限 1024 token（SoEBench 放宽到 2048），temperature=0，正则解析答案、格式非法即判错；一致性统计只在 OCR 完全正确子集上进行。

## 实验关键数据

### 主实验
评测 15 个前沿 MLLM。REST 上的一致性（RER 越高越好、CFR 越低越好，下表为四个 benchmark 平均、OCR✓ 子集）：

| 模型 | RER↑ (OCR✓) | CFR↓ (OCR✓) | OCR✓率 |
|------|------|------|------|
| GPT-5-mini | **90.7** | **8.7** | 99.0 |
| Haiku-4.5 (Claude) | 90.3 | 8.9 | 98.2 |
| Qwen-2.5 (32B) | 84.7 | 13.6 | 97.5 |
| InternVL3 (14B) | 78.4 | 19.6 | 95.3 |
| GPT-4o-mini | 71.3 | 26.0 | 98.8 |
| Gemini-2.5 Fl. Lite | 54.3 | 40.3 | 98.3 |
| Phi-4 | 14.9 | 82.3 | 94.0 |
| Deepseek-Tiny | 6.6 | 98.0 | 70.3 |

RER 跨模型从 **6.6% 到 90.7%**——**没有一个模型做到模态无关一致**。闭源的 GPT-5-mini / Claude Haiku 4.5 领先，开源里 Qwen-2.5-32B 最强。即便最强的 GPT-5-mini，仍有 8.7% 的题「能在某模态答对、却答不全」。关键是：只看 OCR✓ 子集时 RER 普遍**持平或略升**，且模型排名不变，说明剔除识别错误后不一致依然存在——**OCR 不是主因**。

各模态准确率（OCR✓ 子集，MMLU/ARC/GSM-Sym）显示**几乎所有模型在文本模态最强**：t 检验 text>image（$t=17.7,\,p<0.05$）、image<mixed（$t=-7.2$）。例如 GPT-4o-mini 在 MMLU 文本比图像高 >7%。

SoEBench（全新、OCR 近乎完美）进一步坐实「文本模态本质更强」：

| 模型 | RER↑ | Text↑ | Mixed↑ | Image↑ | OCR↑ |
|------|------|------|------|------|------|
| Haiku-4.5 | 92.0 | 96.7 | 95.3 | 97.3 | 100.0 |
| GPT-5-mini | 91.9 | 98.7 | 95.3 | 97.3 | 99.3 |
| Qwen-2.5 (32B) | 69.1 | 87.2 | 87.9 | 75.2 | 99.3 |
| Phi-4 | 1.3 | 13.3 | 13.3 | 17.3 | 100.0 |
| Deepseek-Tiny | 0.0 | 0.7 | 0.0 | 0.0 | 0.0 |

OCR 几乎满分（除 DeepSeek-Tiny 直接复读输入示例），但 Text 依然普遍领先，证明不一致既非 OCR 也非数据污染所致。

### 消融实验

| 配置 / 分析 | 关键指标 | 说明 |
|------|---------|------|
| OCR-first（先转写再解题） | ΔAcc 不稳定 | DeepSeek-Small MMLU **−13.1**、ARC −12.8；GPT-5-mini 全线 +1 左右——有的涨有的崩，OCR 不是 confound |
| REST+（视觉扰动加难版） | 最佳 CFR 仍 27.9% | InternVL3-14B 一致性最高；字体/颜色/分辨率任一变化都能左右答案对错 |
| DPI 50→100→200 | RER 随 DPI 升而升 | GPT-5-mini 71.5→82.7→83.6；低 DPI 掉点常伴 OCR 同步变差（Claude/Gemma） |
| 字体（Sans/Courier/Cursive） | <2% 差异 | 反直觉：草书并不更难，字体几乎无影响（仅 Phi-3.5 差 5.3%） |
| 彩色 vs 黑字 | 多模型 >5% 相对提升 | 反直觉：红/黄字反而更好（DeepSeek-Small、Qwen-32B 明显） |
| 表征余弦相似度 vs RER | 显著正相关（$R^2$ 拟合） | 一致性高的模型，匹配样本图文表征更接近——机制层面解释 |

### 关键发现
- **没有任何 MLLM 跨模态一致**，不一致率从 ~10% 到 >80%；这意味着大量「本可解」的题因为换了个输入格式就答错了（Phi-4 上 ~49% 的题如此）。
- **文本模态系统性更强**，且这不是 OCR 或数据污染造成的（SoEBench 全新题上依然如此）——直接给「文本渲染成图省 token」的路线泼了冷水。
- **视觉特征的影响很反直觉**：字体无所谓、彩色文字反而更好、分辨率有影响（但部分通过 OCR 起作用）。说明评测跨模态不一致时**必须控制 OCR**，否则会把识别能力误算进推理一致性（Claude 在 DPI@50 若不控 OCR 会显得更差）。
- **现象有内部对应**：跨模态表征相似度（模态鸿沟）与 RER 正相关，把行为不一致和共享空间的几何错位连了起来。

## 亮点与洞察
- **「只在 OCR✓ 子集上比较」是这篇的灵魂**：用一个简单的前置 OCR 闸门，干净地把「读不出字」和「读对却答错」分开，让结论第一次能可信地归因于推理而非识别——这套解耦协议可直接迁移到任何「文本 vs 渲染图」的评测里。
- **SoEBench 的双重防污染设计很聪明**：极简符号集压低 OCR 难度 + 全新生成杜绝记忆，一次性挡掉「OCR 没做好」和「文本题被背过」两个最常见的 reviewer 质疑。
- **给热门的 render-text-as-image（DeepSeek-OCR）省 token 路线做了一次冷静的体检**：发现除 Qwen2.5-VL-32B 外，模型都要更多视觉 token 才能追平文本准确率——"能读出字 ≠ 能一样地推理"，对做长上下文压缩的人是重要警示。
- **彩色文字反而更好、字体无影响**这种反直觉发现，提示视觉 token 化过程里存在尚未被理解的偏置，是值得后续挖的口子。
- **RER↔表征相似度的相关性**把一个行为现象和模态鸿沟的几何解释挂钩，并明确指出未来可做的因果实验（主动拉近表征相似度，看 RER 是否上升）。

## 局限性 / 可改进方向
- **只是相关、不是因果**：RQ4 仅证明表征相似度与 RER 相关，作者自己也承认未验证因果——拉近表征是否真能提升一致性还是 open question。
- **任务偏窄**：REST 主要是渲染印刷体文字的 QA（MMLU/ARC/GSM/方程组），虽用 ImageNet 自然图 + 棋局补充验证「不一致超出排版文字」，但对真实复杂自然图像（含深度、布局，无精确文本对应物）上的不一致刻画仍有限。
- **闭源模型分析受限**：拿不到闭源模型 token 数和隐藏激活，token 效率分析和 RQ4 表征分析只能在开源模型上做。
- **指标只看"答案是否相同"**：RER 把「三个都错但一致」也算一致，需配合 CFR/MMC 一起读才不被误导；单看 RER 可能高估弱模型的"稳定性"。
- **未给出修复方案**：本文止于诊断，没有提出降低不一致的训练/对齐方法，留给后续工作。

## 相关工作与启发
- **vs Zhang et al.（vision-language consistency）**：同样测跨模态一致，但他们只评 GPT-4V 一个模型、且不控制 OCR；本文评 15 个模型并严格剥离 OCR，结论更有普适性。
- **vs Omni-R / OmnixR（Chen et al.）**：他们有文本/图像/视频/音频配对题，但只评闭源模型、不控 OCR；本文专门把 OCR 难度压到最低并新增 SoEBench 杜绝记忆。
- **vs MMMU-Pro 的 vision-only 子任务**：MMMU-Pro 把文字渲染进截图/自然照片里（OCR 难），本文反其道把 OCR 做到最简单，从而把推理不一致单拎出来。
- **vs MMIR / Alonso et al. / Sim et al.**：这些研究的是不同形式的多模态不一致（语义错配检测、跨模态实体对齐、模态坍缩）——它们的图文是**互补**信息；本文专攻**语义完全相同**内容下的答案分歧，是一个更纯粹的「渲染等价」设定。
- **vs 模态鸿沟系列（Liang et al. / Shukor & Cord / Eslami & de Melo）**：前人发现并量化了 CLIP/MLLM 的模态鸿沟、且鸿沟越小性能越好；本文把这条线索从「下游性能」推进到「跨模态推理一致性」，并实证二者相关。

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题切口新（render-equivalence + OCR 解耦），但 benchmark 本身复用了大量已有数据集，方法不算原创性极高
- 实验充分度: ⭐⭐⭐⭐⭐ 15 个模型、4 个 RQ、含 OCR-first 反事实/视觉扰动/表征分析，覆盖全面且控变量严谨
- 写作质量: ⭐⭐⭐⭐ RQ 驱动、逻辑清晰、指标定义完整；个别表格信息密集需对照读
- 价值: ⭐⭐⭐⭐⭐ 直接质疑「render text-as-image 省 token」这一热门路线，并提供可复用的解耦评测协议，对做多模态压缩/对齐的人很有警示价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Decoupled and Reusable Adaptation for Efficient Cross-Modal Transfer](decoupled_and_reusable_adaptation_for_efficient_cross-modal_transfer.md)
- [\[CVPR 2026\] Rethinking Cross-Modal Anchor Alignment for Mitigating Error Accumulation](rethinking_cross-modal_anchor_alignment_for_mitigating_error_accumulation.md)
- [\[ICML 2026\] CG-MLLM: Captioning and Generating 3D Content via Multi-modal Large Language Models](../../ICML2026/multimodal_vlm/cg-mllm_captioning_and_generating_3d_content_via_multi-modal_large_language_mode.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](../../ACL2026/multimodal_vlm/cross-modal_taxonomic_generalization_in_vision-_language_models.md)
- [\[CVPR 2026\] Same or Not? Enhancing Visual Perception in Vision-Language Models](same_or_not_enhancing_visual_perception_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
