---
title: >-
  [论文解读] SketchVL: Policy Optimization via Fine-Grained Credit Assignment for Chart Understanding and More
description: >-
  [CVPR 2026][多模态VLM][图表理解] SketchVL 让 MLLM 把每一步图表推理"画"成图像上的标注动作（框/线/点/圈），再用新提出的 FinePO 算法把整条轨迹的粗粒度优势，按一个过程奖励模型 FinePRM 给每个动作打的分**重新分配到每一步**，从而做到 step 级的细粒度信用分配，在图表/自然图像/数学多类基准上平均比基座模型提升 7.23%。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "图表理解"
  - "强化学习"
  - "过程奖励模型"
  - "信用分配"
  - "Reasoning on Image"
---

# SketchVL: Policy Optimization via Fine-Grained Credit Assignment for Chart Understanding and More

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_SketchVL_Policy_Optimization_via_Fine-Grained_Credit_Assignment_for_Chart_Understanding_CVPR_2026_paper.html)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 图表理解, 强化学习, 过程奖励模型, 信用分配, Reasoning on Image

## 一句话总结
SketchVL 让 MLLM 把每一步图表推理"画"成图像上的标注动作（框/线/点/圈），再用新提出的 FinePO 算法把整条轨迹的粗粒度优势，按一个过程奖励模型 FinePRM 给每个动作打的分**重新分配到每一步**，从而做到 step 级的细粒度信用分配，在图表/自然图像/数学多类基准上平均比基座模型提升 7.23%。

## 研究背景与动机

**领域现状**：图表（chart）是高密度的数据可视化载体，自动图表理解需要精确的视觉推理。近年主流做法是用 MLLM（Qwen2.5VL、Gemma3 等）配合强化学习（如 Vision-R1、VLM-R1）来增强多模态理解；其中 "Reasoning on Image (RoI)" 范式（ChartSketcher、DeepEyes）让模型把中间推理外化成图像上的可见标注，形成交互式反馈。

**现有痛点**：图表理解天然是**分步**结构——先定位图例/坐标轴，再读数值、对齐类别、比较趋势、综合结论，任何一步出错（裁剪不准、刻度读错、图例对错）都会让整条推理链崩掉。但当前 MLLM 的 RL 实践大多只给**粗粒度、只看最终结果**的反馈：像 GRPO 这样的方法只从最终答案算一个标量优势，然后**均匀广播**给轨迹里所有 token。

**核心矛盾**：trajectory 级的优势估计**无法区分一条回答内部的对错步骤**。结果是：一条答错的回答里本来正确的中间逻辑被一起惩罚；一条侥幸答对的回答里有缺陷的步骤被一起奖励——给学习信号注入噪声，削弱了 RL 的收益。

**本文目标**：为图表这种组合式、强步骤依赖的任务，实现**沿推理链对每一步单独评估和强化**的细粒度信用分配。

**切入角度**：RoI 范式把推理显式拆成一串离散的视觉标注动作，恰好给"按步打分、按步分配信用"提供了天然的结构载体——既然每一步都是可见、可评估的动作，就能给每个动作单独打过程分。

**核心 idea**：用一个过程奖励模型 FinePRM 给每个绘制动作打分，再用 FinePO 把整条轨迹的全局优势**按各步贡献二次重分配**——全局成功时更强地奖励正确 token，全局次优时更重地惩罚错误 token。

## 方法详解

### 整体框架
SketchVL 是一个迭代式推理的 MLLM：推理时它把"意图（intent）"对应的标注动作（action）画到图表上，把标注后的图再喂回自己以指导下一步决策，形成一条可见的推理轨迹。训练分两阶段（沿用 ColdStart-RL 范式）：**冷启动**让模型学会基础定位和 RoI 推理模式（50K SFT 数据），随后用 **FinePO** 强化学习解锁更复杂的推理能力。FinePO 的核心是借助 **FinePRM**（一个过程奖励模型）给轨迹里每个动作打分，从而把"算优势→分信用"做到 step 级。训练 FinePRM 的数据则由一条**跨模态蒸馏 pipeline**（473K 样本）合成。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入：图表 + 查询"] --> B["RoI Sketch 推理<br/>逐步画标注动作<br/>意图→动作→回喂图像"]
    B --> C["一组 k 条候选轨迹"]
    C --> D["FinePRM 过程评分<br/>每个动作打 4 级<br/>+ KL 动作正则"]
    D --> E["FinePO 信用分配<br/>跨轨迹优势 → 步内重分配"]
    E -->|step 级优势 A(sj)| F["策略更新"]
    F -->|下一轮| B
    G["跨模态蒸馏数据 pipeline<br/>视觉→文本→意图-动作对"] -.训练.-> D
```

### 关键设计

**1. Reasoning-on-Image 标注推理 + ColdStart-RL 两阶段训练**

为了让"按步打分"有可评估的对象，SketchVL 把推理过程外化成图像上的一串标注动作：每一步由一个文本"意图"（描述推理目标，如"标出最大值"）和一个"动作"（在图上画出可见标记来执行意图）组成，动作类型有 Line / Point / Rectangle / Circle / Text 五种。标注后的图回喂给模型作为下一步输入，从而把抽象推理变成可见、可逐步监督的轨迹——这正是 step 级信用分配能成立的前提。训练上先冷启动（2 epoch，50K 蒸馏 SFT 数据，来自 EvoChart/GQA/ChartQA-Train 的合成 QA + ChartSketcher pipeline）让模型掌握定位和 RoI 推理的基本盘，再进入 FinePO RL 阶段（1 epoch，9K 混合域 prompt）。消融里 "w/o Sketch (zero GRPO)" 把画图能力去掉后，所有基准全面崩塌（如 EvoChart-QA 从 47.28 掉到 30.48），说明这个交互式视觉推理过程是解决复杂图表任务的地基。

**2. FinePO：把轨迹优势二次重分配到每一步**

这是本文核心，专门解决"trajectory 级优势无法区分步内对错"的痛点。FinePO 分两步走。**第一步跨轨迹算粗优势**：对一个 prompt 采样 $k$ 条候选回答 $\{y_1,\dots,y_k\}$，每条按整体正确性给终端奖励 $R(y_i)$，优势取相对组内均值的偏差 $A(y_i) = R(y_i) - \frac{1}{k}\sum_{j=1}^{k} R(y_j)$，这是 GRPO 式的高层标量信号。**第二步在轨迹内做信用重分配**：用 FinePRM 给第 $j$ 步打过程分 $p_j = P(\text{intent}_j, \text{action}_j, \text{img}_{j-1}, \text{img}_j)$（看动作执行前后两张图的视觉变化），经 KL 正则修正为 $p'_j$ 后，按 token 长度 $L_j$ 加权求步内均值 $\bar p = \frac{\sum_j L_j p'_j}{\sum_j L_j}$，每步相对均值的偏差 $\Delta_j = p'_j - \bar p$ 表示该步比响应内平均更优还是更差。

关键在于 FinePO **不创造新奖励，只是把已有的粗优势 $A(y_i)$ 更精确地摊到各步**：

$$A'(s_j) = A(y_i) + \alpha \cdot k \cdot \Delta_j$$

其中 $\alpha$ 控制调整强度，$k = \frac{|A(y_i)|}{\max_{j}(0,\Delta_j)+\epsilon}$ 是动态缩放因子，让调整幅度与 $|A(y_i)|$ 成正比。各步调整的加权和被设计为零，从而**守恒总优势**，让细粒度信号始终锚定在响应整体表现上。最后再夹一刀，保证全局更优的回答（$A(y_i)>0$）各步不会拿到负优势、反之亦然：

$$A(s_j) = \begin{cases} \text{clip}(A'(s_j),\ 0,\ \beta A(y_i)) & A(y_i) > 0 \\ \text{clip}(A'(s_j),\ \beta A(y_i),\ 0) & A(y_i) \le 0 \end{cases}$$

这样全局成功的轨迹里正确步被更强奖励、全局次优的轨迹里错误步被更重惩罚，得到比 GRPO 均匀广播更锐利、更低噪声的学习信号。

**3. FinePRM：四级分类的过程奖励模型 + KL 动作正则**

FinePRM 是给 FinePO 提供 step 级信号的过程奖励模型，沿用 VisualPRM 思路用 MLLM（Qwen2.5VL-7B）做骨干。它接收文本意图、动作以及动作执行前后两张图 $\text{img}_{j-1}, \text{img}_j$，被 prompt 成"评审者"角色，判断 Image 2 的视觉修改是否精确、正确地实现了给定意图，输出 Excellent / Acceptable / Poor / Unacceptable 四级，再映射成标量 $[4.0, 3.0, 2.0, 1.0]$ 作为过程分。一个隐患是：策略可能偏好"容易得高分"的动作类型，回避重要但难完美执行的动作——因为对难度不同的动作用了统一打分标准。为此引入 **KL 动作正则**：先算一个截断的 KL 惩罚偏移

$$O_{\text{clipped}}(a_j) = \text{clip}\!\left(-\lambda_{KL}\log\frac{P_k(a_j)+\epsilon}{Q(a_j)+\epsilon},\ -\gamma,\ \gamma\right)$$

其中 $Q(a)$ 是训练集预统计的动作先验分布，$P_k(a)$ 是当前策略在最近 $k$ 个 batch 滑窗内的动作分布；用它修正过程分得 $p'_j = p_j + O_{\text{clipped}}(a_j)$，惩罚策略动作分布偏离先验。⚠️ 公式细节以原文为准。其作用是防止模型坍缩到少数易得分的动作类型（"Action Bias"）。Figure 4 显示去掉 KL 后 7B 模型的动作分布几乎完全坍缩（模型越大越容易钻 FinePRM 的空子），而加上 KL 反而鼓励更均衡、更鲁棒的推理策略。

**4. 跨模态蒸馏数据 pipeline（473K）**

训练 FinePRM 需要大规模"意图-动作"对，但即便 Gemini 2.5 Pro 在密集图像上同时定位+识别大量目标也很吃力，直接蒸馏既贵又不够精确。本文用两阶段跨模态蒸馏：**视觉→文本标注**借鉴 Set-of-Mark，用 SAM 把图切成以物体为中心的 patch，每个 patch 外扩 20% 上下文并用红框高亮目标，喂给 MLLM（Qwen2.5VL-72B）逐物体标注"自身属性"（如"一条紫色折线"）和"交互属性"（如"与绿色折线相交"），把密集视觉内容转成结构化文本——利用 MLLM 单物体识别强、批量定位弱的特性降低认知负担。**文本→图像蒸馏**再用 LLM 把这些标注蒸成模拟的意图-动作对，分两条线：Direct Generation 产单步简单任务，Trajectory-based Simulation 先造 QA 对、再让强 LLM 解题模拟出多步 GT 轨迹（也用于冷启动 SFT）。最后对动作注入噪声制造"意图与动作不匹配"的负样本，按 Excellent:Acceptable:Poor:Unacceptable = 2:4:3:1 配比（刻意非均匀，逼模型聚焦 Acceptable 与 Poor 的细微决策边界），共 473K 样本。

### 一个完整示例
查询"FDI 大约从何时开始持续超过 400？"。SketchVL 不直接答，而是走一条标注轨迹：① 意图"确认 FDI 曲线颜色"→ 在图例上画红框（FinePRM 评 Excellent/Acceptable）；② 意图"在 400 处画水平线"→ 画蓝色横线；③ 意图"从交点向下画竖线到 x 轴"→ 画绿色竖线（评 Excellent）；最后看到绿色竖线在 x 轴约 2005 处与橙色 FDI 曲线相交、此后橙线再没跌回蓝线下方，得出答案 2005。每一步的动作都被 FinePRM 单独打分，再由 FinePO 把整条轨迹的优势按这些分数重分配到各步——正确的定位步被强化，跑偏的步被惩罚。

### 损失函数 / 训练策略
SketchVL 训练 7B 与 3B 两个版本（基座 Qwen2.5VL-Instruct），FinePRM 用 Qwen2.5VL-7B 做骨干。FinePRM 训 4 epoch、冷启动 2 epoch、FinePO 训 1 epoch，全程 16×NVIDIA A800(40G)，框架为 ms-swift。评测用 DeepSeek-R1-Distill-Qwen-14B 当判别器，9 票多数表决判对错。

## 实验关键数据

### 主实验
覆盖图表专家数据集（EvoChart-QA、ChartQA、ChartQA-Pro、ChartBench、PlotQA）和通用数据集（MMStar、MathVista）。SketchVL 全面超越基座，3B 上增益尤其大（小模型更难"hack" FinePRM、更忠实地学其信号）。

| 模型 | EvoChart-QA | ChartQA | ChartBench | MathVista | MMStar |
|------|------|------|------|------|------|
| Qwen2.5VL-7B（基座） | 54.80 | 82.00 | 64.78 | 61.40 | 56.67 |
| **SketchVL-7B（本文）** | **58.64** | **83.96** | **65.11** | **63.50** | **57.13** |
| Qwen2.5VL-3B（基座） | 39.36 | 61.88 | 56.20 | 49.50 | 43.53 |
| **SketchVL-3B（本文）** | **47.28** | **77.20** | **59.96** | **53.80** | **51.00** |
| VLM-R1 | 40.32 | 72.98 | 39.58 | 55.10 | 48.27 |

SketchVL-3B 在 ChartQA 上比基座 +15.32、ChartBench +3.76；且在非图表的 MathVista/MMStar 上仍有提升，说明 FinePO 在强化专项能力的同时保住了通用泛化能力。

### 消融实验（基于 SketchVL-3B）

| 配置 | EvoChart-QA | ChartQA | PlotQA | 说明 |
|------|------|------|------|------|
| Full Model | 47.28 | 77.20 | 48.32 | 完整模型 |
| w/o FinePO（naive GRPO） | 45.60 | 75.12 | 44.72 | 换回均匀广播优势，ChartQA −2.08 |
| w/o FinePRM（随机分） | 48.08 | 76.76 | 46.40 | FinePRM 换成随机奖励，过程分明显掉（见 Table 2） |
| w/o KL 动作正则 | 48.56 | 77.80 | 48.16 | 性能接近，但动作分布会坍缩 |
| w/o Sketch（zero GRPO） | 30.48 | 57.56 | 31.12 | 去掉画图能力，全面崩塌 |
| w/o RL（纯 SFT） | 26.48 | 54.72 | 27.44 | 只冷启动不 RL，最差 |

另有过程质量评估（Table 2，用 FinePRM 当自动评估器打平均过程分）：SketchVL-3B Full 在 PlotQA/ChartBench/MMStar 上得 2.857/2.917/2.914，均高于 naive GRPO（2.705/2.777/2.755）和随机分版本——证明 FinePO 的细粒度信号确实让模型 step 级行为对齐了 FinePRM，且在未参与 RL 训练的 PlotQA 上也成立。

### 关键发现
- **画图能力是地基**：去掉 Sketch 后掉点最猛（EvoChart-QA −16.8），RoI 交互式视觉推理是解复杂图表的根本。
- **FinePRM 必须是有意义的信号**：换成随机分后过程分下降，说明信用分配靠的是真实评估而非噪声。
- **KL 正则的真正价值不在最终分**：去掉它最终精度甚至略高，但动作分布会严重坍缩（7B 尤甚，容量大更易钻 FinePRM 空子）；它换来的是更均衡、更鲁棒的动作使用。
- **泛化外溢**：RL 阶段的提升能扩散到未训练的 PlotQA，FinePO 增强的是内在可泛化的推理能力。

## 亮点与洞察
- **把"信用守恒"做进公式**：FinePO 不新增奖励、只重分配粗优势，且设计各步调整加权和为零，保证细粒度信号始终锚定整体表现——避免过程奖励"喧宾夺主"地把策略带偏，是很可借鉴的 RL 信号设计原则。
- **RoI 范式天然适配 step 级信用分配**：把抽象推理外化成离散可见动作，既给 PRM 提供了可评估对象，也让"按步打分"有了清晰边界——这是"用任务结构换可解释信用分配"的巧思。
- **KL 防奖励 hacking**：用动作类型分布对齐先验来防止策略坍缩到易得分动作，可迁移到任何"过程奖励模型可能被钻空子"的 RLHF/RLVR 场景。
- **跨模态蒸馏绕开 MLLM 定位弱点**：先用 SAM 切块 + 单物体标注把密集图变成结构化文本，再蒸成意图-动作对，是低成本造高精度 grounding 数据的实用配方。

## 局限与展望
- 作者承认 KL 正则在某些被单一动作类型主导的基准上会略微约束最优策略，是精度与多样性的权衡。
- ⚠️（自己观察）整套系统依赖较重：FinePRM 用 72B/80B 级模型蒸 473K 数据，训练用 16×A800，复现门槛高。
- FinePRM 本身是 7B MLLM，过程分的可靠性受其评判能力上限制约；用 FinePRM 同时当训练信号和评估器（Table 2）存在自评循环，结论需谨慎看待。
- 7B 上的增益明显小于 3B（作者归因于大模型更会 hack PRM），如何让强基座也充分受益于细粒度信号仍待解决。

## 相关工作与启发
- **vs GRPO / VLM-R1**：它们算一个 trajectory 级标量优势并均匀广播给所有 token，无法区分步内对错；SketchVL 在此基础上用 FinePRM 把优势二次重分配到每一步，信号更锐利、噪声更低。
- **vs ChartSketcher / DeepEyes（RoI 范式）**：同样让模型在图上画标注做推理，但它们没有 step 级的过程奖励与信用分配；SketchVL 的核心增量是 FinePO + FinePRM 这套细粒度强化机制（且显著超过 ChartSketcher-2B）。
- **vs VisualPRM**：都用 MLLM 当过程奖励模型，但 SketchVL 的 FinePRM 专为图像标注动作设计（看动作前后两张图判断意图实现质量），并配套了 473K 跨模态蒸馏数据与 KL 动作正则。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ FinePO 的"守恒式优势重分配" + RoI 范式承载 step 级信用分配，设计完整且针对性强。
- 实验充分度: ⭐⭐⭐⭐ 7 个基准 + 多角度消融 + 过程质量分析，但缺代码、复现门槛高，且部分评估自评循环。
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰、公式给全，图示直观。
- 价值: ⭐⭐⭐⭐ 为推理类 RL 的细粒度信用分配提供了可迁移的范式，对图表/视觉推理社区有实用参考。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HiconAgent: History Context-aware Policy Optimization for GUI Agents](hiconagent_history_context-aware_policy_optimization_for_gui_agents.md)
- [\[CVPR 2026\] ArtiMuse: Fine-Grained Image Aesthetics Assessment with Joint Scoring and Expert-Level Understanding](artimuse_fine-grained_image_aesthetics_assessment_with_joint_scoring_and_expert-.md)
- [\[CVPR 2026\] Beyond Global Similarity: Multi-Conditional Retrieval for Fine-Grained Cross-Modal Understanding](beyond_global_similarity_multi-conditional_retrieval_for_fine-grained_cross-moda.md)
- [\[CVPR 2026\] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)
- [\[ACL 2026\] Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding](../../ACL2026/multimodal_vlm/learning_more_from_less_exploiting_counterfactuals_for_data-efficient_chart_unde.md)

</div>

<!-- RELATED:END -->
