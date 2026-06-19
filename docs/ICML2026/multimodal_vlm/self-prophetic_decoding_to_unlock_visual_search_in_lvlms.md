---
title: >-
  [论文解读] Self-Prophetic Decoding to Unlock Visual Search in LVLMs
description: >-
  [ICML 2026][多模态VLM][视觉搜索] SeProD 让经过视觉搜索后训练的 LVLM 与其未微调的预训练版本配对，把预训练模型当作"先知"在每一步生成单步草稿前缀，再由后训练模型按概率阈值选择性接受这些前缀，从而在不训练、零额外计算的前提下同时保住单步基础能力与多步推理连贯性。 领域现状：让 LVLM 具备"边…
tags:
  - "ICML 2026"
  - "多模态VLM"
  - "视觉搜索"
  - "LVLM"
  - "预言式解码"
  - "推测解码"
  - "多步推理"
---

# Self-Prophetic Decoding to Unlock Visual Search in LVLMs

**会议**: ICML 2026  
**arXiv**: [2605.28741](https://arxiv.org/abs/2605.28741)  
**代码**: 暂未公布  
**领域**: 多模态VLM  
**关键词**: 视觉搜索, LVLM, 预言式解码, 推测解码, 多步推理  

## 一句话总结
SeProD 让经过视觉搜索后训练的 LVLM 与其未微调的预训练版本配对，把预训练模型当作"先知"在每一步生成单步草稿前缀，再由后训练模型按概率阈值选择性接受这些前缀，从而在不训练、零额外计算的前提下同时保住单步基础能力与多步推理连贯性。

## 研究背景与动机
**领域现状**：让 LVLM 具备"边看边想"的视觉搜索能力目前有两条路。一条是外部工具增强（SEAL、DyFo、ZoomEye 等），通过函数调用把裁剪、放大、定位等操作外包给视觉专家；另一条是内在能力扩展（Pixel Reasoner、DeepEyes、Mini-o3 等），直接对底座模型做视觉搜索后训练，让它在一次前向中自己发起 zoom-in 与 grounding。

**现有痛点**：外部工具路径接口僵硬，把本应连续的多步推理拆成多次独立的工具调用，损失上下文。内在扩展路径表面上更优雅，但论文在 Mini-o3 等模型上测出后训练带来的代价非常具体——grounding 单步准确率掉了 49.3%，OCR 掉 2.3%，空间理解掉 10.9%，计数掉 3.0%。同时多步轨迹一长，早期错误就会向后传染，把无关步骤从上下文里抹掉反而能让 VisualProbe-test 三个 split 各涨 5.66%/2.24%/5.66%。

**核心矛盾**：视觉搜索后训练用的数据有限，又主要靠 RL 在轨迹末端给奖励，缺乏中间步骤的监督信号。优化目标偏向"任务完成"，于是 grounding、计数、OCR 这些原本独立的内在能力开始互相干扰甚至被遗忘。但反过来如果不做后训练，模型又没有跨步规划与发起搜索的能力，单步强、多步弱。

**本文目标**：在不再训练、不增加推理预算的前提下，把"未后训练版本保留的强单步能力"和"后训练版本获得的多步搜索骨架"重新拼回去，并让它们在每一步互相校正。

**切入角度**：作者注意到后训练模型与其预训练底座共享同一套词表与大体相近的输出分布，这个对齐度足以借鉴 LLM 投机解码（speculative decoding）的范式——让一个轻量"草稿模型"先猜，再让目标模型按概率接受。区别在于：这里两模型扮演的不是"加速器+主体"，而是"单步专家+多步规划者"。

**核心 idea**：用预训练 LVLM 作为"先知"为后训练 LVLM 持续生成单步草稿前缀，后训练模型只接受联合概率超过阈值的前缀，从而把单步能力嫁接回多步推理。

## 方法详解

### 整体框架
SeProD 把一对模型耦合在一起：post-training 后的 LVLM 叫 search model，它负责跑多轮搜索轨迹；同一底座未做视觉搜索后训练的版本叫 prophet model，它在每轮被独立调用生成单步草稿。第 $i$ 轮里，search model 维护完整历史 $H_{i-1}=\{(I,Q),(I_1,C_1),\dots,(I_{i-1},C_{i-1})\}$，并以两种模式之一输出：grounding 模式产出推理片段 $R_i$ 加一个候选区域 $G_i$（再裁剪放大得到 $I_i$），answering 模式产出 $R_i$ 加最终答案 $A_i$。每一轮 search 都会触发一次 prophet 调用，prophet 看 $I_i$ 与一个针对当前模式的查询 $Q^p$，在一次前向里输出长度为 $L_d$ 的草稿 $O_i$，这串草稿随后被 search model 用概率阈值过滤式地"吸收"为自己后续 token 的前缀，下一轮再继续。整个回路只在推理期生效，对任何 intrinsic-extended LVLM 即插即用。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["高分辨率图像 I + 问题 Q"] --> B["Search 模型（后训练）第 i 轮<br/>读历史 H(i−1)，决定模式、生成 C_i"]
    B -->|grounding 模式| C["产出区域 G_i → 裁剪放大得 I_i"]
    B -->|answering 模式| C2["进入答案生成（I_i = I(i−1)）"]
    C --> P
    C2 --> P
    P["Search–Prophet 双模型配对与单步聚焦<br/>prophet 只看裁剪图 I_i，不接收文本 C_i"] --> Q2["Grounding 验证与答案草拟两类前缀<br/>按模式切查询 Q^p，自回归出草稿 O_i"]
    Q2 -->|grounding: 查询 Q^g| G["验证 ROI 是否含目标<br/>草稿指引下一轮 R(i+1)"]
    Q2 -->|answering: 查询 Q| H["草拟答案 O_i，对应当前轮 A_i"]
    G --> ACC["概率阈值预言式接受<br/>s_j = p_s^α · p_p^(1−α)，首个 s_j 低于 τ 起改用 p_s 采样"]
    H --> ACC
    ACC -->|grounding: 接受前缀→R(i+1)，回下一轮| B
    ACC -->|answering: 边接受边出 A_i| Z["最终答案"]
```

### 关键设计

**1. Search–Prophet 双模型配对与单步聚焦：让规划者和单步专家各管一摊**

后训练把多步搜索骨架装进了 search model，却让 grounding、OCR 这些单步能力退化；而预训练底座恰恰保住了这些单步能力。SeProD 的做法是让这两个模型分工——search model 管全局轨迹与跨步上下文、决定"这一步看哪、问什么"，prophet model 只盯着当前裁剪图 $I_i$ 做一步"专家级"判断。关键是 prophet **不接收** search 的文本输出 $C_i$，免得被对方的推理痕迹带偏，从而保住它独立的单步能力。

作者发现，若改成"把 prophet 输出当文本提示喂回 search"，就等同于退回工具调用接口——prefix 要么不起作用、要么打断推理连贯性（Appendix Fig. 8 的失败案例）。让 prophet 单独看图、search 决定问什么，才能把"任务相关的关注点"与"独立的单步能力"分两条线传递，这是后两个设计的前提。

**2. Grounding 验证与答案草拟两类前缀：按 search 的模式切换 prophet 该草拟什么**

search 每轮以两种模式之一输出，prophet 的查询 $Q^p$ 随之切换，按 $p_p(O_i\mid I_i, Q^p)=\prod_j p_p(o_{i,j}\mid I_i,Q^p,o_{i,<j})$ 自回归生成长度 $L_d$ 的草稿 $O_i$，两类草稿落在轨迹的不同位置：

- **grounding 模式 → 验证查询 $Q^g$**：让 prophet 判断当前裁剪图 $I_i$ 里有没有目标区域，输出 true/false；为 true 时附上区域细节作为草稿 $O_i$，这串前缀被接受后改写 search **下一轮**"去看哪里"的推理片段 $R_{i+1}$；为 false 则提示 search 重新定位。
- **answering 模式 → 原始查询 $Q$**：让 prophet 直接草拟答案 $O_i$，作为 search **当前轮**最终答案 $A_i$ 的前缀；$A_i$ 不预先完整生成，而是在接受过程中边接受边产出（on-the-fly），省下一次完整答案的解码。

把"指引下一步搜索"和"修正最终答案"拆成两类、落在轨迹的不同位置（前者在后续轮、后者在当前轮末），prophet 的单步能力才能在正确的地方发力。

**3. 概率阈值预言式接受：只吸收 search 本就高似然的草稿 token**

prophet 的草稿不是当外部输入硬塞，而是当作 search 可以选择性接受的 token 前缀——接受的部分像 search 自己生成的一样进 KV cache，和后续 token 一起解码。对 $O_i$ 的每个 token $o_{i,j}$ 算一个几何平均一致性分数

$$s_j = p_s(o_{i,j}\mid H_i,o_{i,<j})^{\alpha} \cdot p_p(o_{i,j}\mid I_i,Q^p,o_{i,<j})^{1-\alpha}$$

其中 $\alpha$ 初始为 0.5，再按该 token 在 search logits 里的归一化排名自动调整（排名越高、$\alpha$ 越大，越偏向 search 的原生分布）。从第一个 $s_j<\tau$ 的位置起拒绝，其后的 token 直接从 $p_s(x_j\mid H_i,x_{<j})$ 采样。所有 $s_j$ 能在一次前向里并行算完（草稿 token 已预先备好、无需顺序生成），所以整段草稿评估的耗时相当于一个普通解码步，没有额外延迟。

这种按联合概率门限接受的方式，好处是只有那些本就落在 search 高似然区的 token 才被吸收——既借到了 prophet 的单步知识，又不会让 search 的多步推理"性格突变"，论文用 Fig. 2(c) 验证 SeProD 的输出分布曲线与原模型几乎重合。

### 损失函数 / 训练策略
SeProD 完全 training-free，不引入任何可训练参数。两个超参数：一致性阈值 $\tau$ 控制接受严格度，平衡因子 $\alpha$ 在线自适应；prophet model 默认用与 search 同款底座（如 Qwen-2.5-VL-3B），也允许选用更小的同源底座以进一步降本。

## 实验关键数据

### 主实验
在 4 个高分辨率视觉搜索 benchmark 共 12 个 split 上接入 Pixel Reasoner 与 DeepEyes 两个 search backbone，prophet 默认 3B。

| Benchmark / Split | Pixel Reasoner | + SeProD | DeepEyes | + SeProD |
|-------------------|---------------:|---------:|---------:|---------:|
| VisualProbe-Hard | 28.7 | 30.2 (+1.5) | 38.4 | 41.9 (+3.5) |
| VisualProbe-Medium | 29.0 | 30.4 (+1.4) | 30.5 | 32.3 (+1.8) |
| VisualProbe-Easy | 58.7 | 61.7 (+3.0) | 61.2 | 64.7 (+3.5) |
| V* Bench Overall | 86.9 | 88.5 (+1.6) | 89.0 | 91.1 (+2.1) |
| HR-Bench 4K Overall | 72.6 | 73.6 (+1.0) | 73.0 | 73.8 (+0.8) |
| HR-Bench 8K Overall | 64.3 | 65.1 (+0.8) | 69.9 | 71.9 (+2.0) |

12/12 split 全部提升，难度越大、空间/跨实例感知越关键的场景增益越显著（VisualProbe-Hard 上 DeepEyes 涨 3.5 个点）。SeProD 还在通用 VQA 上观察到一致的小幅增益，且因前缀评估并行化没有额外延迟。

### 消融与机制分析

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| Search only（baseline） | 单步 grounding 掉 49.3% | 后训练带来的能力退化基线 |
| Prophet as text prompt | 推理被打断、收益不稳 | 论文 Appendix Fig. 8 中失败案例 |
| 删去无关上下文 | VisualProbe-test 三 split 各涨 5.56/2.24/5.66% | 验证长上下文干扰 |
| 概率阈值接受（SeProD） | 输出分布与原 search 几乎重合（Fig. 2(c)） | 保住多步连贯性 |

### 关键发现
- 把 prophet 输出当 text prompt 喂回 search 几乎拿不到稳定增益，必须走概率域的"前缀接受"才能让单步能力真正回到多步轨迹里。
- 增益与搜索难度正相关——VisualProbe-Hard、HR-Bench 8K 这类长轨迹、强空间感知场景拿到的 +2~+3.5 点最大，说明 SeProD 主要在补"后训练丢失的精细单步能力"。
- prophet 默认 3B 即可，前缀评估可一次并行算完，论文报告"无额外计算开销"。

## 亮点与洞察
- 把投机解码的"草稿+接受"范式从 LLM 推理加速搬到 VLM 推理质量上，是少见的把同一技术语义重新用一次的设计，原本用于省时间的并行接受机制这里变成了"模态间能力转移"的接口。
- "用同一底座的预训练版本作为 prophet"是这篇论文最聪明的工程决策——两模型分布天然接近，接受率高；如果换成异源 prophet，$s_j$ 会持续低于 $\tau$，方法退化为 search 自己生成。
- 把"接口设计"上升为核心贡献——作者明确指出 token-level 概率接口和 text-prompt 接口的差距，提示后续做 LVLM 协同/集成时不要默认用文本接口，token 级耦合可能是更稳妥的方向。

## 局限与展望
- 方法依赖 search 与 prophet 共享同一底座，对纯黑盒的商用 LVLM（如 GPT-4o）不可用。
- 只在视觉搜索这一具体形态的"thinking-with-images"上验证，对没有显式 zoom-in/grounding 操作的多模态推理任务（如复杂数学图表）泛化性未知。
- 阈值 $\tau$ 是全局固定的，论文没给自动调参方案；不同 backbone 和不同 benchmark 上的最优 $\tau$ 可能需要小规模验证。
- prophet 推理虽与 search 并行，但显存占用近乎翻倍，对 70B 级 search backbone 部署成本不低。

## 相关工作与启发
- **vs SEAL / DyFo / ZoomEye（外部工具）**: 工具路径用文本/函数接口把多步推理拆开；SeProD 用 token 级概率接口让两模型耦合在同一解码循环里，保住跨步上下文。
- **vs DeepEyes / Mini-o3 / Pixel Reasoner（内在扩展后训练）**: 这些工作通过 RL 教模型自己发起搜索，但付出"单步能力退化"的代价；SeProD 不改这些后训练模型，而是把它们与各自的预训练底座再配对，等于事后修复退化。
- **vs Speculative Decoding (Leviathan et al., 2023)**: 投机解码用小模型加速大模型推理，目标是"等价输出+少计算"；SeProD 借用同样的并行接受机制，但目标变成"用单步专家增强多步推理"，接受准则也从无偏采样改成概率阈值。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把投机解码的接口语义复用到 VLM 多步推理能力修复，角度新；同基座配对是关键工程洞察。
- 实验充分度: ⭐⭐⭐⭐ 4 benchmark × 12 split × 2 backbone 全部一致提升，并有单步退化与上下文干扰的诊断实验支撑核心论点。
- 写作质量: ⭐⭐⭐⭐ Method 一节把"为什么文本接口不行→为什么概率接口可行"讲得清楚，Fig. 2 三联诊断图很有说服力。
- 价值: ⭐⭐⭐⭐ training-free + plug-and-play，对已有视觉搜索后训练模型几乎零成本接入，工业可用性高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)
- [\[ICML 2025\] Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration](../../ICML2025/multimodal_vlm/towards_rationale-answer_alignment_of_lvlms_via_self-rationale_calibration.md)
- [\[ICLR 2026\] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](../../ICLR2026/multimodal_vlm/self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)
- [\[ICML 2026\] LBR/LBP: Language Bias in LVLMs — From In-Depth Analysis to Simple and Effective Mitigation](language_bias_in_lvlms_from_in-depth_analysis_to_simple_and_effective_mitigation.md)
- [\[CVPR 2026\] Revisiting Visual Corruptions in LVLMs: A Shape-Texture Perspective on Model Failures](../../CVPR2026/multimodal_vlm/revisiting_visual_corruptions_in_lvlms_a_shape-texture_perspective_on_model_fail.md)

</div>

<!-- RELATED:END -->
