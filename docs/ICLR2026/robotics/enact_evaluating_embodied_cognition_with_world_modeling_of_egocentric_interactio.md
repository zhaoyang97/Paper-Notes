---
title: >-
  [论文解读] ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction
description: >-
  [ICLR 2026][机器人][具身认知] ENACT 将具身认知评测形式化为基于一人称交互的世界建模 VQA——通过正向/逆向序列重排任务，系统揭示了当前顶级 VLM 在长时域交互推理中相较人类的显著差距及拟人化偏见。 领域现状：具身认知理论认为智能源于感觉运动交互，而非被动观察。近年来 VLM（GPT-5、Gemini…
tags:
  - "ICLR 2026"
  - "机器人"
  - "具身认知"
  - "世界模型"
  - "视觉语言模型评测"
  - "自我中心感知"
  - "POMDP"
---

# ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction

**会议**: ICLR 2026  
**arXiv**: https://enact-embodied-cognition.github.io  
**代码**: https://github.com/enact-embodied-cognition  
**领域**: robotics  
**关键词**: 具身认知、世界模型、视觉语言模型评测、自我中心感知、POMDP

## 一句话总结

ENACT 将具身认知评测形式化为基于一人称交互的世界建模 VQA——通过正向/逆向序列重排任务，系统揭示了当前顶级 VLM 在长时域交互推理中相较人类的显著差距及拟人化偏见。

## 研究背景与动机

**领域现状**：具身认知理论认为智能源于感觉运动交互，而非被动观察。近年来 VLM（GPT-5、Gemini 2.5、Claude 等）凭借大规模非具身训练展现出令人印象深刻的交互能力，使得"VLM 是否具备具身认知"成为一个关键科学问题。  

**现有痛点**：已有工作要么聚焦于静态场景的空间感知，要么仅评估语言规划能力，或只考察原始物体间交互，缺乏将一人称感知与长时域具身交互紧密耦合的统一评测框架。主观分类体系（如 Yang et al., 2025）难以提供可复现的客观度量。  

**核心矛盾**：VLM 在静态视觉理解上表现突出，但多步骤、因果性、部分可观测的具身世界建模能力尚未被严格量化——评测工具的缺失制约了对 VLM 具身能力边界的认识。  

**本文目标**：构建一个可扩展、客观、与具体图像生成质量解耦的具身认知基准，在统一框架下系统测量 VLM 的前向/逆向世界建模能力。  

**核心 idea**：将具身认知评测转化为基于 POMDP 的序列重排 VQA——正向世界建模（给定动作重排打乱的观测序列）与逆向世界建模（给定观测序列重排打乱的动作序列），动作以场景图差分表示，既剥离低层图像合成的干扰，又隐式要求模型具备可供性识别、动作-效果推理、具身感知与长时域记忆能力。

## 方法详解

### 整体框架

```mermaid
flowchart TD
    A[机器人操控轨迹\nBEHAVIOR 仿真器] --> B[关键帧提取\n场景图差分非空时刻]
    B --> C[关键帧轨迹采样\n长度 L∈3..10\n组合式 C(M,L) 扩展]
    C --> D1[正向世界建模 QA\n给定动作序列+初始观测\n要求重排打乱的观测图像]
    C --> D2[逆向世界建模 QA\n给定有序观测序列\n要求重排打乱的动作]
    D1 --> E[ENACT 基准\n8972 QA 对\n29 项家庭活动]
    D2 --> E
    E --> F[VLM 评测\n在线验证器\nTask Acc + Pairwise Acc]
```

### 关键设计

**1. POMDP 形式化的序列重排 VQA：与图像生成解耦**

ENACT 将世界建模定义在 POMDP $(S, O, A)$ 上：状态空间 $S$ 为场景图，观测空间 $O \subset \mathbb{R}^{H \times W \times 3}$ 为机器人一人称 RGB 视图，动作空间 $A$ 为场景图差分 $a_t = \delta(s_t, s_{t-1})$。评测被形式化为两个排列推断任务：

- **正向**：给定 $o_0$ 和有序动作序列 $(a_0,\ldots,a_{L-2})$，以及打乱的观测集合 $O'$，模型输出排列 $\sigma \in \text{Sym}([L-1])$ 使 $(o'_{\sigma(1)}, \ldots, o'_{\sigma(L-1)}) = (o_1, \ldots, o_{L-1})$。
- **逆向**：给定 $o_0$ 和有序观测序列 $(o_1,\ldots,o_{L-1})$，以及打乱的动作集合 $A'$，模型输出排列 $\tau$ 使动作与观测进展一致。

这一设计将长时域交互视觉推理与高保真视频预测彻底解耦，使评测信号干净、可复现，同时隐式考察可供性识别、接触推理和局部可观测下的空间记忆。

**2. 可扩展关键帧轨迹合成：组合式数据爆炸**

原始机器人轨迹（30Hz）中大量时刻无语义变化，ENACT 通过检测场景图差分非空时刻提取关键帧集合 $K = \{t_1 < \cdots < t_M\}$，并用余弦相似度过滤近重复帧（基于谓词级变化特征签名 $c_j$）。从 $M$ 个关键帧中采样长度为 $L$ 的轨迹，由于 $L \ll M$（实践中 $L \leq 10$，$M \gtrsim 30$），单条轨迹可生成最多 $\binom{M}{L}$ 种不同候选，使数据规模从"轨迹数"变为"组合数"，理论上从单条轨迹就能生成百万量级 QA 对，实现真正的可扩展数据生成。

**3. 多粒度评测指标与在线验证器**

ENACT 设计了两层评测指标：任务准确率（Task Accuracy, TA）要求完全匹配——$\text{TA} = \frac{1}{|D|}\sum_{x \in D} \mathbf{1}[\text{accepted}(x)]$；配对准确率（Pairwise Accuracy, PA）给予局部正确的部分分数——$\text{PA} = \frac{\sum_x \#\text{正确相邻对}_x}{\sum_x L_x}$。由于多个合法排列可能同时满足约束，在线验证器接受任何与输入约束一致的排列，避免将多解问题误判为错误，更准确反映模型的因果推理能力。

**4. 精细化错误分析框架：五类错误分类**

通过将模型预测排列转换为对应的动作序列，再与仿真器提供的真值场景图差分做 Venn 图分析，ENACT 将每个原子状态差分归类为：正确（Correct）、遗漏（Omission）、幻觉（Hallucination）、实体替换（Entity Substitution）、极性反转（Polarity Inversion）、谓词替换（Predicate Substitution）五类。这一语义级错误分类比排列级比较更具诊断价值，能直接揭示模型认知失败的根因。

## 实验关键数据

### 主实验（Pairwise Accuracy，部分步长）

| 模型 | 正向 L=3 | 正向 L=6 | 正向 L=10 | 逆向 L=3 | 逆向 L=6 | 逆向 L=10 |
|------|----------|----------|-----------|----------|----------|-----------|
| 人类 | 93.62 | 93.87 | 95.13 | 92.05 | 94.25 | 96.29 |
| GPT-5 | 84.62 | 64.18 | 46.93 | 86.28 | 68.78 | 55.33 |
| GPT-5 mini | 87.50 | 63.41 | 44.11 | 85.05 | 67.67 | 50.02 |
| Gemini 2.5 Pro | 86.10 | 60.80 | 36.98 | 87.94 | 70.03 | 56.62 |
| InternVL3.5-241B | 75.79 | 45.85 | 25.24 | 82.26 | 53.38 | 30.56 |
| Qwen2.5-VL-72B | 78.15 | 41.92 | 25.07 | 77.80 | 48.19 | 36.27 |
| Claude Sonnet 4 | 65.65 | 30.52 | 20.16 | 73.25 | 43.07 | 28.49 |

### 消融实验（图像真实度 vs 相机配置，GPT-5 mini，Pairwise Acc 变化量 Δ）

| 配置变体 | 显著性 (p) | 影响 Δ | 说明 |
|----------|-----------|--------|------|
| 光线追踪（Path Tracing） | p≥0.2 | 小 | 渲染真实度不影响性能 |
| 真实图像（GPT-image-1 转换） | p≥0.2 | 小 | 仿真与真实差距极小 |
| FOV 孔径 60/80/Fisheye | p≤0.01 | 显著下降 | VLM 偏向人眼视角内参 |
| 相机高度 +0.5m（正向） | p<0.05 | Δ=−0.13 | 非标准高度显著损伤性能 |
| 右手 vs 左手（混淆率） | — | 右4.67% vs 左9.38% | 右手显著优于左手 |

### 关键发现

- 逆向任务一致优于正向任务（所有模型、所有步长），表明语言回顾性推理强于视觉前瞻性模拟
- 准确率随轨迹长度单调下降，L=8-10 时多数模型任务准确率接近零，而人类保持稳定 >93%
- GPT-5 和 Gemini 2.5 Pro 仅在 L=3 时接近人类水平，长时域差距迅速拉大
- 最主要错误类型：正向任务幻觉（43.9%）+ 遗漏（37.1%）≈ 81%；逆向任务各占 41.8%
- Cosmos-Reason1-7B（具身数据训练）在 L>5 时比同尺寸模型更稳定
- 仿真与真实世界评测结果高度一致，验证 sim-to-real 差距极小

## 亮点与洞察

- 任务设计极其简洁但覆盖广泛：序列重排 VQA 这一"窄"形式，隐式要求可供性识别、动作-效果推理、部分可观测长时域记忆等多种具身核心能力，同时避免了视频生成质量干扰
- 数据生成流水线真正可扩展：组合式关键帧采样使单条轨迹即可生成百万 QA，为大规模具身认知研究提供数据基础
- 人类对照令人震惊：人类在所有步长保持 ~94% 准确率，而最强 VLM（GPT-5）在 L=10 时降至 47%——这一差距比想象中大得多
- 拟人化偏见的量化揭示了 VLM 训练数据偏差的深层问题：VLM 默认"看世界"的视角与人类高度绑定，难以泛化到非人类机器人视角
- 错误分析框架的语义粒度为未来改进指明方向：主要问题不是误识别具体变化，而是遗漏/幻觉出根本不存在的状态变化

## 局限与展望

- 仅使用仿真数据（BEHAVIOR），尽管 sim-to-real 差距较小，真实世界轨迹的多样性仍受限
- 动作以场景图差分表示，依赖仿真器提供的真值状态，在真实机器人平台上难以直接复制
- 评测集规模（8972 QA）相比 LLM 常见基准偏小，且仅涵盖 29 项家庭活动，场景多样性有待扩展
- 当前未涉及跨模态（语言描述→动作执行）或实际机器人控制，仅测量"理解"层面能力

## 相关工作与启发

- **vs EmbodiedScan / ScanQA 等静态场景 VQA**：ENACT 引入了时序动作链和部分可观测性，从静态空间理解升级到动态因果推理
- **vs Aurora-Bench**：Aurora-Bench 聚焦短时域通用视频世界建模；ENACT 专注长时域机器人操控且有明确动作语义标注
- **vs BEHAVIOR 挑战赛**：ENACT 复用 BEHAVIOR 轨迹数据，但将其转化为评测导向而非训练导向的基准
- **vs Cosmos-Reason1 等具身 VLM**：ENACT 的评测结果直接指出具身数据训练在长时域稳定性上的价值，为未来具身 VLM 训练数据设计提供定量依据

## 评分

- 新颖性: ⭐⭐⭐⭐ 将世界建模×POMDP×序列重排 VQA 三者融合为统一评测框架，概念清晰且形式化严谨
- 实验充分度: ⭐⭐⭐⭐ 评测 30 个模型、8972 QA、多维度偏见分析（视角/FOV/手性）、仿真-真实对照，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，每节都有 Key Takeaways 框，公式与直觉表述兼顾，可读性极强
- 价值: ⭐⭐⭐⭐⭐ 填补了长时域具身认知评测的空白，数据可扩展至百万量级，对 VLM 具身能力研究有重要基础设施价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] World-In-World: World Models in a Closed-Loop World](world-in-world_world_models_in_a_closed-loop_world.md)
- [\[ICLR 2026\] Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments](test-time_mixture_of_world_models_for_embodied_agents_in_dynamic_environments.md)
- [\[CVPR 2026\] TraceGen: World Modeling in 3D Trace Space Enables Learning from Cross-Embodiment Videos](../../CVPR2026/robotics/tracegen_world_modeling_in_3d_trace_space_enables_learning_from_cross-embodiment.md)
- [\[ICLR 2026\] WorldGym: World Model as an Environment for Policy Evaluation](worldgym_world_model_as_an_environment_for_policy_evaluation.md)
- [\[ICLR 2026\] Empowering Multi-Robot Cooperation via Sequential World Models](empowering_multi-robot_cooperation_via_sequential_world_models.md)

</div>

<!-- RELATED:END -->
