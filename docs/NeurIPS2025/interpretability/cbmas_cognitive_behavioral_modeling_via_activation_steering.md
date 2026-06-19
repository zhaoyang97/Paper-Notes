---
title: >-
  [论文解读] CBMAS: Cognitive Behavioral Modeling via Activation Steering
description: >-
  [NeurIPS 2025][可解释性][激活引导] CBMAS提出一个将激活引导作为连续诊断工具的框架，通过密集α扫描和注入-读取层解耦，将认知偏差分析从"有偏差/无偏差"的二元判断升级为可追踪翻转点、传播路径和衰减模式的连续轨迹分析，在GPT-2 Small上揭示了安抚行为在浅层强烈编码但向深层快速衰减的规律。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "激活引导"
  - "认知偏差"
  - "偏差响应曲线"
  - "logit lens"
  - "层敏感性分析"
---

# CBMAS: Cognitive Behavioral Modeling via Activation Steering

**会议**: NeurIPS 2025  
**arXiv**: [2601.06109](https://arxiv.org/abs/2601.06109)  
**代码**: [有](https://github.com/shimamooo/CBMAS)  
**领域**: 可解释性 / LLM行为分析  
**关键词**: 激活引导, 认知偏差, 偏差响应曲线, logit lens, 层敏感性分析

## 一句话总结

CBMAS提出一个将激活引导作为连续诊断工具的框架，通过密集α扫描和注入-读取层解耦，将认知偏差分析从"有偏差/无偏差"的二元判断升级为可追踪翻转点、传播路径和衰减模式的连续轨迹分析，在GPT-2 Small上揭示了安抚行为在浅层强烈编码但向深层快速衰减的规律。

## 研究背景与动机

LLM在不同prompt和上下文中表现出各种认知行为（谄媚、安抚、满足化、顺从等），但这些行为在模型内部的编码方式不可预测且难以控制。现有的偏差评估方法存在两个根本问题：

第一，**偏差被当作二元现象处理**。传统方法通过成对prompt比较（如"He is a doctor" vs "She is a doctor"）来检测偏差的存在或程度，但这种"快照式"方法忽视了偏差在模型内部的连续潜在结构——偏差并非简单地"有"或"没有"，而是存在一个从萌生到翻转到饱和的连续动态过程。

第二，**高层行为评估与低层表示分析之间存在鸿沟**。机制可解释性已经揭示了注意力头、MLP层和残差流中的精细结构，但这些工具很少被应用于认知行为研究。结果是我们既无法解释模型为什么表现出特定认知偏差，也无法在不重新训练的情况下精准干预。

CBMAS的核心想法是将激活引导（activation steering）从一种控制手段转化为一种诊断工具。通过构建认知行为方向上的引导向量，并沿着引导强度α和模型层深度两个维度进行密集扫描，可以产生偏差响应曲线（Bias Response Curve），从而揭示离散快照方法无法捕捉到的翻转点、传播模式和层位敏感性。

## 方法详解

### 整体框架

CBMAS的分析流程分为四步：（1）从对比prompt对数据集中提取引导向量；（2）在指定的注入层添加引导向量，并在多个读取层观察效果；（3）沿α范围密集扫描，收集多维指标；（4）分析偏差响应曲线，识别翻转点和传播规律。整个框架的关键创新在于将注入层（injection layer）和读取层（readout layer）解耦，形成(注入层, 读取层, α)三维分析空间。

### 关键设计

1. **对比prompt数据集与引导向量构建**:
    - 功能：为每种认知行为（谄媚、安抚、满足化、顺从）提取表示偏差方向的向量
    - 核心思路：每个数据点包含一对结构相同但选择相反的prompt。选项A代表表现目标认知行为的回答，选项B代表中性回答。对于给定层L和注入位点S，引导向量定义为 $\mathbf{v}_L^{(S)} = \mathbb{E}[\mathbf{h}_L^{(S)}(p^{(A)}) - \mathbf{h}_L^{(S)}(p^{(B)})]$，即所有对比对在该层的隐状态差异的均值
    - 设计动机：通过多样的prompt对覆盖目标行为的各种表现形式，使引导向量代表行为的一般方向而非某个特定语境。每种行为提供200个对比例子，涵盖建议咨询、技术问题、健康、金融等多个领域

2. **偏差响应曲线（BRC）协议**:
    - 功能：将偏差分析从离散快照扩展为连续轨迹
    - 核心思路：在用户定义的α范围（默认-10到10，步长0.5）内密集扫描。对每个α值，在注入层修改隐状态 $\mathbf{h} \leftarrow \mathbf{h} + \alpha \mathbf{v}$，然后在读取层记录六种指标：Logit差异 $\Delta_{logit}(\alpha) = \text{logit}(y_A|x,\alpha) - \text{logit}(y_B|x,\alpha)$、概率差异、几率比、KL散度（衡量对整体分布的扰动程度）、逐token困惑度（流畅性代理）和排名轨迹（目标token排名变化）
    - 设计动机：强制二元输出——prompt以"I choose ("结尾，迫使模型在A/B之间选择。同时使用随机向量和垂直于偏差向量的正交向量作为控制组，以确认观察到的效果确实来自偏差方向

3. **注入-读取层解耦分析**:
    - 功能：追踪偏差信号从注入点向后续层的传播和变化
    - 核心思路：对所有满足 $L_{read} > L_{inj}$ 的(注入层, 读取层)组合进行分析，形成"偏差传播图谱"。支持对不同注入位点（hook_resid_mid、hook_resid_post等）的对比分析
    - 设计动机：如果只在注入层观察效果，无法知道干预是否真正改变了模型的最终行为，还是被后续层"洗掉"了。解耦设计允许追踪信号的放大、衰减和消散

### 损失函数 / 训练策略

CBMAS是纯分析框架，不涉及模型训练。所有分析在推理时完成，使用TransformerLens进行激活层面的干预和读取。实验固定seed=42，整套实验可在单块A40 GPU上约4-7分钟内完成复现。

## 实验关键数据

### 主实验

在GPT-2 Small（12层）上对reassurance（安抚行为）引导向量的分析结果：

| 注入层→读取层 | α翻转区域 | Logit差异斜率 | KL散度 | 控制组表现 |
|---------------|-----------|--------------|--------|-----------|
| L0→L1 | α≈0 | 陡峭正斜率 | 低且对称 | 平坦 |
| L0→L6 | α≈0 | 中等斜率 | 低 | 平坦 |
| L0→L11 | 无明显翻转 | 近乎平坦 | 低且对称 | 平坦 |
| L1→L6 | α≈0 | 单调递增 | 低 | 平坦 |
| L3→L4 | α≈0 | 排名翻转 | — | — |
| L3→L6 | α≈0 | 连续logit轨迹 | 低 | 平坦 |

### 消融实验

| 对比配置 | 关键观察 | 说明 |
|---------|---------|------|
| 偏差向量 vs 随机向量 | 偏差向量产生单调轨迹，随机向量平坦 | 效果来自偏差方向而非噪声 |
| 偏差向量 vs 正交向量 | 正交向量同样平坦 | 双重控制增强结论可信度 |
| L0注入 vs L1注入 | L1产生更干净的单调信号 | L0更嘈杂，表示结构在L1后更稳定 |
| 浅层读取 vs 深层读取 | 浅层(L1)信号强，深层(L11)衰减殆尽 | 安抚行为在浅层编码后被逐渐稀释 |

### 关键发现

- **翻转点客观存在**：在α≈0附近，模型行为发生质变——从偏好选项A到偏好选项B的转变是突变而非渐变，这是离散评估方法根本无法捕捉到的现象
- **浅层编码、深层衰减**：安抚行为方向在L1处效果极强，但随层深度增加迅速衰减，到L11几乎完全消散。这意味着该认知行为的表示在模型早期就已建立
- **注入位点决定信号质量**：L1比L0更适合作为注入点，产生更干净的因果信号。这暗示token嵌入层（L0）还未建立稳定的行为表示
- **干预不破坏流畅性**：KL散度在整个α扫描范围内保持低水平且对称分布，说明激活引导是可控手段

## 亮点与洞察

- **从二元到连续的范式转变**：传统方法只问"有没有偏差"，CBMAS追问"偏差在哪里产生、如何传播、何时翻转、何时消散"，这是对偏差分析范式的本质性升级
- **注入-读取解耦是核心创新**：这种设计把一维的α扫描扩展为三维的探测空间，使得"偏差传播图谱"的构建成为可能
- **控制组设计严谨**：同时使用随机向量和正交向量作为对照，比单一控制组更有说服力地排除了噪声假说
- **实用工具和数据集**：提供CLI工具和四种认知行为（谄媚、安抚、满足化、顺从）的200例数据集，有较好的可复现性
- **"L1定型"现象**值得关注：暗示认知行为的表示分布可能存在层级组织结构

## 局限与展望

- **模型规模严重不足**：仅在GPT-2 Small（117M参数，12层）上验证，结论是否适用于现代大模型（LLaMA-70B、GPT-4等80+层模型）完全不确定，偏差编码的层级分布可能完全不同
- **仅分析next-token预测**：引导效果在长文本自回归生成中是否持续、累积还是衰减未做研究
- **缺乏因果分析**：α扫描揭示的是相关性模式，但没有回答"哪些注意力头或MLP组件负责偏差编码"，需结合activation patching等因果方法
- **数据集规模和质量**：每种行为仅200个例子，部分由LLM生成，手工构造的对比prompt对可能引入偏差
- **行为维度有限**：仅测试4种认知行为，LLM可能表现出的认知偏差还有确认偏差、锚定效应、框架效应等

## 相关工作与启发

- 在ActAdd和CAA的引导向量构建基础上创新——前者限于单层和窄α范围评估，CBMAS首次做系统性连续分析
- Logit lens和causal mediation等工具被用作分析手段但未深度整合——未来可将因果归因与连续引导分析结合
- 与RLHF/Constitutional AI的根本区别：CBMAS是诊断工具而非对齐方法，它不改变模型，只揭示内部认知行为结构
- 翻转点分析可为安全对齐提供"安全裕量"的量化参考——如果知道某种有害行为在α=X处翻转，就可以评估干预的margin

## 评分

- 新颖性: ⭐⭐⭐⭐ 从离散到连续的分析范式转变有创新性，注入-读取解耦设计巧妙
- 实验充分度: ⭐⭐⭐ 仅在GPT-2 Small上实验规模严重不足，但在该模型上的分析维度覆盖较全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学形式化严谨，图表信息量大
- 价值: ⭐⭐⭐ 框架设计合理但需在现代大模型上验证才能真正发挥影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)
- [\[ICLR 2026\] Activation Steering with a Feedback Controller](../../ICLR2026/interpretability/activation_steering_with_a_feedback_controller.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](../../ICML2026/interpretability/steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[NeurIPS 2025\] Curvature Tuning: Provable Training-free Model Steering From a Single Parameter](curvature_tuning_provable_training-free_model_steering_from_a_single_parameter.md)
- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)

</div>

<!-- RELATED:END -->
