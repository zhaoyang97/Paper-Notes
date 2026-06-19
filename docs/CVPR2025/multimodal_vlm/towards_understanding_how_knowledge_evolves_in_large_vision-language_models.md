---
title: >-
  [论文解读] Towards Understanding How Knowledge Evolves in Large Vision-Language Models
description: >-
  [CVPR 2025][多模态VLM][LVLM可解释性] 首次系统分析LVLM中多模态知识的演化过程，从**单token概率**、**token概率分布**和**特征编码**三个层次揭示知识演化的"关键层-突变层"双节点模式，将演化过程划分为**快速演化→稳定→突变**三个阶段，并发现深层突变与幻觉现象密切相关。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "LVLM可解释性"
  - "知识演化"
  - "early exit"
  - "幻觉分析"
  - "层级分析"
  - "模型压缩"
---

# Towards Understanding How Knowledge Evolves in Large Vision-Language Models

**会议**: CVPR 2025  
**arXiv**: [2504.02862](https://arxiv.org/abs/2504.02862)  
**代码**: [https://github.com/XIAO4579/Vlm-interpretability](https://github.com/XIAO4579/Vlm-interpretability)  
**领域**: 多模态VLM  
**关键词**: LVLM可解释性, 知识演化, early exit, 幻觉分析, 层级分析, 模型压缩

## 一句话总结
首次系统分析LVLM中多模态知识的演化过程，从**单token概率**、**token概率分布**和**特征编码**三个层次揭示知识演化的"关键层-突变层"双节点模式，将演化过程划分为**快速演化→稳定→突变**三个阶段，并发现深层突变与幻觉现象密切相关。

## 研究背景与动机
LVLM（如LLaVA系列）已成为许多AI应用的基础模型，但其内部工作机制仍是"黑箱"。一个核心问题是：**多模态特征是如何在LVLM的transformer层中逐步转化为自然语言的？** 理解这一过程对于优化模型效率和解决幻觉问题至关重要。

**现有痛点**：已有LLM可解释性研究（如probing、attention分析）主要针对纯文本模型，对于LVLM中视觉-语言特征的跨模态转化过程研究非常有限。特别是，LLM中观察到的知识演化模式不能直接推广到LVLM——因为LVLM需要将视觉信息忠实转化为语言描述。

**核心矛盾**：LVLM的语言模型部分参数量和训练数据远大于视觉模型，导致语言模型可能"主导"生成过程。当视觉知识在中间层稳定后，深层的语言先验可能注入与图像无关的知识，造成幻觉。

**切入角度**：从输出往回追溯——先观察token概率如何跨层变化，再分析概率分布的层间差异，最后深入特征编码空间，构建完整的知识演化图景。

## 方法详解

### 整体框架
在LLaVA-1.5（32层Vicuna-1.5后端）上进行分析。利用early exit技术，将语言头应用于每个中间层的隐藏特征，观察token概率、概率分布和特征编码三个层次的变化。不需要额外训练，所有分析直接在预训练模型上进行。

### 关键设计

1. **Token概率跨层分析（Token Probability Analysis）**
    - 功能：揭示单个token的概率在网络不同深度的变化规律
    - 核心思路：用early exit在每个中间层$j$计算token概率 $p_j(x_K^o|x_p) = \text{softmax}(\phi(H^j))_{x_K^o}$，追踪所有预测token从第0层到第32层的概率变化
    - 关键发现：
        - **关键层（~第20层）**：所有token在浅层概率接近零，到约第20层时概率急剧跃升，从词汇表中脱颖而出
        - **稳定token**：标点符号、确定性高的词（如直接复制自输入的"image"）在关键层后概率稳定
        - **突变token**：承载重要语义信息的词（如名词、动词）在深层发生概率突变，候选token之间竞争激烈
    - 设计动机：从最直观的概率视角观察知识何时"成形"

2. **Token概率分布的JS散度分析（Distribution-level Analysis）**
    - 功能：通过相邻层间概率分布的JS散度隐式揭示知识变化的速率
    - 核心思路：在每层计算所有token在词汇表上的概率分布，然后计算相邻层间的JS散度：$JSD(p_i \| p_j) = \frac{1}{2}(KLD(p_i\|A) + KLD(p_j\|A))$
    - 关键发现：
        - 浅层JS散度很大（知识快速变化），约第18层后急剧降低并趋近于零（知识趋于稳定）
        - 部分token在深层出现JS散度突变（知识二次演化），与突变层一致
        - 这一模式将知识演化分为三阶段：**快速演化→稳定→突变**
    - 设计动机：单token概率难以捕捉全局知识变化，分布级分析更全面

3. **特征编码的t-SNE可视化分析（Feature Encoding Analysis）**
    - 功能：在特征空间中直观观察知识演化的几何轨迹
    - 核心思路：将每层的4096维特征向量通过t-SNE压缩至2D，观察不同token/不同图像在各层的特征变化
    - 关键发现：
        - **单图多token**：所有token在初始层特征紧密聚集，随深度增加呈放射状线性扩散，每层特征紧邻前一层（连续性）
        - **多图单token（VQA）**：不同图像的特征形成"吉他状"形态——浅层特征聚集形成"琴颈"，深层特征向不同方向发散形成"琴身"
        - 浅层-深层的分界恰好对应关键层
    - 设计动机：特征级分析揭示知识演化的几何本质——从模态无关的通用表示逐渐分化为token特异性表示

### 跳层连接验证实验（Skip Connection）
- skip.1（跳过关键层到突变层间的稳定阶段）：输出与原始高度相似，甚至幻觉也保留→稳定阶段知识变化极小
- skip.2（仅跳过突变层）：大部分语义保留，部分幻觉被纠正（如"standing"→"playing"）→突变层注入外部知识是幻觉的潜在来源
- skip.3（跳过关键层到最后5层）：输出与原始差异大，幻觉增多→稳定阶段虽慢但仍在积累知识

## 实验关键数据

### 关键层位置统计

| 模型 | 总层数 | 关键层（约） | 突变层 |
|------|-------|------------|--------|
| LLaVA-1.5-7B | 32 | ~18-20层 | 26-30层（因token而异）|

### 幻觉关联分析
- 所有幻觉token（如"water"应为"camera"，"red"应为"black"，"dog"应为"sheep"）都在突变层发生概率逆转
- 正确token在突变前已具有概率优势，但在突变层后概率急速下降，幻觉token概率急速上升

### 跳层实验定性结果
- 跳过稳定阶段（~10层）后输出语义保持率极高→支持模型深度压缩的可行性
- 跳过突变层后部分幻觉被修复→支持突变层干预作为幻觉缓解策略

### LVLM vs. LLM对比
- 在纯文本LLM（LLaMA-1.5）上进行相同分析，**未发现**明显的层级结构或突变现象
- 功能词和信息词的分布变化模式不同于LVLM→知识演化的三阶段模式是LVLM的独有特征

## 亮点与洞察
- **首创性**：首次完整揭示LVLM中多模态知识从视觉特征到自然语言的演化轨迹
- **三层递进分析**：token概率→分布JS散度→特征t-SNE，自顶向下层层深入，结论互相验证
- **实用启示**：
    - 模型压缩：稳定阶段的层可以安全跳过→为深度裁剪提供理论依据
    - 幻觉缓解：突变层是幻觉注入点→针对性干预突变层可修复幻觉
    - 高效微调：浅层特征跨图像高度相似→只需微调深层参数即可泛化到新任务
- **关键层-突变层双节点模型**：简洁而有力地概括了LVLM的知识处理范式

## 局限性
- 仅在LLaVA-1.5-7B（32层）上验证，更大/更新模型（如LLaVA-Next、InternVL等）上的规律可能不同
- 分析主要是观察性的，缺乏因果性证明（如突变层是否*导致*幻觉，还是仅仅与之相关？）
- t-SNE可视化受参数选择影响，可能不完全反映高维空间的真实结构
- 未提出具体的优化方法（如基于发现的幻觉修复算法、压缩算法）

## 相关工作与启发
- DoLA（对比层解码提升事实性）→ 本文的突变层发现为DoLA提供了LVLM视角的理论支撑
- 知识神经元（Knowledge Neurons）→ 本文从层级而非神经元粒度分析知识
- Early exit技术 → 本文将其从加速推理工具转化为分析工具
- 启发：LVLM不是简单地将视觉特征"翻译"为语言，而是经历了一个复杂的知识演化过程，其中语言先验可能在深层"喧宾夺主"——这一视角对理解和改进LVLM有深远意义

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统揭示LVLM知识演化的三阶段模式和关键层-突变层双节点结构
- 实验充分度: ⭐⭐⭐⭐ 三层递进分析相互验证，跳层实验设计巧妙，但缺乏因果验证
- 写作质量: ⭐⭐⭐⭐ 分析路径清晰（概率→分布→特征），图示丰富直观
- 价值: ⭐⭐⭐⭐ 为模型压缩、幻觉缓解和高效微调提供理论依据，但未落地具体算法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge](vila-m3_enhancing_vision-language_models_with_medical_expert_knowledge.md)
- [\[CVPR 2025\] EventGPT: Event Stream Understanding with Multimodal Large Language Models](eventgpt_event_stream_understanding_with_multimodal_large_language_models.md)
- [\[CVPR 2025\] Embodied Scene Understanding for Vision Language Models via MetaVQA](embodied_scene_understanding_for_vision_language_models_via_metavqa.md)
- [\[NeurIPS 2025\] FlySearch: Exploring how vision-language models explore](../../NeurIPS2025/multimodal_vlm/flysearch_exploring_how_vision-language_models_explore.md)
- [\[CVPR 2025\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)

</div>

<!-- RELATED:END -->
