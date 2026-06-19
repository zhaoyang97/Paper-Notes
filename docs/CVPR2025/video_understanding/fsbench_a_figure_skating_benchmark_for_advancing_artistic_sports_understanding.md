---
title: >-
  [论文解读] FSBench: A Figure Skating Benchmark for Advancing Artistic Sports Understanding
description: >-
  [CVPR 2025][视频理解][花样滑冰] 提出 FSAnno/FSBench，首个面向花样滑冰的细粒度、多模态、多层次基准数据集，覆盖从先验知识测试、单个动作识别/评估/解说到整体表演评估/解说的完整任务链，揭示了现有 LLM 在艺术体育理解上的显著不足。 花样滑冰被誉为"冰上艺术"，其评价不仅依赖技术难度（跳跃、旋转…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "花样滑冰"
  - "运动理解基准"
  - "艺术体育"
  - "多模态大语言模型"
  - "动作质量评估"
---

# FSBench: A Figure Skating Benchmark for Advancing Artistic Sports Understanding

**会议**: CVPR 2025  
**arXiv**: [2504.19514](https://arxiv.org/abs/2504.19514)  
**代码**: 无（数据集可获取）  
**领域**: 视频理解  
**关键词**: 花样滑冰, 运动理解基准, 艺术体育, 多模态大语言模型, 动作质量评估

## 一句话总结

提出 FSAnno/FSBench，首个面向花样滑冰的细粒度、多模态、多层次基准数据集，覆盖从先验知识测试、单个动作识别/评估/解说到整体表演评估/解说的完整任务链，揭示了现有 LLM 在艺术体育理解上的显著不足。

## 研究背景与动机

花样滑冰被誉为"冰上艺术"，其评价不仅依赖技术难度（跳跃、旋转），还涉及艺术表达（节奏感、流畅性、情感表达）。现有数据集和研究存在三个关键缺陷：

1. **任务单一**：已有数据集（FSD-10 做动作识别、FisV 做评分、MCFS 做时序分割）各自为战，缺乏将单个动作与整体表演关联的能力
2. **忽略艺术性**：主流运动理解研究聚焦球类运动的战术和策略分析（如 SportQA、SPORTU），完全忽视了花样滑冰等艺术体育的独特评判维度——情感表达、音乐诠释、动作流畅性
3. **模态单一**：长视频中 MLLM 难以关注局部细节，短视频 MLLM 又难处理长序列。骨骼数据缺少动作的形状和外观信息，不足以刻画旋转、过渡等微妙动作

## 方法详解

### 整体框架

本文的贡献分为三部分：
- **FSAnno**：大规模训练/验证数据集，包含 783 个完整花滑表演的多模态、多粒度标注
- **FSBench**：独立评测基准，分为 FSBench-Text（多选题 + 解释）和 FSBench-Motion（多模态QA对）
- **SkateLLM**：基于 MotionGPT 的领域微调模型，用于验证数据集的训练价值

### 关键设计

1. **多层次任务体系**:
    - 功能：从简到难、从局部到全局地评估模型对花样滑冰的理解深度
    - 核心思路：分为三层——先验知识测试（4200+ 规则/赛事信息多选题）、个体动作层（动作识别、单动作评估、单动作解说）、完整表演层（时序分割、表演评估、表演解说）
    - 设计动机：模拟真实裁判的评审流程——先识别每个技术动作并评分，再综合评估整体表现

2. **多模态数据获取与隐私保护**:
    - 功能：提供身份无关的多模态数据，确保公平评估
    - 核心思路：从原始比赛视频中用 4DHumans 提取 motion 数据、用 HRNet 估计骨骼数据，得到去身份化的运动表征。同时保留 RGB 视频链接供社区使用
    - 设计动机：MLLM 可能通过识别运动员外貌或比赛环境来"作弊"，使用 motion/skeleton 模态能更公平地评估模型对动作本身的理解

3. **多源标注策略**:
    - 功能：提供客观、多维度、细粒度的标注
    - 核心思路：标注来自三个权威来源——官方裁判报告（GOE 评分、TES/PCS 多维度评分）、视频内容（时序分割、动作类别）、音频解说（Whisper 语音转文字后按时间戳对齐到动作），对于整体评价使用 LLM 整合解说员零散评论
    - 设计动机：利用裁判报告保证标注准确性，解说员评论提供艺术性描述，多源融合实现技术与艺术的全面覆盖

### 损失函数 / 训练策略

SkateLLM 基于 MotionGPT 进行指令微调：
- 使用通用动作理解数据（HumanML3D 等）+ 花样滑冰专用描述数据
- 用 GPT-4 根据模板生成花滑动作描述，高 GOE 加正面艺术评价，低 GOE 加建设性批评
- 两阶段训练：视觉-文本对齐预训练 + LoRA 指令微调（避免灾难性遗忘）
- 使用交叉熵 loss 做 token 预测

## 实验关键数据

### LLM 先验知识测试

| 模型 | 赛事信息 Acc. | 规则 Acc. | 知识竞答 Acc. |
|------|-------------|---------|-------------|
| GPT-4 | **73.0%** | **78.0%** | **87.9%** |
| GPT-3.5-turbo | 59.0% | 64.8% | 72.7% |
| LLaVA 13B | 47.0% | 51.4% | 63.6% |

### 动作描述生成（AutoDQ 指标）

| 方法 | F1 (↑) | Recall (↑) | Precision (↑) |
|------|--------|-----------|-------------|
| SkateLLM | **38.0** | **58.3** | **61.6** |
| Motion-GPT | 7.1 | 3.8 | 27.5 |

### 关键发现

1. **现有 LLM 对花样滑冰理解严重不足**：即使是 GPT-4 在规则测试中也只达到 78%（"全明星"等级），开源模型表现平庸
2. **Motion-GPT 直接用于花样滑冰效果极差**（F1 仅 7.1），但经过 FSAnno 微调后的 SkateLLM 提升至 38.0，说明领域数据对于专业运动理解至关重要
3. SkateLLM 的 precision 远高于 recall，原因是花滑动作类别间差异微妙（如不同旋转圈数），模型能识别大类但难以区分细类

## 亮点与洞察

- **填补重要空白**：首个同时关注技术性和艺术性的花样滑冰理解基准，且是目前唯一包含动作评估和解说标注的数据集（对比 Table 2）
- **任务设计贴合实际**：模拟了从"观众→裁判→解说员"的多角色理解过程
- **AutoDQ 评估指标**比传统 BLEU/METEOR 更适合评估技术描述的语义准确性，通过提取关键事件进行匹配
- **负样本的重要性**：以往数据集多来自顶级赛事，缺少失误样本；FSAnno 覆盖多级别赛事，包含充足的负 GOE 样本

## 局限与展望

- 目前实验主要聚焦 motion-based 方法，video-based MLLM 的完整评估留待后续
- SkateLLM 仅做了最基础的动作描述任务，更复杂的评分和解说任务尚未展开
- Motion 数据的提取质量依赖于 4DHumans/HRNet，在复杂动作（快速旋转、多人）中可能有误差
- 目前仅覆盖单人滑，双人滑和冰舞的团队配合理解是更大的挑战

## 相关工作与启发

- **与 SportQA/SPORTU 的区别**：这些基准聚焦球类运动的战术理解，FSBench 是首个面向艺术体育的评测基准
- **与传统花滑数据集的进化**：从单任务（FSD-10 仅识别、FisV 仅评分）到多任务全链路评估
- **对未来的启示**：体操、芭蕾、跳水等其他评分类运动也可借鉴此基准构建思路

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个系统性的艺术体育理解基准，任务设计有深度
- 实验充分度: ⭐⭐⭐ 实验偏少，仅展示了先验知识测试和基础动作描述，更多任务的评测缺失
- 写作质量: ⭐⭐⭐ 结构完整但篇幅有限，很多内容推到了附录
- 价值: ⭐⭐⭐⭐ 开辟了艺术体育理解的新赛道，数据集本身有持久价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM](videorefer_suite_advancing_spatial-temporal_object_understanding_with_video_llm.md)
- [\[CVPR 2025\] SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding](seriesbench_a_benchmark_for_narrative-driven_drama_series_understanding.md)
- [\[CVPR 2025\] Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs](q-bench-video_benchmark_the_video_quality_understanding_of_lmms.md)
- [\[ICCV 2025\] Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding](../../ICCV2025/video_understanding/towards_video_thinking_test_a_holistic_benchmark_for_advanced_video_reasoning_an.md)
- [\[CVPR 2025\] M-LLM Based Video Frame Selection for Efficient Video Understanding](m-llm_based_video_frame_selection_for_efficient_video_understanding.md)

</div>

<!-- RELATED:END -->
