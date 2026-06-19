---
title: >-
  [论文解读] Filter-And-Refine: A MLLM Based Cascade System for Industrial-Scale Video Content Moderation
description: >-
  [ACL 2025][多模态VLM][MLLM] TikTok提出一种基于MLLM的两阶段级联内容审核系统（Router-Ranker），通过轻量级嵌入检索路由器过滤97.5%的合规流量，仅将高风险视频送入微调后的LLaVA进行精细分类，F1提升66.5%的同时部署成本降至直接全量部署的1.5%。 短视频平台（如TikTok…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "MLLM"
  - "内容审核"
  - "级联系统"
  - "视频分类"
  - "工业部署"
---

# Filter-And-Refine: A MLLM Based Cascade System for Industrial-Scale Video Content Moderation

**会议**: ACL 2025  
**arXiv**: [2507.17204](https://arxiv.org/abs/2507.17204)  
**代码**: 无  
**领域**: Multimodal VLM / 内容审核  
**关键词**: MLLM, 内容审核, 级联系统, 视频分类, 工业部署

## 一句话总结
TikTok提出一种基于MLLM的两阶段级联内容审核系统（Router-Ranker），通过轻量级嵌入检索路由器过滤97.5%的合规流量，仅将高风险视频送入微调后的LLaVA进行精细分类，F1提升66.5%的同时部署成本降至直接全量部署的1.5%。

## 研究背景与动机
短视频平台（如TikTok、YouTube Shorts）的爆发式增长使内容审核成为刚需。传统视频分类模型能处理显式违规内容，但对隐含有害内容（如隐晦的误导信息、暗示性图像）和上下文依赖的审核场景力不从心。多模态大语言模型（MLLM）凭借跨模态推理和上下文理解能力，理论上可以解决这些痛点，但面临两个核心障碍：

**计算成本过高**：每天数亿条新视频上传，直接对全量流量跑MLLM在算力上不可行

**生成模型做分类的适配难题**：MLLM本质是生成模型，如何高效转化为判别性分类器尚无成熟方案

本文的切入角度是借鉴推荐系统中的"召回-排序"范式，将审核任务拆解为先粗筛再精排，同时提出一种极简的生成模型→分类器转化方法，用极少量标注数据即可完成。

## 方法详解

### 整体框架
系统分为两个阶段：

1. **Router（路由器/召回阶段）**：轻量化模型快速过滤低风险视频，仅放行可疑内容
2. **Ranker（排序器/精排阶段）**：MLLM对高风险子集进行细粒度推理和分类

这种层级化过滤大幅减少了MLLM需要处理的视频量，兼顾了审核精度和可扩展性。

### 关键设计

1. **基于嵌入检索的Router**：

    - 核心思路：维护一个高风险代表性视频库（seed videos），新视频通过语义相似度与seed视频匹配来判断是否高风险
    - 无需标注数据，无监督训练
    - 种子选择策略：质心邻近选择（聚类算法）+ 人工精选（标注员选"黄金种子"）
    - 优势：种子库可快速更新以适应新型违规模式，灵活性强
    - 实际效果：过滤掉97.5%的合规流量，不增加服务延迟

2. **MLLM作为Ranker的微调方法**：

    - 骨架：LLaVA（Mistral-7B + ViT-Large + 2层MLP投影器）
    - 关键创新：将生成任务简化为**单token预测**——通过prompt工程限制模型只输出Yes/No一个token，提取该token的logits做softmax得到概率分数
    - 训练数据仅需约300K样本（传统分类模型数据量的2%），三类数据1:1:1混合——VQA数据（来自LLaVA-Mix665k子集）、视频Caption数据、审核分类数据
    - 这是一种非常务实的转化思路：不改架构，只改训练格式和输出解码方式

3. **Prompt设计**：

    - 设计4种prompt模板，探索不同的提问方式：直接问总体标签、分别问细粒度和总体标签、顺序问两个标签强调关联、先给定义再问
    - 实验发现：分别问两个问题（P2）效果最好，单问题提供的信息不足，将两个标签合在一个prompt中反而引入噪声

### 损失函数 / 训练策略
探索了两种SFT策略：
- **Multi-task Learning（多任务混合训练）**：直接将三类数据混合训练，约20小时/8×A100
- **Mixed Sequential Phased Learning（分阶段训练）**：先在VQA+Caption数据上做视觉指令调优，再在审核分类数据上做第二阶段SFT，总计约20.5小时/8×A100

多任务训练在所有prompt上表现更鲁棒，分阶段训练则更灵活且时间效率更高。

## 实验关键数据

### 主实验

| 模型 | PR-AUC | ROC-AUC | Max-F1 |
|------|--------|---------|--------|
| X-VLM（传统多模态分类） | 30.79 | 65.31 | 36.81 |
| LLaVA（零样本） | 23.17 | 58.59 | 31.32 |
| LLaVA w/ Caption（零样本） | 28.85 | 65.88 | 36.71 |
| Multi-task + P2（最佳） | **68.73** | **87.68** | **61.29** |

MLLM方案在F1上比传统分类模型提升66.50%，微调后比零样本在PR-AUC上提升45.55%。

### 消融实验

| 配置 | PR-AUC | ROC-AUC | 说明 |
|------|--------|---------|------|
| 原始输出 | 68.73 | 87.68 | Multi-task + P2基线 |
| Union概率 | 68.78 | **87.83** | 细粒度+总体标签取并集 |
| 加权求和 | **68.83** | 87.78 | 加权融合，PR-AUC最优 |
| 贝叶斯融合 | 68.67 | 87.79 | 略低于基线 |
| 温度调节(0.2-0.8) | ~不变 | ~不变 | 温度对性能影响不大 |

### 线上实验

| 指标 | 效果 |
|------|------|
| 自动审核量提升 | +41.27% |
| 系统精确度提升 | +19.16% |
| Router流量过滤率 | 97.5% |
| 人工标注数据需求降低 | 比传统模型仅需2% |

### 关键发现
- MLLM在内容审核这种需要深层理解的任务上大幅优于传统分类模型
- 仅需极少量标注数据（2%）即可完成微调，极大节省人力
- 级联部署将计算成本压缩到全量部署的1.5%，实现了工业可行性
- 嵌入可视化清晰显示微调后的模型能学到更好的决策边界

## 亮点与洞察
- **思路极简但有效**：将MLLM生成输出限制为单token Yes/No，再提取logits做softmax，这个转化方法几乎零额外开销，却能充分利用MLLM的语义理解能力
- **借鉴推荐系统架构**：召回-排序的级联设计非常自然地适配了审核场景的"大量合规+少量违规"分布
- **无标注的Router**：使用无监督嵌入检索做第一阶段过滤，避免了标注瓶颈
- **实际落地验证**：不是纯学术实验，而是在TikTok真实生产环境中部署并验证了效果

## 局限与展望
- 仍然依赖少量人工标注数据，可能引入噪声
- Router的召回率有上限，存在漏检风险
- 仅展示了Mistral-7B级别的MLLM，更大/更强模型是否能进一步提升？
- 未讨论多语言和跨文化审核的适应性
- 种子视频库的维护和更新策略未详细讨论

## 相关工作与启发
- **推荐系统架构迁移**：召回-排序范式在搜索/推荐中已经很成熟，本文证明其在内容审核中同样有效，这种跨领域架构借鉴值得学习
- **生成→判别转化**：Sparse Attention Vectors（Mitra et al., 2025）等工作也在探索类似的MLLM分类化方向，本文的单token+softmax方法更加轻量
- **对工业部署的启示**：大模型落地不一定要全量推理，级联设计+精准路由可以大幅降低成本

## 评分
- 新颖性: ⭐⭐⭐ 各组件技术并不新，但组合方式为内容审核提供了实用的工业方案
- 实验充分度: ⭐⭐⭐⭐ 涵盖离线评估、消融、线上A/B测试，数据说服力强
- 写作质量: ⭐⭐⭐⭐ 结构清晰，工业论文风格务实
- 价值: ⭐⭐⭐⭐ 对工业界有直接参考价值，展示了MLLM在审核领域的可行路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](speaking_beyond_language.md)
- [\[CVPR 2026\] Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset](../../CVPR2026/multimodal_vlm/towards_open-vocabulary_industrial_defect_understanding_with_a_large-scale_multi.md)
- [\[CVPR 2025\] Efficient Motion-Aware Video MLLM](../../CVPR2025/multimodal_vlm/efficient_motion-aware_video_mllm.md)
- [\[CVPR 2025\] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](../../CVPR2025/multimodal_vlm/video-xl_extra-long_vision_language_model_for_hour-scale_video_understanding.md)
- [\[CVPR 2025\] V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents](../../CVPR2025/multimodal_vlm/v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)

</div>

<!-- RELATED:END -->
