---
title: >-
  [论文解读] Unleashing Hour-Scale Video Training for Long Video-Language Understanding
description: >-
  [NeurIPS 2025][视频理解][长视频理解] 构建首个大规模小时级视频指令跟随数据集 VideoMarathon（9700小时、330万QA对、22种任务），并提出 Hour-LLaVA 模型，通过记忆仓库+遗忘机制+MemAug模块实现1-FPS下小时级视频的高效训练与推理，在四个长视频基准上全面领先同规模开源模型。
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "长视频理解"
  - "Video-LMM"
  - "记忆增强"
  - "指令跟随数据集"
  - "小时级视频"
---

# Unleashing Hour-Scale Video Training for Long Video-Language Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2506.05332](https://arxiv.org/abs/2506.05332)  
**代码**: [Project Page](https://videomarathon.github.io/)  
**领域**: 视频理解 / 多模态VLM  
**关键词**: 长视频理解, Video-LMM, 记忆增强, 指令跟随数据集, 小时级视频

## 一句话总结

构建首个大规模小时级视频指令跟随数据集 VideoMarathon（9700小时、330万QA对、22种任务），并提出 Hour-LLaVA 模型，通过记忆仓库+遗忘机制+MemAug模块实现1-FPS下小时级视频的高效训练与推理，在四个长视频基准上全面领先同规模开源模型。

## 研究背景与动机

**领域现状**：近期 Video-LMM 在视频QA、视频摘要等任务上取得显著进展，但训练数据大多是短视频（平均不到1分钟），现有数据集如 LLaVA-Video-178K 的平均视频长度仅 0.6 分钟。

**现有痛点**：测试基准（如 LVBench 平均67分钟、Video-MME 平均17分钟）要求模型理解小时级长视频，但训练时只见过几分钟的短视频，存在严重的训练-测试长度不匹配。现有方法（如均匀采样64帧）在处理长视频时信息损失巨大。

**核心矛盾**：缺乏高质量的长视频指令跟随数据 + 现有模型无法高效处理小时级视频的海量token。GPU显存限制使得模型无法直接消费1-FPS采样下数千帧的全部视觉token。

**本文目标**（1）构建大规模长视频训练数据；（2）设计能在有限计算下利用完整视频上下文的模型架构。

**切入角度**：借鉴人类记忆系统——选择性保留和回忆关键信息、系统性丢弃冗余信息，设计记忆增强机制在压缩token与保留信息间取得平衡。

**核心 idea**：用分层视频描述pipeline生成大规模长视频QA数据，用记忆仓库缓存全量视频特征+遗忘压缩+交叉注意力增强，实现可学习的token压缩。

## 方法详解

### 整体框架

系统包含两大部分：（1）**VideoMarathon 数据集**——通过分层视频描述（clip→event→global）+ DeepSeek-V3 生成 330万QA对；（2）**Hour-LLaVA 模型**——视频编码器（SigLIP）以1-FPS提取特征→投影器（2层MLP）→遗忘机制（空间+时间压缩至1/16）→MemAug模块（4层Transformer，交叉+自注意力回忆）→LLM解码器（Qwen2-7B）生成回答。完整视频特征存入记忆仓库，压缩后的衰减token经MemAug增强后输入LLM。

### 关键设计

1. **VideoMarathon 数据集构建**:

    - 从 Panda-70M、Ego4D、ActivityNet、YouCook2、MovieChat-1K 等5个来源收集28K长视频（3-60分钟），总时长约9700小时
    - 分层视频描述pipeline：先用 Qwen2VL-7B 对每个clip从时间性、空间性、物体、动作、场景、总结6个维度生成详细描述；再用 DeepSeek-V3 聚合为事件级和全局级描述
    - 基于分层描述生成覆盖22种任务（6大主题）的QA对，包含173万开放式QA和157万选择题
    - 设计动机：只有在大规模长视频上训练，模型才能显式学习长程依赖；实验证明 LLaVA-Video 即使用 VideoMarathon 训练也无法受益（因为稀疏采样无法学到长期模式）

2. **遗忘机制 + MemAug模块**:

    - **记忆仓库**：1-FPS采样的全量视频token存入记忆仓库，模型始终可访问完整视频上下文
    - **遗忘机制**：空间维度随机丢弃3/4 token（压缩比1/4），时间维度均匀丢弃3/4帧，总压缩比约1/16，得到"衰减视频token" $\tilde{\mathbf{H}}_v$
    - **MemAug模块**：4层Transformer块。交叉注意力中衰减token $\tilde{\mathbf{H}}_v$ + 问题token $\mathbf{H}_q$ 作为query，记忆仓库 $\mathbf{H}_v$ 作为key/value，从完整上下文中按需检索被丢弃的信息。自注意力中问题信息流向视频token，赋予其问题导向性。公式：$\hat{\mathbf{H}}_v = f_{\theta_M}(\tilde{\mathbf{H}}_v, \mathbf{H}_q | \mathbf{H}_v)$
    - 设计动机：与手工设计的关键帧选择、问题引导压缩不同，MemAug是可学习的压缩方式，由最终next-token prediction loss端到端监督

3. **三阶段训练**:

    - Stage 1：图像-语言预训练（3B图文对，仅训练MemAug）
    - Stage 2：视频-语言适应（0.6M混合数据，全参数训练1个epoch）
    - Stage 3：视频指令微调（4.4M混合数据，含0.7M VideoMarathon长视频样本，冻结视觉编码器，长短视频比例3:1最优）

### 损失函数 / 训练策略

标准交叉熵自回归损失。学习率2e-5，cosine退火，AdamW优化器。Hour-LLaVA-7B 用 64 AMD MI300X GPU 训练。

## 实验关键数据

### 主实验

| 方法 | 参数量 | TempCompass (11s) | LongVideoBench (459s) | Video-MME Long (2466s) | LVBench (4037s) |
|------|--------|------------------|-----------------------|----------------------|-----------------|
| GPT-4o | - | 70.9 | 66.7 | 65.3 | 48.9 |
| LLaVA-Video-7B | 7B | 60.0 | 58.2 | 56.3 | 33.7 |
| Apollo-7B | 7B | 60.0 | 56.0 | 60.0 | 37.1 |
| Video-XL | 7B | 59.3 | 55.4 | 55.5 | 38.8 |
| **Hour-LLaVA-7B** | **7B** | **63.2** | **60.1** | **62.2** | **40.5** |

### 消融实验

| 配置 | TempCompass | LongVideoBench | LVBench | 说明 |
|------|-------------|----------------|---------|------|
| Hour-LLaVA (MemAug) | 59.7 | 54.0 | 40.6 | 3B完整版 |
| 均匀压缩（无MemAug） | 59.3 | 52.1 | 38.3 | 可学习 vs 均匀 |
| 关键帧压缩 | 59.1 | 52.0 | 38.9 | 可学习 vs 关键帧 |
| 问题引导压缩 | 56.0 | 51.0 | 37.5 | 最差——过早筛选丢失上下文 |
| 记忆库100% → <10% | - | - | ~35 | 减少记忆库规模掉约5分 |

### 关键发现

- MemAug 在所有基准上一致超越手工压缩，LVBench 上比最佳手工方法高约2分
- LLaVA-Video 模型即使用 VideoMarathon 训练也无法受益于长视频数据——稀疏采样是根本瓶颈
- 空间遗忘保留1/4 token后，MemAug在图像任务上与无压缩基线可比（MMStar 51.9 vs 52.8），验证了对Image-LMM token压缩的通用性
- Hour-LLaVA-3B已超过半数7B模型，且在LVBench（平均视频超出训练最大长度）上展示超训练分布的泛化能力

## 亮点与洞察

- **填补数据空白**：首个大规模小时级视频指令跟随数据集，clip→event→global的三级描述策略巧妙解决了用大模型为长视频生成高质量QA的难题。可迁移到任何长视频标注场景。
- **可学习压缩 vs 手工压缩**：用统一实验框架对比了均匀/关键帧/问题引导三种策略，发现问题引导方法反而最差（过早的信息筛选丢失上下文），揭示了端到端学习的优势。
- **MemAug对Image-LMM的潜力**：空间压缩消融显示1/4 token + MemAug即可匹配全token性能，暗示记忆增强可直接用于Image-LMM推理加速。

## 局限与展望

- 训练数据完全由模型合成（Qwen2VL + DeepSeek-V3），含噪声和幻觉，未设计噪声鲁棒训练策略
- 仅处理视频+文本模态，忽略音频信息
- 评测以多选QA为主，无法全面衡量长视频理解能力
- 最长视频60分钟，超长视频（完整电影2-3小时）的效果有待验证
- 遗忘机制较朴素，可以探索基于内容重要性的自适应遗忘

## 相关工作与启发

- **vs LLaVA-Video**: 64帧均匀采样无法从长视频数据中学习，Hour-LLaVA的记忆库+MemAug架构是关键差异
- **vs LongVU/Video-XL**: 手工启发式压缩（关键帧选择、KV-cache检索），Hour-LLaVA用可学习MemAug在相同token预算下一致更优
- **vs LongVILA**: 扩展上下文到2M但需大量计算资源和序列并行，Hour-LLaVA通过压缩+恢复更高效

## 评分

- 新颖性: ⭐⭐⭐⭐ 数据集构建pipeline完整实用，MemAug设计合理但创新性适中
- 实验充分度: ⭐⭐⭐⭐⭐ 四个基准、详尽消融（遗忘策略、数据混合、记忆规模、压缩方法对比）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细，图表丰富
- 价值: ⭐⭐⭐⭐⭐ VideoMarathon填补了长视频训练数据的巨大空白，Hour-LLaVA证明了长视频训练的必要性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](../../ICCV2025/video_understanding/vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](../../AAAI2026/video_understanding/apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[NeurIPS 2025\] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)

</div>

<!-- RELATED:END -->
