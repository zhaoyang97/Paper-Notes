---
title: >-
  [论文解读] Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension
description: >-
  [NeurIPS 2025][目标检测][检索增强生成] 本文提出Video-RAG，一个免训练、即插即用的RAG管道，通过从视频中提取视觉对齐的辅助文本（OCR、ASR、目标检测）并经检索筛选后输入LVLM，在仅增加约2K token的条件下将7个开源LVLM的Video-MME平均性能提升2.8%，72B模型超越GPT-4o。
tags:
  - "NeurIPS 2025"
  - "目标检测"
  - "检索增强生成"
  - "长视频理解"
  - "辅助文本"
  - "即插即用"
  - "多模态对齐"
---

# Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension

**会议**: NeurIPS 2025  
**arXiv**: [2411.13093](https://arxiv.org/abs/2411.13093)  
**代码**: [https://github.com/Leon1207/Video-RAG-master](https://github.com/Leon1207/Video-RAG-master)  
**领域**: 目标检测  
**关键词**: 检索增强生成, 长视频理解, 辅助文本, 即插即用, 多模态对齐

## 一句话总结

本文提出Video-RAG，一个免训练、即插即用的RAG管道，通过从视频中提取视觉对齐的辅助文本（OCR、ASR、目标检测）并经检索筛选后输入LVLM，在仅增加约2K token的条件下将7个开源LVLM的Video-MME平均性能提升2.8%，72B模型超越GPT-4o。

## 研究背景与动机

现有大型视频-语言模型（LVLM）在理解长视频时受限于有限的上下文长度。面对这一挑战，当前有两条技术路线：

**微调长上下文LVLM**：如LongVA通过在扩展文本上预训练来增加token容量，但需要海量高质量数据和大量GPU资源，且实验表明简单增加采样帧数反而降低性能（LongVA从128帧增加到384帧，准确率从52.6%降至51.8%）

**基于GPT的Agent方法**：如VideoAgent、DrVideo等，使用多轮交互和专有模型处理长视频，但计算成本极高（在Video-MME上运行VideoAgent约需20天和~$2000的API费用）且依赖闭源模型

本文的动机是找到一种**免训练、低成本、兼容任意LVLM**的解决方案。核心思路是：与其增加视觉token数量，不如用更精炼的辅助文本来补充视觉信息不足的问题——这些文本既与视觉内容对齐，又能提供超出视觉的额外信息（如音频内容）。

## 方法详解

### 整体框架

Video-RAG包含三个阶段：
1. **查询解耦**：将用户问题分解为辅助文本的检索请求
2. **辅助文本生成与检索**：并行生成三种辅助文本并通过RAG检索相关内容
3. **整合与生成**：将检索到的辅助文本与查询和视频帧一起输入LVLM

### 关键设计

1. **查询解耦（Query Decouple）**:

    - LVLM仅处理文本输入（不访问视频帧），将用户查询分解为三类检索请求：
        - R_asr：关于语音识别的请求（提取音频信息）
        - R_det：物体检测请求（识别视频中的物理实体）
        - R_type：物体信息类型请求（位置、数量、关系）
    - 输出为JSON格式，可为NULL表示不需要该类信息

2. **辅助文本生成与RAG检索**:

    - **OCR数据库**：使用EasyOCR对每帧进行文字识别，用Contriever编码为向量，存入FAISS索引
    - **ASR数据库**：使用Whisper转录音频，分块后同样编码存入FAISS
    - **DET数据库**：先通过CLIP相似度筛选关键帧（阈值t=0.3），再用APE（开放词汇目标检测）在关键帧上检测查询相关物体
    - 检索时用Contriever编码查询+请求，通过FAISS计算相似度，保留超过阈值的文本块

3. **目标检测信息的场景图处理**:

    - 将原始检测结果（"类别: [bbox]"）处理为三种结构化信息：
        - 物体位置（A_loc）：精确描述物体类别和坐标
        - 物体计数（A_cnt）：统计各类物体数量
        - 相对位置关系（A_rel）：描述物体间的空间关系
    - 通过场景图组织，使LVLM更容易理解物体关系

4. **整合与生成**:

    - 将OCR、ASR、DET辅助文本按时间顺序合并
    - 与用户查询和视频帧一起输入LVLM生成答案
    - 全过程为单轮检索，无需多轮交互

## 实验关键数据

### 主实验：Video-MME基准

| 模型 | 参数量 | 帧数 | 无字幕 | 有字幕 | +Video-RAG | 提升 |
|------|--------|------|--------|--------|-----------|------|
| Video-LLaVA | 7B | 8 | 39.9% | 41.6% | 45.0% | **+3.4%** |
| LLaVA-NeXT-Video | 7B | 16 | 43.0% | 47.7% | 50.0% | +2.3% |
| LongVA | 7B | 128 | 52.6% | 56.0% | 62.0% | **+6.0%** |
| Long-LLaVA | 7B | 64 | 52.9% | 57.8% | 62.6% | **+4.8%** |
| Qwen2-VL | 72B | 32 | 64.9% | 71.9% | 72.9% | +1.0% |
| LLaVA-Video | 72B | 64 | 70.3% | 75.9% | **77.4%** | +1.5% |
| GPT-4o | - | 384 | 71.9% | 77.2% | - | - |

### 辅助文本消融实验

| RAG | DET | OCR | ASR | Short | Medium | Long | Overall |
|-----|-----|-----|-----|-------|--------|------|---------|
| - | - | - | - | 60.3 | 51.4 | 44.1 | 52.0 |
| ✓ | ✓ | - | - | 62.2 | 55.4 | 54.4 | 57.4 |
| ✓ | ✓ | ✓ | - | 64.0 | 56.2 | 55.0 | 58.4 |
| ✓ | - | - | ✓ | 63.0 | 57.3 | 56.4 | 58.9 |
| ✓ | ✓ | ✓ | ✓ | **66.4** | **60.2** | **59.8** | **62.1** |
| - | ✓ | ✓ | ✓ | 64.3 | 58.8 | 56.3 | 59.8 |

### 跨基准性能

| 基准 | 模型 | 原始 | +Video-RAG | 提升 | 对标 |
|------|------|------|-----------|------|------|
| MLVU | LLaVA-Video-7B | 70.8% | 72.4% | +1.6% | > Oryx-1.5 (32B) |
| MLVU | LLaVA-Video-72B | 73.1% | 73.8% | +0.7% | 新SOTA |
| LongVideoBench | LLaVA-Video-7B | 56.6% | 58.7% | +2.1% | - |
| LongVideoBench | LLaVA-Video-72B | 61.9% | 65.4% | +3.5% | > Gemini-1.5-Pro |

### 关键发现

- Video-RAG仅增加约2K token（≈14帧的token量），即可实现平均2.8%的性能提升
- 72B LLaVA-Video + Video-RAG超越GPT-4o（77.4% vs 77.2%）
- ASR对长视频提升最大（+14.7%在Long类别），OCR和DET在短视频上更有效
- RAG检索比直接输入所有辅助文本更好（62.1% vs 59.8%），说明检索筛选减少了噪声
- 额外GPU内存开销仅8GB，每个问题额外推理时间约5秒

## 亮点与洞察

- 极其务实的设计哲学：不微调、不依赖闭源模型、全部使用开源工具（EasyOCR、Whisper、APE、Contriever）
- 辅助文本的作用不仅是提供额外信息，更重要的是**促进跨模态对齐**——Grad-CAM可视化证实，辅助文本帮助LVLM将注意力聚焦在与查询相关的关键帧上
- 查询解耦设计避免了为每个问题都生成所有类型辅助文本的浪费
- 单轮检索的设计远比多轮Agent高效，同时保持了竞争性的性能

## 局限与展望

- 视觉工具的性能上限决定了辅助文本的质量上限——OCR、ASR、目标检测本身的错误会传播到最终结果
- 未探索自适应帧选择策略——目前使用均匀采样，可能遗漏关键帧
- 对于视觉信息主导的问题（如动作识别），辅助文本可能帮助有限
- 相似度阈值（t=0.3）对不同LVLM和任务类型是否通用未做充分验证
- 场景图处理中的相对位置关系表述仍然比较粗糙

## 相关工作与启发

- 与RAG在NLP中的应用思路一致，但创新性地将检索目标从文本文档扩展到从视频中自动提取的多模态辅助文本
- 与VideoAgent等Agent方法相比，Video-RAG用单轮检索替代多轮交互，实现了效率与性能的良好平衡
- 启发：对于多模态模型，跨模态对齐可能比增加同一模态的信息量更重要
- 辅助文本作为"语义桥梁"帮助LVLM更好地理解视觉内容的思路，可以推广到其他多模态任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将RAG创新性地应用于视频理解，设计简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 7个LVLM、3个基准、详细的消融实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，方法描述详细
- 价值: ⭐⭐⭐⭐⭐ — 即插即用的特性使其具有极强的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Beyond Caption-Based Queries for Video Moment Retrieval](../../CVPR2026/object_detection/beyond_caption-based_queries_for_video_moment_retrieval.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](../../ICCV2025/object_detection/large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[NeurIPS 2025\] MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling](mstar_box-free_multi-query_scene_text_retrieval_with_attention_recycling.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](../../ICCV2025/object_detection/voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)
- [\[CVPR 2025\] Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval](../../CVPR2025/object_detection/search_and_detect_training-free_long_tail_object_detection_via_web-image_retriev.md)

</div>

<!-- RELATED:END -->
