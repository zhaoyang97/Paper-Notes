---
title: >-
  [论文解读] DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation
description: >-
  [ICCV 2025][视频生成][人脸视频数据集] 推出 DH-FaceVid-1K，一个包含 1,200+ 小时、270,043 个视频片段、20,000+ 个人身份的大规模高质量人脸视频数据集，重点解决现有数据集中亚洲人脸严重不足的问题，并通过系统实验验证了数据规模与模型参数的缩放定律。 人脸视频生成是当前视频生成领域…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "人脸视频数据集"
  - "文本到视频"
  - "图像到视频"
  - "扩散模型"
---

# DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation

**会议**: ICCV 2025  
**arXiv**: [2410.07151](https://arxiv.org/abs/2410.07151)  
**代码**: [项目页面](https://luna-ai-lab.github.io/DH-FaceVid-1K/)  
**领域**: 图像生成 / 人脸视频生成  
**关键词**: 人脸视频数据集, 视频生成, 文本到视频, 图像到视频, 扩散模型

## 一句话总结

推出 DH-FaceVid-1K，一个包含 1,200+ 小时、270,043 个视频片段、20,000+ 个人身份的大规模高质量人脸视频数据集，重点解决现有数据集中亚洲人脸严重不足的问题，并通过系统实验验证了数据规模与模型参数的缩放定律。

## 研究背景与动机

人脸视频生成是当前视频生成领域最热门的任务之一，支撑着说话人脸视频、文本驱动视频生成等多种应用。然而，许多先进方法依赖于不公开的私有数据，而公开数据集存在三个核心限制：

**总时长不足**：CelebV-HQ 68h、CelebV-Text 279h，远不能满足预训练需求

**质量与数量的权衡**：VoxCeleb2 有 2,400h 但分辨率仅 224×224；TalkingHead-1KH 分辨率同样受限

**多样性不足**：现有数据集中亚洲人脸严重缺乏，限制了模型在不同种族上的泛化能力

此外，论文识别了现有公开数据集的多个常见质量问题：低清晰度/分辨率、单帧多人脸、手部/物体遮挡、字幕/噪声叠加等，这些问题严重影响训练效果。

## 方法详解

### 整体框架

DH-FaceVid-1K 的构建分为四个关键阶段：
1. **原始视频采集**：从众包平台收集采访节目和 vlog 类视频（单人、专业环境、高质量设备），原始素材超过 2,000 小时
2. **人脸检测与裁剪**：裁剪至包含完整面部和上肩区域，确保人脸区域至少 256×256
3. **噪声过滤**：字幕检测（OCR）、黑边检测、多人脸排除、手部/遮挡物手动筛选、模糊增强（CodeFormer）
4. **标注生成**：使用 DWPose 提取面部关键点，PLLaVA 生成初始文本描述，100+ 人工标注员历时 6 个月交叉验证

### 关键设计

1. **多种族覆盖与亚洲人脸补充**: 自采数据约 1,000 小时（95% 亚洲面孔），另从 CelebV-HQ、CelebV-Text、TalkingHead-1KH 清洗出 200 小时补充数据，最终数据集中亚洲面孔占 80%、白人 11%、非裔 4%、其他 5%。通过专注于亚洲人脸的补充，解决了现有数据集中的人口学偏差问题。

2. **严格的数据质量控制流程**: (1) 字幕检测：随机采样 5 帧进行 OCR，超过 10 字符判定有字幕；(2) 黑边检测：检测超过 20 像素的连续黑边区域；(3) 人脸过滤：OpenCV 检测多人脸并丢弃，FaceXFormer 过滤 22 岁以下个体；(4) 两阶段人工验证：第一阶段按 ISO 2859 标准交叉核验静态和动态属性，第二阶段对标记问题样本做进一步审查。

3. **音频过滤与唇音同步**: 针对非英语语音 SyncNet 评分偏差问题，重新训练 SyncNet 模型对每个视频生成同步评分，过滤唇音同步差的样本，确保数据集可用于音频驱动的说话人脸生成任务。

### 损失函数 / 训练策略

作为数据集论文，本身无训练策略。但对下游模型训练给出了重要经验：
- 2B 参数 DiT 模型最佳数据规模约 600 小时
- 5B/6B 参数模型需要完整 1,000+ 小时数据才能发挥优势
- 小模型在大数据集上出现过拟合，大模型在小数据集上训练不稳定

## 实验关键数据

### 主实验（Text-to-Video）

| 方法 | 数据集 | FVD (↓) | FID (↓) | CLIP (↑) |
|---|---|---|---|---|
| CogVideoX | HDTF | 127.88 | 17.83 | 0.9247 |
| CogVideoX | CelebV-HQ | 137.62 | 17.31 | 0.9305 |
| CogVideoX | CelebV-Text | 129.59 | 15.82 | 0.9388 |
| CogVideoX | **DH-FaceVid-1K** | **98.01** | **11.73** | **0.9401** |
| EasyAnimate | CelebV-Text | 121.22 | 16.53 | 0.9274 |
| EasyAnimate | **DH-FaceVid-1K** | **113.27** | **13.91** | 0.9240 |

### 消融实验（数据规模缩放定律，CogVideoX T2V）

| 数据规模 | 2B FVD | 2B FID | 5B FVD | 5B FID |
|---|---|---|---|---|
| 100h | 215.06 | 17.01 | 237.52 | 18.17 |
| 200h | 185.06 | 16.23 | 203.52 | 16.37 |
| 400h | 177.18 | 14.16 | 180.99 | 14.25 |
| 600h | 145.14 | 12.50 | 143.25 | 12.83 |
| 800h | 148.82 | 13.15 | 121.63 | 12.05 |
| 1000h | 150.27 | 13.31 | **98.01** | **11.73** |

2B 模型在 600h 时达到性能拐点（再增加数据反而略微下降），5B 模型则持续受益于更多数据。

### 关键发现

1. **所有模型在 DH-FaceVid-1K 上训练后均显著超越公共数据集**：CogVideoX 在 T2V 上 FVD 从 129.59 降至 98.01（降 24.3%），FID 从 15.82 降至 11.73（降 25.9%）
2. **缩放定律**：2B 模型约需 600h 数据达到最优性价比，5B+ 模型需要完整数据集才能充分发挥
3. **DiT vs UNet**：DiT 架构（Latte、CogVideoX）整体优于 UNet 架构（AnimateDiff），但训练资源需求更高
4. **CelebV-Text 训练的模型生成亚洲面孔时出现明显伪影**（随机手部噪声、多人脸），DH-FaceVid-1K 训练的模型无此问题
5. **I2V 任务同样受益**：CogVideoX I2V 在 DH-FaceVid-1K 上 FVD 92.31 vs CelebV-Text 123.25

## 亮点与洞察

- **数据工程价值突出**：100+ 标注员 6 个月的人工质量控制流程，包含两阶段交叉验证，确保了业界领先的数据质量
- **缩放定律实验**提供了实用指导：不同参数规模的模型对应不同的最优数据量，避免了盲目堆数据
- **DiT vs UNet 的经验性对比**为研究者选择 backbone 提供了有价值的参考
- **解决了亚洲人脸数据缺乏的实际痛点**，对实际应用部署具有重要意义

## 局限与展望

- 亚洲面孔占 80% 导致其他种族仍然不足，可能引入新的偏差
- 数据集获取流程严格（需提交申请表、签署协议），可能限制学术社区的广泛使用
- 未探索更长视频（>15s）生成的效果
- 缺少与最新 Hallo3 等专门的 talking head 方法的直接对比
- 音频驱动生成任务的实验较少

## 相关工作与启发

- 验证了领域特定高质量数据集对预训练模型微调的重要性，这一结论可推广到其他垂直领域
- 数据处理流水线（字幕检测 → 黑边去除 → 多人脸过滤 → 手动审核 → 音频同步）可作为视频数据集构建的标准参考
- 缩放定律分析框架（模型参数 × 数据规模 × 性能指标）为未来数据集论文提供了评估方法论

## 评分

- **新颖性**: ⭐⭐⭐ 数据集构建流程规范但方法论新意有限，核心贡献在数据本身
- **实验充分度**: ⭐⭐⭐⭐⭐ 5 个 T2V 模型 + 5 个 I2V 模型 + 6 个数据规模 + 2 种参数规模的系统实验
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数据统计完整，缩放定律分析有深度
- **价值**: ⭐⭐⭐⭐ 填补了高质量大规模人脸视频数据集的空白，缩放定律分析具有实用指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)
- [\[CVPR 2025\] HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation](../../CVPR2025/video_generation/hoigen-1m_a_large-scale_dataset_for_human-object_interaction_video_generation.md)
- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[CVPR 2026\] Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset](../../CVPR2026/video_generation/scaling_instruction-based_video_editing_with_a_high-quality_synthetic_dataset.md)
- [\[CVPR 2026\] CineBrain: A Large-Scale Multi-Modal Audiovisual Brain Dataset for Brain-Conditioned Video Generation](../../CVPR2026/video_generation/cinebrain_a_large-scale_multi-modal_audiovisual_brain_dataset_for_brain-conditio.md)

</div>

<!-- RELATED:END -->
