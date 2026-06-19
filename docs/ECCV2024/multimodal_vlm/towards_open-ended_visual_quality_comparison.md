---
title: >-
  [论文解读] Towards Open-ended Visual Quality Comparison
description: >-
  [ECCV 2024][多模态VLM][图像质量评估] 本文提出 Co-Instruct，首个面向开放式视觉质量比较的大型多模态模型，通过从两种"弱监督源"（LLM合并的单图描述 + GPT-4V伪标签）构建562K指令微调数据集，实现比 GPT-4V（其教师模型）更高的多图质量比较准确率，并提出首个多图比较基准 MICBench。
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "图像质量评估"
  - "多图比较"
  - "大型多模态模型"
  - "指令微调"
  - "视觉质量"
---

# Towards Open-ended Visual Quality Comparison

**会议**: ECCV 2024  
**arXiv**: [2402.16641](https://arxiv.org/abs/2402.16641)  
**代码**: [https://huggingface.co/q-future/co-instruct](https://huggingface.co/q-future/co-instruct)  
**领域**: 多模态VLM  
**关键词**: 图像质量评估, 多图比较, 大型多模态模型, 指令微调, 视觉质量

## 一句话总结

本文提出 Co-Instruct，首个面向开放式视觉质量比较的大型多模态模型，通过从两种"弱监督源"（LLM合并的单图描述 + GPT-4V伪标签）构建562K指令微调数据集，实现比 GPT-4V（其教师模型）更高的多图质量比较准确率，并提出首个多图比较基准 MICBench。

## 研究背景与动机

图像质量评估(IQA)是视觉计算的重要领域。近年来，大型多模态模型（LMM）已被探索用于将IQA从输出标量分数扩展到开放式场景，即可以回答开放性问题并提供推理解释。

**核心痛点**：现有的开放式IQA方法都基于单图评估，面临一个根本问题 —— **绝对评估的模糊性**。不同观察者对同一张图片的曝光度、清晰度等属性持有不同标准，导致绝对评价不一致。然而在比较设置下（如"哪张图更亮？"），所有人的回答趋于一致。

**现有差距**：
1. 现有数据集和方法仅支持整体质量的简单比较，未扩展到开放式场景
2. 开源LMM通常仅用单图数据微调，缺乏多图比较能力
3. 从人类标注收集比较数据集成本极高

**核心创意**：提出"协同指令微调"策略（Co-Instruct），利用两个不完美的弱监督源互补合作：(1) LLM将单图质量描述"合并"为比较文本；(2) GPT-4V在未标注数据上生成伪标签。两者互补构成562K训练数据集。

## 方法详解

### 整体框架

Co-Instruct 采用 mPLUG-Owl2 为基础模型，包含CLIP-ViT-L14视觉编码器、visual abstractor（将视觉token从1025压缩到65）和 LLaMA-2 LLM。通过图文交错格式处理多图输入，使用Co-Instruct-562K数据集微调。

### 关键设计

1. **Merge2Compare（LLM合并比较）**: 从Q-Pathway数据集的19K张图的单图质量描述出发，随机配对/分组成100K组（2-4图），通过E5-Mistral文本嵌入模型去除最相似描述对，再用LLM将多个独立描述"合并"为比较文本。核心思路是将已有的单图评价"转换"为比较评价，人工验证正确率达96%。

2. **Teach2Compare（GPT-4V教师比较）**: 收集9K张多样化未标注图片（包含野外图片、人工退化图片、AI生成图片），随机分组成30K组，喂给GPT-4V获取两种响应：(a) 整体质量比较描述；(b) 针对特定质量属性（清晰度、色彩等）的Q&A对（共230K对）。GPT-4V的准确率约94%，虽略低于Merge2Compare，但包含更多内容信息。

3. **图文交错格式与Visual Token压缩**: 为处理多图输入，采用visual abstractor将每张图的token数从1025压缩至65，并设计图文交错格式："The first image: <img₀> The second image: <img₁> ... <query>"，让模型明确区分每张图的信息。实验表明此格式显著优于简单拼接或可学习分隔符。

4. **MICBench基准**: 构建首个多图质量比较评估基准，包含2000个多选题，涵盖3或4张图的质量比较，包括Which问题(60%)、Yes-or-No问题(22%)和其他类型(18%)。图片来源于LLVisionQA和未标注数据库，10位人类专家标注并交叉验证。

### 损失函数 / 训练策略

- 基于 mPLUG-Owl2 的已发布checkpoint微调
- 学习率 2e-5，batch size 192，训练2个epoch
- 所有参数均更新，总训练时间约25小时（8×A100）
- 图像padding到正方形后resize到448×448

## 实验关键数据

### 主实验

| 数据集 | 指标 | Co-Instruct | GPT-4V | 提升 |
|--------|------|-------------|--------|------|
| Q-Bench^PAIR-A1 | Overall Accuracy | 80.18% | 78.07% | +2.7% |
| Q-Bench^PAIR-A1 | Compare子集 | 74.22% | 68.00% | +6.2% |

Co-Instruct在Q-Bench^PAIR-A1上：
- 比基础模型mPLUG-Owl2高**64%**
- 比无多图比较数据的变体高**51%**  
- 比最佳开源LMM（InternLM-XComposer2）高**23%**
- **超越人类非专家水平**（80.18% vs 80.12%），是唯一做到这点的LMM

### 消融实验

| 配置 | Overall Accuracy | 说明 |
|------|-----------------|------|
| 无多图比较数据 | 53.15% | 基线 |
| + Merge2Compare | 显著提升 | LLM合并有效 |
| + Teach2Compare | 进一步提升 | 两源互补 |
| 图片简单拼接 | 较低 | 图片信息混淆 |
| 图文交错格式 | 最高 | 明确区分各图 |

### 关键发现

1. 学生（Co-Instruct）超越教师（GPT-4V）：虽然MCQ训练数据来自GPT-4V，但Co-Instruct在MCQ评估中反超教师，说明协同教学策略有效
2. 两个弱监督源的互补性至关重要：Merge2Compare准确率更高但缺乏细粒度比较，Teach2Compare虽略不准确但包含更多内容信息和Q&A多样性
3. 图文交错格式大幅优于图片拼接或可学习特殊token分隔

## 亮点与洞察

- **弱监督协同**的思路非常巧妙：不直接收集昂贵的人类比较标注，而是从已有的单图描述和GPT-4V伪标签两个不完美来源互补学习
- **比较优于绝对评价**：这是心理物理学中的经典认知，本文将其系统性地引入LMM领域
- **学生超越教师**现象：Co-Instruct同时从LLM合并（高准确率）和GPT-4V（多样性）学习，整合后超越任一单独来源
- Visual token压缩策略使多图输入变得可行，解决了LLaVA等模型context窗口不够的实际问题

## 局限与展望

- MICBench仅评估MCQ形式，未涵盖开放式回答的评估
- 依赖GPT-4V作为教师，数据质量受限于GPT-4V的视觉感知能力
- Visual abstractor的压缩（1025→65 tokens）可能丢失部分细节信息
- 未探索更多图片数量（如5张以上）的场景
- 可尝试用更强的视觉编码器（如InternViT）替代CLIP-ViT

## 相关工作与启发

- Q-Bench/Q-Instruct/Q-Align系列（同一团队）为本文奠定了单图质量评估的基础
- 数据构建策略（Merge2Compare）可推广到其他需要比较数据但人工标注昂贵的场景
- 协同弱监督的思路适用于其他模态的比较任务（如音频质量、视频质量比较）

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地将LMM扩展到开放式多图质量比较，数据构建策略创新
- 实验充分度: ⭐⭐⭐⭐⭐ 多基准全面评估，消融实验详细，包含人类对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰、结构完整、图表丰富
- 价值: ⭐⭐⭐⭐ 在图像质量评估和LMM多图理解两个方向上都有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](../../ICLR2026/multimodal_vlm/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [\[CVPR 2026\] GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](../../CVPR2026/multimodal_vlm/guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](../../ACL2025/multimodal_vlm/craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)
- [\[CVPR 2025\] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](../../CVPR2025/multimodal_vlm/opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)
- [\[ICML 2026\] ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation](../../ICML2026/multimodal_vlm/eca_efficient_continual_alignment_for_open-ended_image-to-text_generation.md)

</div>

<!-- RELATED:END -->
