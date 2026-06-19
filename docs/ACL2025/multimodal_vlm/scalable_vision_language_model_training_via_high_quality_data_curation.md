---
title: >-
  [论文解读] Scalable Vision Language Model Training via High Quality Data Curation
description: >-
  [ACL 2025][多模态VLM][视觉语言模型] 提出 SAIL-VL 系列开源视觉语言模型（2B/8B），核心贡献在于：构建了3亿规模最高质量的 SAIL-Caption 数据集，首次揭示了VLM预训练中的数据量对数缩放定律（655B token实验），并通过课程式三阶段SFT将缩放曲线从对数提升至近线性，在18个基准上达到SOTA。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "视觉语言模型"
  - "数据质量"
  - "数据规模定律"
  - "课程学习"
  - "预训练扩展"
  - "指令微调"
---

# Scalable Vision Language Model Training via High Quality Data Curation

**会议**: ACL 2025  
**arXiv**: [2501.05952](https://arxiv.org/abs/2501.05952)  
**作者**: Hongyuan Dong*, Zijian Kang*, Weijie Yin*, Xiao Liang, Chao Feng, Jiao Ran (ByteDance Douyin)  
**代码**: [HuggingFace](https://huggingface.co/BytedanceDouyinContent)  
**领域**: 多模态VLM  
**关键词**: 视觉语言模型, 数据质量, 数据规模定律, 课程学习, 预训练扩展, 指令微调  

## 一句话总结

提出 SAIL-VL 系列开源视觉语言模型（2B/8B），核心贡献在于：构建了3亿规模最高质量的 SAIL-Caption 数据集，首次揭示了VLM预训练中的数据量对数缩放定律（655B token实验），并通过课程式三阶段SFT将缩放曲线从对数提升至近线性，在18个基准上达到SOTA。

## 研究背景与动机

### 问题背景
轻量级VLM性能不佳的根本原因在于两点：（1）预训练阶段视觉理解能力不足——要么训练预算有限（如LLaVA系列仅做轻量预训练），要么数据质量不高（如MiniCPM-V和Qwen2-VL虽投入数百亿token预训练但数据质量拖后腿）；（2）SFT阶段缺乏系统性的数据质量评估和阶段化训练方法论。

### 已有工作的不足
- LLaVA系列用少量低质量caption数据做轻量预训练，视觉理解能力受限
- MiniCPM-V-2.5、Qwen2-VL虽分配了数百亿token预训练预算，但数据质量不足，未能充分发挥预训练效果
- 现有工作几乎没有提供可靠结论来理解预训练预算和数据质量如何影响VLM性能
- SFT数据集的分布调配和训练阶段划分缺乏公认方法论
- Infinity-MM虽探索了多阶段SFT，但缺乏广泛认可的系统化框架

### 核心动机
用数据工程驱动模型性能——在预训练和SFT两个阶段都实现"高质量+可扩展"，并量化揭示数据量与模型性能之间的缩放关系。

## 方法详解

### 整体框架
SAIL-VL以InternViT-300M为视觉编码器，Qwen2.5系列为语言模型，通过五个训练阶段逐步构建能力：
1. **Pretrain-Alignment**（131B token）：仅训练MLP投影层，用SAIL-Caption和OCR数据对齐视觉-语言空间
2. **Pretrain-Advance**（524B token）：解锁视觉编码器联合训练，进一步提升视觉理解
3. **SFT-Knowledge**（21M样本）：学习基本指令遵循和世界知识
4. **SFT-Instruction**（12M样本）：增强视觉指令遵循能力
5. **SFT-Preference**（3.5M样本）：用少量高复杂度数据处理复杂推理任务

### 关键设计1：可扩展的高质量数据构建管线（SAIL-Caption）

数据构建分四步：
- **数据收集**：从LAION-COCO、TextCaps、SA1B等公开数据集广泛收集，确保分布多样性
- **参考数据标注**：选择分布均衡的子集，用GPT-4O标注高质量细节描述，辅以alt-text提供世界知识
- **标注器模型训练**：用参考数据微调InternVL2-8B得到SAIL-Captioner，兼具caption和recaption能力
- **大规模生产**：通过LMDeploy部署SAIL-Captioner，实现多任务、多节点、多进程异步标注

最终产出SAIL-Caption：3亿张图片的细节描述数据集。人工评估质量分87.2/88.2（10分制换算后），远超DataComp-LLaVA-Caption的70.0和BLIP3-KALE的73.2。在语言多样性指标上（unique n-gram、名词、动词、形容词数量）也全面领先。

### 关键设计2：数据量对数缩放定律

在预训练阶段，随着数据量从数十亿token指数级增长到655B token，模型性能呈现清晰的线性-对数缩放关系：
- Pretrain-Alignment阶段：caption和OCR任务性能稳步提升，横轴取对数后近似线性
- Pretrain-Advance阶段：解锁视觉编码器后性能显著跃升，维持类似的对数缩放曲线
- SFT后的性能同样保持了数据量的缩放关系——无论用自有SFT数据还是LLaVA-Next SFT数据，预训练数据量越大，SFT后的最终性能越好

这是首次在VLM预训练中明确提出并验证数据量缩放定律。

### 关键设计3：课程式SFT与数据复杂度扩展

**三层数据质量评估体系**：
- **Quick Quality Evaluation**：用2M子集快速训练评估数据集质量，假设不同数据集在不同数据量下保持一致的性能排序（经实验验证）
- **Composition Evaluation**：将SFT数据按格式分组（closed-form VQA、open-ended VQA、document VQA等），逐组减半测试以优化比例
- **Incremental Evaluation**：新数据集逐一加入并评估，保留提升或维持性能的数据集

**课程学习策略**：三阶段SFT数据的任务难度（1.90→2.15→2.20）、数据复杂度（2.44→2.62→2.74）和图文相关性（3.94→4.45→4.55）逐步升高。实验表明，课程式SFT的缩放曲线近似线性，显著优于将所有数据混合一起训练的对数缩放曲线。

## 实验关键数据

### 表1：SAIL-VL与同规模SOTA模型在18个基准上的对比

| 模型 | 规模 | 平均分 | General VQA | OCR VQA | Math&Knowledge | Hallucination |
|------|------|--------|-------------|---------|----------------|---------------|
| **SAIL-VL** | **2B** | **69.1** | 60.4 | **75.9** | **79.0** | **66.2** |
| InternVL2.5-MPO | 2B | 67.7 | **63.1** | 71.1 | 75.3 | 64.5 |
| Qwen2-VL | 2B | 64.4 | 58.3 | 72.5 | 59.0 | 62.9 |
| DeepSeekVL-2 | 2B | 67.0 | 59.4 | 74.4 | 71.3 | 63.6 |
| **SAIL-VL** | **8B** | **74.5** | 68.3 | **79.8** | **83.3** | 68.7 |
| InternVL2.5-MPO | 8B | 74.3 | **71.2** | 76.3 | 83.2 | **69.7** |
| Qwen2-VL | 8B | 73.0 | 68.5 | 79.6 | 71.0 | 67.5 |
| DeepSeekVL-2 | 8B | 72.7 | 66.8 | 79.0 | 79.0 | 65.3 |

SAIL-VL-2B在整体平均分上比第二名InternVL2.5-MPO-2B高1.4分（+2.06%），在OCR和数学知识子领域优势突出。8B模型同样领先，但优势缩小。

### 表2：预训练数据质量对模型性能的影响

| Caption数据 | OCR数据 | Overall | Caption | OCR |
|------------|---------|---------|---------|-----|
| **SAIL-Caption** | 高质量 | **54.36** | **51.80** | 55.38 |
| SA1B-QwenVL-Caption | 高质量 | 48.43 | 39.57 | 51.97 |
| DataComp-LLaVA-Caption | 高质量 | 49.08 | 42.70 | 51.63 |
| BLIP3-KALE | 高质量 | 53.06 | 46.00 | **55.89** |
| SAIL-Caption | 高质量+低质量 | 52.13 | 51.22 | 52.50 |
| SAIL-Caption | 高质量(重复) | 54.05 | 52.63 | 54.62 |

SAIL-Caption在视觉理解性能上显著优于其他开源数据集。值得注意的是，用高质量OCR数据重复训练，效果优于混入多样但低质量的数据——在冻结LLM的预训练设定下，重复高质量数据不会过拟合。

### 表3：SFT各阶段数据的2M子集训练评估

| 训练数据 | Overall | General | OCR | Math | Hallucination |
|---------|---------|---------|-----|------|---------------|
| SFT-Knowledge | 57.8 | 53.2 | 60.9 | 56.9 | 63.9 |
| **SFT-Instruction** | **61.9** | 55.8 | **67.1** | **65.4** | 61.7 |
| SFT-Preference | 61.3 | **57.1** | 65.8 | 59.5 | 61.3 |

SFT-Instruction数据集质量最高，但SFT-Preference因复杂度过高反而在单独训练时性能略降——这正好验证了课程学习的必要性。

## 关键发现

- **数据质量 > 数据多样性**：在预训练中，用高质量数据重复训练优于混入低质量数据；在SFT中，精心筛选的12M数据超过其他开源数据集数十倍的数据量
- **2B模型也能从数据扩展中获益**：655B token预训练实验表明，即使是紧凑模型也能从更大的预训练数据中持续获得性能提升，打破了"小模型不需要大预训练"的偏见
- **课程SFT将缩放曲线从对数提升至线性**：分阶段由易到难训练的策略，比一次性混合训练在同等计算量下产出更优性能
- **Quick Quality Evaluation有效**：用2M子集评估的数据集质量排序与全量训练一致，为高效数据筛选提供了实用方法

## 亮点与洞察

- **首次提出VLM预训练的数据量缩放定律**：这一发现为预训练资源分配提供了量化指导，具有重要的工程参考价值
- **完整的数据工程方法论**：从数据构建（SAIL-Caption管线）到数据评估（三层评估体系）到数据使用（课程SFT），形成了完整闭环
- **数据构建管线可复制**：先用GPT-4O标注参考数据，再蒸馏到InternVL2-8B做大规模标注——这一"强模型标注→弱模型蒸馏→大规模生产"的范式具有通用性
- **复杂度过高的数据反而有害**：SFT-Preference单独训练不如SFT-Instruction，揭示了训练数据复杂度与模型当前能力需要匹配的重要原则

## 局限性

- **缩放定律仅在特定数据量级验证**：尽管观察到性能趋于饱和，但更优训练设定下是否还有提升空间不确定
- **8B模型优势明显缩小**：作者承认8B模型训练数据量相对较小（预训练仅52B token vs 2B模型的655B token），未能充分发挥大模型潜力
- **未探索更大模型**：所有结论基于2B/8B模型，缩放定律是否在更大规模成立尚需验证
- **General VQA子领域未达最优**：在MMVet等需要长文本生成的基准上性能不稳定
- **SAIL-Caption未开源完整数据集**：虽然模型开源，但300M的完整数据集仅部分开放
- **可能存在幻觉和偏见**：作者承认模型在某些场景下仍会产生错误信息

## 相关工作与启发

- **与LLaVA系列的区别**：LLaVA用轻量预训练+精细SFT的路线，SAIL-VL则证明了大规模高质量预训练的不可替代性
- **与InternVL2.5-MPO的对比**：MPO需要额外的强化学习阶段，SAIL-VL仅靠数据质量和课程SFT就达到了可比甚至更优的性能
- **与Infinity-MM的关系**：SAIL-VL直接使用了Infinity-MM的Stage2和Stage4数据作为SFT的一部分，但在方法论层面有更系统的数据质量评估框架
- **对数据飞轮的启示**：SAIL-Captioner的构建过程本质上是一个数据飞轮——高质量参考数据→训练标注器→大规模生产→更好的模型→更高质量的数据

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统揭示VLM预训练数据量缩放定律，课程SFT策略有新意
- 实验充分度: ⭐⭐⭐⭐⭐ — 655B token预训练、18个基准评测、多组消融实验，工程量极大
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，方法论阐述系统
- 价值: ⭐⭐⭐⭐ — 对VLM训练的数据工程提供了实用指导，缩放定律和课程SFT策略有广泛参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Active Data Curation Effectively Distills Large-Scale Multimodal Models](../../CVPR2025/multimodal_vlm/active_data_curation_effectively_distills_large-scale_multimodal_models.md)
- [\[ACL 2025\] R-VLM: Region-Aware Vision Language Model for Precise GUI Grounding](r-vlm_region-aware_vision_language_model_for_precise_gui_grounding.md)
- [\[ACL 2025\] Error-driven Data-efficient Large Multimodal Model Tuning](error-driven_data-efficient_large_multimodal_model_tuning.md)
- [\[ACL 2025\] Response Wide Shut: Surprising Observations in Basic Vision Language Model Capabilities](response_wide_shut_surprising_observations_in_basic_vision_language_model_capabi.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](../../ICCV2025/multimodal_vlm/effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)

</div>

<!-- RELATED:END -->
