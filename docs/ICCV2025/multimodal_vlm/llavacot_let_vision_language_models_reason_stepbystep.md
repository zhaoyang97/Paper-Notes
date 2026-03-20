# LLaVA-CoT: Let Vision Language Models Reason Step-by-Step

**会议**: ICCV 2025  
**arXiv**: [2411.10440](https://arxiv.org/abs/2411.10440)  
**代码**: [https://github.com/PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)  
**领域**: 多模态VLM / 推理  
**关键词**: VLM推理, Chain-of-Thought, 多阶段推理, test-time scaling, 结构化推理  

## 一句话总结
通过构建包含结构化推理标注的LLaVA-CoT-100k数据集，训练VLM自主执行"总结→视觉解读→逻辑推理→结论"四阶段推理，配合测试时SWIRES搜索策略，11B模型超越GPT-4o-mini和Gemini-1.5-pro等大模型。

## 背景与动机
大语言模型在推理能力上已取得显著进步（如CoT prompting），但当前的视觉语言模型（VLM）在面对复杂视觉问答任务时，仍然难以进行系统性的结构化推理。传统的chain-of-thought提示方法对VLM效果有限，因为视觉信息的解读需要额外的结构化步骤。现有VLM通常直接从问题跳到答案，缺乏中间的系统性思考过程，这在涉及空间推理、科学计算、图表理解等复杂任务时表现尤为明显。

## 核心问题
如何让VLM自主地、系统性地进行多阶段推理？关键挑战在于：(1) 缺乏结构化推理的训练数据；(2) 需要一种方法让模型学会在不同推理阶段之间自然过渡；(3) 如何在推理时进一步提升推理质量（test-time scaling）。

## 方法详解

### 整体框架
LLaVA-CoT基于Llama-3.2-11B-Vision-Instruct进行微调。输入为图像和问题，模型自主生成四个阶段的推理过程，每个阶段用特殊标签包裹（如`<SUMMARY>...</SUMMARY>`），最终输出结论。整个推理过程是端到端生成的，不需要额外的prompt工程。

### 关键设计
1. **四阶段结构化推理**：模型被训练成自动产生四个推理阶段：
   - **Summary阶段**：理解问题，明确需要做什么（"What's the problem? What should I do?"）
   - **Caption阶段**：从图像中提取与问题相关的视觉信息（"What can I know from the image?"）
   - **Reasoning阶段**：基于提取的信息进行逐步逻辑推理（"How to solve the problem step-by-step?"）
   - **Conclusion阶段**：综合前面的分析给出最终答案
   
   这种设计的核心洞察是：VLM需要先"看懂"图像再"想清楚"问题，而不是混在一起处理。

2. **LLaVA-CoT-100k数据集构建**：从多个开源VQA数据集采集图像和问题（ShareGPT4V 31.3k、ChartQA 17.2k、A-OKVQA 16.1k、AI2D 11.4k、GeoQA+ 11.4k、ScienceQA 5.6k等，共约98.6k样本），使用GPT-4o为每个样本生成四阶段的结构化推理标注。数据覆盖通用VQA和科学推理两大类。

3. **SWIRES（Stage-Wise Retracing Search）**：测试时的阶段式回溯搜索策略，实现test-time scaling。在每个推理阶段结束后，模型可以生成多个候选答案，然后选择最优路径继续，类似于beam search但在推理阶段的粒度上进行。这使得模型在推理时可以通过增加计算量来提升准确率。

### 损失函数 / 训练策略
使用标准的自回归语言建模损失进行微调。训练配置：8卡并行，学习率1e-5，3个epoch，batch size 4，使用FSDP分布式训练。训练数据中的特殊标签（`<SUMMARY>`等）被视为普通token参与训练。

## 实验关键数据
| 数据集 | 指标 | LLaVA-CoT (11B) | Llama-3.2-90B-Vision | GPT-4o-mini | Gemini-1.5-pro |
|--------|------|---------|----------|------|------|
| MMStar | Acc | 显著提升 | 低于LLaVA-CoT | 低于LLaVA-CoT | 低于LLaVA-CoT |
| MMBench | Acc | 显著提升 | 低于LLaVA-CoT | 低于LLaVA-CoT | 低于LLaVA-CoT |
| MathVista | Acc | 显著提升 | 低于LLaVA-CoT | 低于LLaVA-CoT | 低于LLaVA-CoT |

- 相比基座模型Llama-3.2-11B-Vision-Instruct，在6个多模态推理benchmark上平均提升**9.4%**
- 11B模型超越了8倍大的Llama-3.2-90B-Vision-Instruct以及闭源模型GPT-4o-mini和Gemini-1.5-pro
- SWIRES进一步带来额外的性能提升，且计算开销可控

### 消融实验要点
- 四阶段结构缺一不可：去掉Caption阶段或Reasoning阶段均导致显著性能下降
- 数据集规模重要，但100k已足够达到强效果
- SWIRES相比普通贪心解码和标准beam search更高效，在阶段粒度搜索比token粒度搜索更有效

## 亮点
- **小模型超大模型**：仅11B参数，用100k训练数据就超越90B和闭源大模型，证明结构化推理的重要性远超规模
- **四阶段设计很直觉**：Summary→Caption→Reasoning→Conclusion的流程模拟了人类解题的思维过程，特别是先看懂图再推理的分离设计
- **SWIRES是一种通用的test-time scaling方法**：在推理阶段粒度的搜索比token级别更高效，且可以灵活调控推理时间和精度的trade-off
- **数据集构建方法可复用**：用GPT-4o生成结构化推理标注的pipeline可以迁移到其他推理任务

## 局限性 / 可改进方向
- 依赖GPT-4o生成训练数据，数据质量受限于GPT-4o的能力上限
- 四阶段的划分是固定的，某些简单问题不需要全部阶段（计算浪费）
- 目前仅在Llama-3.2-Vision上验证，其他VLM架构（如Qwen-VL、InternVL）的兼容性未探索
- SWIRES的搜索空间随阶段数增加指数增长，扩展到更多阶段可能需要更好的剪枝策略

## 与相关工作的对比
- **vs. CoT prompting**：传统CoT是通过prompt引导模型展示推理过程，LLaVA-CoT是通过训练让模型内化结构化推理能力，不需要复杂的prompt设计
- **vs. LLaVA系列**：保持了LLaVA的简洁架构，核心改进在训练数据的结构化标注，而非模型架构改变
- **vs. o1-like模型**：与OpenAI o1的思路类似（通过训练时间和推理时间计算提升推理能力），但在多模态领域的开源实现

## 启发与关联
- 结构化推理标注的方法可以启发其他VLM任务（如视觉定位、图像描述）的数据构建
- SWIRES的阶段级搜索思路可以与ideas/multimodal_vlm/中的推理验证相关idea结合
- test-time scaling在VLM中的有效性值得进一步探索

## 评分
- 新颖性: ⭐⭐⭐⭐ 四阶段推理的思路直觉且简单，但有效；SWIRES有巧思
- 实验充分度: ⭐⭐⭐⭐ 6个benchmark全面验证，消融实验充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，demo展示有说服力
- 价值: ⭐⭐⭐⭐⭐ 开源的多模态推理方案，影响力大（2.1k stars），实用性强
