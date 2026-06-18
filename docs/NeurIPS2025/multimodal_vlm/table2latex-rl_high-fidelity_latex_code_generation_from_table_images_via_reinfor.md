---
title: >-
  [论文解读] Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models
description: >-
  [NeurIPS 2025][代码智能][表格识别] 提出VSGRPO——基于GRPO的双奖励强化学习策略，联合优化结构级奖励（TEDS-Structure）和视觉保真度奖励（CW-SSIM渲染图比较），使微调后的MLLM（仅3B参数）在表格图像到LaTeX代码生成任务上超越GPT-4o和72B+规模模型，尤其在复杂表格上提升显著。
tags:
  - "NeurIPS 2025"
  - "代码智能"
  - "表格识别"
  - "LaTeX生成"
  - "GRPO强化学习"
  - "双奖励机制"
  - "多模态大模型"
---

# Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.17589](https://arxiv.org/abs/2509.17589)  
**代码**: [GitHub](https://github.com/newLLing/Table2LaTeX-RL)  
**领域**: 代码智能  
**关键词**: 表格识别, LaTeX生成, GRPO强化学习, 双奖励机制, 多模态大模型

## 一句话总结

提出VSGRPO——基于GRPO的双奖励强化学习策略，联合优化结构级奖励（TEDS-Structure）和视觉保真度奖励（CW-SSIM渲染图比较），使微调后的MLLM（仅3B参数）在表格图像到LaTeX代码生成任务上超越GPT-4o和72B+规模模型，尤其在复杂表格上提升显著。

## 研究背景与动机

表格是科学文档的核心组件，从表格图像自动生成可编译的高质量LaTeX代码对文档数字化至关重要。然而，现有工作主要集中在生成HTML表示，缺乏LaTeX所需的结构表达力和排版精度。这一任务面临三大核心挑战：

**复杂表格的处理难度**：大尺寸、深层嵌套结构（多行/多列合并）和语义丰富的单元格内容（数学公式等）对视觉编码器和语言解码器都构成挑战。视觉编码器需提取细粒度的视觉和结构线索，语言解码器需生成长且语法敏感的LaTeX序列。任一环节出错都可能导致幻觉输出甚至编译失败。

**SFT的固有局限**：监督微调使用teacher forcing，训练信号是token级的next prediction。但LaTeX本身有语法歧义——不同的语法形式可能产生完全相同的视觉输出。这导致训练目标和评估目标之间存在不匹配，尤其对复杂表格影响严重。

**评估指标不完善**：TEDS对细粒度错误不敏感且在HTML和LaTeX之间存在适配问题；像素级指标聚焦局部视觉相似度但忽略全局结构正确性。需要一种混合评估策略。

这些挑战驱动了VSGRPO的设计——通过将渲染后的视觉反馈引入RL优化，直接优化最终视觉输出质量，绕过LaTeX语法歧义问题。

## 方法详解

### 整体框架

三阶段流程：（1）大规模数据收集——从arXiv爬取120万表格-LaTeX配对；（2）MLLM监督微调（SFT）——获得初步的表格到LaTeX生成能力；（3）VSGRPO强化微调——利用双奖励机制进一步提升复杂表格上的性能。

### 关键设计

1. **大规模Table2LaTeX数据集构建**：从2017年10月至2023年4月的arXiv论文中爬取LaTeX源码，用正则表达式提取tabular环境，清理引用、颜色设置等控制命令后获得**1,209,986**个表格-LaTeX对。按复杂度分为三级：

    - Simple：基础结构（94%）
    - Medium：含2+个\multirow或\multicolumn且100-160个单元格（3%）
    - Complex：超过160个单元格（3%）
   
   这种分级使得评估更加细粒度，能准确反映模型在不同复杂度上的真实能力。

2. **VSGRPO双奖励强化学习策略**：核心创新在于将LaTeX渲染（不可微操作）引入RL优化环路。对每个表格图像输入，模型采样一组LaTeX输出 $\{o_1, ..., o_N\}$，分别计算两种奖励：

    - **视觉奖励（Visual Reward）**：将生成的LaTeX编译渲染为图像，与ground truth渲染图计算CW-SSIM。超过阈值（0.6）奖励为1，否则为0。CW-SSIM针对黑白表格图像做了专门适配：转灰度→统一尺寸→行列对齐→2×2 Haar小波分解为4个子带→各子带独立计算SSIM→取平均。
   
    - **结构奖励（Structure Reward）**：将生成和ground truth的LaTeX转换为HTML，计算TEDS-Structure。超过阈值（0.9）奖励为1，否则为0。TEDS-Structure通过最小树编辑距离衡量表格结构对齐度。

   优化目标基于GRPO框架：
    $J_{\text{RFT}}(\theta) = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i, \text{clip}(\cdot, 1-\varepsilon, 1+\varepsilon)A_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$
   
   其中优势函数 $A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$，$\varepsilon=0.2$，$\beta=0.02$。

3. **训练策略的精心设计**：

    - VSGRPO仅在**5,936个复杂表格**上训练（ground truth LaTeX<3000字符），平衡复杂度和计算可行性
    - SFT是必要前置步骤——不经SFT直接做RL效果极差（消融验证）
    - 编译失败的输出自动获得0奖励

### 损失函数 / 训练策略

- **SFT阶段**：标准负对数似然损失 $\mathcal{L}_{\text{SFT}} = -\sum \log p_\theta(\mathbf{y}^{(i)}|\mathbf{x}^{(i)})$，全参数微调，训练一个epoch
- **RFT阶段**：PPO风格的目标函数+KL正则化。InternVL2-1B使用VLM-R1框架（num_gens=8），Qwen2.5-VL-3B使用ms-swift框架（num_gens=4）
- **混合评估策略**：采用TEDS-Structure+CW-SSIM的组合评估，前者衡量全局结构正确性，后者衡量局部视觉保真度

## 实验关键数据

### 主实验——CW-SSIM与编译率

| 模型 | Simple CW-SSIM | Medium CW-SSIM | Complex CW-SSIM | Complex编译率 |
|------|:---:|:---:|:---:|:---:|
| Mathpix（商业） | 0.6884 | 0.5647 | 0.4862 | 0.9889 |
| GPT-4o | 0.6792 | 0.5612 | 0.4747 | 0.9917 |
| Qwen2.5-VL-72B | 0.7077 | 0.6009 | 0.5112 | 0.9335 |
| Nougat（专家） | 0.7401 | 0.5505 | 0.4699 | 0.3352 |
| **Qwen2.5-VL-3B-VSGRPO** | **0.8186** | **0.7236** | **0.6145** | **0.9917** |
| 对比GPT-4o提升 | +0.1394 | +0.1624 | +0.1398 | 持平 |

### 主实验——TEDS与TEDS-Structure

| 模型 | Complex TEDS | Complex TEDS-Struct | 说明 |
|------|:---:|:---:|------|
| Mathpix | 0.7176 | 0.8100 | 商业工具 |
| GPT-4o | 0.5865 | 0.7745 | 复杂表格下降严重 |
| Qwen2.5-VL-72B | 0.7448 | 0.8334 | 72B开源最佳 |
| Nougat | 0.0424 | 0.0527 | 复杂表格几乎崩溃 |
| **Qwen2.5-VL-3B-VSGRPO** | **0.8673** | **0.9218** | **首个突破0.9的模型** |

### 消融实验

| 配置 | Complex CW-SSIM | TEDS | TEDS-Struct | 说明 |
|------|:---:|:---:|:---:|------|
| SFT only | 0.5806 | 0.8481 | 0.9047 | 基线 |
| +仅TEDS-Struct奖励 | 0.5925 | 0.8608 | 0.9155 | 结构奖励有效 |
| +仅CW-SSIM奖励 | 0.6064 | 0.8607 | 0.9133 | 视觉奖励有效 |
| +双奖励（VSGRPO） | **0.6145** | **0.8673** | **0.9218** | 互补最优 |
| VSGRPO w/o SFT | 0.4695 | 0.6884 | 0.8167 | SFT前置必要性 |

### 关键发现

- **3B模型全面超越72B+模型和商业工具**：VSGRPO使小模型在所有复杂度级别上超越Mathpix、GPT-4o和72B开源模型，证明针对性RL策略比单纯scaling更有效
- **复杂度越高优势越大**：VSGRPO在Complex表格上的提升最为显著（CW-SSIM +0.1398 vs GPT-4o），体现了"在难题上练"的训练策略效果
- **双奖励互补**：结构奖励和视觉奖励各有侧重，联合使用效果最佳
- **SFT是RL的必要基础**：跳过SFT直接做RL导致全面性能崩塌

## 亮点与洞察

1. **Visual-in-the-loop RL**是核心创新亮点：将渲染（不可微操作）通过RL的奖励信号引入训练循环，绕过了可微性约束
2. **"在难题上练"的训练策略**：仅用5,936个复杂表格做RL，效果好于使用混合数据或简单数据，说明RL阶段应聚焦于模型的薄弱环节
3. **小模型+精准RL > 大模型+通用能力**：3B参数的专精模型全面胜过72B通用模型，对"是否一定需要大模型"这一问题给出了重要反例
4. **复杂度分级评估**是对该领域评估方法的重要补充

## 局限与展望

- VSGRPO训练过程中每个LaTeX输出都需要渲染为PDF再转PNG用于CW-SSIM计算，是严重的训练瓶颈
- 受限于GPU资源，RL仅在5,936个复杂表格上训练，更多数据可能带来更大提升
- 目前仅支持tabular环境，未处理其他LaTeX表格格式（如longtable、tabularx）
- 奖励设计为二元（0/1），连续奖励可能提供更精细的优化信号

## 相关工作与启发

- **GRPO在数学推理中的成功**：本文将GRPO从文本生成扩展到"文本+渲染"的多模态奖励场景
- **对其他代码生成任务的启发**：任何"生成代码→编译/执行→评估输出"的任务都可以借鉴visual-in-the-loop RL思路，如HTML生成、SVG绘图、Markdown排版等
- **与Nougat的对比**：Nougat端到端生成LaTeX但在复杂表格上完全崩溃（TEDS仅0.04），说明单纯的端到端训练不足以处理结构复杂的输出

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 双奖励RL+渲染反馈的设计高度创新，对RL在视觉-代码生成任务中的应用有方法论贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ 多个基线（商业/通用/专家）、复杂度分级评估、人工评价、详尽消融
- **写作质量**: ⭐⭐⭐⭐ 问题分析透彻，方法动机说明清楚
- **价值**: ⭐⭐⭐⭐⭐ 实用价值极高（直接服务于科学文档数字化），方法论价值也大（visual-in-the-loop RL的新范式）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](../../ACL2025/code_intelligence/texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](../../ACL2025/code_intelligence/personality_guided_code_gen.md)
- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](../../ACL2026/code_intelligence/from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](../../ACL2025/code_intelligence/codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](../../ACL2025/code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)

</div>

<!-- RELATED:END -->
