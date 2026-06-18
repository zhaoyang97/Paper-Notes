---
title: >-
  [论文解读] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant
description: >-
  [CVPR 2025][信息检索/RAG][多模态检索] 将生成式大语言模型（LMM）改造为通用多模态检索器+重排器，通过两阶段训练（语言预训练+多模态指令微调）和联合逐点/列表重排训练，仅插入轻量LoRA模块即可在16种检索任务上显著超越双编码器方法，且在10个未见数据集上展现强泛化能力。 领域现状：多模态信息检索正变得越…
tags:
  - "CVPR 2025"
  - "信息检索/RAG"
  - "多模态检索"
  - "大语言模型"
  - "通用检索"
  - "重排序"
  - "LoRA微调"
---

# LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant

**会议**: CVPR 2025  
**arXiv**: [2412.01720](https://arxiv.org/abs/2412.01720)  
**代码**: [https://github.com/Code-kunkun/LamRA](https://github.com/Code-kunkun/LamRA)  
**领域**: 信息检索  
**关键词**: 多模态检索、大语言模型、通用检索、重排序、LoRA微调

## 一句话总结
将生成式大语言模型（LMM）改造为通用多模态检索器+重排器，通过两阶段训练（语言预训练+多模态指令微调）和联合逐点/列表重排训练，仅插入轻量LoRA模块即可在16种检索任务上显著超越双编码器方法，且在10个未见数据集上展现强泛化能力。

## 研究背景与动机

**领域现状**：多模态信息检索正变得越来越复杂——从简单的文本→图像检索扩展到组合图像检索（文本+图像→图像）、长文本→图像检索、多模态文档检索等。传统方法（CLIP、ALIGN）基于双编码器+对比学习，在跨模态检索上表现出色。

**现有痛点**：(1) 双编码器方法对复杂查询理解不足——CLIP的文本编码器对长文本、复杂组合指令的理解有限；(2) 现有方法需要针对每个任务单独微调，无法统一处理多种检索格式；(3) 少有研究探索将LMM用于通用检索——LMM是为生成任务训练的，直接用于检索效果很差（Qwen2-VL-7B 在M-BEIR上仅23.0分）。

**核心矛盾**：LMM具有强大的多模态理解和自然语言理解能力，但其生成式训练目标（next-token prediction）与检索任务（embedding相似度排序）之间存在根本差异。

**本文解决什么**：如何用最小改动将生成式LMM转化为通用的多模态检索器和重排器？

**切入角度**：用 LoRA 模块适配 LMM 的输出，两阶段渐进式训练先建立检索能力基础再扩展到多模态任务。

**核心 idea**：通过"适配+渐进训练"策略，让LMM用最后一层隐状态作为统一的多模态embedding，实现10+种检索任务的统一处理。

## 方法详解

### 整体框架
LamRA = LamRA-Ret（检索）+ LamRA-Rank（重排序）。检索部分：在LMM中插入LoRA，用 "Summarize in one word: `<emb>`" 提示获取embedding，两阶段训练。重排部分：另一个LoRA模块，同时支持逐点（YES/NO判断）和列表（直接输出位置编号）重排。推理时先用LamRA-Ret检索top-K，再用LamRA-Rank重排。

### 关键设计

1. **显式单词限制（EOL）特征提取**:
    - 功能：将LMM的生成式输出转化为可用于检索的embedding
    - 核心思路：设计三种提示模板——图像输入用 "`<image>` Summarize above image in one word: `<emb>`"，文本输入用 "`<text>` Summarize above sentence in one word: `<emb>`"，混合输入用对应组合模板。取 `<emb>` token前的最后隐状态作为embedding
    - 设计动机：生成式LMM的输出分布与检索所需的embedding空间差异大，通过明确的summarization提示引导模型将多模态信息压缩到一个隐状态中，相当于给LMM一个"任务切换"信号

2. **两阶段渐进训练（LamRA-Ret）**:
    - 功能：从零开始建立LMM的检索能力
    - 核心思路：**Stage-I** 在NLI数据集上做纯文本对比学习预训练——让模型学会输出有意义的embedding（而非next-token概率分布）；**Stage-II** 在M-BEIR数据集的8种检索任务上指令微调——用任务特定指令（如"retrieve a similar image"）引导模型理解不同检索意图。两阶段都用 InfoNCE 对比损失：$\mathcal{L}_{ret} = -\frac{1}{B}\sum_{n=1}^B \log \frac{\exp[\kappa(\text{LMM}(q_n), \text{LMM}(c_n))/\tau]}{\sum_{m=1}^B \exp[\kappa(\text{LMM}(q_n), \text{LMM}(c_m))/\tau]}$
    - 设计动机：消融显示去掉预训练阶段性能下降3个点（53.6→56.6），去掉指令微调则更糟（36.2）。两阶段缺一不可——预训练解决"如何输出embedding"的基础问题，指令微调解决"如何理解不同检索任务"的扩展问题

3. **联合逐点+列表重排（LamRA-Rank）**:
    - 功能：在初始检索结果上进一步提升排序质量
    - 核心思路：训练另一个LoRA模块，从LamRA-Ret的top-100候选中做hard negative mining。**逐点重排**：对每个候选独立判断YES/NO，损失为 $\mathcal{L}_{point} = \mathcal{L}_{ce}(\text{YES}, \text{Reranker}(q, c_{pos})) + \mathcal{L}_{ce}(\text{NO}, \text{Reranker}(q, c_{neg}))$。**列表重排**：将正样本随机插入2-5个负样本中，让模型直接输出正样本的位置编号。最终融合分数 $S = \alpha \cdot S_{ret} + (1-\alpha) \cdot S_{rank}$，$\alpha=0.5$
    - 设计动机：逐点重排精度高但计算开销大（K次推理），列表重排高效但受限于LLM上下文长度。联合训练让模型同时具备两种能力，实际部署时可按需选择。重排带来平均7.1个点的提升

### 损失函数 / 训练策略
- 检索阶段：InfoNCE对比损失，温度参数 $\tau$
- 重排阶段：交叉熵损失，$\mathcal{L}_{rank} = \mathcal{L}_{point} + \mathcal{L}_{list}$
- 硬件配置：预训练8×A100（batch=576，lr=4e-5，2 epochs），指令微调16×A100（batch=960，lr=1e-4，1 epoch），重排16×A100（batch=64，lr=4e-5，1 epoch）
- 视觉编码器参数全程冻结，仅用 LoRA 微调 LLM

## 实验关键数据

### 主实验（M-BEIR 16种任务平均 Recall@5）

| 方法 | 类型 | 平均R@5 | 提升 |
|------|------|---------|------|
| Qwen2-VL-7B（零样本） | LMM | 23.0 | - |
| UniIR-CLIP-SF | 双编码器 | 50.6 | +27.6 |
| LamRA-Ret | LMM+LoRA | 56.6 | +33.6 |
| **LamRA** | LMM+LoRA+重排 | **63.7** | **+40.7** |

### 未见数据集泛化（零样本 R@1）

| 数据集 | 任务 | LamRA | EVA-CLIP-18B | Long-CLIP-L | E5-V |
|--------|------|-------|-------------|-------------|------|
| ShareGPT4V | T→I | **97.9** | 92.1 | 95.6 | 86.7 |
| Urban-1K | T→I | **98.8** | 81.7 | 86.1 | 84.0 |
| CIRCO | (I,T)→I | **42.8** | 6.0 | 5.7 | 24.8 |
| Visual Dialog | Dialog→I | **70.9** | 24.7 | 37.9 | 54.6 |
| MSR-VTT（视频） | T→V | 44.7 | - | - | - |

### 消融实验

| 配置 | M-BEIR 平均 | 说明 |
|------|------------|------|
| 完整两阶段 | 56.6 | 基线 |
| 去掉预训练 | 53.6 (-3.0) | 缺失embedding基础能力 |
| 去掉指令微调 | 36.2 (-20.4) | 无法理解多种检索任务 |
| 两者都去掉 | 23.0 (-33.6) | 裸LMM不具备检索能力 |
| Qwen2-VL-2B 版本 | 51.6/58.3 | 模型越大效果越好 |
| Qwen2-VL-7B 版本 | 56.6/63.7 | - |

### 关键发现
- **LMM作为检索器全面超越双编码器**：在M-BEIR上LamRA-Ret(56.6)已超越UniIR-CLIP(50.6)，加重排后达63.7
- **在复杂检索任务上优势极大**：InfoSeek 数据集上的文本图像→文本检索，LamRA-Ret 超出 UniIR-CLIP 24.2 个点
- **零样本泛化惊人**：在 Urban-1K（长文本→图像）上 R@1=98.8，超越 EVA-CLIP-18B 17个点
- **重排一致性提升**：对全部16个M-BEIR检索任务均有提升，平均+7.1个点
- **视频检索零样本可行**：MSR-VTT上R@1=44.7（超越InternVideo 4.7个点），模型完全没见过视频训练数据

## 亮点与洞察
- **"LMM+LoRA做检索"的范式非常实用**——无需改动模型架构，仅插入轻量LoRA，就能让通用LMM获得SOTA级检索能力
- **两阶段训练的逻辑非常清晰**——先用文本对比预训练"修复"embedding空间，再用多模态数据"扩展"任务支持
- **联合逐点+列表重排**给了部署灵活性——精度优先选逐点，速度优先选列表
- **在M-BEIR上从23.0提升到63.7**（+40.7个点），展示了LMM作为检索器的巨大潜力

## 局限与展望
- LMM推理开销高——每个候选都需要完整前向传播计算embedding，对大规模候选集不友好
- 视觉编码器全程冻结——可能限制了对视觉细节的感知能力
- 重排阶段需要额外的LoRA模块和训练——增加了系统复杂度
- 仅在 Qwen2-VL 上验证——未知是否能推广到其他LMM（如LLaVA、InternVL）

## 相关工作与启发
- **vs UniIR (CLIP/BLIP)**：双编码器方法，统一多任务但受限于CLIP的文本理解能力。LamRA利用LLM的强大语言理解在复杂查询上大幅胜出
- **vs E5-V**：同样用LMM做检索但仅做TEXT-only微调，在复杂多模态任务上效果有限。LamRA的两阶段训练和指令微调更全面
- **vs GENIUS**：生成式检索方法，效率高但精度不如embedding检索。LamRA走embedding路线但用LMM替代CLIP作为编码器

## 评分
- 新颖性: ⭐⭐⭐⭐ 将LMM用于通用检索非全新（E5-V在前），但两阶段训练+联合重排的完整框架设计更成熟
- 实验充分度: ⭐⭐⭐⭐⭐ 16种M-BEIR任务+10个未见数据集+视频检索+完整消融，实验极其充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，消融分析有说服力，表格虽多但组织合理
- 价值: ⭐⭐⭐⭐⭐ 证明了LMM作为通用检索器的可行性，对工业界多模态搜索系统有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Your Language Model Secretly Contains Personality Subnetworks](../../ICLR2026/information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)
- [\[CVPR 2025\] GENIUS: A Generative Framework for Universal Multimodal Search](genius_a_generative_framework_for_universal_multimodal_search.md)
- [\[ACL 2025\] SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](../../ACL2025/information_retrieval/saferag_benchmarking_security_in_retrieval-augmented_generation_of_large_languag.md)
- [\[ECCV 2024\] Towards Open-Ended Visual Recognition with Large Language Model](../../ECCV2024/information_retrieval/towards_open-ended_visual_recognition_with_large_language_models.md)
- [\[NeurIPS 2025\] The Transparent Earth: A Multimodal Foundation Model for the Earth's Subsurface](../../NeurIPS2025/information_retrieval/the_transparent_earth_a_multimodal_foundation_model_for_the_earths_subsurface.md)

</div>

<!-- RELATED:END -->
