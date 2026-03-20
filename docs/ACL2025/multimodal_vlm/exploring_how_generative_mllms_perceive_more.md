# Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder

**会议**: ACL 2025  
**arXiv**: [2411.05195](https://arxiv.org/abs/2411.05195)  
**代码**: [GitHub](https://github.com/lst627/CLIP-Embeds)  
**领域**: NLP理解  
**关键词**: CLIP, 多模态LLM, 视觉推理, 空间推理, 对比学习

## 一句话总结

系统探究为何生成式多模态LLM（如LLaVA）使用与CLIP相同的视觉编码器却能在视觉推理任务上大幅超越CLIP，发现patch token、位置编码和prompt加权是关键因素。

## 研究背景与动机

1. **领域现状**: CLIP在零样本分类上表现出色，但在组合性推理、空间理解、细粒度视觉理解等任务上表现不佳。
2. **现有痛点**: 人们认为CLIP视觉编码器丢失了关键信息，但生成式MLLM使用相同编码器在这些任务上表现优异，说明问题不在编码器。
3. **核心矛盾**: 相同的视觉编码器和权重，CLIP提取不到的信息，生成式MLLM却能提取，说明信息提取策略更为关键。
4. **本文要解决什么**: 找出生成式MLLM优于CLIP的核心设计选择，为改进CLIP类模型提供方向。
5. **切入角度**: 在What'sUp、Winoground等挑战性基准上进行对照实验，逐一剖析训练数据、token使用、位置编码、语言模型、训练目标等因素。
6. **核心idea一句话**: 生成式MLLM的优势来自架构设计（patch tokens + RoPE + prompt加权），而非更好的训练数据或更强的文本编码器。

## 方法详解

### 整体框架

以CLIP-ViT-L/14-336px和LLaVA-1.5-7B为核心对比对象，系统评估多个视觉推理基准，然后通过控制变量实验逐一分析因素。

### 关键设计

1. **Token使用实验**: 对比[CLS] token vs patch tokens。使用PACL方式的patch token聚合，pair accuracy在What'sUp上从1.9%提升到9.7%。添加RoPE后进一步提升到22.3%。
2. **训练数据实验**: 用LLaVA-1.5的训练数据微调CLIP/SigLIP/EVA-CLIP，包括加入hard negative，仍然接近随机水平，说明数据不是关键。
3. **文本编码器实验**: 使用更强的LLM转换文本编码器（LLM2CLIP），仍不足以解决问题。
4. **对比式微调实验**: 将LLaVA转换为类CLIP编码器进行对比微调后仍优于CLIP，说明细粒度视觉推理不依赖于自回归损失。
5. **Prompt作为加权**: 问题作为prompt与图像完全融合后，重新加权图像token，显著增强相关信息提取。

### 评估策略

使用VQAScore作为统一评估协议，定义为P("Yes"|image, question)，确保生成式MLLM与CLIP的公平比较。同时报告个体准确率和配对准确率。

## 实验关键数据

### 主实验（What'sUp空间推理）

| 模型 | Left/Right Pairs | On/Under Pairs | Front/Behind Pairs |
|------|-----------------|----------------|-------------------|
| CLIP-ViT-L/14-336px | 1.9% | 23.3% | 7.8% |
| LLaVA-1.5-7B | 93.2% | 52.4% | 52.9% |
| Phi-3-V-3.8B | 95.1% | 58.3% | 26.5% |
| LLaMA-3-V-8B | 96.1% | 64.1% | 47.1% |
| 随机基线 | 25.0% | 25.0% | 25.0% |

### 消融实验（CLIP + 各种改进）

| 配置 | What'sUp A Pairs | What'sUp B Pairs |
|------|-----------------|-----------------|
| CLIP原始 | 1.9% | 10.8% |
| + Patch Tokens | 9.7% | 9.8% |
| + PT + RoPE | 22.3% | 20.6% |
| + PT + RoPE + 多文本token | 0.0% | 6.9% |
| + PT + RoPE + 更强文本编码器 | 10.7% | 15.7% |

### 关键发现

- Patch tokens比[CLS] token提供更丰富的空间信息
- RoPE位置编码对空间推理至关重要
- 仅靠更强文本编码器或更多训练数据不够
- MLLM转对比式后仍保持优势，说明架构本身就具备优势

## 亮点与洞察

- 颠覆性发现：CLIP视觉编码器其实保留了足够的视觉信息，问题在于信息提取方式
- 精心设计的对照实验，逐一排除因素，方法论值得学习
- 细粒度推理不是生成模型独有的能力，对比模型也可以通过架构改进获得

## 局限性 / 可改进方向

- 主要聚焦于What'sUp空间推理，其他视觉推理任务的泛化性待验证
- 对照实验中freeze了视觉编码器，未充分探索解冻后的效果
- 改进后的CLIP在绝对性能上仍与MLLM有差距

## 相关工作与启发

- 与NegCLIP、SPARC等改进CLIP的工作互补
- 对VLM设计（如下一代CLIP模型）有直接指导意义
- 启发：架构选择可能比训练数据规模更重要

## 技术细节补充

- VQAScore定义：$P(\text{"Yes"} | \text{image}, \text{"Does this figure show 'text'? Please answer yes or no."})$
- CLIP使用[CLS] token + cosine similarity进行图文匹配
- LLaVA架构：CLIP视觉编码器的patch tokens → 2层MLP连接器 → 生成式语言模型
- 对比微调实验：将LLaVA的hidden states通过投影层映射到对比嵌入空间
- SigLIP解冻视觉编码器微调后仍无改善，进一步排除了编码器本身作为瓶颈的可能

## 评分

- 新颖性: ⭐⭐⭐⭐ 视角新颖，首次系统阐明CLIP vs MLLM差异的根因
- 实验充分度: ⭐⭐⭐⭐⭐ 控制变量实验设计严谨，覆盖多个基准
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，roadmap式实验展示易于跟随
- 价值: ⭐⭐⭐⭐ 对VLM架构设计有实际指导价值
