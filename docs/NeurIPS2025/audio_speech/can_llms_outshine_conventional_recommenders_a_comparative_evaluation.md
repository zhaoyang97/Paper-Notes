---
title: >-
  [论文解读] Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation
description: >-
  [NeurIPS 2025][音频/语音][LLM-as-RS] 提出 RecBench 综合评估框架，在5个领域数据集上系统对比17个LLM与10个传统DLRM，发现LLM推荐器在CTR任务上AUC提升最高5%、在序列推荐上NDCG@10提升最高170%，但推理速度慢10-1000倍，而传统DLRM结合LLM语义嵌入（LLM-for-RS）可以20倍更快的速度达到LLM约95%的性能，是当前最具工业可行性的方案。
tags:
  - "NeurIPS 2025"
  - "音频/语音"
  - "LLM-as-RS"
  - "RecBench"
  - "CTR预测"
  - "序列推荐"
  - "物品表示"
  - "推理效率"
---

# Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation

**会议**: NeurIPS 2025  
**arXiv**: [2503.05493](https://arxiv.org/abs/2503.05493)  
**代码**: [RecBench](https://recbench.github.io)  
**领域**: 音频语音  
**关键词**: LLM-as-RS, RecBench, CTR预测, 序列推荐, 物品表示, 推理效率

## 一句话总结

提出 RecBench 综合评估框架，在5个领域数据集上系统对比17个LLM与10个传统DLRM，发现LLM推荐器在CTR任务上AUC提升最高5%、在序列推荐上NDCG@10提升最高170%，但推理速度慢10-1000倍，而传统DLRM结合LLM语义嵌入（LLM-for-RS）可以20倍更快的速度达到LLM约95%的性能，是当前最具工业可行性的方案。

## 研究背景与动机

**领域现状**：LLM与推荐系统的融合（LLM+RS）是近年热点，分为两种范式：LLM-for-RS（LLM作为特征增强插件）和 LLM-as-RS（LLM直接作为推荐器）。后者在冷启动、可解释推荐等场景展现潜力，但缺乏系统性评估。

**现有痛点**：已有基准（LLMRec、PromptRec、OpenP5等）存在三大不足——(a) 仅评估单一推荐场景（pair-wise或list-wise）；(b) 物品表示形式覆盖不全，通常仅用文本或唯一ID；(c) 评估模型数量有限，且完全忽略推理效率指标。

**核心问题**：LLM在推荐任务上的准确率优势是否足以弥补推理效率的巨大劣势？不同物品表示方式如何影响LLM的推荐能力？

**本文切入角度**：构建迄今最全面的LLM推荐评测基准RecBench，首次同时评估准确率和效率两个维度，覆盖4种物品表示、2种推荐场景、27个模型、5个数据集。

## 方法详解

### 整体框架

RecBench 评估矩阵：**5个数据集**（H&M时尚、MIND新闻、MicroLens视频、Goodreads书籍、Amazon CDs音乐）× **4种物品表示**（唯一ID、文本、语义嵌入、语义标识符）× **2种推荐任务**（CTR预测、序列推荐）× **27个模型**（17个LLM + 10个DLRM），同时测量准确率指标和推理延迟。

### 四种物品表示方式

1. **Unique Identifier（唯一ID）**：传统方法，每个物品分配一个随机初始化的embedding向量，通过协同过滤信号学习语义。
2. **Text（文本描述）**：使用物品标题等文本特征，通过词嵌入取均值得到物品表示，天然适配LLM的文本理解能力。
3. **Semantic Embedding（语义嵌入）**：用预训练LLM（如Llama-1 7B）编码物品文本得到稠密向量，作为DLRM的初始化输入，引入丰富的通用语义。
4. **Semantic Identifier（语义标识符）**：先用SentenceBERT提取物品嵌入，再用RQ-VAE离散化为4层×256码本的编码序列。语义相似的物品共享更长的公共子序列，既压缩词表又保持语义关系。

### 两种推荐场景

**Pair-wise推荐（CTR预测）**：输入用户-物品对，预测点击概率。模型分为6组（A-F）：

- **Group A**：传统DLRM + 唯一ID（DNN、DeepFM、DCN、DCNv2、AutoInt、GDCN等9个模型）
- **Group B**：传统DLRM + 文本（DNN_text、DCNv2_text等4个模型）
- **Group C**：传统DLRM + 语义嵌入（DNN_emb、GDCN_emb等4个模型，LLM-for-RS范式）
- **Group D**：LLM + 唯一ID（P5系列，物品ID作为特殊token）
- **Group E**：LLM + 文本（GPT-3.5、Llama系列、Qwen系列等，支持零样本和微调）
- **Group F**：LLM + 语义标识符（SID-BERT、SID-OPT）

**List-wise推荐（序列推荐）**：输入用户历史交互序列，预测下一个物品。模型分为4组（G-J），引入**条件Beam Search（CBS）**技术——利用语义标识符树约束解码路径，确保生成的token序列对应有效物品。

### 训练策略

- LLM微调采用LoRA：CTR任务rank=32/alpha=128，序列推荐rank=128/alpha=128
- 学习率：LLM用1e-4，DLRM用1e-3
- 所有实验在单张A100 GPU上完成，结果取5次运行平均

## 实验关键数据

### CTR预测（Pair-wise，AUC指标）

| 物品表示 | 代表模型 | Overall AUC | CPU延迟(ms) | GPU延迟(ms) |
|---------|---------|-------------|-----------|-----------|
| 唯一ID（DLRM最佳） | GDCN | 0.6825 | 1.20 | 2.02 |
| 文本（DLRM最佳） | GDCN_text | 0.6923 | 5.09 | 3.77 |
| **语义嵌入（DLRM最佳）** | **DNN_emb** | **0.7171** | **1.42** | **2.09** |
| 文本（LLM微调最佳） | Mistral-2 7B | 0.7578 | 7680 | 76.14 |
| 零样本LLM最佳 | GLM-4 9B | 0.6231 | 9690 | 83.38 |

**关键发现**：Mistral-2微调后AUC达0.7578，比最佳DLRM（DNN_emb 0.7171）高约5.7%，但CPU推理慢**5400倍**（7680ms vs 1.42ms）。

### 序列推荐（List-wise，NDCG@10指标）

| 物品表示 | 代表模型 | Overall NDCG@10 | CPU延迟(ms) |
|---------|---------|----------------|-----------|
| 唯一ID（DLRM最佳） | SASRec_24L | 0.0698 | 103.41 |
| 唯一ID（LLM最佳） | P5-BERT_base | 0.1025 | 41.54 |
| **语义ID（LLM+CBS最佳）** | **SID-BERT_base-CBS** | **0.1877** | **1900** |
| 语义ID（大模型+CBS） | SID-Llama-3 7B-CBS | 0.1607 | 177540 |

**关键发现**：SID-BERT_base-CBS的NDCG@10（0.1877）比SASRec_24L（0.0698）提升**169%**，但推理时间增加18倍。SID-Llama-3 7B-CBS推理时间高达177秒/样本，完全不可实际部署。

### 零样本LLM表现

大多数LLM在零样本CTR任务上AUC在0.50附近徘徊（接近随机），仅Mistral（0.6199）和GLM-4（0.6231）表现尚可。专用推荐模型RecGPT（0.4952）和P5_Beauty（0.5049）的零样本泛化能力极差。Qwen-2系列展现出与模型规模正相关的零样本推荐能力（0.5B→1.5B→7B: 0.5413→0.5707→0.6075）。

### 微调带来的提升

指令微调使LLM的CTR AUC相对提升22%-43%。例如Llama-3 8B从零样本0.5252提升至微调后0.7508。

## 亮点与洞察

1. **"LLM-for-RS"是当前最优权衡**：DLRM+LLM语义嵌入（Group C）以极低延迟（~2ms GPU）达到DNN_emb AUC=0.7171，约为最佳LLM（0.7578）的94.6%性能，速度快36倍。这是工业部署最实际的方案。

2. **语义标识符在序列推荐中优势巨大**：SID表示使浅层网络即可捕获用户兴趣模式，SID-SASRec_3L-CBS（0.0306）已远超SASRec_3L（0.0096）。但随着层数增加优势递减，说明深层ID-based模型也能学到类似信息。

3. **条件Beam Search（CBS）是关键技术**：CBS通过语义标识符树约束解码，确保生成有效物品。SID-BERT_base从0.0941提升至CBS后0.1877，几乎翻倍。

4. **预训练语言模式与用户兴趣模式存在抽象相似性**：BERT_base用唯一ID（无文本）在序列推荐上（0.1025）超过同架构的SASRec_12L（0.0672），暗示语言建模的序列模式可迁移到用户行为建模。

5. **模型规模不是万能的**：在序列推荐中，SID-BERT_base-CBS（0.1877）反而大幅超过SID-Llama-3 7B-CBS（0.1607），小模型在特定设置下可能更优。

## 局限与展望

- 仅测量单样本推理延迟，未考虑批量推理和KV-cache等加速技术的影响
- 未评估LLM在冷启动、跨域推荐等特殊场景的优势——这可能是LLM-as-RS真正有价值的方向
- 语义嵌入仅使用Llama-1 7B，未探索更强编码器（如更新的Llama-3或领域专用模型）的效果
- 5个数据集均经过统一预处理裁剪为相似规模，可能不反映真实工业场景的数据分布

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个同时覆盖准确率和效率、4种物品表示、2种任务的全面LLM推荐基准
- 实验充分度: ⭐⭐⭐⭐⭐ 27模型×5数据集×4物品表示，结果取5次平均，规模罕见
- 写作质量: ⭐⭐⭐⭐ 实验分析清晰有条理，结论有说服力
- 价值: ⭐⭐⭐⭐⭐ 对推荐系统社区有重要战略指导意义——明确了LLM-for-RS优于LLM-as-RS的结论

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Inductive Transfer Learning for Graph-Based Recommenders](inductive_transfer_learning_for_graph-based_recommenders.md)
- [\[ACL 2025\] Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models](../../ACL2025/audio_speech/who_can_withstand_chat-audio_attacks_an_evaluation_benchmark_for_large_audio-lan.md)
- [\[ICML 2025\] Teaching Physical Awareness to LLMs through Sounds](../../ICML2025/audio_speech/teaching_physical_awareness_to_llms_through_sounds.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](../../ACL2026/audio_speech/jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[ICML 2026\] Position: Towards Responsible Evaluation for Text-to-Speech](../../ICML2026/audio_speech/position_towards_responsible_evaluation_for_text-to-speech.md)

</div>

<!-- RELATED:END -->
