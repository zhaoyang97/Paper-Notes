---
title: >-
  [论文解读] Vocabulary Customization for Efficient Domain-Specific LLM Deployment
description: >-
  [NeurIPS 2025][预训练][词表扩展] 提出一种保证编码效率单调不降的BPE tokenizer扩展算法，将领域高频token追加到Llama 3.1词表中（+30K token），在电商场景实现输入序列缩短20%、推理吞吐量提升20-30%，经10K步继续训练后模型质量不降，且约98%情况下模型主动生成新token。
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "词表扩展"
  - "tokenizer适配"
  - "域适应"
  - "BPE"
  - "推理加速"
---

# Vocabulary Customization for Efficient Domain-Specific LLM Deployment

**会议**: NeurIPS 2025  
**arXiv**: [2509.26124](https://arxiv.org/abs/2509.26124)  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: 词表扩展, tokenizer适配, 域适应, BPE, 推理加速

## 一句话总结

提出一种保证编码效率单调不降的BPE tokenizer扩展算法，将领域高频token追加到Llama 3.1词表中（+30K token），在电商场景实现输入序列缩短20%、推理吞吐量提升20-30%，经10K步继续训练后模型质量不降，且约98%情况下模型主动生成新token。

## 研究背景与动机

**领域现状**：LLM在特定领域部署时，通用tokenizer常常无法高效编码领域术语。例如电商场景中品牌名、SKU编号、多语言商品描述等高频词被拆分为多个子词token，导致token fertility偏高，直接增加推理延迟和成本。

**现有痛点**：已有tokenizer扩展工作主要面向新语言适配（中文、泰语等），领域适配的系统研究极少。Yamaguchi et al.的方法将新merge操作前置到merge list头部，但这会改变已有merge的优先级，导致通用文本编码效率反而下降。AdaptiVocab用n-gram token替换已有token，无法保证编码效率单调不降。

**核心矛盾**：词表扩展面临效率-兼容性trade-off——加入太多新token会增大embedding和projection矩阵，拖慢单次forward pass；加入太少则压缩效果有限。更关键的是，对于自回归LLM，模型是否会在生成时主动使用新token是一个从未被系统研究过的问题。

**切入角度**：将新merge操作追加到merge list末尾而非前置，利用BPE的顺序执行特性保证原有分词行为完全不变，任何输入的token数只减不增。

**核心 idea**：追加式词表扩展 + 编码效率-forward pass速度联合优化 + 新token采用率分析，三位一体解决领域LLM推理效率。

## 方法详解

### 整体框架

五步流程：在领域数据上训练BPE tokenizer → 从中选取新token追加到原tokenizer → 初始化新embedding → 继续训练LLM → 评估效率和质量。

### 关键设计

1. **追加式Tokenizer扩展算法**:

    - 在领域数据集上从头训练BPE tokenizer获取领域高频token候选，筛选不在原词表中的新token
    - **核心创新**：将新merge操作追加到merge list末尾。BPE按顺序执行merge，追加保证原有merge优先级不变，新merge只在原有分词完成后才激活
    - **保证性质**：对任意输入，扩展后token数 $\leq$ 原tokenizer。新merge只会将已有的多个token合并为一个，不可能增加token数
    - 对比前置策略：前置会改变优先级，在通用文本上可能增加token数（实验验证）

2. **词表大小-效率权衡分析**:

    - 无需训练模型即可完成：扫描1K-80K新增token数，测编码效率和forward pass耗时
    - 8B模型上30K是最优平衡点：forward pass仅慢1%，电商任务平均缩短8%，最高缩短20%
    - 权衡与模型大小相关：模型越大embedding占比越小，可加更多token

3. **Embedding初始化与继续训练**:

    - 新token的embedding/projection向量用组成子token向量均值初始化
    - 混合数据（50%通用+50%领域），cosine学习率（1e-5→5e-7），10K步
    - 480块H100 GPU，Megatron-LM框架，训练不到24小时

### 损失函数 / 训练策略

标准自回归语言建模损失。cosine学习率调度。

## 实验关键数据

### 主实验

**推理吞吐量**（vLLM, H100, Llama-3.1 8B）：

| 输入/输出长度 | 原始RPS | 扩展后RPS | 吞吐提升 |
|-------------|---------|----------|---------|
| 300 words | 29.19 | 35.23 | 20.7% |
| 3000 words | 1.95 | 2.52 | 29.2% |

**模型质量**（14个电商任务）：

| 模型 | 通用NLU(En) | MMLU | 电商(En) | 电商(non-En) |
|------|-----------|------|---------|------------|
| 8B LLM | 71.6 | 63.5 | 60.5 | 47.9 |
| +扩展词表30K | 71.8 | 63.4 | 60.1 | 47.6 |

质量完全持平，通用和领域任务均无损失。

### 消融实验

| 配置 | 说明 |
|------|------|
| 追加式 vs 前置式 | 追加在通用文本上不增加token数；前置在加>20K后通用文本反增 |
| 30K新token | forward pass慢1%，编码缩短8% |
| 新token采用率(>15词) | 约98%情况下模型生成新token |
| 新token采用率(<15词) | 约95.3%，短序列偶尔回退旧分词 |

### 关键发现

- **新token采用率**：首次实证表明自回归LLM确实会主动使用新token（98%），解答了社区长期疑虑
- 长序列受益更大：3000词时吞吐提升29.2%（vs 300词的20.7%），因为注意力计算的二次复杂度放大了token减少的效果
- 追加策略在通用文本上保证不退化，前置策略在Wikipedia上可能增加token数
- 词表扩展与量化、投机解码等方法正交，可组合使用获得乘法级加速

## 亮点与洞察

- **保证单调的追加策略**：利用BPE顺序执行语义保证向后兼容，不需要额外验证。可迁移到任何需要增量更新tokenizer的场景。
- **首次分析新token采用率**：填补了领域tokenizer扩展研究的关键空白。98%的采用率说明均值初始化+10K步训练足够让模型"接受"新词表。
- **实用的效率-速度分析框架**：不需要训练模型就能通过扫描找到最优词表大小，对工业界LLM部署优化直接可用。

## 局限与展望

- 仅在电商领域验证，其他领域（医疗/法律/金融）的压缩效果有待验证
- 所有实验基于Llama 3.1 8B，更大模型的权衡可能不同
- 均值初始化对语义复杂的新token可能不够好，可探索更智能的初始化
- 未讨论多领域同时服务场景的tokenizer管理策略
- 10K步继续训练的最优步数缺乏消融分析

## 相关工作与启发

- **vs Yamaguchi et al. (2024)**: 前置merge策略领域效率增长更快但破坏通用文本编码，本文追加策略更适合生产环境
- **vs AdaptiVocab**: 用n-gram替换token无法保证编码单调不降，分布外输入可能反而更差
- **vs 语言适配类工作**: 语言适配追求质量提升，本文聚焦效率提升，两者方法论互补

## 评分

- 新颖性: ⭐⭐⭐ 核心idea直观清晰，创新更多在工程层面，但新token采用率分析是首次
- 实验充分度: ⭐⭐⭐⭐ 编码效率、质量、速度、采用率全覆盖，但仅一个领域
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，问题定义精准，分析透彻
- 价值: ⭐⭐⭐⭐ 对LLM领域部署有直接实用价值，新token采用率分析开创新评估维度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)
- [\[ICCV 2025\] Make Your Training Flexible: Towards Deployment-Efficient Video Models](../../ICCV2025/llm_pretraining/make_your_training_flexible_towards_deployment-efficient_video_models.md)
- [\[ACL 2025\] TokAlign: Efficient Vocabulary Adaptation via Token Alignment](../../ACL2025/llm_pretraining/tokalign_vocab_adaptation.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/llm_pretraining/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[NeurIPS 2025\] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models](learning_the_wrong_lessons_syntactic-domain_spurious_correlations_in_language_mo.md)

</div>

<!-- RELATED:END -->
