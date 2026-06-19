---
title: >-
  [论文解读] Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech
description: >-
  [ACL2026][医疗NLP][痴呆检测] 这篇论文用 4000 条英菲平行 DementiaBank 对话转写系统评测 TF-IDF、BERT、NeoBERT、XLM-R 和 RoBERTa-Tagalog，发现痴呆检测的跨语言鲁棒性主要来自训练阶段的语言覆盖，而不是更现代的编码器架构。 领域现状：从自发语音或对话转写中…
tags:
  - "ACL2026"
  - "医疗NLP"
  - "痴呆检测"
  - "Filipino-English"
  - "NeoBERT"
  - "跨语言迁移"
  - "双语微调"
---

# Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech

**会议**: ACL2026  
**arXiv**: [2605.26007](https://arxiv.org/abs/2605.26007)  
**代码**: https://github.com/rezsam09/Filipino-English-Dementia-Classification  
**领域**: 临床 NLP / 低资源语言  
**关键词**: 痴呆检测, Filipino-English, NeoBERT, 跨语言迁移, 双语微调  

## 一句话总结
这篇论文用 4000 条英菲平行 DementiaBank 对话转写系统评测 TF-IDF、BERT、NeoBERT、XLM-R 和 RoBERTa-Tagalog，发现痴呆检测的跨语言鲁棒性主要来自训练阶段的语言覆盖，而不是更现代的编码器架构。

## 研究背景与动机
**领域现状**：从自发语音或对话转写中检测痴呆是临床 NLP 的重要方向，因为认知衰退会反映在词汇多样性、重复、停顿、句法简化和指称连贯性下降等语言现象中。DementiaBank 的 Cookie Theft 图像描述任务是这个方向最常用的数据源之一。

**现有痛点**：绝大多数痴呆 NLP 系统以英语为中心，低资源语言、东南亚语言、以及真实双语 code-switching 场景缺少系统评估。菲律宾日常和临床交流中 Filipino-English code-switching 非常普遍，但此前没有 NLP-based dementia detection 工作专门研究 Filipino speech。

**核心矛盾**：模型在英语 DementiaBank 上高分，不代表它学到了语言无关的认知衰退信号。性能下降可能来自语言迁移失败、预训练语料偏置、临床域变化或数据采集差异。要 isolating language shift，就必须让英语和 Filipino 样本在临床内容、任务结构、类别比例上尽可能平行。

**本文目标**：作者构建一个受控双语数据集，比较 English-only、multilingual、language-matched 和现代化 encoder 在三种训练设置下的表现：英语单语、Filipino 单语、英菲双语。特别关注 NeoBERT 这种现代 encoder 架构在临床和跨语言条件下是否真正更鲁棒。

**切入角度**：论文不加入手工语言特征或复杂声学特征，而是固定相同预处理和 fine-tuning pipeline，让性能差异主要反映 representation 和 pretraining language exposure。这样能更清楚地区分“架构更强”和“语言覆盖更充分”。

**核心 idea**：在域和任务都受控的平行双语数据上，如果单语训练仍跨语言失败，就说明瓶颈是表示空间的语言对齐；如果双语微调能消除差距，则训练语言覆盖比模型架构更新更关键。

## 方法详解

### 整体框架
这篇论文不提新模型，而是搭一个受控的双语评测台，去回答一个具体问题：痴呆检测的跨语言鲁棒性，到底来自更现代的编码器架构，还是来自训练阶段的语言覆盖。为此作者构建了一个 balanced bilingual binary classification 设置——每种语言 2000 条、dementia-positive 与 healthy control 各 1000 条，共 4000 条；英文样本取自 DementiaBank，Filipino 样本则由完整 2000 条英文转写人工翻译而来，翻译时刻意保留重复、犹豫、false starts 和句法退化这些 discourse-level 的认知衰退标记。

所有转写走同一条预处理流水线：Unicode normalization、空白归一化、小写化，但保留 filled pauses、repetitions、hesitation markers，不做 stemming/lemmatization/parsing，输入截断到 128 tokens。在这之上铺开五类模型——TF-IDF + Logistic Regression 作为可解释的 lexical baseline，BERT-base 与 NeoBERT 代表英语预训练（NeoBERT 带 RoPE、Pre-LayerNorm、RMSNorm、SwiGLU 等现代化设计），XLM-RoBERTa 代表 100 语言的 multilingual baseline，RoBERTa-Tagalog 代表 Filipino 的 language-matched 预训练。每类模型再跑三种训练设置：English-only、Filipino-only、English+Filipino bilingual，分别在同语言 in-domain、跨语言 zero-shot 和 bilingual mixed-language 下评估，主指标是 Accuracy 和 Macro-F1，并额外报告 class-wise F1 与 dementia recall，避免平均指标掩盖临床敏感性。

### 关键设计

**1. 平行双语数据构造：把“语言迁移”单独拎出来当唯一变量**

如果直接拿两种语言各自独立的临床语料来比，性能差异会被患者群体、任务设计、录音协议这些混杂因素污染，根本说不清掉分是不是语言造成的。这篇论文的做法是不另找语料，而是对完整的 DementiaBank 转写做人工翻译，强制让 Filipino 侧与英文侧在 class distribution、discourse structure、elicitation task 和 clinical content 上全部对齐。这样一来，英菲两侧唯一系统性变动的就是语言本身，跨语言掉分才能被干净地归因到 language shift，而不是数据采集差异。

**2. 保留 disfluency 的统一预处理：别把诊断信号“润色”掉**

痴呆的语言信号恰恰藏在不流畅和组织失败里——重复、犹豫、停顿标记和句法碎片本身就是认知衰退的证据。所以预处理刻意保留这些标记，也刻意不用机器翻译来生成 Filipino 侧（机器翻译会把不流畅语音顺成流畅文本，等于抹掉诊断线索）。如果像普通文本分类那样做过度规范化，任务会退化成识别语义内容、削弱临床意义；保留 disfluency 才让模型有机会真正学到退化模式，而不是表面词汇。

**3. 架构与语言覆盖的对照实验：用一组模型矩阵把两个假设拆开**

要判断鲁棒性来自架构还是语言覆盖，就得让这两个因素在模型选择上可分离。BERT 与 NeoBERT 都是 English-only、但架构一旧一新，对比它们能单独读出“架构现代化”的贡献；XLM-R 带显式多语言预训练、RoBERTa-Tagalog 带语言匹配预训练，对比它们能读出“语言覆盖”的贡献。为排除其他干扰，所有 transformer 统一用 masked mean pooling、dropout 0.1 和 linear head。逻辑很干净：如果 NeoBERT 比 BERT 强、却仍跨语言不稳，说明现代架构只是把英语侧拟合得更紧、并不等于跨语言临床鲁棒；如果双语训练让所有模型收敛到一起，那核心就是监督的语言覆盖，而非架构更新。

### 损失函数 / 训练策略
TF-IDF 使用 unigram + bigram，sublinear TF scaling，min document frequency 2，max document frequency 0.95，max vocabulary 20,000，并用 $l_2$ 正则 Logistic Regression 和 liblinear solver，最多 2000 iterations。

Transformer 模型端到端 fine-tune，使用 attention-masked mean pooling：$h=\sum_i m_iH_i/\sum_i m_i$，再接 dropout 和线性分类头。优化目标是标准 cross-entropy：$L(\theta)=-E_{(x,y)}\log p_\theta(y|x)$，优化器 AdamW。

超参通过 grid search 选择，learning rate 包括 $5e^{-6}$、$6e^{-6}$、$1e^{-5}$、$2e^{-5}$、$3e^{-5}$，weight decay 包括 $1e^{-2}$ 和 $1e^{-5}$，batch size 包括 4 和 8。训练最多 10 epochs，搜索阶段用 validation Macro-F1 early stopping，最终报告用 stratified 10-fold cross-validation 衡量稳定性。

## 实验关键数据

### 主实验

| 模型 | 训练语言 | EN Macro-F1 | TL Macro-F1 | Combined Macro-F1 | Gap |
|--------|------|------|----------|------|------|
| TF-IDF + LR | EN | 0.930±0.013 | 0.649±0.008 | 0.836±0.005 | 0.281 |
| BERT | EN | 0.952±0.014 | 0.455±0.012 | 0.744±0.008 | 0.497 |
| NeoBERT | EN | 0.952±0.013 | 0.617±0.109 | 0.802±0.045 | 0.335 |
| XLM-RoBERTa | EN | 0.948±0.017 | 0.936±0.018 | 0.942±0.016 | 0.013 |
| RoBERTa-Tagalog | EN | 0.951±0.014 | 0.934±0.005 | 0.942±0.015 | 0.017 |
| BERT | EN+TL | 0.954±0.009 | 0.984±0.009 | 0.969±0.007 | 0.030 |
| XLM-RoBERTa | EN+TL | 0.953±0.010 | 0.990±0.007 | 0.972±0.006 | 0.037 |
| RoBERTa-Tagalog | EN+TL | 0.958±0.010 | 0.988±0.007 | 0.973±0.006 | 0.030 |
| NeoBERT | EN+TL | 0.956±0.015 | 0.983±0.009 | 0.970±0.007 | 0.027 |

### 临床敏感性与类别表现

| 模型 / 设置 | Healthy F1 (TL) | Dementia F1 (TL) | Dementia Recall (TL) | 解读 |
|------|---------|------|------|------|
| BERT EN→TL | 0.216 | 0.695 | 0.931 | 表面 recall 高，但几乎把大量 Filipino 样本判成 dementia，Healthy 崩溃 |
| NeoBERT EN→TL | 0.504 | 0.729 | 0.939 | 比 BERT 好，但 variance 很大，决策边界不稳定 |
| XLM-RoBERTa EN→TL | 0.950 | 0.947 | 0.920 | 类别平衡更好，适合筛查场景 |
| RoBERTa-Tagalog EN→TL | 0.952 | 0.950 | 0.928 | 语言匹配预训练也能支持 English-to-Filipino 迁移 |
| NeoBERT EN+TL | 0.956 | 0.955 | 0.938 | 双语训练后恢复稳定 |

### 关键发现
- English-trained BERT 在英语上 Macro-F1 0.952，但到 Filipino 只有 0.455，说明强 in-domain 表现无法保证跨语言临床鲁棒。
- NeoBERT 的架构现代化没有解决跨语言问题；English-trained NeoBERT 到 Filipino 为 0.617±0.109，方差最大，说明它可能形成更紧的英语侧边界。
- XLM-RoBERTa 和 RoBERTa-Tagalog 在 English-to-Filipino transfer 上 gap 极小，分别为 0.013 和 0.017。
- 双语微调让所有 transformer 的 combined Macro-F1 收敛到 0.969-0.973，差异几乎消失，说明瓶颈主要是语言覆盖和表示对齐。
- TF-IDF 双语训练也达到 0.954，但仍低于 transformer，说明表面词汇保留了部分痴呆信号，稳定迁移仍需要上下文表示。

## 亮点与洞察
- **实验控制非常干净**：人工翻译保持 clinical content 和 discourse structure，让“语言迁移失败”这个结论更有说服力。
- **NeoBERT 负结果有价值**：现代 encoder 架构可以提升通用 benchmark，但在低资源临床跨语言任务中并不会自动带来鲁棒性。
- **类别级分析避免误读**：BERT EN→TL 的 dementia recall 很高，但 Healthy F1 只有 0.216，说明模型是在过度阳性预测，而不是理解了 Filipino 痴呆线索。
- **RoBERTa-Tagalog 的结果很有趣**：即便只在 Filipino 语料上预训练，它也能处理 English-to-Filipino transfer，可能因为 Filipino 本身含大量英语借词和 code-switching 结构。

## 局限与展望
- Filipino 数据是人工翻译的 DementiaBank，而非菲律宾本地患者自然临床语音。即使保留了 disfluency，语义内容和会话结构仍来自英文源数据。
- 数据规模只有 4000 条，虽然做了 10-fold CV，但跨语言和临床部署的泛化仍需更大本地 cohort 验证。
- 论文只研究文本转写，暂未纳入 pitch variance、pause duration、phonation rate 等声学和非语言线索；真实痴呆筛查很可能需要多模态语音。
- 模型解释机制不足。临床部署前需要 feature attribution、错误案例审查、语言学标记分析，确认模型不是依赖翻译痕迹或数据构造 artifact。
- 当前任务是二分类，未覆盖 MCI、不同痴呆阶段、纵向变化和 code-switching 的真实比例变化。

## 相关工作与启发
- **vs ADReSS-M / English-Greek transfer**: 这些工作多在印欧语言之间迁移；Filipino-English 更能检验 typological distance 和 code-switching 场景。
- **vs Nepali / Amis 低资源痴呆检测**: 这些研究使用翻译或增强改善低资源性能，但本文更系统地比较 zero-shot cross-lingual、language-matched pretraining 和 bilingual fine-tuning。
- **vs ClinicalBERT / AD-BERT**: 域适配或疾病专门预训练并不等同于低资源语言鲁棒；本文提示临床 NLP 不能只在英语域内扩展。
- **启发**：在低资源临床 NLP 中，优先级可能是收集本地语言监督、保留真实 disfluency 和 code-switching，而不是盲目换更大的英文模型。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 首个 Filipino 痴呆 NLP 系统评测，并把 NeoBERT 放到临床跨语言场景中；方法本身是基准评测而非新模型。
- 实验充分度: ⭐⭐⭐⭐☆ 模型、训练设置、10-fold、类别级指标都很完整；最大短板是 Filipino 数据来自翻译而非原生临床采集。
- 写作质量: ⭐⭐⭐⭐☆ 问题定义清楚，结果解释克制，尤其对 Macro-F1 与 clinical recall 的拆解有帮助。
- 价值: ⭐⭐⭐⭐☆ 对低资源临床 NLP 和多语言医疗筛查非常有启发，清楚说明语言覆盖比架构升级更关键。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[ACL 2026\] MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts](medfact_benchmarking_the_fact-checking_capabilities_of_large_language_models_on_.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ICML 2026\] MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings](../../ICML2026/medical_nlp/medcase-structured_a_text-to-fhir_dataset_for_benchmarking_diagnostic_reasoning_.md)

</div>

<!-- RELATED:END -->
