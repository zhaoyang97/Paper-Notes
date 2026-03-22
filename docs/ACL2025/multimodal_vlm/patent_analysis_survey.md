# A Survey on Patent Analysis: From NLP to Multimodal AI

**会议**: ACL 2025  
**arXiv**: [2404.08668](https://arxiv.org/abs/2404.08668)  
**领域**: LLM NLP / 综述 / 专利分析  
**关键词**: 专利分析, NLP, 多模态AI, 专利分类, 预训练语言模型

## 一句话总结

全面综述基于 NLP 和多模态 AI 的专利分析方法，按专利生命周期中的四大任务（分类、检索、质量分析、生成）提出新的分类体系，覆盖从传统神经网络到 PLM/LLM 的技术演进。

## 研究背景与动机

- **专利分析的重要性**：专利数据量呈指数增长，人工审查面临效率瓶颈；NLP 和 AI 技术可显著加速专利分类、检索、质量评估和撰写流程
- **现有综述的不足**：已有的专利 NLP 综述（Gomez & Moens 2014、Krestel et al. 2021 等）未覆盖近年来 LLM 和多模态方法的进展，且缺乏按任务细分的方法论视角
- **本文贡献**：提出基于"专利生命周期任务 × 方法类型"的双维度分类体系，系统整理近年进展，特别补充了 LLM 和多模态方法的最新趋势

## 方法详解

### 整体框架

按专利生命周期将任务分为四类，每类下按方法性质进一步分组：

**1. 专利分类 (Patent Classification)**
**2. 专利检索 (Patent Retrieval)**
**3. 专利质量分析 (Patent Quality Analysis)**
**4. 专利生成 (Patent Generation)**

### 关键设计

#### 1. 专利分类

**挑战**：多类多标签（IPC 有 70788 子组，CPC 约 250,000 条目）、层级结构的依赖关系、长文档中关键段落识别

三代方法演进：
- **传统神经网络**：Word2Vec + LSTM（Grawe et al. 2017）、固定层级向量 + LSTM（Shalaby et al. 2018）、FastText + Bi-GRU（Risch & Krestel 2018/2019）
- **集成模型**：多种词向量 + 多种 RNN 架构的组合；CLIP + MLP 用于专利图像分类（Ghauri et al. 2023）
- **预训练语言模型**：BERT 微调在 USPTO-2M/3M 数据集上（Lee & Hsiang 2020b）；XLNet 在分类精度上超越 BERT（Roudsari et al. 2022，precision 从 0.53 提升至 0.82）；SciBERT 在技术语言理解上优于通用 BERT（Althammer et al. 2021）；Sentence-BERT 的整句理解能力带来最高 recall 和 F1（Bekamiri et al. 2024）

#### 2. 专利检索

**挑战**：同一发明可用不同措辞描述（文本检索歧义）；专利图像多为黑白草图且带标号（图像检索特殊性）

方法分类：
- **传统机器学习**：SVM、随机森林、朴素贝叶斯用于先有技术检索（Setchi et al. 2021）
- **深度学习**：BiLSTM-CRF + BiGRU-HAN（Chen et al. 2020）；DUAL-VGG 用于图像检索（Jiang et al. 2021）
- **预训练模型**：BERT 用于专利文本检索（Kang et al. 2020）；RoBERTa + CLIP 用于文本+图像联合检索（Pustu-Iren et al. 2021）
- **前沿方法**：深度度量学习 + 自监督（Higuchi & Yanai 2023）；BLIP-2 + GPT-4V 用于多模态专利检索（Lo et al. 2024）

#### 3. 专利质量分析

**挑战**：质量度量指标的模糊性（引用数、权利要求数、授权延迟等权重不清）；多指标综合分析的复杂性

代表方法：
- 属性网络嵌入 + 注意力CNN（Lin et al. 2018，评估指标 RMSE）
- DNN + PCA（Trappey et al. 2019）
- BiLSTM-ATT-CRF 基于专利维护期预测（Li et al. 2022）
- MSABERT 基于专利文本的质量评分（Krant 2023）

#### 4. 专利生成

**挑战**：专利各部分（摘要、独立权利要求、从属权利要求）间的依赖关系；精确技术语言的生成；生成内容的评估标准

## 实验关键数据

### 主实验

由于本文为综述，主要梳理各方法的性能趋势：

**专利分类性能演进**：
| 方法 | 代表工作 | 数据集 | 最高精度 |
|------|---------|--------|---------|
| FastText + GRU | Risch & Krestel 2018 | USPTO | 0.53 (P) |
| BERT/XLNet/RoBERTa | Roudsari et al. 2022 | USPTO-2M | **0.82** (P) |
| SciBERT | Althammer et al. 2021 | USPTO | 优于 BERT |
| Sentence-BERT | Bekamiri et al. 2024 | USPTO | 最高 recall & F1 |

**专利检索方法对比**（覆盖文本、图像、多模态三类数据）：
- 13 篇代表性工作，从传统 ML 到 BLIP-2+GPT-4V
- 数据集：USPTO、DeepPatent/DeepPatent2、EPO 等
- 趋势：从监督学习向预训练+自监督迁移

**专利质量分析**：
- 8 篇代表性工作，使用 2-12 个专利指标
- 评估指标包括 MAE、RMSE、准确率、F1 等

### 关键发现

1. **PLM/LLM 采用率快速上升**：从早期的 Word2Vec+LSTM 到 BERT/XLNet/RoBERTa，再到最新的 GPT-4V 和 BLIP-2
2. **专利域适应很重要**：在科学文献上预训练的 SciBERT 优于通用 BERT，表明专利语言的特殊性需要领域适应
3. **多模态方法是新趋势**：专利包含大量草图和技术图纸，单纯文本方法存在信息损失
4. **现有研究与前沿 LLM 之间存在差距**：专利领域仍主要使用 BERT 级别模型，GPT/LLaMA 级别的应用尚不充分
5. **跨方法比较困难**：不同研究使用的数据集子集、层级粒度和评估指标差异大

## 亮点与洞察

- **双维度分类体系**：按"任务 × 方法"组织文献，比单一维度更有检索和参考价值
- **覆盖多模态**：不仅关注文本，也系统梳理了专利图像分析和多模态方法（蓝色标注使用图像的工作）
- **实用导向**：为专利局和专利分析师提供了技术选型路线图
- **GitHub 仓库维护**：持续更新的论文列表和资源

## 局限性

- 综述侧重方法梳理，缺少不同方法的定量横向对比（受限于数据集/指标不统一）
- 对 LLM（GPT-4、LLaMA 等）在专利领域的应用讨论偏少，可能因为相关工作刚起步
- 未深入讨论专利分析中的法律和伦理问题（如 AI 生成专利的法律效力）
- 专利数据的多语言问题未充分覆盖

## 相关工作

- **早期专利 NLP 综述**：Gomez & Moens 2014、Hanbury et al. 2011、Krestel et al. 2021
- **NLP 基础技术**：BERT（Devlin et al. 2019）、GPT（Radford et al. 2019）、XLNet、RoBERTa
- **专利数据集**：USPTO-2M/3M、DeepPatent/DeepPatent2、EPO 等
- **多模态方法**：CLIP、BLIP-2、GPT-4V、Vision Transformer

## 评分

- **创新性**: ★★★☆☆ — 作为综述，分类体系有一定新意但无方法论创新
- **实用性**: ★★★★★ — 对专利 AI 领域的研究者和从业者极具参考价值
- **实验充分度**: ★★★☆☆ — 综述性质，表格梳理充分但缺乏实验验证
- **写作质量**: ★★★★☆ — 组织结构清晰，表格丰富，但部分任务描述偏浅
