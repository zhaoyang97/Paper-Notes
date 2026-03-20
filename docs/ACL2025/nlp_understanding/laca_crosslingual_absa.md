# LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation

**会议**: ACL 2025  
**arXiv**: [2508.09515](https://arxiv.org/abs/2508.09515)  
**代码**: [https://nlp.kiv.zcu.cz](https://nlp.kiv.zcu.cz)  
**领域**: NLP 理解  
**关键词**: Cross-lingual ABSA, LLM Data Augmentation, Pseudo-labelled Data, Multilingual Sentiment, Zero-shot Transfer  

## 一句话总结

提出 LACA 框架，利用 LLM 为目标语言生成高质量伪标注数据（而非依赖机器翻译），在六种语言上显著提升跨语言 ABSA 性能，在 mBERT 和 XLM-R 上分别平均超过前 SOTA 1.50% 和 2.62%。

## 研究背景与动机

### 跨语言 ABSA 的挑战

Aspect-Based Sentiment Analysis (ABSA) 旨在识别句子中与特定 aspect 相关的情感极性。例如 "Great tea but terrible service" 中 "tea" 为正面，"service" 为负面。由于大多数标注数据集中在英语，低资源语言严重缺乏标注数据，使得跨语言 ABSA 成为重要研究方向。

### 现有方法的不足

1. **翻译方法的局限性**：传统方法依赖机器翻译将源语言数据翻译为目标语言，但翻译过程中 aspect term 容易错位或丢失，导致模型无法正确识别目标语言中的 aspect term
2. **直接迁移的语言鸿沟**：直接在源语言微调后应用于目标语言，会受到语言特有的 aspect term、俚语、缩写等影响
3. **低资源语言的 mPLM 覆盖不足**：一些低资源语言在多语言预训练模型的训练语料中占比很低

### 核心动机

LLM 的数据增强能力为跨语言 ABSA 提供了新思路——可以直接在目标语言中生成多样化的训练样本，避免翻译带来的噪声问题。

## 方法详解

### 整体框架

LACA (LLM Augmented Cross-lingual ABSA) 是一个两阶段框架：

**第一阶段：ABSA 模型预测**
1. 在带标签的英语源语言数据 $\mathcal{D}_\mathcal{S}$ 上微调 ABSA 模型
2. 将微调后的模型应用于目标语言的无标注数据 $\mathcal{D}_\mathcal{T}$，得到噪声预测标签 $\hat{y}^\mathcal{T}$

**第二阶段：LLM 数据增强**
1. 将预测标签 $\hat{y}^\mathcal{T}$ 输入 LLM，要求 LLM 在目标语言中生成与标签对齐的自然句子 $\hat{x}^\mathcal{T}$
2. 组成伪标注数据集 $\mathcal{D}_\mathcal{G} = \{(\hat{x}_i^\mathcal{T}, \hat{y}_i^\mathcal{T})\}$
3. 将 $\mathcal{D}_\mathcal{G}$ 与 $\mathcal{D}_\mathcal{S}$ 合并，在该混合数据集上进一步训练 ABSA 模型

### 关键设计

**ABSA 模型的多种架构支持**：

- **Encoder 模型**（mBERT, XLM-R）：序列标注方式，使用 BIO 标注 + 3 种情感极性（POS/NEG/NEU），token 级预测
- **Encoder-Decoder 模型**（mT5）：文本生成方式，输出格式为 "[A] aspect [P] polarity"
- **Decoder-only 模型**（LLaMA 3.1, Orca 2）：自回归生成方式

**LLM 生成的质量控制**：

1. **预处理**：确保预测标签中至少包含一个 sentiment element
2. **Prompt 设计**：指定目标语言、要求不引入额外的 sentiment element，并提供 10 个源语言的 few-shot 示例
3. **后处理过滤**：
   - 过滤生成文本中缺少预测 aspect term 的实例
   - 过滤 ABSA 模型对生成文本的重新预测与原始标签不一致的实例

**处理类别不平衡**：修改 20% 过度表征的正面情感样本，以 60% 概率生成中性、40% 概率生成负面情感的新实例。

### 损失函数 / 训练策略

- Encoder 模型使用 token 级交叉熵损失：$\mathcal{L} = \frac{1}{|\mathcal{D}|}\sum -\frac{1}{n}\sum y_i \log P_\Theta(y_i|x_i)$
- Encoder-Decoder 模型使用序列级交叉熵损失
- 训练分两步：先在 $\mathcal{D}_\mathcal{S}$ 上微调，再在 $\mathcal{D}_\mathcal{S} \cup \mathcal{D}_\mathcal{G}$ 上继续微调
- 使用源语言验证集做模型选择，确保真正的无监督设置

## 实验关键数据

### 主实验

**数据集**：SemEval-2016，包含英语（en）、西班牙语（es）、法语（fr）、荷兰语（nl）、俄语（ru）、土耳其语（tr）六种语言的餐饮评论。

**主要结果**（micro-F1，5 次实验平均）：

| 方法 | mBERT Avg | XLM-R Avg |
|------|-----------|-----------|
| Zero-shot | 45.68 | 60.35 |
| Equi-XABSA (前 SOTA) | 54.40 | 63.47 |
| LACA_LLaMA8 | 56.25 | 65.18 |
| LACA_Orca13 | 57.07 | 66.18 |
| **LACA_LLaMA70** | **57.29** | **66.35** |
| Supervised 上界 | 61.34 | 67.15 |

**扩展到更多 backbone 模型**（Avg F1）：

| Backbone | Zero-shot | +LACA_LLaMA70 | 提升 |
|----------|-----------|---------------|------|
| mBERT | 45.68 | 57.29 | +11.61 |
| XLM-R | 60.35 | 66.35 | +6.00 |
| mT5 | 59.77 | 65.90 | +6.13 |
| LLaMA 3.1 | 63.79 | 68.75 | +4.96 |

### 关键发现

1. **LACA 全面超越翻译方法**：在 mBERT 上超过 Equi-XABSA 1.50%，在 XLM-R 上超过 2.62%
2. **XLM-R + LACA 接近有监督性能**：在西班牙语上匹配有监督结果（71.89 vs 71.93），在荷兰语上甚至超过有监督结果（65.35 vs 64.28）
3. **英语中心的 Orca 2 13B 表现出色**：尽管主要针对英语，却几乎匹配多语言 LLaMA 3.1 70B，可能得益于其高级推理能力
4. **LLM 规模效应**：LLaMA 70B > Orca 13B ≈ LLaMA 8B，但更大模型推理更慢
5. **语言相似性影响效果**：与英语相似的西班牙语效果最好，俄语因语系差异较大效果略低
6. **Fine-tuned LLM 优于小型多语言模型**：LLaMA 3.1 作为 ABSA 模型本身表现最优

## 亮点与洞察

1. **核心创新——以生成代替翻译**：与翻译方法产生语义相似的数据不同，LLM 生成的数据语义多样性更强，增强了模型的泛化能力
2. **噪声标签的优雅处理**：通过让 LLM 根据（可能有噪声的）预测标签生成对齐的文本，而非直接使用噪声预测的 (原文, 预测标签) 对，有效减轻了预测噪声的影响
3. **无需翻译工具**：完全不依赖第三方翻译工具，适用于翻译质量低的低资源语言场景
4. **框架通用性强**：同时支持 encoder、encoder-decoder 和 decoder-only 三类模型架构
5. **实际可行性**：XLM-R + LACA 在无目标语言标注数据的情况下接近有监督水平

## 局限性 / 可改进方向

1. **依赖 LLM 在目标语言的生成质量**：对于 LLM 训练数据中覆盖较少的语言（如土耳其语），生成质量可能下降
2. **计算成本**：使用 LLaMA 70B 进行数据生成成本较高
3. **两阶段流程的误差传播**：第一阶段预测的质量直接影响 LLM 生成数据的标签准确性
4. **仅在餐饮评论领域验证**：未在其他领域（如电子产品、酒店等）进行验证
5. **未探索 LLM 的迭代优化**：可以考虑多轮生成-训练-预测的迭代方案进一步提升

## 相关工作与启发

- **跨语言 NLP 的数据增强新范式**：用 LLM 替代传统翻译工具，可推广至其他跨语言 NLP 任务
- **Self-training 的改进**：不同于直接将噪声预测作为伪标签，LACA 通过 LLM 生成与标签对齐的文本来"净化"伪标注数据
- **LLM 在标注数据稀缺场景的应用**：展示了 LLM 在低资源语言场景中的数据增强潜力
- Zhang et al. (2021) 的 ACS-Distill 方法为在无标注目标语言数据上蒸馏提供了思路
- Lin et al. (2024) 的 Equi-XABSA 关注类别不平衡问题，与 LACA 的不平衡处理策略互补

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-------------|------|
| 创新性 | 7 | LLM 数据增强替代翻译的思路新颖，但整体框架较直观 |
| 实验充分性 | 9 | 6种语言、5种backbone模型、多种LLM的全面对比 |
| 写作质量 | 8 | 结构清晰，实验分析详尽 |
| 实用价值 | 8 | 接近有监督水平且无需翻译工具 |
| 总分 | 8 | 扎实的工作，实验全面，方法实用 |
