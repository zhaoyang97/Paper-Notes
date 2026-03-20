# LangSAMP: Language-Script Aware Multilingual Pretraining

**会议**: ACL 2025  
**arXiv**: [2409.18199](https://arxiv.org/abs/2409.18199)  
**代码**: [GitHub](https://github.com/cisnlp/LangSAMP)  
**领域**: llm_nlp  
**关键词**: multilingual pretraining, language embedding, script embedding, crosslingual transfer, language neutrality  

## 一句话总结
提出 LangSAMP 方法，在多语言预训练中将语言和文字系统 (script) embedding 添加到 Transformer 输出端（而非输入端），使模型主干学到更语言中立的表示，在 500+ 语言的零样本跨语言迁移中一致优于基线。

## 研究背景与动机
1. **领域现状**：多语言预训练模型（如 XLM-R、mBERT）是通用文本编码器，支持零样本跨语言迁移。早期模型（XLM）使用语言 embedding，但最新模型已放弃使用。
2. **现有痛点**：取消语言 embedding 后，token 表示需要同时编码语言/文字特定信息和语义信息，导致表示不够语言中立，影响跨语言迁移效果。
3. **核心矛盾**：使用语言 embedding 会使模型依赖语言 ID 输入（推理时不便）；不使用则损害表示的语言中立性。
4. **本文要解决什么？** 如何在不影响推理灵活性的前提下，利用语言和文字 embedding 提升多语言表示的质量？
5. **切入角度**：把语言/文字 embedding 放在 Transformer 输出端而非输入端——预训练时用于辅助 MLM 解码，微调时完全不需要，模型主干可照常作为通用编码器。
6. **核心 idea 一句话**：在输出端加语言/文字 embedding 分担解码负担，让 Transformer 主干学到更语言中立的表示。

## 方法详解

### 整体框架
基于 XLM-R 做持续预训练，在 Glot500-c 语料（500+ 语言、30 种文字系统）上训练。输入通过 Transformer 块后得到隐藏表示 $\boldsymbol{h}_i$，加上对应的语言 embedding $\boldsymbol{E}^{Lang}_l$ 和文字 embedding $\boldsymbol{E}^{Script}_s$，形成 $\boldsymbol{o}_i = \boldsymbol{h}_i + \boldsymbol{E}^{Lang}_l + \boldsymbol{E}^{Script}_s$，再送入 MLM head 预测被 mask 的 token。微调时直接用 $\boldsymbol{h}_i$ 做下游任务，不需要语言/文字 ID。

### 关键设计
1. **输出端嵌入（核心创新）**:
   - 做什么：把语言和文字 embedding 加在 Transformer 输出端而非输入端
   - 核心思路：$\boldsymbol{o}_i = \boldsymbol{h}_i + \boldsymbol{E}^{Lang}_l + \boldsymbol{E}^{Script}_s$，辅助 MLM 解码时提供语言/文字"提示"
   - 设计动机：传统做法（XLM）在输入端加语言 embedding，导致下游任务也需要语言 ID；放在输出端则只有预训练需要，微调时模型主干独立工作

2. **语言和文字双重 embedding**:
   - 做什么：同时维护语言 embedding ($\mathbb{R}^{610 \times 768}$) 和文字 embedding ($\mathbb{R}^{30 \times 768}$)
   - 核心思路：语言捕获词汇/语法差异，文字捕获字形/编码差异，两者互补
   - 设计动机：消融显示两者组合效果最好，单用语言 embedding 对检索任务帮助更大，单用文字 embedding 对序列标注帮助更大

3. **副产物：语言/文字 embedding 用于源语言选择**:
   - 做什么：学到的语言 embedding 之间的相似度反映语言类型学特征
   - 核心思路：用语言 embedding 相似度选择跨语言迁移的源语言
   - 设计动机：传统依赖语言学先验知识选源语言，embedding 相似度提供数据驱动的替代方案

## 实验关键数据

### 主实验（零样本跨语言迁移，表 1）
| 任务 | LangSAMP (all) | Baseline (all) | 提升 |
|------|----------------|-----------------|------|
| SR-B (句检索) | 45.1 | 42.9 | +2.2 |
| SR-T (Tatoeba) | 71.1 | 69.7 | +1.4 |
| Taxi1500 (分类) | 53.4 | 50.3 | +3.1 |
| SIB200 (分类) | 75.9 | 75.0 | +0.9 |
| NER | 62.6 | 62.2 | +0.4 |
| POS | 71.6 | 71.5 | +0.1 |

### 消融实验
| 配置 | SR-B (all) | Taxi1500 (all) |
|------|------------|----------------|
| Vanilla | 23.2 | 28.4 |
| + Lang only | 24.5 | 28.5 |
| + Script only | 23.9 | 28.2 |
| + Lang + Script | 24.9 | 30.3 |

### 关键发现
- 低资源尾部语言受益更大：SR-B 上 tail 提升 2.6% vs head 0.7%
- 非拉丁文字语言提升更大：SR-B 上 non-Latin +2.3% vs Latin +2.1%
- 语言 embedding 捕获了语系关系——UMAP 可视化中同语系自然聚类
- 文字 embedding 距离与 Unicode 块关系一致
- 语言中立性提升：跨语言 pairwise cosine 相似度提高

## 亮点与洞察
- 输出端嵌入的设计非常优雅——预训练时用、微调时不用，零额外推理成本，保持了通用编码器的无语言 ID 特性
- 语言/文字 embedding 作为副产物可用于源语言选择，这种"一石二鸟"的设计很实用
- 覆盖 500+ 语言的评估非常全面，对低资源语言研究有直接贡献

## 局限性 / 可改进方向
- 仅在 encoder-only 模型（XLM-R）上验证，未扩展到 decoder-only 或 encoder-decoder
- 需要可靠的语言 ID 检测作为预训练输入（错误的语言标签可能引入噪声）
- 绝对提升幅度在 NER/POS 等序列标注任务上较小
- 计算成本：需要在完整 Glot500-c 上重新持续预训练（4 周 4×RTX6000）

## 相关工作与启发
- **vs XLM**: XLM 在输入端加语言 embedding，微调时也需要语言 ID；LangSAMP 仅预训练时需要
- **vs XLM-R/mBERT**: 完全不用语言 embedding，表示的语言中立性受限
- **vs Adapter 方法**: Adapter 在微调时增加参数，LangSAMP 零额外推理参数

## 评分
- 新颖性: ⭐⭐⭐⭐ 输出端嵌入的想法简单但有效，设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 500+语言、6 个任务、消融+分析全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法简洁
- 价值: ⭐⭐⭐⭐ 对多语言 NLP 社区有直接贡献，特别是低资源语言
