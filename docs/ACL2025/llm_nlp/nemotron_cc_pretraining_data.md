# Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset

**会议**: ACL 2025  
**arXiv**: [2412.02595](https://arxiv.org/abs/2412.02595)  
**代码**: [https://github.com/NVIDIA/NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator)  
**领域**: LLM NLP  
**关键词**: 预训练数据, Common Crawl, 数据质量, 合成数据, 长期训练  

## 一句话总结
Nemotron-CC 通过分类器集成、合成数据改写和减少启发式过滤三种策略，从 Common Crawl 构建了 6.3T token 的长期预训练数据集，在 15T token 训练中超越 Llama 3.1 8B。

## 研究背景与动机
1. **领域现状**：FineWeb-Edu 和 DCLM 等数据集通过激进的模型过滤提升了基准性能，但丢弃了 90% 的数据
2. **现有痛点**：激进过滤导致数据量不足，DCLM 仅 1T 唯一 token，FineWeb-Edu 仅 0.2T——在 15T token 长训练中必须大量重复
3. **核心矛盾**：数据质量与数据数量之间的权衡——高质量过滤牺牲了训练所需的多样性和数据量
4. **本文要解决什么**：在保持甚至提升质量的前提下，大幅增加可用唯一 token 数量
5. **切入角度**：多分类器集成扩大高质量数据召回 + 合成改写扩展 token 数量 + 对高质量数据取消启发式过滤
6. **核心idea一句话**：用分类器集成 + 合成数据 + 减少过滤来同时提升数据质量和数量

## 方法详解

### 整体框架
Common Crawl HTML → JusText 提取（比 Trafilatura 多 28.6% 高质量 token）→ 语言过滤+全局去重 → 三分类器集成打分 → 质量分桶（20 桶 → 5 级）→ 高质量数据不做启发式过滤 → 合成数据生成（低质量改写 + 高质量多样化）→ 6.3T token 数据集。

### 关键设计
1. **分类器集成 (Classifier Ensembling)**:
   - 三个分类器：FineWeb-Edu classifier (Nemotron-340B 标注)、Mixtral classifier (Mixtral-8x22B 标注)、DCLM classifier (fastText)
   - 每个分类器按排名映射到 0-19 分，取三者最大值作为最终分数
   - 设计动机：单一分类器召回率仅约 10%，集成可以扩大高质量数据的覆盖面

2. **质量分级 (Quality Labeling)**:
   - 将 20 个细粒度桶通过退火实验分组为 5 级（High/Medium-High/Medium/Medium-Low/Low）
   - 退火实验：在 70% 训完的 8B 模型上用 50B token 继续训练，评估每个桶的下游效果
   - 设计动机：让质量标签直接与下游任务性能对齐，而非仅依赖分类器分数

3. **合成数据生成 (Synthetic Data)**:
   - 低质量数据：用 Wikipedia 风格 prompt 改写，减少噪声和错误（336B token）
   - 高质量数据：用四种 prompt 生成多样变体——QA 对、蒸馏、知识提取、知识列表（1.5T token）
   - 使用 Mistral NeMo 12B (FP8) 生成，总计 1.9T 合成 token
   - 设计动机：为高质量数据创造"新的唯一 token"以避免多 epoch 的边际递减

4. **减少启发式过滤**:
   - 对高质量数据（分类器高分）不应用传统启发式过滤（C4 filter/Gopher filter/PPL filter）
   - 实验验证移除过滤反而提升了高质量 token 产量 (+57.4%) 而不损失质量

## 实验关键数据

### 主实验（8B 模型，1T token 训练）
| 数据集 | MMLU | Avg (10任务) | 唯一token |
|--------|------|-------------|----------|
| FineWeb-Edu | 42.9 | 53.2 | 0.2T |
| DCLM | 53.4 | 57.0 | 1.0T |
| Nemotron-CC | 53.0 | 57.8 | 4.4T |
| **Nemotron-CC-HQ** | **59.0** | **60.1** | 0.6T |

### 长期训练（8B 模型，15T token 训练）
| 模型 | MMLU | ARC-C | Avg |
|------|------|-------|-----|
| Llama 3.1 8B | 65.3 | 55.0 | 64.2 |
| **Ours** | **70.3** | **58.1** | **64.7** |

### 关键发现
- 分类器集成 + 合成数据在短期训练上就能超 DCLM 5.6 MMLU
- 4x 更多唯一 token 在 15T 长训练中优势明显——超越同样 15T 训练的 Llama 3.1
- JusText 比 Trafilatura 多提取 28.6% 高质量 token
- 对高质量数据移除启发式过滤提升 token 产量且不损质量

## 亮点与洞察
- "从静态启发式管线转向学习型飞轮"的理念很有前瞻性——随着模型变强，数据质量也会提升
- 退火实验做质量分级是一个实用的方法——直接用下游性能定义数据质量
- 合成数据不是生成新知识而是改写/蒸馏/多样化，降低了幻觉风险
- 数据集开源在 Common Crawl 上，有很高的复用价值

## 局限性 / 可改进方向
- 只覆盖英语 Common Crawl，多语言版本未涉及
- 合成数据使用 12B 模型生成，更强的模型可能产出更好的合成数据
- 15T 训练实验仅在 8B 模型上验证，更大模型的效果未知
- 去重策略可能仍然不够激进，存在语义级近似重复

## 相关工作与启发
- **vs DCLM**: DCLM 用单一 fastText 分类器，仅保留约 10% 数据；Nemotron-CC 用三分类器集成保留更多高质量数据
- **vs FineWeb-Edu**: FineWeb-Edu 激进过滤到 0.2T 唯一 token，Nemotron-CC 保留 4.4T
- **vs DSIR/QuRating**: 专注数据选择但不做合成扩展，Nemotron-CC 同时做选择和合成

## 评分
- 新颖性: ⭐⭐⭐⭐ 分类器集成+合成数据+减少过滤的组合策略在预训练数据领域有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 1T和15T训练+详细消融+超越Llama 3.1
- 写作质量: ⭐⭐⭐⭐ 清晰有条理，表格丰富
- 价值: ⭐⭐⭐⭐⭐ 6.3T开源数据集+超越Llama 3.1的实验结果，极高实用价值
