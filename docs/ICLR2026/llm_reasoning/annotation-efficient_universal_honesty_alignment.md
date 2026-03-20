# Annotation-Efficient Universal Honesty Alignment

**会议**: ICLR 2026  
**arXiv**: [2510.17509](https://arxiv.org/abs/2510.17509)  
**代码**: 有（GitHub 链接）  
**领域**: LLM推理  
**关键词**: honesty alignment, confidence calibration, self-consistency, annotation efficiency, LLM trustworthiness  

## 一句话总结
提出 EliCal（先激发后校准）两阶段框架，先用无标注的 self-consistency 信号教 LLM 表达内部置信度，再用极少量正确性标注（仅 1k 个，占 0.18%）进行校准，在 HonestyBench（560K 训练 + 70K 评估）上达到接近全量标注 98% 的诚实性对齐性能，并在未见 MMLU 任务上泛化优于仅校准基线。

## 研究背景与动机
1. **领域现状**：LLM 诚实性对齐（honesty alignment）要求模型准确认识自己的知识边界并表达校准后的置信度。现有方法分两类：免训练的置信度估计（token 概率、self-consistency）和基于训练的校准（需正确性标注）。
2. **现有痛点**：基于训练的方法效果更好，但实现跨任务的"通用"诚实性对齐需要大规模正确性标注——对每个问题都需要 ground truth 答案来判断模型是否回答正确。这成本极高。
3. **核心矛盾**：正确性标注同时承担两个角色——(1) 教模型表达置信度；(2) 将置信度与正确性校准。如果第一个角色可以用更廉价的信号实现，那么只需少量标注做第二步。
4. **本文要解决什么？** 如何用最少的正确性标注实现高质量的诚实性对齐？
5. **切入角度**：观察到 self-consistency 置信度（多次采样的语义一致性比例）与实际正确率高度相关，且是免费生成的。用它先教模型表达置信度（Stage 1），再用少量标注校准（Stage 2）。
6. **核心idea一句话**："先激发，后校准"——用 self-consistency 做预训练级别的置信度学习，用极少标注做微调级别的校准。

## 方法详解

### 整体框架
EliCal 分两阶段：Stage 1 用 560K 问题的 self-consistency 信号训练模型表达内部置信度（无需 ground truth）；Stage 2 用仅 1K 个有正确性标注的样本校准置信度。模型架构：冻结 LLM 参数 + LoRA + 线性头输出置信度分数。

### 关键设计

1. **Stage 1: Confidence Elicitation（置信度激发）**:
   - 做什么：训练模型一次性输出其内部置信度，替代昂贵的多次采样一致性估计
   - 核心思路：对每个问题采样 k=20 个回答，计算与 greedy 回答的语义一致性比例作为 self-consistency 目标。用 MSE 损失训练 LoRA + 线性头来预测这个目标
   - 设计动机：self-consistency 与真正正确率高度相关（Figure 2），且是免费信号——不需要 ground truth。这一步教会模型"感知自己有多确定"

2. **Stage 2: Confidence Calibration（置信度校准）**:
   - 做什么：用少量标注数据将 Stage 1 学到的置信度校准到真实正确率
   - 核心思路：从 Stage 1 的参数出发，继续用 MSE 损失微调，但目标改为 Accuracy（基于 ground truth 的正确率）。仅需 ~1K 个标注样本
   - 设计动机：类似预训练-微调范式——Stage 1 已经学到了置信度的基本表达能力，Stage 2 只需少量标注做"最后一公里"的校准

3. **HonestyBench 基准**:
   - 做什么：构建大规模诚实性对齐 benchmark
   - 核心思路：整合 10 个 free-form QA 数据集，560K 训练 + 38K in-domain 评估 + 33K OOD 评估。每个模型-问题对标注 20 个采样回答的一致性和正确性。覆盖 3 个 LLM（Qwen-7B/14B, Llama-8B）
   - 设计动机：此前的诚实性研究只在小数据集上做 in-domain 评估，缺乏通用性测试

### 损失函数 / 训练策略
两个阶段都用 MSE 损失。冻结 LLM 参数，只训练 LoRA 和线性头。Stage 1 在全量 560K 上训练（self-consistency 目标），Stage 2 在 1K 标注样本上微调（correctness 目标）。

## 实验关键数据

### 主实验

| 方法 | 标注量 | In-Domain 性能 | OOD 性能 |
|------|--------|---------------|----------|
| 最佳免训练方法 (Self-Consistency) | 0 | 基线 | 基线 |
| Cal-Only（全量标注） | 560K | Upper bound | - |
| EliCal + Cal-Only (全量) | 560K | Upper bound（比免训练高 17%+） | - |
| **EliCal (仅 1K 标注)** | **1K (0.18%)** | **~98% of upper bound** | **显著优于 Cal-Only** |
| Cal-Only (仅 1K 标注) | 1K | 显著低于 EliCal | 较差 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Cal-Only (从头校准) | 需要 >>1K 标注 | 没有 elicitation 阶段，大量标注才能收敛 |
| EliCal 1K | ~98% upper bound | 预训练-微调范式极大提升标注效率 |
| MMLU (OOD) | EliCal >> Cal-Only | 泛化到未见任务 |

### 关键发现
- EliCal 仅用 0.18% 的标注量达到 98% 的最佳性能，标注效率提升超 500 倍
- 在 MMLU（完全 OOD 的任务）上，EliCal 一致优于 Cal-Only——说明 self-consistency 预训练提供了更好的泛化基础
- Self-consistency 置信度与正确率的相关性在多个模型上都很高，但模型普遍过度自信——这正是需要 Stage 2 校准的原因
- 全量标注时 EliCal 和 Cal-Only 性能持平（都达到 upper bound），但少标注时 EliCal 显著更好

## 亮点与洞察
- **预训练-微调范式迁移到置信度学习**：用廉价信号做"预训练"、用昂贵标注做"微调"的思路非常优雅，具有方法论上的泛用性
- **Self-consistency 作为免费监督信号的价值**：虽然 self-consistency ≠ correctness（模型可能一致地错），但它是一个足够好的代理信号来教会模型"表达置信度"
- **HonestyBench 基准的实用价值**：560K 规模、三个模型、10 个数据集的标注资源，为社区提供了标准化的诚实性评估平台

## 局限性 / 可改进方向
- Self-consistency 需要多次采样（k=20）来生成训练信号，虽然只在 Stage 1 构建数据时需要，推理时是 one-shot
- 仅在 free-form QA 上验证，对 reasoning/math 等需要更精确置信度的任务未覆盖
- 线性头 + LoRA 的架构选择是否最优尚未充分探索
- 校准后的置信度是否在 RAG 触发等下游应用中真正有效尚需验证

## 相关工作与启发
- **vs Cal-Only (Zhang et al., 2024)**: Cal-Only 在少标注时性能大幅下降，EliCal 通过 elicitation 阶段解决了这个问题
- **vs 训练免方法 (Self-Consistency)**: EliCal 的 one-shot 推理比 self-consistency 的 20 次采样高效得多，且校准后更准确
- **vs R-Tuning (Yang et al., 2023)**: R-Tuning 只在单数据集上训练和测试，EliCal 瞄准跨任务的通用诚实性

## 评分
- 新颖性: ⭐⭐⭐⭐ 两阶段框架的设计理念有创新，免标注的 elicitation 阶段很巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 大规模 benchmark、多模型、in-domain+OOD、详细的标注效率分析
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，形式化完整，叙事流畅
- 价值: ⭐⭐⭐⭐⭐ HonestyBench + EliCal 构成了诚实性对齐方向的重要基础设施和方法论
