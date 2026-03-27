# CAPability: A Comprehensive Visual Caption Benchmark for Evaluation

**会议**: NeurIPS 2025
**arXiv**: [2502.14914](https://arxiv.org/abs/2502.14914)
**代码**: 有
**领域**: 多模态VLM / 评估
**关键词**: visual captioning, benchmark, multi-dimensional evaluation, VLM, KT gap

## 一句话总结
构建 CAPability——11K 标注的图片/视频描述评估基准，从 6 个视角 12 个维度评估 VLM 的描述能力，引入 KT（know-but-cannot-tell）指标衡量 VLM 在 QA 中已知但描述中遗漏的信息差距。

## 研究背景与动机
1. **领域现状**：视觉描述评估多用 CIDEr/METEOR 等自动指标，但这些指标与人类判断相关性低且只评估整体质量。
2. **现有痛点**：(1) 缺乏细粒度多维度评估；(2) 不区分"模型不知道"和"模型知道但没说"两种失败模式；(3) 静态和动态描述缺乏统一评估框架。
3. **本文要解决什么？** 提供多维度、支持 KT gap 分析的全面描述评估基准。

## 方法详解

### 关键设计
1. **6个视角12个维度**：Object（物体识别/属性/关系/数量）、Global（场景/情感）、Text（OCR）、Camera（角度/运动）、Temporal（时序）、Knowledge（常识）
2. **Precision + Hit 指标**：Precision 衡量描述准确性，Hit 衡量覆盖全面性
3. **KT 指标**：对比 QA 和 Caption 性能差异——如果模型在 QA 中能回答但在 Caption 中遗漏，说明存在"知道但不说"的能力差距

## 实验关键数据
| 模型 | Precision 最佳 | Hit 最佳 | KT Gap |
|------|-------------|---------|--------|
| GPT-4o | ✓ | | 显著 |
| Gemini-1.5-pro | | ✓ | 显著 |
| 开源 VLM | 中等 | 中等 | 更大 |

### 关键发现
- GPT-4o 在 Precision 上最好（描述准确），Gemini-1.5-pro 在 Hit 上最好（描述全面）
- 所有模型都存在显著的 KT gap，说明描述能力弱于 QA 能力

## 评分
- 新颖性: ⭐⭐⭐⭐ KT 指标是新颖贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 11K数据+多模型+多维度
- 写作质量: ⭐⭐⭐⭐ 评估框架描述清晰
- 价值: ⭐⭐⭐⭐ 对VLM描述能力评估有重要推动
