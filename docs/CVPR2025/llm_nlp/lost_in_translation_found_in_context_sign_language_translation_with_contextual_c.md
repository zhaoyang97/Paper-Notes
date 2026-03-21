# Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues

**会议**: CVPR 2025  
**arXiv**: 见CVF  
**代码**: 待确认  
**领域**: NLP理解  
**关键词**: 待补充

## 一句话总结
> 基于摘要：Our objective is to translate continuous sign language into spoken language text. Inspired by the way human interpreters rely on context for accurate translation, we incorporate additional contextual cues together with the signing video, into a new translation framework. Specifically, besides visual

## 研究背景与动机
1. **领域现状**：本文研究的问题属于 NLP理解 方向。Our objective is to translate continuous sign language into spoken language text. Inspired by the way human interpreters rely on context for accurate translation, we incorporate additional contextual cues together with the signing video, into a new translation framework.
2. **现有痛点**：现有方法存在局限性——效率、精度或泛化性方面有改进空间。
3. **核心矛盾**：需要在效果与效率/泛化性之间找到更好的平衡。
4. **本文要解决什么？** 针对上述问题，作者提出了新方法。
5. **切入角度**：从新的技术视角或观察出发。
6. **核心idea一句话**：Specifically, besides visual sign recognition features that encode the input video, we integrate complementary textual information from (i) captions describing the background show, (ii) translation of

## 方法详解

### 整体框架
本文提出的方法概述如下（基于摘要信息）：

Specifically, besides visual sign recognition features that encode the input video, we integrate complementary textual information from (i) captions describing the background show, (ii) translation of previous sentences, as well as (iii) pseudo-glosses transcribing the signing. These are automatically extracted and inputted along with the visual features to a pre-trained large language model (LLM), which we fine-tune to generate spoken language translations in text form. Through extensive ablation studies, we show the positive contribution of each input cue to the translation performance.

### 关键设计

1. **核心模块**:
   - 做什么：解决上述痛点的关键技术组件
   - 核心思路：详见论文方法部分
   - 设计动机：提升性能或效率

### 损失函数 / 训练策略
详见论文全文（缓存不足，无法提取具体训练细节）。

## 实验关键数据

### 主实验
基于摘要的实验信息：We train and evaluate our approach on BOBSL -- the largest British Sign Language dataset currently available. We show that our contextual approach significantly enhances the quality of the translations compared to previously reported results on BOBSL, and also to state-of-the-art methods that we implement as baselines. Furthermore, we demonstrate the generality of our approach by applying it also to How2Sign, an American Sign Language dataset, and achieve competitive results.

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见论文 | - | - | - | - |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整模型 | 最优 | 完整方法 |
| 去除核心模块 | 下降 | 验证核心贡献 |

### 关键发现
- 本文方法在目标任务上取得显著改进
- 各核心模块均对最终性能有贡献

## 亮点与洞察
- 问题定义清晰，方法针对性强
- 核心设计思路可能可以迁移到相关场景

## 局限性 / 可改进方向
- 需要阅读全文才能深入分析方法细节和局限
- 泛化性和可扩展性有待进一步验证

## 相关工作与启发
- 本文在该领域的既有方法基础上做出了改进

## 评分
- 新颖性: ⭐⭐⭐ 基于摘要初评，有一定创新
- 实验充分度: ⭐⭐⭐ 需读全文验证
- 写作质量: ⭐⭐⭐ 基于摘要初评
- 价值: ⭐⭐⭐ 在该领域有贡献
