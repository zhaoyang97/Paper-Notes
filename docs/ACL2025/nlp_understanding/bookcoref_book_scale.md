# BookCoref: Coreference Resolution at Book Scale

**会议**: ACL 2025  
**arXiv**: [2507.12075](https://arxiv.org/abs/2507.12075)  
**代码**: [GitHub](https://github.com/sapienzanlp/bookcoref)  
**领域**: NLP理解  
**关键词**: 共指消解, 长文档, 书级别, 自动标注管线, 角色链接

## 一句话总结

提出首个书级别共指消解基准BookCoref，通过角色链接+LLM过滤+窗口扩展的自动标注管线，在50本完整小说上生成高质量银标注数据，平均文档长度超过20万tokens。

## 研究背景与动机

1. **领域现状**: 共指消解系统通常在短/中等长度文档上评估（OntoNotes平均467 tokens，LitBank截断到2000 tokens）。
2. **现有痛点**: 缺乏书级别基准，现有系统无法有效处理跨越数十万tokens的共指关系。LongtoNotes仅679 tokens/文档，MovieCoref仅9个文档。
3. **核心矛盾**: 人工标注长文本成本极高（需增量式阅读整本书），但自动标注系统（如Maverick）在长文本上性能急剧下降（Animal Farm仅36% CoNLL-F1）。
4. **本文要解决什么**: 设计可靠的自动标注管线，构建首个书级别共指消解训练和评估资源。
5. **切入角度**: 利用人物名单初始化共指簇，通过LLM过滤提高精度，再用窗口化CR模型扩展到代词和其他指称。
6. **核心idea一句话**: 通过角色链接→LLM过滤→窗口级CR扩展→分组窗口扩展四步管线，实现书级别的高质量自动共指标注。

## 方法详解

### 整体框架

BookCoref Pipeline四步流程：(1) 角色链接初始化显式提及簇；(2) LLM过滤去除错误链接；(3) 在小窗口中用CR模型扩展到代词等非显式提及；(4) 分组窗口二次扩展提高召回。

### 关键设计

1. **角色链接（Cluster Initialization）**: 基于LitBank微调ReLiK实体链接系统，将文本中的人名提及链接到预定义角色列表。相比简单模式匹配，F1从29.2%提升到44.5%。
2. **LLM过滤（Cluster Refinement）**: 使用Qwen2-7B验证每个提及是否正确关联角色（基于上下文判断），精度提升+5.2%，减少误差传播。
3. **窗口式CR扩展**: 将书划分为1500词窗口，在每个窗口中用Maverick扩展角色簇（加入代词、名词短语等），然后按角色名合并窗口间的簇。
4. **分组窗口扩展**: 将10个连续窗口合并为一组，使用Maverick_xl在更大上下文中二次扩展，解决跨窗口边界的共指遗漏。

### 评估策略

使用标准CR指标：MUC、B³、CEAFφ4和CoNLL-F1。在手工标注的BookCoref_gold（3本书：Animal Farm、Siddhartha、Pride and Prejudice）上评估管线和现有系统。

## 实验关键数据

### 管线评估（BookCoref_gold）

| 管线步骤 | 角色链接F1 | CoNLL-F1 |
|---------|-----------|----------|
| Pattern Matching | 29.2 | 17.9 |
| Character Linking | 44.5 | 34.2 |
| + LLM Filtering | 43.5 | 33.9 |
| + Window CR | 84.7 | 77.7 |
| + Grouping Step | **86.3** | **80.5** |

### 现有系统对比（BookCoref_gold）

| 系统 | CoNLL-F1 (全书) | CoNLL-F1 (分割版) |
|------|----------------|-----------------|
| BookNLP | 42.2 | 50.6 |
| Longdoc | 46.6 | 61.2 |
| Dual cache | 42.5 | 64.8 |
| Maverick_xl | 41.2 | - |

### 关键发现

- 管线达到80.5 CoNLL-F1，MUC 93.3接近人工标注一致性（96.1）
- 现有系统在全书设置下性能远低于分割版（差距高达20+ CoNLL-F1）
- BookCoref_silver包含10.8M tokens、968k mentions，规模远超OntoNotes
- 书级别的平均共指链距离达73,432 tokens，前所未有

## 亮点与洞察

- 管线设计精巧：高精度初始化+逐步扩展，避免误差传播
- LLM作为过滤器（而非生成器）是一个实用范式
- 角色链接→CR扩展的两阶段思路可推广到其他长文档NER/CR任务

## 局限性 / 可改进方向

- 仅标注角色（人物）的共指，不覆盖物品、地点等实体
- Gold标注仅3本书，评估集规模有限
- Pipeline中窗口边界可能导致跨窗口共指遗漏（分组步骤只部分缓解）
- 仅覆盖英语经典文学作品

## 相关工作与启发

- 与LitBank、MovieCoref互补，将CR推向真正的书级别
- 自动标注管线的方法论可用于构建其他长文档NLP资源
- 增量式CR系统（Longdoc、Dual cache）需要在BookCoref上进一步改进
- Pipeline的"高精度初始化→逐步扩展"策略对其他长文档标注任务有借鉴意义

## 技术细节补充

- 底层资源：Project Gutenberg全文 + Wikidata角色名 + LiSCU角色信息，共53本书平均27个角色
- 窗口大小：1500词（接近Maverick训练配置），分组大小G=10
- 标注工作量：3位专家标注约120小时，覆盖194,280词
- 标注者一致性：MUC score 96.1%，与LitBank (95.5%)和OntoNotes (83.0%)相比更高
- BookCoref_silver规模：50本书、10.8M tokens、968k mentions，是OntoNotes的6.75倍

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个书级别CR基准，管线设计新颖实用
- 实验充分度: ⭐⭐⭐⭐ 管线评估详尽，但现有系统对比略少
- 写作质量: ⭐⭐⭐⭐⭐ 论述清晰，方法与评估紧密结合
- 价值: ⭐⭐⭐⭐⭐ 填补书级别CR的资源空白，具有长期研究价值
