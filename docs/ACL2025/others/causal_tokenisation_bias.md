# Causal Estimation of Tokenisation Bias

**会议**: ACL 2025
**arXiv**: [2506.03149](https://arxiv.org/abs/2506.03149)
**代码**: [GitHub](https://github.com/pietrolesci/tokenisation-bias)
**机构**: University of Cambridge & ETH Zürich
**领域**: AI安全
**关键词**: tokenisation bias, causal inference, regression discontinuity, BPE, WordPiece, vocabulary, language model

## 一句话总结

本文首次将 tokeniser 选择对语言模型输出的影响定义为"分词偏差"(tokenisation bias)，并利用因果推断中的断点回归设计(RDD)来量化这一效应——发现当一个 subword 被纳入词表时，其对应字符串的概率最高可提升 17 倍（小模型），揭示分词是语言建模中一个被低估的关键设计选择。

## 研究背景与动机

### 现状

- 现代语言模型在 subword 序列上训练，但最终定义的是字符串上的概率分布
- 不同 tokeniser（不同词表）会将相同字符串映射为不同的 subword 序列（如 "hello" → ⟨he,llo⟩ 或 ⟨hello⟩）
- 理想情况下，tokeniser 的选择不应影响模型给字符串分配的概率；但实际中确实会影响

### 痛点

- 估计分词偏差面临根本性的因果推断难题：每个模型只用一个 tokeniser 训练，无法直接比较
- 不能简单比较词表内/外词的概率——词表是按频率等标准构建的，两组词有系统性差异
- 暴力方法（为每种可能的 tokeniser 分别训练模型）计算上不可行
- 现有研究知道 tokeniser 影响模型性能，但对"如何影响"缺乏定量理解

### 核心洞察

- Tokeniser 词表通常通过增量算法构建（BPE 按频率、WordPiece 按似然增益），这产生了一个 subword 排名
- 词表大小 K 是一个任意的截断点：排名前 K 的 subword 入选，后面的被排除
- 这个截断点构成了一个天然的"断点"(discontinuity)，可以用断点回归设计来估计因果效应
- 截断点附近的 subword 具有相似的特征，但"是否入选词表"是准随机分配的

## 方法详解

### 整体框架

将分词偏差形式化为因果效应：比较 subword v 在词表中时（观察值）和不在词表中时（反事实值）模型对其字符串的 log-probability 差异。利用 BPE/WordPiece 的排名机制和词表大小截断作为断点，应用 regression discontinuity design (RDD) 进行因果估计。

### 关键设计 1：因果框架定义

- 处理变量 W_v = 1{v ∈ T}：subword v 是否在 tokeniser 词表中
- 潜在结果 Y_v(w)：在处理条件 w 下，模型对字符序列 c_v 赋予的 log-probability
- 因果效应 τ_v = Y_v(1) - Y_v(0)：入选词表 vs 不入选对 log-probability 的影响
- 平均因果效应 ATE = E[Y(1) - Y(0)]（对截断点附近 subword 取平均）

### 关键设计 2：Regression Discontinuity Design (RDD)

- 运行变量 R_v：subword v 的排名（BPE 按合并频率，WordPiece 按似然增益）
- 截断点 c = K（词表大小）：R_v ≤ K 则 v 入选词表，R_v > K 则未入选
- 在截断点附近，排名相差 1 的 subword 特征几乎相同（频率、长度等），但入选/未入选是准随机的
- 使用局部线性回归拟合截断点两侧的 outcome，断点处的跳变即为因果效应
- 带宽 h 通过 Imbens-Kalyanaraman 方法自动选择

### 关键设计 3：实验控制与验证

- 训练多种规模的模型（100M、850M 参数）
- 使用 BPE 和 WordPiece 两种 tokeniser
- 在多种词表大小（1K、2K、4K、8K、16K、32K）上实验
- McCrary 密度检验验证排名没有被操纵
- Covariate balance 检验确认截断点两侧 subword 特征均衡
- Placebo 检验（在非截断点位置估计）确认效应不存在

### 训练策略

- 在固定数据集上从头训练 transformer 语言模型
- 每种词表大小 × tokeniser 组合训练独立模型
- 使用字符级 marginalization 将 subword 概率转换为字符串概率
- 跟踪训练过程中偏差的变化趋势

## 实验关键数据

### 主实验：不同规模模型的分词偏差

| 模型规模 | Tokeniser | 平均偏差 (nats) | 概率倍数 |
|---------|-----------|----------------|---------|
| 100M 参数 | BPE | 2.88 | ~17.8x |
| 100M 参数 | WordPiece | 2.51 | ~12.3x |
| 850M 参数 | BPE | ~1.0 | ~2.7x |
| 850M 参数 | WordPiece | ~1.0 | ~2.7x |

### 词表大小的影响

- 偏差在所有词表大小（1K-32K）上都一致存在
- 较小词表通常产生较大偏差
- 偏差随训练进行持续增长，未出现收敛趋势

### 关键发现

1. **分词偏差普遍存在**：在所有测试的模型规模、词表大小和 tokeniser 类型上都观察到了显著偏差
2. **小模型偏差更大**：100M 参数模型的偏差可达 2.88 nats（概率提升 17 倍），比 850M 模型的 ~1 nat 大得多
3. **偏差随训练增长**：偏差在训练过程中持续增加，说明模型越来越"依赖"词表中 subword 的整体性
4. **BPE 偏差略大于 WordPiece**：不同 tokeniser 产生的偏差有差异但方向一致
5. **因果性验证**：McCrary 检验、covariate balance、placebo 检验均通过，确认是因果效应而非混淆

## 亮点与洞察

- **方法论创新**：首次将因果推断的 RDD 方法应用于语言建模中的分词问题，非常优雅
- **定量揭示被忽视的问题**：虽然"分词会影响模型"是直觉认知，但本文首次给出了精确的因果量化
- **17 倍概率差**：这个数字非常惊人——仅仅因为一个 subword 是否在词表中，其字符串的概率就能差 17 倍
- **对实践的警示**：在多语言、专业领域等场景中，tokeniser 设计可能比模型架构选择更重要

## 局限性 / 可改进方向

- RDD 只能估计截断点附近（marginal subwords）的局部因果效应，不能推广到词表核心区域
- 实验模型规模有限（最大 850M），未在 7B+ 模型上验证偏差是否进一步缩小
- 仅考虑 subword 是否入选词表这一维度的偏差，未分析 tokeniser 内部排序的影响
- 未提出减轻分词偏差的具体方法（如改进的训练策略或正则化）
- 多语言场景下的分词偏差分析是重要的未来方向

## 相关工作与启发

- Pimentel & Meister (2024) 提出了 subword-to-character probability marginalization 的理论基础
- Rust et al. (2021) 和 Ali et al. (2024) 等实证研究表明 tokeniser 影响模型性能，但未给出因果证据
- 启发：tokeniser 设计应该被视为与模型架构、训练数据同等重要的设计选择

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 因果推断 + 分词的跨领域创新非常独到
- **技术深度**: ⭐⭐⭐⭐⭐ — RDD 的应用严谨完整，验证充分
- **实用性**: ⭐⭐⭐ — 主要是诊断性工具，暂未提出解决方案
- **实验充分度**: ⭐⭐⭐⭐ — 多规模、多 tokeniser、完整因果验证
