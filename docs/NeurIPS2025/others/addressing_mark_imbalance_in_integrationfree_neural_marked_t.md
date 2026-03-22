Title: Addressing Mark Imbalance in Integration-free Neural Marked Temporal Point Processes
Authors: Sishun Liu, Ke Deng, Yongli Ren, Yan Wang, Xiuzhen Zhang
Venue: NeurIPS 2025 (poster)
arXiv: 2510.20414

One-line summary
- Proposes a thresholding approach for marked temporal point processes to correct highly imbalanced mark distributions by learning thresholds that normalize mark probabilities by priors; predict mark first then time.

Background & Motivation
- Marked Temporal Point Processes (MTPP) predict the next event's mark and time. Real-world mark distributions are often heavily imbalanced, hurting rare-mark prediction.

Core idea / Method
- Learn per-mark thresholds to rescale mark probabilities relative to each mark's prior, optimizing for mark prediction rather than relying solely on raw probability outputs.
- Pipeline: predict mark first (with thresholded normalization), then sample/estimate time using an integration-free neural MTPP design that avoids costly improper integrals.

Experiments (from abstract)
- Extensive experiments on real-world datasets show improved performance on next-event mark and time prediction compared to baselines. Code: https://github.com/undes1red/IFNMTPP.

Highlights
- Simple thresholding trick targeted at class-imbalance in discrete marks.
- Integration-free time estimation reduces computation overhead.

Limitations / Open questions
- Abstract-level description; would check experiments for per-mark gains on rare classes and sensitivity to threshold learning.
- Need to verify scalability to very large mark vocabularies.

Rating: 3.5 / 5

Notes
- Follow-up: inspect released code for threshold learning details and default priors; consider relevance to event prediction tasks in multimodal streams.
# Addressing Mark Imbalance in Integration-free Neural Marked Temporal Point Processes

## 基本信息
- **arXiv**: 2510.20414
- **会议**: NeurIPS 2025 Poster
- **作者**: Sishun Liu, Ke Deng, Yongli Ren, Yan Wang, Xiuzhen Zhang
- **代码**: https://github.com/undes1red/IFNMTPP
- **领域**: Marked Temporal Point Process / Long-tail Event Modeling

## 一句话总结
论文针对现实事件流中常见的 mark 类别长尾失衡问题，提出基于先验归一化概率的阈值学习策略，并设计 integration-free 的神经 MTPP 架构，先预测 mark 再预测 time，在避免昂贵数值积分的同时显著提升稀有事件的 mark 与到达时间预测性能。

## 背景与动机
MTPP 用于建模带类别标签的事件流，例如：
- 用户行为序列；
- 金融交易事件；
- 医疗监测告警；
- 视频/多模态交互事件。

但现实中 mark 分布往往严重失衡：头部类频繁，尾部类稀少。现有研究多数直接基于 mark probability 做预测，忽视了 long-tail 带来的决策偏置，导致 rare mark 预测能力很弱。

## 核心问题
如何在不引入高昂积分计算成本的前提下，显式缓解 MTPP 中 mark imbalance 对 next-event mark/time prediction 的负面影响？

## 方法详解

### 1. 基于先验校正的阈值学习
论文的核心不是直接取最大 mark probability，而是：
- 用 mark 的预测概率除以该 mark 的先验概率；
- 通过学习阈值来调整决策边界；
- 使稀有类别不会因先验过小而系统性被压制。

这本质上把“类别频率偏置”显式地纳入推断过程。

### 2. 先预测 mark，再预测 time
作者采用两阶段预测：
- 先确定下一个事件的 mark；
- 再基于 mark 预测到达时间。

这种顺序更贴近“稀有事件识别优先”的任务目标，也利于后续时间建模聚焦在更明确的条件上。

### 3. Integration-free 神经 MTPP
为避免传统 TPP 中常见的数值不当积分开销，作者提出 integration-free 模型：
- 支持高效时间采样；
- 能估计 mark probability；
- 训练和推理成本更可控。

## 实验结论
根据摘要：
- 在多个真实数据集上优于多种 baseline；
- 对 next event 的 mark 与 time prediction 都有提升；
- 尤其在 rare mark 上更有优势。

## 亮点
1. **抓住实际痛点**：MTPP 长尾失衡在真实应用里非常常见，却常被忽略。
2. **方法直接有效**：阈值学习与先验校正思路简洁。
3. **兼顾效率**：integration-free 设计避免昂贵积分计算。
4. **任务定义清晰**：既提升类别预测，也保持时间预测能力。

## 局限性
1. 主要改进聚焦监督预测层面，未必解决表示层中的类间重叠问题。
2. 阈值学习对不同数据集的稳定性和可迁移性需进一步验证。
3. 与更强的多模态事件编码器结合的潜力尚未展开。

## 与相关工作的对比
- 相比标准 neural MTPP：更显式地处理 mark imbalance。
- 相比重采样/重加权：方法直接作用于预测决策规则，更贴近最终目标。
- 相比依赖积分的 TPP：计算更高效，更利于大规模应用。

## 启发
- 可与多模态 TPP benchmark（如 DanmakuTPPBench）结合，检验 long-tail 多模态事件建模。
- 对 agent 交互日志、驾驶事件流、医疗预警等稀有事件场景很有实际意义。
- 可以探索把该 prior-corrected thresholding 思路迁移到 VLM 的长尾动作/事件识别。

## 评分
- 新颖性：★★★★☆
- 技术深度：★★★☆☆
- 实用价值：★★★★★
- 实验说服力：★★★★☆
