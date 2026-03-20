# Empower Words: DualGround for Structured Phrase and Sentence-Level Temporal Grounding

## 基本信息
- **arXiv**: 2510.20244
- **会议**: NeurIPS 2025
- **作者**: Minseok Kang, Minhyeok Lee, Minjung Kim, Donghyeong Kim, Sangyoun Lee
- **领域**: Video Temporal Grounding / Video-Language Alignment
- **任务**: Moment Retrieval, Highlight Detection

## 一句话总结
论文指出现有视频时序定位模型在跨模态注意力中往往过度依赖句末 [EOS] token 的全局语义、忽视词级局部信号，提出 DualGround 双分支架构，将句子级全局语义与短语级局部语义显式解耦建模，在 QVHighlights 和 Charades-STA 上实现 Moment Retrieval 与 Highlight Detection 的 SOTA。

## 背景与动机
Video Temporal Grounding 需要在长视频中定位与自然语言查询对齐的时间区间，通常包含两类子任务：
- **Moment Retrieval**：找出对应片段；
- **Highlight Detection**：识别关键高亮时刻。

现有方法虽然使用 CLIP、InternVideo2 等强视觉语言 backbone，但常把所有文本 token 一视同仁处理。作者通过控制实验发现，这会导致模型：
- 过度依赖 [EOS] 的全局句义；
- 对单词/短语级的定位信号利用不足；
- 难以实现细粒度 temporal alignment。

## 核心问题
如何同时保留句子级整体语义和词/短语级局部语义，使模型在视频时序定位中既能做粗粒度匹配，又能做细粒度边界对齐？

## 方法详解

### 1. 语义角色解耦
DualGround 的核心思想是把文本语义按结构拆开：
- **句子级路径**：专门处理 [EOS] token 所承载的全局语义；
- **短语级路径**：将单词 token 聚类为 phrase-level 单元，用于局部定位。

### 2. Token-role aware cross-modal interaction
论文为不同语义角色设计不同的跨模态交互策略：
- 全局路径强调视频整体与句子语义的一致性；
- 局部路径强调视频片段与短语语义的细粒度对应。

这比统一注意力机制更符合任务结构。

### 3. Joint modeling 框架
DualGround 不是简单地把两路特征拼起来，而是联合优化：
- 提升全局 sentence-level alignment；
- 同时增强 phrase-aware temporal grounding；
- 让上下文建模更具结构性和可解释性。

## 实验结论
根据摘要：
- 在 QVHighlights 和 Charades-STA 上，Moment Retrieval 与 Highlight Detection 均达到 SOTA；
- 证明显式解耦 global/local semantics 对视频语言时序对齐有效。

## 亮点
1. **问题诊断精准**：指出 [EOS]-driven global semantics 是当前 VTG 的隐性偏置。
2. **结构设计合理**：全局与局部语义分路处理，很符合 grounding 本质。
3. **统一解决双任务**：同时兼顾 MR 与 HD，而非只优化一个子任务。
4. **解释性较强**：比黑盒 attention 更容易分析模型为何定位成功/失败。

## 局限性
1. 短语聚类策略的质量可能影响整体效果。
2. 方法针对 VTG 设计，向开放式视频问答或 agent 规划迁移仍需验证。
3. 增加了结构复杂度，对轻量模型部署可能不够友好。

## 与相关工作的对比
- 相比统一 token attention 方法：DualGround 明确利用 token 语义角色差异。
- 相比纯全局文本表示方法：能更好支持细粒度边界定位。
- 相比短语级 grounding 方法：又保留了全局句义，避免局部过拟合。

## 启发
- 这类 global/local 解耦思想可迁移到 Video-LLM 的时空推理 token 设计。
- 对多模态 agent 理解复杂指令中的动作短语与目标短语也有参考价值。
- 可进一步与 CoT 或工具调用结合，形成可解释 temporal reasoning pipeline。

## 评分
- 新颖性：★★★★☆
- 技术深度：★★★★☆
- 实验完整度：★★★★☆
- 实用价值：★★★★☆