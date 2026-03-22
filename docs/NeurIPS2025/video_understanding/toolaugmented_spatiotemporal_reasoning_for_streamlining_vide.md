# Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task

## 基本信息
- **arXiv**: 2512.10359
- **会议**: NeurIPS 2025 Main Track
- **作者**: Sunqi Fan, Jiashuo Cui, Meng-Hao Guo, Shuojin Yang
- **代码**: https://github.com/fansunqi/VideoTool
- **领域**: VideoQA / MLLM / Tool Use / Spatiotemporal Reasoning

## 一句话总结
论文为复杂 VideoQA 提出一套轻量但可扩展的 Video Toolkit，并设计 STAR（Spatiotemporal Reasoning Framework）来调度时间工具与空间工具的调用顺序，逐步定位视频关键区域，显著增强 GPT-4o 的时空推理能力，在 VideoMME 上提升 8.2%，在 LongVideoBench 上提升 4.6%。

## 背景与动机
现有 MLLM 做 VideoQA 的主要难点是：
- 难同时建模帧内空间关系与跨帧因果变化；
- 工具增强方案常工具太杂、调用无序；
- 容易出现 toolchain shortcut，模型不是真正推理，而是投机性调用。

作者的核心观点是：**视频推理不仅需要工具，还需要严格的时空调度策略。**

## 核心问题
如何设计一个既足够强大又不会失控的工具增强框架，使 MLLM 在视频问答中循序渐进地完成 temporal reasoning 与 spatial grounding？

## 方法详解

### 1. Comprehensive and Extensible Video Toolkit
作者构建一套轻量视频工具集，用于辅助 MLLM：
- 覆盖时序分析与空间分析；
- 注重工具数量与多样性的平衡；
- 保证增强能力同时避免系统过重。

### 2. STAR：Spatiotemporal Reasoning Framework
STAR 的关键不是简单让模型“自由调用工具”，而是：
- 战略性安排 temporal tools 与 spatial tools 的执行顺序；
- 先粗定位时间，再细定位空间，或按任务需求交替推进；
- 逐步缩小视频中关键区域和关键时段。

这比无约束 tool-use 更接近真实推理流程。

### 3. 避免 Toolchain Shortcut
作者明确关注一个很实际的问题：工具链捷径。STAR 通过控制调用序列来降低模型跳过关键推理步骤的风险，使工具调用更可解释。

## 实验结论
- 在 VideoMME 上提升 8.2%；
- 在 LongVideoBench 上提升 4.6%；
- 说明轻量工具配合合理调度，足以显著增强强基座模型的 VideoQA 推理能力。

## 亮点
1. **非常贴近 agent 设计**：工具集 + 调度框架本质上就是视频分析 agent。
2. **关注顺序控制**：不是盲目堆工具，而是做 reasoning orchestration。
3. **结果清晰直接**：对 GPT-4o 增益显著。
4. **可扩展性强**：工具集和框架都适合后续扩展。

## 局限性
1. 性能依赖底座 MLLM 的原生能力。
2. 工具调度策略在更多任务上的泛化还需验证。
3. 可能增加系统复杂度和推理时延。

## 与相关工作的对比
- 相比纯 prompt-based VideoQA：STAR 显式引入外部工具与推理流程。
- 相比无约束 tool-use agent：更强调时空顺序控制与 shortcut 避免。
- 相比单一视频特征增强：该方法在系统层面优化 reasoning pipeline。

## 启发
- 可把 STAR 思路迁移到长视频理解、具身视频回放分析、视频监控 agent。
- 与 FutureSightDrive 的 visual CoT 思路互补：一个加强工具推理，一个加强视觉中间表示。
- 对通用多模态 agent 来说，tool scheduling 可能比工具本身更关键。

## 评分
- 新颖性：★★★★☆
- 技术深度：★★★★☆
- Agent 相关性：★★★★★
- 实用价值：★★★★★
