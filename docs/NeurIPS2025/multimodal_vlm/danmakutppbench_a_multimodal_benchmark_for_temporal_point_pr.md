# DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding

## 基本信息
- **arXiv**: 2505.18411
- **会议**: NeurIPS 2025
- **作者**: Yue Jiang, Jichu Li, Yang Liu, Dingkang Yang, Feng Zhou, Quyu Kong
- **代码**: https://github.com/FRENKIE-CHIANG/DanmakuTPPBench
- **领域**: 多模态时序建模 / Temporal Point Process / LLM Benchmark

## 一句话总结
论文提出首个面向多模态 Temporal Point Process 的系统 benchmark：一方面构建来自 Bilibili 弹幕视频的时间戳-文本-视频联合事件数据集 DanmakuTPP-Events，另一方面通过多智能体 LLM/MLLM pipeline 构建复杂时序推理问答集 DanmakuTPP-QA，系统揭示当前 TPP 模型与 MLLM 在多模态事件动态理解上的明显短板。

## 背景与动机
传统 TPP 研究主要处理单模态事件序列，例如纯时间戳或附带简单 mark 的离散事件流，但现实世界中的事件通常同时具备：
- 时间属性；
- 文本语义；
- 视觉上下文；
- 多主体交互和因果依赖。

现有数据集过于单一，导致模型开发与评测严重脱离 LLM/MLLM 时代的真实需求。作者的核心动机是：**把 TPP 从统计建模问题推进到多模态理解与推理问题。**

## 核心问题
如何构建一个既能支持经典 TPP 建模，又能评估大模型在时序-文本-视觉联合推理上的综合能力的 benchmark？

## 方法详解

### 1. DanmakuTPP-Events
数据来自 Bilibili 的视频弹幕场景：
- 弹幕天然带有精确时间戳；
- 文本内容反映用户即时反应；
- 对应视频帧提供视觉语境。

因此单个“事件”可表示为时间、文本和视频三元组，是非常自然的多模态 TPP 数据源。

### 2. DanmakuTPP-QA
作者在事件数据基础上进一步构建 QA benchmark：
- 通过多智能体 LLM + MLLM pipeline 自动生成问题；
- 问题面向复杂 temporal-textual-visual reasoning；
- 不只测试事件预测，还测试多模态时序理解。

### 3. 评测覆盖面
论文同时评测：
- 经典 TPP 模型；
- 最近的多模态大模型；
- 不同类型时序推理任务。

这样的设计使 benchmark 既服务传统时序社区，也服务大模型社区。

## 实验结论
根据摘要，作者发现：
- 现有方法在多模态事件动态建模上存在显著性能缺口；
- 经典 TPP 模型难以处理复杂视觉/语义上下文；
- MLLM 虽具多模态能力，但对时间动态和事件机制的建模仍不足。

这说明“多模态理解强”不等于“多模态时间事件建模强”。

## 亮点
1. **问题定义升级**：将 TPP 从单模态统计问题扩展到多模态语言建模语境。
2. **数据构造自然**：弹幕是极少数天然同时具备时间、文本、视频三要素的数据源。
3. **双组件设计完整**：既有事件建模数据，也有高层推理 QA。
4. **社区桥梁作用强**：连接 TPP、时序推理、MLLM benchmark 三个方向。

## 局限性
1. 数据域集中在 Bilibili 弹幕生态，跨平台泛化待验证。
2. 自动生成 QA 可能带来一定标注噪声与风格偏差。
3. benchmark 强调评测，不直接提供新的强建模方法。

## 与相关工作的对比
- 相比传统 TPP benchmark：首次系统纳入视觉与文本联合事件语境。
- 相比一般 MLLM benchmark：更强调事件发生机制和时间点过程建模。
- 相比纯视频 QA：引入 point process 视角，更关注事件动态结构。

## 启发
- 可进一步探索“TPP + VLM”混合架构，显式建模事件强度函数。
- 对 streaming agent、视频助手、用户行为预测都很有参考价值。
- 这类 benchmark 也可用于检验 test-time scaling 是否真正改善时间推理，而非只提升语言流畅度。

## 评分
- 新颖性：★★★★★
- 技术深度：★★★★☆
- 基准价值：★★★★★
- 研究启发性：★★★★★
