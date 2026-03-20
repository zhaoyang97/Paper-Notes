# Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2505.18079](https://arxiv.org/abs/2505.18079)  
**代码**: [https://github.com/microsoft/DeepVideoDiscovery](https://github.com/microsoft/DeepVideoDiscovery)  
**领域**: LLM Agent / 视频理解  
**关键词**: video understanding, agentic search, tool use, long-form video, video QA, multi-granular, adaptive workflow  

## 一句话总结
提出 DVD（Deep Video Discovery）agent，通过在分段视频片段上进行代理式搜索实现长视频理解——不同于使用预定义工作流的先前 video agent，DVD 利用 LLM 的推理能力在多粒度视频数据库上自主规划、策略性选择搜索工具、根据已获取信息动态编排自适应工作流，在 LVBench 上达到 74.2% 准确率（SOTA，显著超越先前所有工作），使用字幕时提升至 76.0%。

## 背景与动机
长视频理解面临两大挑战：(1) 巨大的时空复杂度使得直接处理整个视频不可行，(2) LLM 虽然在视频分析上有进展但处理信息密集的小时级视频仍力不从心。先前的 video agent 依赖**预定义的固定工作流**，无法根据不同查询和获取的信息自适应调整策略。

## 核心问题
如何让 agent 在长视频上进行**自主、自适应的搜索发现**，像人类分析视频一样根据问题灵活调整搜索策略？

## 方法详解

### 关键设计
1. **多粒度视频数据库**: 将长视频分段并建立多粒度索引（帧级、片段级、场景级），支持不同粒度的搜索操作
2. **搜索工具集**: 提供一组以搜索为中心的工具（片段检索、帧分析、OCR、ASR、时间定位等），agent 可按需调用
3. **自主自适应工作流**: LLM agent 根据当前观察状态规划下一步操作——先搜什么、搜到后该如何深入、何时足够得出答案。不同查询产生不同的工作流
4. **迭代信息积累**: 每轮搜索的结果成为下一轮决策的输入，agent 通过迭代搜索逐步积累理解

### 训练策略
无训练框架，完全基于 LLM 的推理能力+工具使用能力。

## 实验关键数据
- LVBench: **74.2%** 准确率（SOTA，显著超越先前所有工作）
- 使用字幕: **76.0%**
- 在多个长视频基准上全面领先

### 消融实验要点
- 自适应工作流 vs 固定工作流：自适应显著优于固定
- 多粒度搜索的必要性
- 不同 LLM backbone 的效果

## 亮点
- **自主搜索**而非固定管线——更像人类分析长视频的方式
- LVBench 74.2% SOTA 是决定性的领先
- 微软出品，代码开源
- 与 AdaVideoRAG（同系列笔记）的关键区别：AdaVideoRAG 按查询难度路由到不同检索策略；DVD 让 agent 完全自主决定搜索轨迹——更灵活但也更依赖 LLM 推理能力

## 局限性 / 可改进方向
- 搜索过程需要多轮 LLM 调用，延迟较高
- 依赖 LLM 的规划能力——弱模型可能搜索策略不佳
- 工具集需要手动设计，可扩展性受限

## 与相关工作的对比
- **vs AdaVideoRAG（同系列笔记）**: AdaVideoRAG 用规则路由（L1/L2/L3）；DVD 用 agent 自主搜索——更灵活
- **vs AHA（同系列笔记）**: AHA 做在线流式高光检测；DVD 做离线长视频深度理解

## 评分
- 新颖性: ⭐⭐⭐⭐ Agent 式视频搜索范式
- 实验充分度: ⭐⭐⭐⭐⭐ LVBench SOTA + 多基准
- 写作质量: ⭐⭐⭐⭐ 搜索工具设计清晰
- 价值: ⭐⭐⭐⭐⭐ 长视频理解的新范式，显著超越先前工作
