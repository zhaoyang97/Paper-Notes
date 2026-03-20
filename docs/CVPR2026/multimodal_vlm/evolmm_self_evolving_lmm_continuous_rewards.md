# EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards

**会议**: CVPR 2026 (Findings)  
**arXiv**: [2511.16672](https://arxiv.org/abs/2511.16672)  
**代码**: [https://github.com/mbzuai-oryx/EvoLMM](https://github.com/mbzuai-oryx/EvoLMM) (有)  
**领域**: 多模态VLM / 自我进化  
**关键词**: 自进化LMM, 无监督推理增强, 自奖励, Proposer-Solver, 连续反馈  

## 一句话总结
提出 EvoLMM，一个完全无监督的 LMM 自进化框架——从单一 backbone 模型实例化两个协作 Agent（Proposer 生成图像相关问题 + Solver 通过内部一致性求解），通过连续自奖励过程提升推理能力，仅用原始训练图像（无标注数据或外部奖励模型）在 ChartQA、MathVista、MathVision 上获得约 3% 的一致提升。

## 背景与动机
现有 LMM 训练流水线依赖人工标注数据或外部验证的奖励模型，限制了自主性和可扩展性。在缺乏标注数据的场景中（如新领域的快速适配），需要 LMM 能够自主提升推理能力。

## 核心问题
如何在完全无监督条件下（无标注数据、无奖励蒸馏）提升 LMM 的推理能力？

## 方法详解

### 关键设计

1. **双 Agent 协作**: 从同一 backbone 模型分化出两个角色：
   - **Proposer**：生成多样化的、以图像为基础的问题
   - **Solver**：通过内部一致性检查求解问题

2. **连续自奖励**: 动态反馈机制鼓励 Proposer 生成更有信息量的问题、Solver 精化结构化推理。无需 ground truth 或人类判断——奖励完全来自模型内部的一致性信号。

3. **纯无监督**: 仅使用原始训练图像，不需要任何标注或外部模型。基于 Qwen2.5-VL 作为 base model。

## 实验关键数据
- ChartQA / MathVista / MathVision 上约 **~3%** 一致提升
- 完全无监督，仅用原始图像

## 亮点
- **完全自主**：无需任何标注数据或外部奖励模型
- **Proposer-Solver 双 Agent 设计**：自生成训练信号的闭环系统
- **通用性强**：可应用于任何 LMM backbone

## 局限性 / 可改进方向
- 3% 提升的绝对幅度有限
- CVPR 2026 Findings（非主会议）
- 自奖励的质量天花板受限于模型自身能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 完全无监督的 LMM 自进化是重要且新颖的方向
- 实验充分度: ⭐⭐⭐ 多个推理基准验证但提升幅度有限
- 写作质量: ⭐⭐⭐⭐ 摘要清晰
- 价值: ⭐⭐⭐⭐ 为无监督 LMM 改进提供了实用基线
