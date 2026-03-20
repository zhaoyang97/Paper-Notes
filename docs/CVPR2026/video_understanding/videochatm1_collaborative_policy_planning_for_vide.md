# VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning

**会议**: CVPR 2026  
**arXiv**: [2511.19524](https://arxiv.org/abs/2511.19524)  
**代码**: 无  
**领域**: 视频理解 / 多智能体 / 强化学习  
**关键词**: 多智能体协作, 视频理解, 工具调用策略, MARL, GRPO  

## 一句话总结
VideoChat-M1 提出了多智能体协作策略规划（CPP）范式 + 多智能体强化学习（MARL）训练框架，让 4 个异构 VLM agent 动态生成和更新工具调用策略来理解视频，在 LongVideoBench 上超过 Gemini 2.5 Pro 3.6%，超过 GPT-4o 15.6%。

## 背景与动机
现有的 agent-based 视频理解框架普遍使用静态、固定的工具调用策略——预定义好调用什么工具、什么顺序，不会根据视频内容动态调整。这导致它们无法充分挖掘时空复杂视频中的多样线索。另一方面，这些框架通常基于单 agent 或无训练的多 agent，无法通过学习提升协作效果。

## 核心问题
如何让多个 agent 动态地生成、执行和协调工具调用策略以应对复杂视频理解任务？如何联合训练多个异构 agent 使其学会有效协作？

## 方法详解
VideoChat-M1 的核心由两部分组成：Collaborative Policy Planning（CPP）推理范式和 Multi-Agent Reinforcement Learning（MARL）训练方法。

### 整体框架
输入：用户问题 + 视频 → 4 个异构 agent（Qwen3-8B, Qwen3-4B, Qwen2.5-7B, Qwen2.5-3B）各自生成工具调用计划 → 逐步执行工具并通过共享内存交换中间结果 → 根据对等反馈动态更新计划 → 各 agent 汇总答案 → 多数投票得出最终答案。总参数约 37B（含工具模型）。

### 关键设计
1. **Collaborative Policy Planning (CPP)**: 包含三个交替进行的阶段：
   - **Policy Generation**: 每个 agent 根据问题自主生成工具调用计划（如 Video Retrieval → Rough Browser → Fine Browser）
   - **Policy Execution**: 逐步调用工具（Global Sampling、Video Retrieval、Image Retrieval、Rough/Fine Browser、Spatial Tool、Grounding Tool 等）获取视频线索
   - **Policy Communication**: 执行每一步后，agent 将中间结果写入共享 memory buffer，其他 agent 可以据此决定是否修改自己后续的计划。这意味着计划是动态更新的，不是一成不变的

2. **Multi-Agent Reinforcement Learning (MARL)**: 首创的多 agent 视频理解联合训练方法：
   - **Policy SFT**: 先用 GPT-4o + DeepSeek-R1 标注高质量策略数据（筛选条件：答案正确 + 计划无需修改），对每个 agent 做 SFT warm-up
   - **GRPO 联合优化**: 三种奖励信号联合训练——结果奖励（答案对错）、格式奖励（工具调用是否合法可执行）、协作奖励（GPT-4o 评估中间协作过程质量，binary 奖励）
   - **Agent Dropout**: 训练时随机采样通信拓扑（DAG），防止 agent 间过度适应特定队友，增强鲁棒性

3. **异构 Agent 组设计**: 实验证明异构组（不同架构的模型）优于同构组（全用同一型号），因为结构多样性带来讨论多样性

### 损失函数 / 训练策略
SFT 阶段用交叉熵损失，MARL 阶段用 GRPO 目标函数（带 KL 散度正则化）。SFT 学习率 1e-6，MARL 学习率 1e-7，仅需 200 步 RL 训练即达最佳性能。

## 实验关键数据
| 数据集 | 指标 | VideoChat-M1 (37B) | GPT-4o | Gemini 2.5 Pro | 前 SOTA Agent |
|--------|------|------|----------|------|------|
| LongVideoBench | Accuracy | **82.3%** | 66.7% | 78.7% | 71.6% (DeepVideoDiscovery) |
| Video-MME (Avg) | Accuracy | **83.2%** | 71.9% | 84.3% | 75.7% (VideoRAG-72B) |
| VideoMMMU | Accuracy | **83.4%** | — | — | 76.2% (VideoChat-A1) |
| Video-Holmes | Accuracy | **60.5%** | 42.0% | 45.7% | — |
| VSIBench (Avg) | Accuracy | **71.9%** | 34.0% | 45.4% | — |
| Charades-STA | mIOU | **67.7%** | — | — | 65.9% (Eagle-2.5) |

效率对比：VideoChat-M1 平均只需 69.9 帧/视频（其他模型 384~568 帧），推理耗时 19.8s（其他 90~227s）。

### 消融实验要点
- **Agent 数量**: 从 1 到 4 性能持续提升，4 以上饱和
- **Agent Dropout 最关键**: 去掉后掉 2 个点，是"最重要的正则化器"
- **协作奖励和格式奖励各贡献约 1 个点**
- **SFT + RFT 缺一不可**: 仅 SFT 55.2/75.9，仅 RFT 57.9/80.2，两者结合 60.5/82.3
- **投票 > 指定 agent 决策 > 打分选择**
- **工具不依赖**: 替换 Spatial Tool 为通用 Qwen2.5-VL 后仍保持 SOTA，说明增益来自 CPP 框架而非特定工具

## 亮点
- CPP 范式非常自然：生成-执行-通信的迭代循环，agent 可以根据同伴信息动态修改策略
- MARL 训练中引入协作过程奖励（process reward），不仅评价结果还评价中间协作质量
- Agent Dropout 是优雅的正则化——随机化通信拓扑避免共适应，灵感来源于网络 Dropout
- 37B 总参数（4 个小模型 + 工具）在多个 benchmark 上匹敌或超越 235B 级模型
- 仅需 69.9 帧和 19.8s 推理时间，效率远超直接喂大量帧给单模型

## 局限性 / 可改进方向
- 需要 4 个模型并行推理 + 多个工具模型，部署复杂度高
- 协作奖励依赖 GPT-4o 当评判者，训练成本不低且引入外部依赖
- 当前工具集较为固定（7 种预定义工具），未探索让 agent 自主发现/创建新工具
- 对于不需要复杂推理的简单视频问题，多 agent 可能是过度设计

## 与相关工作的对比
- **vs VideoChat-A1/VideoRAG**: 这些方法用单 agent 或无训练多 agent，策略固定。VideoChat-M1 的 CPP 范式使策略动态可学习
- **vs Video-R1/VideoChat-R1**: 这些用 RL 优化单模型推理，VideoChat-M1 是首个多 agent 联合 RL 训练框架
- **vs GPT-4o/Gemini**: 即使用 GPT-4o 组成多 agent 团队走 CPP 流程，也只有 56.2 分（vs VideoChat-M1 60.5），说明 MARL 训练注入了零样本推理无法发现的协调模式

## 启发与关联
- 多 agent 协作 + RL 优化的范式可以迁移到其他多模态任务（如图像生成质量控制、多文档 QA）
- Agent Dropout 的思想——随机化通信拓扑来增强鲁棒性——可能对其他 multi-agent 系统也有启发
- 可以考虑把 CPP 中的工具集做成可扩展的，让 agent 在 RL 过程中学会组合新工具

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个多 agent RL 视频理解框架，CPP 范式新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个 benchmark × 4 个任务类型 + 丰富消融
- 写作质量: ⭐⭐⭐⭐ 整体清晰，但表格太多略显臃肿
- 价值: ⭐⭐⭐⭐⭐ 提供了一个新范式，效果显著超越 SOTA
