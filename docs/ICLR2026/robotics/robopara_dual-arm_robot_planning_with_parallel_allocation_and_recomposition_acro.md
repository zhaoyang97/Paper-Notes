# RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks

**会议**: ICLR 2026  
**arXiv**: [2506.06683](https://arxiv.org/abs/2506.06683)  
**代码**: [GitHub](https://github.com/AiDuanshiying/RoboPARA)  
**领域**: Robotics / Task Planning  
**关键词**: dual-arm manipulation, LLM task planning, DAG, parallel scheduling, System-1+2  

## 一句话总结
提出 RoboPARA，一个 LLM 驱动的双臂机器人并行任务规划框架，通过依赖图生成与图重遍历调度两阶段方法，最大化双臂协同并行性，执行时间减少 30%-50%。

## 背景与动机
1. 双臂机器人在复杂多任务场景中具有关键优势，但现有方法多关注任务成功率而忽略并行性
2. 现有 LLM 规划方法（如 RoCo、FLTRNN）主要产生单臂顺序执行计划
3. 类比人类日常行为：烧水时同时刷牙，关键在于推理哪些任务可并行、哪些需同步
4. 双臂协同调度问题（何时同步、何时解耦）在现有研究中缺乏探索
5. 缺少专门评估双臂并行规划的标准化基准数据集
6. System-1+System-2 架构中，System-2 层面的长时程并行双臂规划是关键空白

## 方法（框架/设计）
- **两阶段架构**:
  - **Stage 1 — 依赖图生成**: 利用 RAG 从记忆系统（短期观察 + 长期执行历史）中检索任务知识，构造结构化 prompt 让 LLM 生成 DAG；通过 3 类错误检测（跨对象依赖/跳过步骤/无关依赖）进行迭代修正
  - **Stage 2 — 图重遍历并行调度**: 维护调度队列，根据依赖满足度、臂空闲状态和对象持有一致性进行任务分配；单臂任务分配给空闲臂，双臂任务需两臂同步
- **形式化约束**: ①任务依赖（拓扑序）②臂互斥（同一臂不重叠）③臂锁定一致性（pick-use-place 同臂执行）④死锁预防（回滚较后执行的 pick 链）
- **目标**: 最小化 makespan $C_{\max} = \max_{v}(\sigma(v) + t_v)$

## 实验关键数据
| 指标 | RoboPARA vs 基线 |
|------|-----------------|
| 并行/协同步数 | 平均 4.5× 于其他方法 |
| 执行时间减少 | 30%-50% |
| Hard 任务成功率 | 比基线平均高 34% |
| 任务失败率(TFR) | Kitchen 场景为 0 (最优) |
| TEI (效率指标) | Kitchen 1.407, Office 1.553 (远超基线) |

- 基线: LLM³、ChatGPT-Prompts、VOYAGER、Embodied TaPA、LLM-Planner、FLTRNN、RoCo
- LLM 基础: GPT-4o 和 DeepSeek V3; 物理验证: Franka Research 3 + UR5e
- X-DAPT 数据集: 10 场景 × 3 难度 × 1000+ 任务包, 首个专注双臂并行评估的 benchmark

## 亮点
- 首次系统性定义和解决双臂协同调度问题，填补了 LLM 规划中并行性优化的空白
- DAG + 图重遍历的两阶段设计优雅且可解释，死锁预防机制实用
- X-DAPT 数据集覆盖 10 个贴近日常生活的场景，具有较高 benchmark 价值
- 硬件验证（人形机器人 + 工业臂）增强了可信度

## 局限性
- 任务基元（pick/use/place）需预定义，未处理更灵活的操作空间
- DAG 构建依赖 LLM 的正确理解，实际复杂场景可能仍有未覆盖的错误模式
- 未讨论执行失败后的在线重规划能力
- 场景以桌面操作为主，缺乏移动操作或人机交互场景

## 相关工作
- **双臂操作**: HybridVLA、RoboTwin、RoboCodeX (端到端 vs 组合技能)
- **LLM 规划**: RoCo (多智能体对话)、FLTRNN (RNN+记忆)、LLM-Planner (子目标分解)
- **图结构规划**: Tree-Planner (Hu et al.)、Graph-based planners (Byeon & Oh)
- **System-1+2**: Stanovich、Kahneman 理论在具身 AI 中的应用

## 评分
- 新颖性: ⭐⭐⭐⭐ (双臂并行调度问题定义+DAG 调度方案均为新颖贡献)
- 实验充分度: ⭐⭐⭐⭐ (多场景多难度多基线+硬件验证)
- 写作质量: ⭐⭐⭐⭐ (形式化清晰，图表丰富)
- 价值: ⭐⭐⭐⭐ (问题定义和数据集对后续研究有推动作用)
