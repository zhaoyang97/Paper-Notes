---
title: >-
  [论文解读] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation
description: >-
  [NeurIPS 2025][机器人][长程操作] 提出RoboCerebra长程机器人操作基准，包含1000条人类示范轨迹（平均2972步，约为现有基准的6倍），通过分层规划与执行框架和多维评估协议，系统测评VLM在规划、反思和记忆三个System 2认知维度上的能力。 当前VLM在机器人操作中的应用主要停留在快速反应式的…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "长程操作"
  - "基准评测"
  - "System 2推理"
  - "分层规划"
  - "VLM评估"
---

# RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation

**会议**: NeurIPS 2025  
**arXiv**: [2506.06677](https://arxiv.org/abs/2506.06677)  
**代码**: [robocerebra.github.io](https://robocerebra.github.io)  
**领域**: 机器人  
**关键词**: 长程操作, 基准评测, System 2推理, 分层规划, VLM评估

## 一句话总结

提出RoboCerebra长程机器人操作基准，包含1000条人类示范轨迹（平均2972步，约为现有基准的6倍），通过分层规划与执行框架和多维评估协议，系统测评VLM在规划、反思和记忆三个System 2认知维度上的能力。

## 研究背景与动机

当前VLM在机器人操作中的应用主要停留在快速反应式的System 1层面——VLA模型将多模态输入直接映射为低层控制信号。然而VLM真正的优势在于语义抽象、关系理解和上下文推理，这些恰好对应慢思考的System 2过程（长程规划、子目标分解、自适应调整）。现有基准数据集的不足直接阻碍了System 2能力的研究：

**时间尺度过短**：LIBERO-Long和RoboCasa等"长程"基准通常只有2-5个子任务，不超过500步动作，难以测试记忆维持、时间抽象等真正的长程推理需求。

**缺乏动态场景和记忆需求**：真实世界中物体会被移动、遮挡、状态改变，机器人需要记住之前探索的柜子里有什么、在哪里放过东西。现有基准几乎没有这类设计。

**评估维度单一**：大多基准仅用二值任务成功率评估，无法区分规划能力、感知判断、记忆利用等具体认知维度的表现。

RoboCerebra致力于填补这一空白，构建真正的长程任务环境来全面测评VLM作为System 2高层推理器的能力。

## 方法详解

### 整体框架

RoboCerebra包含三大组件：(1) 基于LLM生成和人类执行的大规模仿真数据集；(2) 分层规划与执行(HPE)框架——VLM做高层规划+VLA做低层控制；(3) 多维评估协议——固定System 1评估不同VLM的System 2能力。

### 关键设计

1. **自上而下的数据生成管线**：

    - **级联任务生成**：从LIBERO物品库随机采样物体，转换为结构化表示（类别、功能、空间上下文），喂给GPT生成高层任务描述（如"在微波炉中加热牛奶"），再分解为子任务序列。通过affordance-aware的提示设计确保时间一致性和物理可行性。
    - **场景初始化与双重验证**：将结构化计划解析为模拟器可执行代码，通过符号检查（物体状态一致性）和视觉语言验证（GPT-4o评判多视角渲染的空间合理性）的双重循环保证场景质量。
    - **人类示范与标注**：人类操作员在仿真中执行任务，生成多样化动作轨迹，并标注精细的子任务时间边界。共投入400小时用于轨迹采集+200小时用于质量验证。

2. **六类子任务设计**：

    - **Ideal**：静态全可观察基线
    - **Memory Exploration**：需主动探索环境构建内部表征（如检查柜子各隔间内容）
    - **Memory Execution**：需利用记忆完成目标（感知线索被移除）
    - **Random Disturbance**：引入意外的环境变化（物体位移、碰撞）
    - **Observation Mismatching**：需应对计划-感知不一致
    - **Mix**：结合记忆和动态因素，需持续在不确定性下重新规划

3. **分层规划与执行(HPE)框架**：

    - **VLM规划器**：处理低频观测，生成和更新子任务级子目标，存入记忆库。训练时用成功/失败标注的视频-指令对进行对比学习，使VLM能评估任务进度。
    - **VLA控制器**：基于OpenVLA，在子任务级别训练，消费高频视觉输入执行精细动作。
    - **记忆库**：连接两个模块的共享状态，VLM检测到子目标完成或偏差时更新记忆和下一个子目标。

### 损失函数 / 训练策略

- VLA训练：从长程示范中采样(图像, 指令, 动作)三元组，将连续动作离散化为token序列，用next-token prediction训练。200K步，batch size 64，256×256输入。
- VLM训练：用子任务级视频片段配合成功/失败标签进行对比学习，使VLM学会判断任务完成状态。

## 实验关键数据

### 主实验

**不同System 1+System 2组合的长程任务表现**

| 方法 | 平均SR | Random | Obs.Mis. | Mem.Exp. | Mem.Exe. | Mix | Ideal |
|------|--------|--------|----------|----------|----------|-----|-------|
| OpenVLA（仅System 1） | 2.00% | 4.59% | 1.35% | 0.18% | 1.86% | 0.00% | 4.05% |
| OpenVLA*（微调） | 4.57% | 7.84% | 8.65% | 1.06% | 2.06% | 0.00% | 7.84% |
| Planner+OpenVLA* | 16.04% | 18.63% | 19.45% | 8.04% | 16.69% | 11.48% | 21.92% |
| HPE框架 | **16.55%** | 18.63% | 19.18% | **9.06%** | **17.83%** | **13.21%** | 21.10% |

**不同VLM作为System 2 Planner的对比**

| Planner模型 | 平均SR | Mem.Exp. | Mix | Ideal |
|-------------|--------|----------|-----|-------|
| GPT-4o | **16.04%** | **8.04%** | **11.48%** | **21.92%** |
| GPT-4o-Blind | 15.10% | 7.02% | 10.48% | 20.00% |
| Qwen2.5-VL | 11.19% | 2.63% | 6.67% | 16.71% |
| LLaVA-Next-Video | 11.37% | 1.07% | 3.70% | 19.73% |
| GT-plan（上界） | 25.16% | 19.47% | 19.26% | 31.23% |

### 消融实验

**System 2多维评估**

| 模型 | 规划准确率↑ | 反思能力↑ | 成功率↑ | 规划长度↓ | 规划效率↑ |
|------|-----------|----------|--------|----------|----------|
| GPT-4o | **68.33%** | 32.66% | **16.04%** | 10.67 | **1.50** |
| GPT-4o-Blind | 61.37% | 0.00% | 15.10% | 10.73 | 1.41 |
| Qwen2.5-VL-7B | 44.67% | 47.74% | 11.19% | 8.30 | 1.34 |
| Qwen2.5-VL-7B-SFT | 30.00% | **66.83%** | 9.33% | 6.95 | 1.32 |

### 关键发现

- **System 1在长程任务中彻底失败**：即使微调后的OpenVLA在Ideal设置下也仅4-8%成功率，在Mix设置中完全失败（0%），证实了System 2的不可或缺性。
- **System 2提升在复杂任务中更显著**：HPE框架在Mix任务（需记忆+动态适应）上从0%提升到13.21%，但在简单的Ideal任务中可能因推理开销反而不如纯Planner方案。
- **规划能力>感知能力**：GPT-4o即使不看图（Blind模式）依然保持15.10%成功率，而Qwen2.5-VL微调后反思能力提升（66.83%）但规划准确率下降（30%），总成功率反而更低。说明当前长程任务中，规划推理比感知判断更为关键。
- **与GT-plan仍有9%差距**：说明VLM的环境交互不足和视觉领域差距仍是主要瓶颈。

## 亮点与洞察

- **数据规模突破性**：平均轨迹长度2972步，是现有长程基准的6倍，真正考验了长时间记忆和多步推理。
- **认知维度解耦评估**：通过固定System 1来隔离评估System 2的规划、反思、记忆能力，这种方法论值得学习。
- **System 1+2协同范式**：明确了VLM应作为高层推理器而非直接控制器使用，为机器人AI系统架构提供了清晰参考。

## 局限与展望

- System 1和System 2之间的交互仍较有限，缺乏细粒度的双向反馈机制。
- 评估协议可进一步扩展执行级信号，如子任务排序正确性、失败恢复能力等。
- 仿真到真实世界的迁移未验证，虽然论文认为System 2关注的是高层推理而非低层控制，sim-to-real gap影响较小。
- 数据生成管线依赖GPT和人类操作员，扩展到更多环境和任务类型的成本较高。

## 相关工作与启发

- 与LIBERO（500步以内）、CALVIN（34个短程任务）、RoboCasa（LLM生成但无人类轨迹）形成对比，RoboCerebra在时间跨度和标注丰富度上有质的飞跃。
- 受Kahneman System 1/System 2理论启发，为机器人操作领域提供了认知科学视角的系统设计。
- 启发：未来可探索让System 2 VLM在执行过程中接收System 1的失败反馈，实现更紧密的闭环协作。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 长程操作基准的系统化构建和认知维度解耦评估设计原创
- **实验充分度**: ⭐⭐⭐⭐ 多个VLM的横向对比+多维度评估+不同VLA后端验证，但绝对性能偏低
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数据集构建和评估方法描述详尽
- **价值**: ⭐⭐⭐⭐⭐ 填补了长程操作评测的重要空白，为VLM在机器人中的角色定位提供了实验依据

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method](../../CVPR2025/robotics/towards_long-horizon_vision-language_navigation_platform_benchmark_and_method.md)
- [\[NeurIPS 2025\] RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks](rdd_retrieval-based_demonstration_decomposer_for_planner_alignment_in_long-horiz.md)
- [\[CVPR 2025\] Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation](../../CVPR2025/robotics/lift3d_policy_lifting_2d_foundation_models_for_robust_3d_robotic_manipulation.md)
- [\[ICML 2025\] Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling](../../ICML2025/robotics/closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling.md)
- [\[CVPR 2026\] AGiLe: Learning Robust Long-Horizon Manipulation via Affordance-Grounded Bidirectional Latent Planning](../../CVPR2026/robotics/agile_learning_robust_long-horizon_manipulation_via_affordance-grounded_bidirect.md)

</div>

<!-- RELATED:END -->
