---
title: >-
  [论文解读] InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild
description: >-
  [ICCV 2025][LLM评测][文本到动作生成] 提出 InterSyn 框架，通过交错学习策略（Interleaved Learning）将单人与多人动作在统一序列中联合建模，配合相对协调精修（REC）模块，生成更自然、更协调的人体交互动作，在 InterHuman 测试集上 FID 较 FreeMotion 降低 6.1%，R Precision Top-1 提升 2.8%。
tags:
  - "ICCV 2025"
  - "LLM评测"
  - "文本到动作生成"
  - "多人交互"
  - "扩散模型"
  - "交错学习"
  - "动作协调"
---

# InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild

**会议**: ICCV 2025  
**arXiv**: [2508.10297](https://arxiv.org/abs/2508.10297)  
**代码**: 未公开（计划开源）  
**领域**: 动作生成  
**关键词**: 文本到动作生成, 多人交互, 扩散模型, 交错学习, 动作协调

## 一句话总结

提出 InterSyn 框架，通过交错学习策略（Interleaved Learning）将单人与多人动作在统一序列中联合建模，配合相对协调精修（REC）模块，生成更自然、更协调的人体交互动作，在 InterHuman 测试集上 FID 较 FreeMotion 降低 6.1%，R Precision Top-1 提升 2.8%。

## 研究背景与动机

文本驱动的人体动作生成（T2M）在动画、虚拟现实等方面有广泛应用。尽管在 HumanML3D 等单人数据集上取得了显著进展，但在现实世界中生成多样化的交互动作仍面临挑战。

现有方法存在两个关键限制：

**单人与交互动作被割裂处理**：语义动作（走路、说话等）在 HumanML3D 上独立学习，社交动作（拥抱、握手等）在 InterHuman 上独立建模。然而现实中，人的动作是在独处和社交之间**流畅切换**的——这种动态交错（interleaving）是自然运动的本质特征

**缺乏互动线索的捕捉**：多人场景中，个体会持续根据他人的细微信号调整自己的运动。现有方法无法有效建模这种**相互适应**的动态过程

论文的核心假设受**情境学习理论**和**文化-历史活动理论**等教育心理学理论启发：语义动作和社交动作是相互强化的，联合学习能提供更好的泛化能力。

## 方法详解

### 整体框架

InterSyn 包含两个阶段：
1. **Interleaved Interaction Synthesis (INS)**：将单人和多人交互动作融合为统一的交错序列，使用条件扩散模型生成
2. **Relative Coordination Refinement (REC)**：通过协调器网络精修交互动作中多角色之间的空间关系和时序同步

### 关键设计

1. **交错数据构建 (Interleaved Data Construction)**: 初始化一个动作桶 $u = (u_x, u_y) \in \mathbb{R}^{2 \times T \times K \times C}$，随机采样单人动作 $p_s$ 和双人交互动作对 $(p_x, p_y)$，通过融合函数 $U(\cdot)$ 将它们组合为连续序列：$u = U(p_x, p_y, p_s, t_i, t_s)$。其中 $t_i$ 和 $t_s$ 分别是交互和独处动作的起始时间索引。函数 $U(\cdot)$ 负责动作对齐、平滑过渡和方向调整。由于两个数据集的骨骼不同，需要通过前向动力学（FK）进行旋转和平移对齐。

2. **条件动作扩散模型 (CMDM)**: 基于 Transformer 的扩散网络 $M_s$，以交互时间步 $t_i$ 和 $t_s$ 作为条件输入。包含时间嵌入层编码 $t_i, t_s$，文本嵌入层编码拼接后的描述 $w_u$。训练时在动作 $u$ 上加噪获得 $u^t$，模型预测去噪后的动作：$\hat{u} = M_s(u, w_u, t_i, t_s)$。通过条件化时间信号，模型学会生成在个体和交互动作之间平滑切换的连续序列。

3. **相对协调精修 (REC)**: 使用 Transformer 架构的协调器网络 $M_c$ 精修交互动作。对于双人交互，第一人的预测动作可以参考第二人的动作进行精修：$\phi_x = M_c(\hat{u}_x, \hat{u}_y, w_u)$。然后用精修后的 $\phi_x$ 反过来微调 $\hat{u}_y$：$\phi_y = M_c(\hat{u}_y, \phi_x, w_u)$。关键约束是**相对协调损失**：当 $\phi_x$ 已经是相对于 $\hat{u}_y$ 的合理交互动作时，对 $\hat{u}_y$ 的微调应该极小：$\mathcal{L}_{\text{rela}} = \|\phi_y - \hat{u}_y\|_2$。

### 损失函数 / 训练策略

两阶段训练：
- **第一阶段** (INS)：$\mathcal{L}_I = \lambda_1 \mathcal{L}_{\text{rec}} + \lambda_2 \mathcal{L}_{\text{smooth}}$，其中 $\mathcal{L}_{\text{smooth}}$ 在融合边界 ±5 帧窗口内强制平滑
- **第二阶段** (REC)：冻结 INS，训练协调器。$\mathcal{L}_R = \lambda_3 \mathcal{L}_{\text{rela}} + \lambda_4 \mathcal{L}_{\text{dm}}$，$\mathcal{L}_{\text{dm}}$ 是来自 InterGen 的掩码关节距离图损失
- 超参数：$\lambda_1=1, \lambda_2=0.1, \lambda_3=1, \lambda_4=0.5$
- 扩散时间步 1,000，推理使用 DDIM 采样
- 文本编码器：冻结的 CLIP-ViT-L-14
- 单 H100 GPU 训练 31 小时，batch size 256，44GB 显存
- 交替训练策略：在单人数据和交错数据上交替计算重建损失

## 实验关键数据

### 主实验（InterHuman 测试集）

| 方法 | R Precision Top1 ↑ | FID ↓ | MM Dist ↓ | Diversity → | MModality ↑ |
|---|---|---|---|---|---|
| TEMOS | 0.224 | 17.375 | 5.342 | 6.939 | 0.535 |
| MDM | 0.153 | 9.167 | 6.125 | 7.602 | **2.355** |
| InterGen | 0.264 | 13.404 | 3.882 | 7.770 | 1.451 |
| FreeMotion | 0.326 | 6.740 | **3.848** | 7.828 | 1.226 |
| **InterSyn** | **0.335** | **6.332** | 3.856 | **7.763** | 1.601 |

### 消融实验

| 消融配置 | R Precision Top1 | FID ↓ | MM Dist ↓ |
|---|---|---|---|
| s-i-s (默认) | 0.298 | 0.417 | 3.707 |
| s-i-s-i (更多切换) | 0.242 | 0.469 | 3.958 |
| s-i-s-i-s (5段) | 0.115 | 0.638 | 4.436 |
| w/o coordinator | 0.103 | 0.847 | 5.842 |
| w/o $\mathcal{L}_{\text{rela}}$ | 0.283 | 0.537 | 3.838 |
| w/o $\mathcal{L}_{\text{smooth}}$ | 0.295 | 0.431 | 3.712 |

### 关键发现

1. **交错段数**：s-i-s（单人-交互-单人，3段）效果最佳，增加切换次数（4段、5段）导致性能显著下降。原因是有限帧数内过多切换会截断关键动作阶段
2. **协调器至关重要**：移除 coordinator 后 FID 暴增 103.1%，MM Dist 增加 57.6%
3. **相对协调损失 $\mathcal{L}_{\text{rela}}$ 的特有作用**：移除后 Top-1 R Precision 下降 5.0%，它专门负责对齐多角色交互的互动动态
4. **平滑损失的定性作用**：虽然移除 $\mathcal{L}_{\text{smooth}}$ 在定量指标上影响较小（FID 仅增 3.4%），但定性分析显示步态切换时出现明显抖动
5. **动态环境评估**：在交错动作基准上（统一 HumanML3D + InterHuman 测试集），InterSyn FID 较 FreeMotion 降低 42.1%
6. **时间步设置**：$t_s=0, \text{random } t_i$ 是最优配置，平衡了精度、多样性和多模态性

## 亮点与洞察

- **学习范式创新**：从"分开学单人+交互"到"联合学交错序列"，更符合人类运动学习的认知过程
- **第一人称视角统一**：所有单人和交互动作都从第一人称视角处理，配合骨骼对齐的 FK 变换，解决了跨数据集骨骼不兼容问题
- **REC 模块的对称设计**：通过两步精修（先修 $u_x$，再用 $\phi_x$ 反向微调 $u_y$），隐式建模了交互的双向依赖
- **与 FreeMotion 的关键差异**：FreeMotion 通过条件运动分布统一单人和多人生成，但分开处理两种模态；InterSyn 在统一潜在空间中联合编码，实现无缝转换

## 局限与展望

- 仅在双人交互上训练和评测，多人（>2人）扩展仅通过推理阶段的 coordinator 设计实现，质量未知
- 交错最多 2 段时效果最好，限制了生成的复杂度
- 骨骼对齐依赖 FK 变换的精度，复杂动作可能引入累积误差
- 未探索音频/场景等额外条件的整合
- 生成的最大帧数受限于固定的 196 帧

## 相关工作与启发

- 继承了 MDM 的直接预测 $x_0$ 的扩散范式，但扩展到多人交错动作
- InterGen 的 mutual attention 和 FreeMotion 的 number-free 生成是重要前作
- "交错学习"的思想可推广到其他多智能体交互场景（如多机器人协作、多车辆轨迹预测）

## 评分

- **新颖性**: ⭐⭐⭐⭐ 交错学习策略是新颖的切入点，符合人类运动学习的认知理论
- **实验充分度**: ⭐⭐⭐⭐ 消融实验详细覆盖了数据组织方式、时间步设置、损失函数设计三个维度
- **写作质量**: ⭐⭐⭐⭐ 方法介绍详细，可视化对比直观，但部分公式符号不够一致
- **价值**: ⭐⭐⭐⭐ 为人体交互动作生成提供了新范式，动态环境评估基准也有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DISTA-Net: Dynamic Closely-Spaced Infrared Small Target Unmixing](dista-net_dynamic_closely-spaced_infrared_small_target_unmixing.md)
- [\[ACL 2025\] CoPrUS: Consistency Preserving Utterance Synthesis Towards More Realistic Benchmark](../../ACL2025/llm_evaluation/coprus_consistency_preserving_utterance_synthesis_towards_more_realistic_benchma.md)
- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](a_conditional_probability_framework_for_compositional_zerosh.md)
- [\[ACL 2026\] WildIFEval: Instruction Following in the Wild](../../ACL2026/llm_evaluation/wildifeval_instruction_following_in_the_wild.md)
- [\[ICLR 2026\] Benchmarking LLM Tool-Use in the Wild](../../ICLR2026/llm_evaluation/benchmarking_llm_tool-use_in_the_wild.md)

</div>

<!-- RELATED:END -->
