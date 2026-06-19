---
title: >-
  [论文解读] CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games
description: >-
  [ICCV 2025][机器人][视觉语言] 提出CombatVLA，一个针对3D动作角色扮演游戏战斗任务的高效3B参数VLA模型，通过Action-of-Thought数据格式和截断推理策略，实现比现有VLM游戏框架快50倍的推理速度，且战斗成功率超越人类玩家。 近年来VLA模型在具身智能领域取得了显著进展…
tags:
  - "ICCV 2025"
  - "机器人"
  - "视觉语言"
  - "3D游戏"
  - "实时决策"
  - "Action-of-Thought"
  - "高效推理"
---

# CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games

**会议**: ICCV 2025  
**arXiv**: [2503.09527](https://arxiv.org/abs/2503.09527)  
**代码**: [https://combatvla.github.io/](https://combatvla.github.io/)  
**领域**: 机器人  
**关键词**: Vision-Language-Action, 3D游戏, 实时决策, Action-of-Thought, 高效推理

## 一句话总结

提出CombatVLA，一个针对3D动作角色扮演游戏战斗任务的高效3B参数VLA模型，通过Action-of-Thought数据格式和截断推理策略，实现比现有VLM游戏框架快50倍的推理速度，且战斗成功率超越人类玩家。

## 研究背景与动机

近年来VLA模型在具身智能领域取得了显著进展，但在复杂3D环境中的实时决策方面仍面临巨大挑战。以《黑神话：悟空》为代表的3D ARPG战斗任务，对模型提出了三项严苛要求：（1）高分辨率视觉流的实时处理；（2）对动态变化的敌人行为进行战术适应；（3）秒级动作执行能力。

现有方法存在明显不足：基于API的方法（如Voyager）虽然可取得较好效果但无法模拟人类视觉交互方式；基于RL的方法需要大量预定义奖励函数和反复试错训练；基于GPT-4o等大规模VLM的框架（如Cradle、VARP）虽有潜力但单次推理延迟高达60-90秒，根本无法满足实时战斗需求。这些痛点驱动了CombatVLA的诞生。

## 方法详解

### 整体框架

CombatVLA的整体流程包含四个核心环节：（1）动作追踪器收集人类玩家的操控数据；（2）将数据构造为Action-of-Thought（AoT）格式用于训练；（3）采用三阶段渐进学习范式训练3B参数模型；（4）将训练好的模型集成到动作执行框架中进行实时推理。

### 关键设计

1. **动作追踪器（Action Tracker）**: 开发了一个轻量级Python工具，在游戏后台运行，通过两个独立线程分别监控键鼠操作和截取游戏画面。关键技术是基于时间戳的帧-动作对齐：对于每个动作 $a_j$，找到满足 $i_j = \arg\min_i (t_{f_i} \geq t_{a_j})$ 的最近未来帧进行配对。这保证了每个动作与其对应的视觉上下文正确关联。

2. **Action-of-Thought（AoT）数据格式**: 受CoT启发，将追踪器收集的帧集合 $F$ 和动作集合 $A$ 及其对齐关系转化为JSON格式的AoT数据。每条数据包含[action]（如"press space"）和[explanation]（描述当前敌人状态和动作的物理含义）。特别引入了 $\langle\text{TRUNC}\rangle$ 特殊token，用于在推理时截断输出以加速。

3. **三阶段渐进学习**: 

    - **Stage 1 - 粗粒度Video-AoT微调**: 每段视频含 $n=20$ 帧，帧率 $m=10$ fps，将各帧对应的动作按时间顺序排列生成video-AoT数据对。模型学习战斗环境的整体理解，训练3个epoch。
    - **Stage 2 - 细粒度Frames-AoT微调**: 创建动作-帧对齐数据，从当前动作时间戳向前回溯 $k=4$ 帧，形成精确的因果推理序列。模型学习战斗场景的时序逻辑，训练1个epoch。
    - **Stage 3 - 截断Frames-AoT微调**: 引入 $\langle\text{TRUNC}\rangle$ token重组AoT数据，将action放在explanation之前。实时推理时在遇到该token后停止生成，从而将推理速度提升约2倍，训练3个epoch。

4. **自适应动作加权损失**: 最终训练损失包含三个组成部分：语言建模损失 $\mathcal{L}_{lang}$、动作对齐损失 $\mathcal{L}_{align}$ 和模态对比损失 $\mathcal{L}_{con}$。采用优先级感知的匹配准则 $\mathcal{M}(A_l, A_o)$ 判断模型输出动作与标签动作是否匹配，依据匹配结果调整视觉EOS嵌入和动作EOS嵌入之间的距离。权重 $\alpha_i = 2^{(k-i-1)}$ 按优先级指数递减归一化到[0.1, 1.0]，确保高优先级的稀有关键动作（如闪避、回血）获得更多关注。

### 损失函数 / 训练策略

总损失为 $\mathcal{L} = \mathcal{L}_{lang} + \alpha \cdot \mathcal{L}_{act}$，其中 $\mathcal{L}_{act}$ 根据匹配结果在拉近（$\mathcal{L}_{con}^{pull}$）和推远+对齐（$\mathcal{L}_{con}^{push} + \mathcal{L}_{align}$）之间切换。骨干网络为Qwen2.5-VL-3B，学习率1e-5，batch size为1，温度0.7。训练冻结视觉编码器参数，仅微调语言模型参数。推理时使用截断策略，通过pyautogui库将动作转化为键鼠操作。

## 实验关键数据

### 主实验

| 模型 | CUBench Avg. | 推理延迟(s) | 模型调用次数 |
|------|-------------|------------|-------------|
| GPT-4o | 57.29 | 61.68(Cradle) | 5 |
| Gemini-2.0-flash | 57.90 | - | - |
| Qwen2.5-VL-3B(骨干) | 55.87 | - | - |
| **CombatVLA-3B** | **63.61** | **1.85** | **1** |
| VARP框架 | - | 90.23 | 10 |

| 任务类型 | CombatVLA | Cradle | VARP | 人类 |
|---------|-----------|--------|------|------|
| Easy零样本(BMW) | ~90% | ~30% | ~60% | ~80% |
| Hard(BMW) | ~80% | ~10% | ~30% | ~70% |
| Very Hard(BMW) | ~60% | 0% | 0% | ~50% |
| 跨游戏零样本(SSDT) | ~70% | ~10% | ~20% | ~60% |

### 消融实验

| 训练阶段 | Gathering | Comprehension | Reasoning | Avg. | 推理时间(s) |
|---------|-----------|---------------|-----------|------|-----------|
| Stage1 | 53.89 | 57.35 | 60.57 | 57.27 | 3.73 |
| Stage2 | 59.17 | 62.25 | 62.86 | 61.43 | 3.73 |
| Stage3(完整) | **60.83** | 60.29 | **69.71** | **63.61** | **1.85** |

| 损失设置 | Reasoning | Avg. |
|---------|-----------|------|
| 完整模型 | **69.71** | **63.61** |
| 去掉 $\mathcal{L}_{con}$ | 63.14 | 61.58 |
| 去掉 $\mathcal{L}_{align}$ | 63.71 | 61.64 |

### 关键发现

- 在高阶推理任务上，CombatVLA超过第二名Claude3.5-sonnet达14.28分，归功于AoT数据增强的推理能力
- 截断策略使Stage3推理速度约为Stage2的2倍（1.85s vs 3.73s），同时性能反而提升
- 通用benchmark（MME/VideoMME/OCRBench）上与骨干模型表现持平，证明特定任务训练未损害通用能力
- 跨游戏（BMW→SSDT）零样本测试成功率高，验证了良好的泛化性

## 亮点与洞察

- AoT数据格式巧妙融合了CoT的推理增强和截断策略的效率优势，是"先思考再行动"与"只取行动丢弃思考"的最佳折中
- 仅在Very Hard任务上训练，就能零样本泛化到Easy-Hard任务，甚至跨游戏泛化，说明高难度战斗中学到的战术逻辑具有可迁移性
- 模态对比损失通过对齐视觉和动作语义空间，有效缓解了动作类别不均衡问题

## 局限与展望

- 推理时需要暂停游戏等待模型输出，尚未实现真正的实时操控
- 训练数据仅约5K条高质量AoT，数据规模有限
- 仅验证了两款游戏，对更多游戏类型的泛化性有待检验
- 动作空间相对固定（10个动作），更复杂的连招系统可能需要扩展

## 相关工作与启发

- 与RT-2等机器人VLA模型的思路类似，但针对游戏场景做了实时性优化
- AoT格式可推广到其他需要实时决策的具身AI场景（如自动驾驶、机器人操控）
- 截断推理策略的思想可应用于任何需要降低VLM推理延迟的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个面向3D ARPG战斗的高效VLA，AoT+截断推理是亮点
- 实验充分度: ⭐⭐⭐⭐ 多维度评测（benchmark+实战+跨游戏），消融全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表美观，故事线完整
- 价值: ⭐⭐⭐⭐ 为游戏AI和具身智能的实时决策提供了实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] QUAR-VLA: Vision-Language-Action Model for Quadruped Robots](../../ECCV2024/robotics/quar-vla_vision-language-action_model_for_quadruped_robots.md)
- [\[ICCV 2025\] DexVLG: Dexterous Vision-Language-Grasp Model at Scale](dexvlg_dexterous_vision-language-grasp_model_at_scale.md)
- [\[NeurIPS 2025\] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](../../NeurIPS2025/robotics/safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)
- [\[CVPR 2025\] ShowUI: One Vision-Language-Action Model for GUI Visual Agent](../../CVPR2025/robotics/showui_one_vision-language-action_model_for_gui_visual_agent.md)
- [\[CVPR 2025\] g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks](../../CVPR2025/robotics/g3d-lf_generalizable_3d-language_feature_fields_for_embodied_tasks.md)

</div>

<!-- RELATED:END -->
