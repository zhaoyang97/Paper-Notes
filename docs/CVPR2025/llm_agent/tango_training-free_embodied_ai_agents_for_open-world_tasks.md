# TANGO: Training-free Embodied AI Agents for Open-world Tasks

**会议**: CVPR 2025  
**arXiv**: [2412.10402](https://arxiv.org/abs/2412.10402)  
**代码**: 待确认  
**领域**: LLM Agent  
**关键词**: Embodied AI, 导航, LLM程序合成, 零样本, 开放世界

## 一句话总结
提出 TANGO，通过 LLM 的程序组合能力编排两个最小化的导航基础原语（PointGoal Navigation + 记忆驱动探索策略），无需任何任务特定训练，仅用 few-shot 示例即可在 Open-Set ObjectGoal Navigation、Multi-Modal Lifelong Navigation 和 Open Embodied QA 三个不同的具身 AI 任务上达到 SOTA，体现了"最小原语集 + LLM 组合"的通用性。

## 研究背景与动机
1. **领域现状**：LLM 已展示在图像推理上组合多模块进行复杂任务的能力（如 VisProg、ViperGPT），但将这种程序组合能力拓展到具身智能体（在 3D 环境中导航和交互）仍是重大挑战。
2. **现有痛点**：现有具身 AI 方法针对特定任务设计和训练——ObjectNav 模型只能做 ObjectNav、EQA 模型只能做 EQA——一旦遇到新任务类型就需要重新收集数据、设计 reward、训练策略。
3. **核心矛盾**：需要一个统一框架在不额外训练的情况下处理多种开放世界具身任务，但具身任务的多样性似乎要求任务特定的训练。
4. **本文要解决什么？** 利用 LLM 的程序组合能力，将简单的导航原语自动组合成解决复杂具身任务的程序。
5. **切入角度**：将 LLM 在 2D 图像上的程序组合能力扩展到 3D 具身环境，仅用 few-shot 示例教 LLM 如何编排导航原语。
6. **核心idea一句话**：PointGoal 导航 + 记忆探索作为基础原语，LLM 通过 in-context learning 编排它们。类似于用“前进”和“转弯”两个基本指令组合出复杂的导航路径。

### 损失函数 / 训练策略
Training-free 框架不涉及任何任务级训练。PointGoal 导航模型是预训练的（在 Habitat 环境中用 RL 训练），但不针对任何下游任务做微调。LLM 仅通过 few-shot prompting 学习原语组合方式。

## 方法详解

### 整体框架
输入具身任务指令，LLM 接收任务描述和可用导航/感知原语，通过 few-shot prompting 生成解决程序。运行时调用 PointGoal Navigation 和记忆探索策略。

### 关键设计

1. **基础原语设计（仅两个）**:
   - **PointGoal Navigation**：给定目标坐标，agent 自动导航到该位置。使用预训练的 PointGoal 导航模型，仅接受 (x, y, z) 坐标输入
   - **Memory-based Exploration**：系统性探索未知环境，维护已探索区域的记忆图，优先探索未访问区域以最大化覆盖率
   - 设计动机："如果你能走到任何地方（PointGoal）并且能系统性地探索环境（Exploration），大多数导航类任务都可以被分解为这两个操作的组合"——这是一个 elegant 的最小化设计

2. **LLM 程序组合**:
   - 做什么：LLM 接收任务描述和可用原语的 API 说明，通过 few-shot prompting 生成组合这些原语的可执行 Python 程序
   - 核心思路：不同任务需要不同的原语组合方式——ObjectGoal Nav 需要"探索→检测目标→导航到目标"，Lifelong Nav 需要"依次导航到多个位置"，Embodied QA 需要"探索→收集信息→推理回答"
   - 设计动机：利用 LLM 的代码生成和推理能力，无需为每个任务单独训练。少量 in-context 示例足以让 LLM 理解原语的使用方式

### 训练策略
Training-free——不需要任何任务特定的训练。仅使用预训练 PointGoal 导航模型和通用 LLM。

## 实验关键数据

### Open-Set ObjectGoal Navigation（HM3D-OVON val-unseen）
| 方法 | SR (%) | SPL (%) | 训练需求 |
|------|:---:|:---:|------|
| Baseline RL | 18.6 | - | 强化学习训练 |
| VLFM | 35.2 | - | 任务特定训练 |
| DAgRL+OD | 37.1 | 19.9 | 任务特定训练 |
| **TANGO** | **35.5** | **19.5** | **零样本** |

### Multi-Modal Lifelong Navigation（GOAT-Bench val-unseen）
| 方法 | SR (%) | SPL (%) |
|------|:---:|:---:|
| Modular GOAT | 24.9 | - |
| SenseAct-NN Skill Chain | 29.5 | - |
| **TANGO** | **32.1** | **16.5** |

TANGO 在 GOAT-Bench 上 +2.6% 超越之前 SOTA

### Open Embodied QA（OpenEQA, LLM-match 1-5 分）
| 方法 | Score |
|------|:---:|
| Blind LLMs | 35.5 ± 1.7 |
| Socratic LLMs + Frame Captions | 38.1 ± 1.8 |
| **TANGO** | **37.2 ± 1.8** |
| 人类 Agent | 85.1 ± 1.1 |

### 失败分析
| 错误类型 | 占比 |
|---------|:---:|
| 检测失败（目标遗漏/误检） | ~34% |
| LLM 伪代码错误 | 18% |
| 探索策略未覆盖 | ~10% |
| 提示相关问题 | 11% |

### 关键发现
- 单一框架在 3 个不同具身 AI 任务上均达或接近 SOTA——验证了最小原语集 + LLM 组合的通用性
- 主要瓶颈在感知层（检测失败 34%）而非规划层（LLM 错误 18%），说明 LLM 组合逻辑整体可靠
- 记忆机制在 GOAT-Bench 终身导航上贡献显著（+2.6%），因为可以跨目标复用空间记忆

## 亮点与洞察
- **"最小原语集"设计哲学**非常优雅——不是设计越多的原语越好，而是找到一组完备且基础的导航原语，让 LLM 的组合能力来覆盖复杂任务。这类似于编程语言设计中的"正交性"原则
- **从图像推理到具身环境的程序合成扩展**自然且有效——VisProg/ViperGPT 在 2D 图像上验证了 LLM 程序合成的能力，TANGO 将其推广到 3D 具身环境
- **Training-free 的实用价值**：无需为每个新环境/任务收集数据和训练，大幅降低部署门槛

## 局限性 / 可改进方向
- PointGoal 导航模型仍需预训练——不是完全 training-free，只是任务级不需要训练
- 精确操作任务（如抓取、开门）可能不足——当前的原语只支持导航类操作
- 探索效率受 memory-based exploration 策略限制，在大规模开放环境中可能不够高效
- 对 LLM 的 prompt 质量敏感——bad few-shot 示例可能导致错误的原语组合
- 未验证在真实机器人上的效果（仅在仿真器中测试）

## 相关工作与启发
- **vs SayCan**: SayCan 需要大量预训练技能原语，且每个技能需要单独训练。TANGO 仅用两个通用原语就更灵活，体现了"少即是多"
- **vs ProgPrompt**: 同属 LLM 程序合成范式，但 ProgPrompt 使用固定 API，TANGO 的原语设计更面向具身导航的实际需求
- **vs VOYAGER**: Voyager 也是 LLM 驱动的 agent，但面向 Minecraft 的技能库构建。TANGO 面向真实世界的具身任务，原语设计更基础
- **对通用 Embodied AI 的启发**: 如果两个简单原语就能支撑三个任务，加入更多基础原语（如抓取、推拉）可能支撑更广泛的具身任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统性应用 LLM 程序组合到多个具身任务，"最小原语集"理念新颖
- 实验充分度: ⭐⭐⭐⭐ 三个不同任务的零样本 SOTA，验证了方法的通用性
- 写作质量: ⭐⭐⭐⭐ 清晰简洁
- 价值: ⭐⭐⭐⭐ 对 training-free 具身 AI 通用框架有重要参考价值
