# OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2412.19723](https://arxiv.org/abs/2412.19723)  
**代码**: [https://qiushisun.github.io/OS-Genesis-Home/](https://qiushisun.github.io/OS-Genesis-Home/)  
**领域**: Agent  
**关键词**: GUI Agent, 轨迹数据合成, 反向任务合成, 交互驱动探索, 轨迹奖励模型  

## 一句话总结
提出OS-Genesis，一种"先交互探索再逆向生成任务"的GUI agent轨迹数据合成范式，通过无人监督的UI元素遍历收集状态转移三元组，逆向合成任务指令后用Trajectory Reward Model质量控制，在AndroidWorld上将Qwen2-VL-7B性能从9.82%提升至17.41%，接近GPT-4o的23.70%。

## 背景与动机
GUI agent训练严重依赖高质量轨迹数据（含高级指令+低级指令+动作+截图状态），但现有收集方式存在瓶颈：(1) 人类标注成本高且难以规模化；(2) 基于预定义任务的合成方法（task-driven）面临任务多样性受限和质量不可控问题——预设任务可能与动态环境不匹配，且执行中错误会导致不完整或不连贯的轨迹。这些问题严重阻碍了GUI agent的发展。

## 核心问题
如何在不依赖人类监督或预定义任务的情况下，高效合成高质量、多样化的GUI agent训练轨迹数据？

## 方法详解

### 整体框架
OS-Genesis采用"交互驱动"而非"任务驱动"的数据构建范式：(1) 无需预定义任务，agent先在GUI环境中进行规则化的UI元素遍历，收集大量⟨pre-screenshot, action, post-screenshot⟩三元组；(2) 通过反向任务合成，将三元组逆向生成低级指令→高级指令；(3) 用高级指令驱动GPT-4o在环境中执行采集完整轨迹；(4) 用Trajectory Reward Model评估轨迹质量进行采样训练。

### 关键设计
1. **交互驱动功能发现（Interaction-Driven Functional Discovery）**：在Android模拟器和Chrome浏览器中，通过规则化遍历（CLICK/TYPE/SCROLL）所有可交互UI元素。遍历过程是model-free的（仅TYPE动作需GPT-4o生成上下文合适的输入内容），产出大量⟨s_pre, a, s_post⟩三元组。这模拟了人类"先试试按钮再理解功能"的学习方式。

2. **反向任务合成（Reverse Task Synthesis）**：两阶段逆向构建任务指令。先用GPT-4o根据三元组生成低级指令τ_low（如"点击下拉菜单显示选项"），再基于低级指令推导高级指令τ_high（如"配置应用设置"）。核心创新在于从底层交互出发构建任务，而非从抽象任务出发尝试执行，天然保证了任务与环境的对齐性。

3. **轨迹奖励模型（Trajectory Reward Model, TRM）**：不同于传统labeler直接丢弃不完整轨迹（浪费大量有价值的部分轨迹），TRM基于GPT-4o对轨迹进行1-5分的细粒度评分（评估完成度和连贯性），用分数做概率采样训练。TRM与人类标注的Spearman相关系数达0.81（mobile）和0.80（web）。

### 损失函数 / 训练策略
两种训练目标：(1) Planning Training——给定截图+高级指令+历史，预测低级指令和动作（公式1）；(2) Action Training——给定截图+低级指令，仅预测动作（公式2）。全参数微调VLM（InternVL2-4B/8B, Qwen2-VL-7B），8×A100 80GB。训练数据量统一用1K轨迹（Self-Instruct用1.5K）。

## 实验关键数据

**AndroidWorld (Success Rate %)**：
| 模型 | Zero-Shot | Task-Driven | Self-Instruct | **OS-Genesis** |
|------|------|------|------|------|
| GPT-4o (M3A) | 23.70 | - | - | - |
| InternVL2-4B | 0.00 | 4.02 | 7.14 | **15.18** |
| InternVL2-8B | 2.23 | 4.46 | 5.36 | **16.96** |
| Qwen2-VL-7B | 0.89 | 6.25 | 9.82 | **17.41** |

**WebArena (Success Rate %)**：
| 模型 | Zero-Shot | Task-Driven | Self-Instruct | **OS-Genesis** |
|------|------|------|------|------|
| GPT-4o | 16.25 | - | - | - |
| InternVL2-4B | 0.00 | 4.98 | 5.81 | **7.88** |
| Qwen2-VL-7B | 7.47 | 7.05 | 5.39 | **10.79** |

### 消融实验要点
- **数据多样性优势**：OS-Genesis在指令和轨迹的cosine距离上都远超Task-Driven和Self-Instruct，甚至轨迹多样性超过人类数据（人类指令多样但动作偏好固定）。
- **TRM vs Labeler vs No RM**：TRM在高级任务上优于直接过滤（labeler），labeler过滤完整轨迹时会丢失低级步骤的学习信号导致低级任务表现下降。
- **数据规模效应**：性能随轨迹数从200到1000持续提升，但>1000后趋于饱和，受限于VLM容量和GPT-4o环境探索能力。
- **vs 人类指令**：即使使用人类写的高级指令，效果仍不如OS-Genesis的逆向生成指令——因为预定义任务可能与动态环境不匹配。
- **vs 人类轨迹**：OS-Genesis轨迹达到人类标注轨迹80%+的训练效果保留率，特别在高级规划任务上差距最小。
- **OOD泛化**：在AndroidControl的833个app中仅有20个在训练中出现过，OS-Genesis在未见过的app上仍表现优异。

## 亮点
- "先探索后定义任务"的反向范式是GUI agent数据构建的范式转变，优雅解决了任务-环境对齐难题
- 交互驱动探索几乎完全model-free（仅TYPE需LLM），成本极低可大规模扩展
- TRM的分级评分比二元过滤更合理，充分利用部分完成的轨迹
- 在3个不同平台（Android/Web）4个benchmark上全面超越task-driven方法
- 深入的分析实验（多样性、scaling、vs人类数据、OOD泛化）让结论非常扎实

## 局限性 / 可改进方向
- 轨迹执行仍依赖GPT-4o（闭源模型），成本和可复制性受限
- 交互探索的功能覆盖受限于规则遍历的深度（如多层嵌套菜单可能遗漏）
- 仅用文本+视觉表示训练，未探索仅用视觉或仅用文本的方案
- Web端平均轨迹长度偏短（4.46步 vs Task-Driven的8.74步），可能遗漏长任务

## 与相关工作的对比
- **vs AgentTuning (Zeng et al. 2024)**：AgentTuning基于预定义任务的SFT数据，任务多样性受限；OS-Genesis通过探索自动发现任务，不受预设限制。
- **vs OpenWebVoyager (He et al. 2024)**：OpenWebVoyager用labeler过滤+迭代优化；OS-Genesis用TRM分级评估保留更多训练信号。
- **vs BAGEL (Murty et al. 2024)**：BAGEL用语言引导探索但仍需初始种子任务；OS-Genesis完全无需种子任务，从零开始交互驱动。

## 启发与关联
- 反向任务合成的范式可迁移到其他agent领域（如代码agent：先在代码库中探索，再逆向生成编程任务）
- TRM的分级评估思路可应用于任何agent数据筛选（如SWE-Bench agent轨迹质量评估）
- 交互驱动探索+逆向标注的流程本质上是一种自动化课程学习——先发现容易的原子操作，再组合成复杂任务
- GUI agent的OOD泛化能力说明"功能理解"比"app记忆"更重要

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 反向任务合成是根本性的范式创新，解决了GUI数据构建的核心瓶颈
- 实验充分度: ⭐⭐⭐⭐⭐ 3平台4benchmark、3种backbone、5种分析维度，极为全面
- 写作质量: ⭐⭐⭐⭐⭐ 方法描述清晰，实验设计严谨，分析深入
- 价值: ⭐⭐⭐⭐⭐ 为GUI agent训练数据构建提供了可规模化的解决方案，影响力大
