# History-Aware Reasoning for GUI Agents

**会议**: AAAI 2026  
**arXiv**: [2511.09127](https://arxiv.org/abs/2511.09127)  
**代码**: [https://github.com/BigTaige/HAR-GUI](https://github.com/BigTaige/HAR-GUI)  
**领域**: Agent / GUI自动化  
**关键词**: GUI Agent, 短期记忆, 强化学习, 反思学习, 历史感知推理  

## 一句话总结
提出 HAR 框架，通过构建反思学习场景、合成纠错指南、设计混合 RL 奖励函数（含 Memory-Augmented Reward），将 GUI Agent 的推理模式从"历史无感知"转变为"历史感知"，3B 模型在 AITW/Mind2Web/GUI-Odyssey 等多个 benchmark 上超越更大模型。

## 背景与动机
现有 GUI Agent（如 UI-R1、GUI-R1、InfiGUI-R1）使用 System-2 CoT + RL 增强推理，但存在一个被忽视的关键问题：它们的推理模式是"历史无感知"的——将链式多步交互退化为离散的单屏幕理解，忽略了历史交互上下文中的关键线索。例如在 11 步的长序列任务中，Agent 在第 8 步推理时完全不考虑前 7 步做了什么。这源于基础 MLLM 的内在 CoT 模式，而现有 RL 训练使用推理格式指令仅优化动作预测，不改变推理模式。

## 核心问题
如何让 GUI Agent 在长序列情景推理中具备稳定的短期记忆——即在 System-2 CoT 中显式整合和分析历史交互信息？核心挑战是"历史无感知"推理模式根深蒂固于预训练阶段的 CoT 中，普通 RL 训练无法改变它（仅缩小 pass@k 到 pass@1 的差距）。

## 方法详解

### 整体框架
HAR 包含两个关键训练阶段：(1) GUI 场景热身（SFT 注入领域知识）；(2) 从失败中学习（反思 RL 增强短期记忆）。

### 关键设计
1. **GUI 场景热身（SFT）**：
   - 收集 GUI 理解数据（caption、问答、grounding 等）
   - 合成 Action-to-Summary（Act2Sum）数据：用教师模型为每个动作生成目标导向的语义总结，增强动作语义理解
   - System-2 CoT 蒸馏：用 Qwen2.5-VL-72B 为每个样本合成 System-2 推理链，过滤正确样本后用于训练

2. **反思学习场景构建**：
   - 用热身后的模型推理，收集错误样本 $\mathbb{D}_{his}$
   - 用教师模型为每个错误样本生成最多 3 条纠错指南 $\mathbb{G}$（分析错误原因，提供线索但不泄露答案）
   - 构建反思格式指令：将错误预测、错误 CoT 和纠错指南一起提供给模型，要求先自述错误（statement）再重新推理

3. **混合 RL 奖励函数**：
   - **Format Reward** $r^{format}$：输出是否符合反思格式
   - **Action Reward** $r^{action}$：对坐标类动作（CLICK），使用多尺度欧几里得距离奖励（归一化坐标距离 + 绝对坐标距离），正确时额外奖励坐标精确度（$r=1+F_{abs}$），错误时基于绝对距离给予部分奖励
   - **Memory-Augmented Reward (MAR)** $r^{memory}$：用 Qwen3-235B 判断 CoT 中是否包含对历史交互的分析。这是关键创新——显式奖励“在推理中考虑了前面做了什么”
   - 混合：$r = r^{format} \times (r^{action} + \gamma \times r^{memory})$，$\gamma=0.2$
   - 设计优势：相比在指令中强制要求关注历史（GRPO*），MAR 通过 RL 信号让模型自主习得何时需要参考历史

4. **Round-2 RL + Task Mixing**：Round-1 RL 在反思场景中训练后，Round-2 RL 切换到推理格式指令（对齐推理时用法），同时混合 grounding 任务（TMTS）防止 grounding 能力退化

### 损失函数 / 训练策略
- GRPO 算法进行 RL 优化
- 基于 Qwen2.5-VL-3B-Instruct，LoRA rank=64 alpha=128
- SFT: 1 epoch, lr=5e-6; RL: 2 epochs, lr=2e-6

## 实验关键数据
| Benchmark | 指标 | HAR-GUI-3B | InfiGUI-R1-3B | GUI-R1-3B | UI-R1-3B | Qwen2.5-VL-7B |
|-----------|------|-----------|--------------|----------|---------|--------------|
| AITW | SSR (avg) | **70.2** | 67.7 | 65.6 | 59.9 | - |
| Mind2Web | SSR (Cross-Task) | **42.2** | 37.2 | 38.8 | 36.8 | - |
| GUI-Odyssey | SSR (avg) | **62.31** | 50.62 | 48.35 | 46.71 | 58.39 |
| ScreenSpot | Avg | **83.3** | - | - | - | 79.8 |
| ScreenSpot-V2 | Avg | **86.2** | - | - | - | - |

OOD 评估（中文支付宝小程序）：HAR-GUI-3B 步骤成功率 76.5%，远超 GUI-R1-3B 的 69.99% 和 Qwen2.5-VL-72B 的 86.91%（3B vs 72B 的差距缩小到 10%）。

### 消融实验要点
- 仅用推理格式 RL（GRPO）：推理仍是历史无感知的，性能提升有限
- 在指令中强制要求关注历史（GRPO*）：反而导致性能下降，说明不应强制约束而应让模型自主习得
- HAR 的反思场景 + 纠错指南 + MAR：模型自主发展出历史感知推理模式
- 仅用情景推理数据训练 RL 会削弱 grounding 能力，TMTS 有效缓解
- 后训练（post-training）中，HAR-GUI 作为初始化 checkpoint 始终优于 GRPO 和基础 Qwen2.5-VL

## 亮点
- **精准定位问题**：发现并系统分析了现有 GUI Agent 的"历史无感知"推理缺陷，连 72B 模型也存在
- **Memory-Augmented Reward**：直接奖励"CoT 中是否考虑了历史信息"，用 RL 信号引导推理模式转变，而非手动约束
- **反思学习范式**：通过构建"错误+纠错指南"的反思场景，注入外部领域推理知识，比单纯 RL 探索更有效
- **多尺度坐标奖励**：归一化+绝对坐标双尺度奖励，精细化 CLICK 动作的优化
- **3B 模型超 7B+**：在 GUI-Odyssey 上超越 Qwen2.5-VL-7B（62.31 vs 58.39）

## 局限性 / 可改进方向
- 依赖 72B 教师模型合成纠错指南和 CoT，蒸馏质量受限于教师模型
- MAR 使用模型判断 CoT 是否包含历史信息，可能存在误判
- 仅在 CLICK-only 场景评估 OOD 泛化，TYPE 等复杂动作未充分验证
- 训练流程相对复杂（SFT + Round-1 RL + Round-2 RL + 后训练）

## 与相关工作的对比
- **vs UI-R1/GUI-R1/InfiGUI-R1**：这些方法用 RL 增强推理但使用推理格式指令，仅优化动作预测不改变推理模式；HAR 通过反思场景从根本上改变推理模式
- **vs UI-TARS**：UI-TARS 引入 System-2 推理但未针对性解决短期记忆问题；HAR 的 3B 模型在 ScreenSpot 上超越 UI-TARS-2B
- **vs 传统 Agent 框架（ReAct/Reflexion）**：传统方法依赖手工 prompt 进行反思，HAR 通过训练让模型内在化反思能力

## 启发与关联
- "短期记忆缺陷"可能是所有基于 CoT 的 Agent 的通病——CoT 倾向于就当前状态独立推理而忽略历史上下文
- MAR 的设计思路（RL 奖励推理过程的某种属性）可以推广到其他需要特定推理模式的场景
- 反思学习场景（提供错误+纠错指南进行 RL）是一种有效的知识注入范式

## 评分
- 新颖性: ⭐⭐⭐⭐ 精准识别并解决了 GUI Agent 的历史无感知问题，MAR 设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖 3 类 benchmark（情景推理/grounding/理解），OOD 评估，消融完整
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，分析有说服力，案例丰富
- 价值: ⭐⭐⭐⭐ 对 GUI Agent 的短期记忆问题提出了有效解决方案，代码开源

## 补充说明
- 该工作的方法论和实验设计对相关领域有参考价值
- 后续工作可在更多场景和更大规模上验证方法的泛化性和可扩展性
- 与近期相关工作的结合（如与 RL/MCTS/多模态方法的交叉）有潜在研究价值
- 建议结合实际应用需求评估该方法的部署可行性和计算效率
- 数据集和评估指标的选择可能影响结论的普适性，需在更多 benchmark 上交叉验证
