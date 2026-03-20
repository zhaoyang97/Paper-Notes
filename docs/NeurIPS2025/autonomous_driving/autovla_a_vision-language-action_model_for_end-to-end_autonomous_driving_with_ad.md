# AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2506.13757](https://arxiv.org/abs/2506.13757)  
**代码**: [https://autovla.github.io/](https://autovla.github.io/)  
**领域**: 自动驾驶 / VLA  
**关键词**: Vision-Language-Action, 端到端驾驶, 自适应推理, GRPO强化微调, Action Tokenization  

## 一句话总结
提出AutoVLA——基于Qwen2.5-VL-3B的端到端自动驾驶VLA模型，将连续轨迹离散化为物理action tokens嵌入语言模型词表，支持fast/slow thinking双模式推理，通过GRPO强化微调同时提升10.6%性能和66.8%推理效率，在NAVSIM和Bench2Drive上达SOTA。

## 研究背景与动机

1. **领域现状**：端到端自动驾驶正从专用规划模型向VLA（Vision-Language-Action）模型演进。VLM已展示出对驾驶场景的理解能力，但如何将语言理解能力转化为可靠的驾驶动作仍是挑战。
2. **现有痛点**：（a）文本waypoint表示精度低且推理慢（需要逐token生成坐标数字）；（b）推理（CoT）增加延迟，对简单场景不必要；（c）缺乏从驾驶环境反馈中学习的机制。
3. **核心idea**：物理action tokenization（K-disk聚类离散化轨迹为2048个token）+ Fast/Slow双模推理 + GRPO强化微调自动学习何时深思何时快行。

## 方法详解

### 整体框架
Qwen2.5-VL-3B接收多视角多帧图像+导航指令+车辆状态，输出action tokens（直接映射为轨迹）或CoT推理+action tokens。

### 关键设计

1. **物理Action Tokenization**：用K-disk聚类将nuPlan训练轨迹离散化为K=2048个物理动作token，每个token代表0.5s的轨迹段。直接加入LLM词表，端到端预测。相比文本waypoint，PDM Score提升+9.23，推理时间减半（3.95s vs 7.65s）。

2. **Fast/Slow双模推理**：简单场景直接输出action token（Fast, ~1s），复杂场景先CoT推理再输出（Slow, ~10s）。训练时混合两种数据，CoT样本加权$\lambda_{cot}=40$确保充分学习。

3. **GRPO强化微调**：用PDMS/ADE作驾驶奖励 + CoT长度惩罚（鼓励简单场景不推理）。GRPO无需critic模型，直接用采样组内比较估计基线。LoRA微调，lr=3e-5。效果：PDMS 80.54→89.11，且自动学会简单场景用fast thinking。

## 实验关键数据

### NAVSIM (nuPlan)

| 方法 | PDMS |
|------|------|
| Centaur | 92.10 |
| AutoVLA (One-shot) | 80.54 |
| AutoVLA (Post-RFT) | 89.11 |
| AutoVLA (Best-of-N) | 92.12 |

### Bench2Drive (CARLA闭环)

| 方法 | Driving Score | Success Rate |
|------|-------------|-------------|
| Orion | 77.74 | 54.62% |
| **AutoVLA** | **78.84** | **57.73%** |

### 关键发现
- RFT同时提升性能和效率（PDMS +10.6%，推理时间-66.8%）。
- 物理action token显著优于文本waypoint。
- CoT推理在>50k样本时超越action-only，体现可扩展优势。

## 亮点与洞察
- **自适应推理的实现方式**：通过GRPO中的长度惩罚自动学习何时推理，比硬编码规则更灵活。
- **action tokenization的简洁优雅**：轨迹→离散token→嵌入LLM词表，使VLM原生具备动作能力。

## 局限性 / 可改进方向
- 3B模型推理仍需~4s/帧，实时部署有挑战。
- K-disk离散化存在精度上限，可探索更细粒度的tokenization。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Action tokenization + 自适应推理 + GRPO的组合很新
- 实验充分度: ⭐⭐⭐⭐ NAVSIM + Bench2Drive + Waymo跨数据集验证
- 写作质量: ⭐⭐⭐⭐ 方法清晰，消融充分
- 价值: ⭐⭐⭐⭐⭐ 为VLA自动驾驶提供了完整且高效的技术方案
