# DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO

**会议**: NeurIPS 2025  
**arXiv**: [2506.07464](https://arxiv.org/abs/2506.07464)  
**代码**: 有  
**领域**: LLM/NLP / 视频理解 / 强化学习  
**关键词**: VideoLLM, GRPO, reinforcement learning, video reasoning, Reg-GRPO, difficulty-aware augmentation  

## 一句话总结
探索GRPO在VideoLLM中的应用，发现"安全门依赖"和"优势消失"两个阻碍有效学习的问题，提出Reg-GRPO（将GRPO loss重建为直接回归优势值的任务，消除clipping/min等安全门操作）和难度感知数据增强策略，在多个视频推理benchmark上显著提升性能。

## 背景与动机
GRPO在LLM推理（如数学、代码）中取得了R1级别的成功，但在VideoLLM中的效果尚未充分探索。视频推理任务相比文本推理更具挑战——需要时间理解、多帧关系推理等。直接将标准GRPO应用于VideoLLM面临两个新问题：(1) 过度依赖安全门机制（clipping和min操作）导致策略更新过于保守；(2) 组内奖励差异太小导致优势函数趋近于零（vanishing advantage），模型无法从中有效学习。

## 核心问题
如何让GRPO在VideoLLM中真正有效？标准GRPO的哪些设计在视频场景下失效？

## 方法详解

### 整体框架
DeepVideo-R1包含两个关键改进：Reg-GRPO改善优化目标，难度感知数据增强改善训练数据。

### 关键设计
1. **Reg-GRPO（Regressive GRPO）**：将GRPO的策略优化loss从PPO风格（带clipping/min的比率优化）重新建模为直接回归任务——模型预测每个输出的优势值（advantage），loss是预测优势与实际优势的回归差异。这消除了clipping和min等安全门操作，让模型更直接地与优势信号对齐，提供更清晰的梯度引导。

2. **难度感知数据增强**：标准GRPO中如果所有rollout都答对或都答错（组内奖励方差为零），优势消失无法学习。通过对输入prompt/视频进行增强（如改变问题难度、遮挡部分视频帧等），确保组内既有成功也有失败的rollout——产生多样化的奖励信号。增强时目标是"可解决的难度"——不太简单也不太难。

### 损失函数 / 训练策略
Reg-GRPO loss = 优势回归loss，替代标准GRPO的PPO-style loss。

## 实验关键数据
- 在多个视频推理benchmark上显著优于标准GRPO训练的VideoLLM
- Reg-GRPO解决了优势消失问题——训练更稳定
- 难度感知增强提供了更丰富的学习信号
- 两个改进叠加效果最佳

### 消融实验要点
- Reg-GRPO > 标准GRPO（消除安全门的效果明显）
- 难度感知增强 > 无增强（解决vanishing advantage）
- 两者联合 > 任一单独

## 亮点
- **首次系统探索GRPO在VideoLLM中的应用**并识别了具体失效模式
- **Reg-GRPO**将PPO风格loss化简为回归——更直接、更稳定——可能对其他RL for LLM场景也有启发
- **难度感知增强**与NoisyRollout的思路类似——都是通过输入变换改善RL探索——但NoisyRollout扰动视觉输入，DeepVideo-R1调整任务难度
- 与GTR的发现互补：GTR聚焦thought collapse（推理崩塌），DeepVideo-R1聚焦vanishing advantage（学习信号消失）

## 局限性 / 可改进方向
- Reg-GRPO的理论收敛保证未提供
- 难度感知增强的策略可能需要任务特定调优
- 仅在视频推理任务验证，其他视频任务（如视频描述）未测试

## 与相关工作的对比
- **vs. NoisyRollout**：NoisyRollout扰动图像增加感知多样性；DeepVideo-R1调整难度增加奖励多样性——互补
- **vs. GTR**：GTR用过程引导防止thought collapse；DeepVideo-R1用Reg-GRPO解决vanishing advantage——不同failure mode

## 评分
- 新颖性: ⭐⭐⭐⭐ Reg-GRPO和vanishing advantage的识别有价值
- 实验充分度: ⭐⭐⭐⭐ 多benchmark验证，消融详尽
- 写作质量: ⭐⭐⭐⭐ 问题诊断→解法的逻辑清晰
- 价值: ⭐⭐⭐⭐ VideoLLM RL训练的重要改进
