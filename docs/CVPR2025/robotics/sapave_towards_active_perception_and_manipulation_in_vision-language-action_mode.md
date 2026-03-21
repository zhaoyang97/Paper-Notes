# SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics

**会议**: CVPR 2025  
**arXiv**: [2603.12193](https://arxiv.org/abs/2603.12193)  
**代码**: https://lmzpai.github.io/SaPaVe  
**领域**: 机器人  
**关键词**: VLA, 主动感知, 主动操作, 相机控制, 人形机器人

## 一句话总结
SaPaVe 提出了一种端到端的主动操作框架，通过解耦相机运动和操作动作的 action space，采用自底向上的两阶段训练策略（先学语义相机控制，再联合优化），在 200K 语义相机运动数据集上训练主动感知先验，配合 3D 几何感知模块增强视角变化下的执行鲁棒性，在真实世界任务中比 GR00T N1 和 $\pi_0$ 分别高 31.25% 和 40% 成功率。

## 研究背景与动机
1. **领域现状**：VLA 模型（$\pi_0$, GR00T-N1）在固定近最优视角下训练和部署，取得了良好的操作能力。
2. **现有痛点**：（1）现实世界中存在遮挡和视野外物体，固定视角无法覆盖所有情况；（2）将相机运动直接加入 VLA 的 action space 会破坏已有的固定视角操作先验，且需要大量昂贵的主动操作数据；（3）VLA 模型缺乏 3D 几何理解，在视角变化时执行不稳定。
3. **核心矛盾**：主动操作需要两种互补能力——语义主动感知（选择合适视角）和主动视角执行（在非最优视角下仍能操作）——但现有方法要么不支持语义输入的主动感知，要么无法应对视角变化下的操作。
4. **本文要解决什么？** 如何以数据高效的方式在 VLA 中同时实现语义驱动的主动感知和视角鲁棒的执行？
5. **切入角度**：相机运动是 embodiment-agnostic 的（与机器人本体无关），可以用大规模图像-语言-相机运动数据单独学习；操作动作是 embodiment-specific 的，需要联合优化。
6. **核心idea一句话**：解耦 action space（相机 vs 操作）+ 自底向上训练（先学主动感知，再学主动操作）+ 3D 几何注入。

## 方法详解

### 整体框架
输入 RGB 图像 $I_t$ + 语言指令 $L$ + 可选 3D 信息 $G_t$ → VLM backbone → 解耦的 action heads 输出相机运动 $A_{\text{head}}$（pitch/yaw）和操作动作 $A_{\text{other}}$（26-DoF 关节角度增量）。

### 关键设计

1. **解耦动作头 + Camera Adapter**:
   - 做什么：将相机控制和操作动作分开为两个解码器，用 LoRA adapter 学习相机运动
   - 核心思路：Camera Adapter 是在 VLM 上的 LoRA 模块，专门学习语义相机控制先验。两个独立的 action decoders 分别预测 $A_{\text{head}} \in \mathbb{R}^2$（pitch/yaw）和 $A_{\text{other}} \in \mathbb{R}^{26}$（双臂+双手关节）
   - 设计动机：解耦避免了相机运动和操作动作在统一 action space 中的干扰；Camera Adapter 保留 VLM 原始权重不变

2. **Universal Spatial Knowledge Injection**:
   - 做什么：将 3D 几何信息（深度图、相机内外参等）注入动作生成过程
   - 核心思路：用预训练 3D 几何模型的 encoder 编码几何信息为 spatial tokens，element-wise 加到 VLM 输出 tokens 上，在 action denoising 过程中指导动作预测
   - 设计动机：为视角变化下的主动操作提供 3D 空间理解，无需重训练或修改架构

3. **两阶段自底向上训练**:
   - Stage 1（主动感知对齐）：用 ActiveViewPose-200K 训练 Camera Adapter + Camera Decoder，MSE loss 监督相机运动。学会"在什么场景下往哪里看"
   - Stage 2（主动操作微调）：冻结 Camera Adapter，用混合数据（ActiveViewPose-200K + 操作数据）训练两个 Action Decoders，联合优化相机和操作

### 数据集 & Benchmark
- **ActiveViewPose-200K**：200K 图像-语言-相机运动对，4K 精标注 3D assets + 500 场景，半自动流程生成
- **ActiveManip-Bench**：首个主动操作仿真 benchmark，12 任务×100 物体×20 场景

## 实验关键数据

### 主实验（ActiveManip-Bench 仿真）

| 方法 | Unoccluded | Occluded | Out-of-View | Average |
|------|-----------|----------|-------------|---------|
| GR00T-N1 | 50.0 | 24.2 | 5.0 | 17.2 |
| $\pi_0$ | 31.7 | 17.5 | 8.3 | 14.2 |
| **SaPaVe** | **83.3** | **76.7** | **70.0** | **75.2** |

### 真实世界结果

| 方法 | Occluded PnP | OoV PnP | Occluded Arti | OoV Arti | Avg |
|------|-------------|---------|---------------|----------|-----|
| GR00T-N1 | 70 | 45 | 55 | 40 | 52.5 |
| $\pi_0$ | 55 | 35 | 45 | 30 | 41.25 |
| **SaPaVe** | **90** | **85** | **85** | **80** | **85.0** |

### 消融实验

| 消融 | Avg Success Rate |
|------|-----------------|
| w/o Stage 1 | 53.75% |
| w/o Stage 2 | 66.25% |
| w/o Decoupled Head | 71.25% |
| w/o Camera Adapter (full finetune) | 73.75% |
| w/o Spatial Knowledge | 71.25% |
| **Full SaPaVe** | **85.0%** |

### 关键发现
- 主动视角比固定视角+腕部相机组合更优——Out-of-View 任务固定视角成功率 <20%，主动视角 >70%
- 两阶段训练缺一不可：去掉 Stage 1 导致 Out-of-View 成功率腰斩
- 解耦 > 统一：统一 action space 比解耦低 ~14%
- LoRA adapter > 全参数微调：全微调破坏 VLM 语义理解
- 2B 参数的 SaPaVe 在语义主动感知上超越 Gemini 2.5 Pro 16%

## 亮点与洞察
- **"相机运动是 embodiment-agnostic 的"洞察**：这个观察非常关键——相机应该"往哪里看"不取决于机器人本体，因此可以用大规模图像数据单独学习，然后迁移到任意机器人
- **自底向上训练策略**：先构建感知先验，再在其基础上学习操作，比端到端联合训练更数据高效
- **首个主动操作 benchmark (ActiveManip-Bench)**：填补了评估空白，包含遮挡/视野外等关键场景

## 局限性 / 可改进方向
- 仅在 Unitree G1 人形机器人上验证，泛化到其他机器人平台（如机械臂）待测
- 相机运动仅 2 DoF（pitch/yaw），未考虑平移和全 6 DoF 运动
- ActiveViewPose-200K 是合成数据，sim-to-real gap 可能影响真实世界主动感知质量
- 未与 NBV 方法或多视角融合方法对比

## 相关工作与启发
- **vs GR00T-N1**: 固定视角 VLA，直接加入相机运动微调效果差。SaPaVe 的解耦+两阶段策略在真实世界超越 31.25%
- **vs $\pi_0$**: 同样的固定视角局限。SaPaVe 超越 40%
- **vs VQA-based 主动感知**: 离散候选视角选择，无法连续控制相机。SaPaVe 直接输出连续相机运动

## 评分
- 新颖性: ⭐⭐⭐⭐ 解耦+自底向上训练的设计巧妙，主动操作框架的系统性贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 仿真+真实世界+消融+泛化+对比实验非常充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，系统设计合理
- 价值: ⭐⭐⭐⭐⭐ 首个系统性解决 VLA 主动操作的框架，数据集和 benchmark 有长期价值
