# World2Act: Latent Action Post-Training via Skill-Compositional World Models

**会议**: CVPR 2025  
**arXiv**: [2603.10422](https://arxiv.org/abs/2603.10422)  
**代码**: https://wm2act.github.io/  
**领域**: 视频理解 / 机器人  
**关键词**: World Model, VLA后训练, 潜在空间对齐, 技能分解, 对比学习

## 一句话总结
World2Act 提出了一种基于潜在空间对齐的 VLA 后训练方法：通过对比学习将 World Model 的视频动态潜表示与 VLA 的动作表示对齐（而非在像素空间监督），并引入 LLM 驱动的技能分解流水线实现任意长度视频生成，在 RoboCasa 和 LIBERO 上以 50 条合成轨迹即达到 SOTA，真实世界提升 6.7%。

## 研究背景与动机
1. **领域现状**：VLA（$\pi_0$, GR00T-N1.6）通过行为克隆学习，但对环境变化和新接触条件泛化不足。World Model（Cosmos-Predict2）可以生成物理一致的滚动轨迹。
2. **现有痛点**：（1）WM 后训练通常用像素空间监督（逆动力学模型/像素奖励），但 WM 的像素滚动会放大噪声和幻觉；（2）视频扩散模型在固定长度 clip 上训练，而机器人任务时长差异大，任意长度生成是瓶颈；（3）收集同时包含相机运动和操作标签的真实数据极其昂贵。
3. **核心矛盾**：WM 包含丰富的动态先验，但像素级别传递这些先验时会引入幻觉和伪影。
4. **本文要解决什么？** 如何在不依赖像素的情况下将 WM 的动态先验转移到 VLA 策略中？如何让 WM 支持任意长度的视频生成？
5. **切入角度**：在 WM 的潜在空间而非像素空间中进行动作-视频对齐；用 LLM 将长任务分解为原子技能段以实现稳定长视频生成。
6. **核心idea一句话**：WM 潜在表示 + VLA 动作表示的对比对齐 + 技能分解的任意长度 WM = 数据高效的 VLA 后训练。

## 方法详解

### 整体框架
两阶段后训练：Stage 1 用对比学习训练 Video Adapter 和 Action Adapter 将两种模态映射到共享潜在空间；Stage 2 冻结 VLA backbone，用轻量级残差策略（Residual Policy）驱动 VLA 动作向 WM 动态先验靠拢。

### 关键设计

1. **技能分解的 World Model (Skill-WM)**:
   - 做什么：将长任务分解为原子技能段，支持任意长度视频生成
   - 核心思路：通过夹爪状态变化分割视频流，用 LLM（DeepSeek）将全局指令分解为有序的原子技能描述，同步视频段和语言。推理时 LLM 生成技能列表，WM 逐段生成，上一段最后帧作为下一段初始条件
   - 设计动机：原子技能的长度分布更均匀集中（密度提升 17-72%），减少长尾导致的误差累积

2. **Stage 1: 潜在空间对齐**:
   - 做什么：训练 Video Adapter $\mathcal{B}_v$（CNN）和 Action Adapter $\mathcal{B}_a$（MLP）将视频潜表示和动作映射到共享空间
   - 核心思路：双向 InfoNCE 对比损失 + 动作重建 MSE 损失。chunk-wise 对齐（每 $M$ 帧一个 chunk）而非全局轨迹对齐，防止模型用任务身份等捷径匹配。Hard negatives 来自同一 skill 的不同 demo
   - 架构细节：Video Adapter 是 3 层 1D 时序 CNN，将 WM 的 DiT 隐层特征（token 维度 ~4096）映射到 256 维共享空间；Action Adapter 是 2 层 MLP（隐藏层 512），将 $M$ 步动作向量拼接后也映射到 256 维。InfoNCE 温度参数 $\tau = 0.07$，batch 内所有非配对 chunk 为 easy negatives
   - 设计动机：chunk-wise 对齐鼓励细粒度时间动态匹配，不同于全局 embedding 可能忽略时序细节

3. **Stage 2: 残差策略后训练**:
   - 做什么：冻结 VLA backbone，学习轻量残差修正 $f^\theta$，使 $a_{\text{final}} = a_{\text{base}} + a_{\text{residual}}$
   - 核心思路：在线滚动当前增强策略，用冻结的 WM 生成视频潜表示作为目标，计算 $z^v$ 和 $z^a$ 的对比损失来训练残差网络。无需奖励或环境成功信号
   - 设计动机：残差策略保留了 VLA 原始能力（避免灾难性遗忘），且样本效率高（仅需轻量网络）

### 损失函数
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{contrastive}}$$

## 实验关键数据

### 主实验（RoboCasa）

| 方法 | Real Demos | Synthetic | SR |
|------|-----------|-----------|-----|
| $\pi_0$ | 300 | 0 | 62.5% |
| GR00T-N1.6 | 300 | 0 | 66.2% |
| Cosmos Policy | 50 | 0 | 65.7% |
| GR00T-N1.6-ft + DreamGen | 350 | +50 | 70.5% |
| **GR00T-N1.6-ft + World2Act** | 350 | **+50** | **72.6%** |
| **Cosmos + World2Act** | 50 | **+50** | **66.3%** |

### LIBERO（4 个 suite 平均）

World2Act 在 LIBERO-Long 上将 Cosmos Policy 从 85.2% 提升到 89.6%，GR00T-N1.6-ft 从 87.6% 提升到 91.2%。

| Suite | Cosmos | +World2Act | GR00T-N1.6-ft | +World2Act |
|-------|--------|-----------|--------------|------------|
| LIBERO-Spatial | 91.0% | 93.4% | 92.8% | 95.0% |
| LIBERO-Object | 93.2% | 95.0% | 94.6% | 96.4% |
| LIBERO-Goal | 88.4% | 91.8% | 90.2% | 93.4% |
| LIBERO-Long | 85.2% | 89.6% | 87.6% | 91.2% |

LIBERO-Long（需要多步长序列推理）提升最大，验证了 Skill-WM 在长任务上的优势。

### 消融/关键发现
- 潜在空间对齐 vs 像素空间监督：潜在空间方法在有幻觉的 WM 滚动下更鲁棒
- Skill-WM vs Base-WM：技能分解后视频生成时间一致性大幅提升（FVD 降低）
- chunk-wise 对比 > trajectory-wise 对比：细粒度时间对齐更有效
- 50 条合成轨迹即可实现有意义的提升，极高的数据效率
- 真实世界实验提升 6.7%，验证 sim-to-real 迁移能力

## 亮点与洞察
- **潜在空间对齐替代像素监督**：核心洞察是 WM 的潜表示比像素更抗幻觉——像素级监督放大噪声，潜在表示保留了动态先验的本质
- **LLM 驱动的自动技能分解**：用夹爪状态做视觉流分割 + LLM 做指令分解，全自动且同步率 >86%，是实用的数据工程方案
- **残差策略的优雅设计**：不碰原始 VLA 权重，仅学习轻量修正，兼顾能力保留和新知识注入
- **对比学习连接 WM 和 VLA**：InfoNCE 作为无奖励的 WM→VLA 知识转移信号，避免了 RL 的不稳定性

## 局限性 / 可改进方向
- WM 选择（Cosmos-Predict2）对后训练效果有重要影响，不同 WM 的适用性差异未充分探讨；作者仅在 Cosmos 家族上验证，开源替代（如 OpenSora）效果未知
- 残差策略假设 base VLA 已有基本能力——对于完全失败的 base policy 可能无效，本质上是"微调"而非"从零学习"
- 技能分解依赖夹爪状态变化，不适用于非抓取任务（如推/滑）
- 仅在操作任务上验证，导航等其他具身任务待测

## 相关工作与启发
- **vs DreamGen**: DreamGen 用像素空间的 IDM 从 WM 滚动中推断伪动作，受幻觉影响。World2Act 在潜在空间操作，更鲁棒（RoboCasa 72.6% vs 70.5%）
- **vs UWM/Cosmos Policy**: 统一视频-动作表示方法，但高维联合 embedding 不稳定。World2Act 用对比学习在适度维度对齐
- **vs VLA-RFT/Ctrl-World**: 基于奖励的后训练方法，依赖 RL 的 policy gradient，World2Act 免奖励

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 潜在空间对齐 + 技能分解 WM 的组合非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ RoboCasa + LIBERO + 真实世界 + 多基线对比 + 消融
- 写作质量: ⭐⭐⭐⭐ 两阶段设计清晰，技术细节详尽
- 价值: ⭐⭐⭐⭐⭐ WM→VLA 知识转移的新范式，实用且高效
