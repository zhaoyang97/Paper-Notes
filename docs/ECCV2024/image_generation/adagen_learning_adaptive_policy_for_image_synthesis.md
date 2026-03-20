# AdaGen: Learning Adaptive Policy for Image Synthesis

**会议**: ECCV 2024 (AdaNAT的扩展版本)  
**arXiv**: [2603.06993](https://arxiv.org/abs/2603.06993)  
**代码**: [https://github.com/LeapLabTHU/AdaGen](https://github.com/LeapLabTHU/AdaGen)  
**领域**: 图像生成 / 生成模型优化  
**关键词**: 强化学习, 自适应调度策略, 对抗奖励, 多范式生成模型, 推理优化  

## 一句话总结

将多步生成模型（MaskGIT/AR/Diffusion/Rectified Flow）的步级参数调度（温度、mask ratio、CFG scale、timestep等）统一建模为MDP，用轻量RL策略网络实现样本自适应调度，并提出对抗奖励设计防止策略过拟合，在四种生成范式上一致提升性能（VAR FID 1.92→1.59，DiT-XL推理成本降3倍同时性能更优）。

## 背景与动机

现代图像生成模型（MaskGIT、自回归VAR、扩散DiT、整流流SiT）的共同特点是将生成过程分解为多步迭代，每一步都有多个需要配置的参数（mask ratio、采样温度、CFG scale、ODE timestep等）。以MaskGIT为例，32步生成需要配置128个策略参数。

现有做法依赖人工设计的静态调度规则（如余弦schedule、固定常数），这存在两个核心痛点：
1. **手动调参代价高**：参数组合空间随步数指数增长，需要专家知识和大量trial-and-error
2. **静态调度次优**：所有样本共享同一套调度，无法根据每个样本的特性（简单/复杂结构）自适应调整

## 核心问题

能否用一个**通用的、可学习的、样本自适应的**框架来自动配置多步生成模型的迭代策略？核心挑战在于：（1）端到端反向传播计算代价过高且部分操作不可微；（2）如何设计有效的奖励信号引导策略学习——简单的FID或预训练奖励模型都容易被"hack"。

## 方法详解

### 整体框架

输入：预训练好的生成模型（冻结不动）+ 当前生成中间状态 → 轻量策略网络输出本步的生成策略参数 → 生成模型执行一步生成 → 反复迭代直到完成 → 对抗奖励模型评估最终图像质量并提供RL训练信号。

核心思想：**不改生成器本身，只学一个旁路策略网络来"指挥"生成过程**。策略网络用PPO训练，奖励模型与策略网络对抗训练。

### 关键设计

1. **统一MDP建模**：将四种生成范式（MaskGIT、AR、Diffusion ODE、Rectified Flow）的调度问题统一为MDP。状态=（当前步t, 中间生成结果），动作=该步的策略参数向量，奖励仅在最终步给出。不同范式的区别体现在状态转移函数——MaskGIT/AR是随机转移，Diffusion/Flow是确定性ODE求解。策略网络架构为Conv+MLP+AdaLN（注入步数信息），利用生成模型的中间特征而非原始中间结果作为输入，极其轻量（仅占生成器0.07%-0.40%计算量）。

2. **对抗奖励模型**：这是本文最关键的设计。作者发现直接用FID或预训练奖励模型（如ImageReward）作为RL奖励会导致策略过拟合——FID改善但图像质量实际很差，或者生成风格单一多样性丧失。解决方案是引入一个GAN式判别器作为奖励模型，与策略网络对抗训练：策略网络最大化奖励时，判别器同步更新以更好区分真假图像。这让奖励信号"活的"，有效避免策略过拟合到某个静态目标上。

3. **动作平滑（Action Smoothing）**：当生成步数增大（如T=32）时，策略网络的动作序列出现剧烈波动，训练不稳定。作者发现这些高频波动是不必要的——简单线性插值反而效果更好。分析原因是PPO的高斯探索在每步独立加噪，产生不合理的探索轨迹。解决方案是对策略输出施加EMA滤波（指数移动平均，β=0.8），这是一个因果低通滤波器，既平滑了动作序列又保持了MDP的马尔可夫性。

4. **推理时精化**：训练后的辅助网络可复用——（a）用对抗奖励模型做repeated sampling（多次生成选最高分的）；（b）对随机转移模型（如MaskGIT），用值网络V做lookahead sampling（每步采K个候选状态，选V值最高的继续）。两者结合使MaskGIT-L的FID从2.28降至1.81。

5. **可控保真-多样性权衡**：引入第二个面向保真度的策略网络，用参数λ线性混合两个策略的输出，同时混合对抗奖励和ImageReward。λ=0偏多样性，λ=1偏保真度。

### 损失函数 / 训练策略

- 策略网络用PPO（clipped surrogate objective）优化，包含advantage估计和值函数损失
- 对抗奖励模型用标准GAN的minimax objective优化：$\max_\phi \min_\psi \mathbb{E}[\log r_\psi(x_{\text{fake}})] + \mathbb{E}[\log(1-r_\psi(x_{\text{real}}))]$
- 两者交替优化：先用策略网络采样更新策略，再用真/假图像更新判别器
- 探索噪声σ=0.6，PPO clip ε=0.2，值函数系数c=0.5
- T>10时启用action smoothing β=0.8

## 实验关键数据

| 数据集 | 模型 | 指标 | Baseline | AdaGen | 提升 |
|--------|------|------|----------|--------|------|
| ImageNet 256 | DiT-XL (16步) | FID-50K | 3.31 | 2.19 | -1.12 |
| ImageNet 256 | DiT-XL (8步) | FID-50K | 5.18 | 2.82 | -2.36 |
| ImageNet 256 | SiT-XL (16步) | FID-50K | 2.99 | 2.12 | -0.87 |
| ImageNet 256 | VAR-d30 (10步) | FID-50K | 1.92 | 1.59 | -0.33 |
| ImageNet 256 | MaskGIT-L (16步) | FID-50K | 3.79 | 2.41 | -1.38 |
| ImageNet 512 | MaskGIT-L (32步) | FID-50K | 7.32 | 2.46 | -4.86 |
| MS-COCO | MaskGIT-S (16步) | FID-30K | 5.78 | 4.92 | -0.86 |
| LAION-5B→COCO | Stable Diffusion (32步) | FID-30K | 9.03 | 8.14 | -0.89 |

系统级对比：AdaGen-DiT-XL 16步(4.1 TFLOPs) FID=2.19 优于 原始DiT-XL 50步(12.2 TFLOPs) FID=2.29，推理成本降约3倍。

推理时精化：MaskGIT-L + repeated + lookahead sampling: FID 2.28→1.81

### 消融实验要点

- **可学习 vs 自适应**的增量贡献：可学习(非自适应)已带来显著提升（MaskGIT FID 7.65→5.40），自适应进一步改善（FID 5.40→4.54），两者都重要
- **奖励设计**：FID做奖励→图像模糊；PretrainedRewardModel→多样性丧失；对抗奖励→质量和多样性平衡最好
- **Action smoothing**β=0.8最优，β=0(不平滑) FID=3.97，β=0.8 FID=3.36
- **策略网络输入**：用生成器中间特征(FID 4.54) >> 原始中间结果(FID 6.55)
- **步数条件**：去掉步数条件FID从4.54升至6.13，策略决策需要知道"现在走到哪一步了"
- **策略网络架构**：Conv+MLP(12M参数)已经足够，更大模型边际收益极小
- **判别器架构**：Transformer-based(FID 4.54) >> Conv-based(FID 5.86)
- **PPO超参**：对ε、c、σ均不敏感，各种设置都显著优于baseline

## 亮点

- **"不改模型只改调度"的范式**极其优雅——策略网络仅占生成器0.07-0.40%计算量,却能带来17-54%的性能提升或1.6-3.6倍推理加速，实用价值极高
- **对抗奖励设计**精准解决了RL优化生成模型时的奖励过拟合问题，这个insight对整个RLHF for vision领域都有参考价值
- **Action smoothing**的信号处理视角非常巧妙——将RL策略探索中的高频噪声问题用经典IIR滤波解决，简单有效
- **统一四种生成范式**的MDP建模思路清晰，Table I和Table II的总结非常赏心悦目
- 推理时将训练副产品（判别器+值网络）复用做精化，不浪费任何训练成果

## 局限性 / 可改进方向

- 虽然策略网络本身轻量，但RL训练过程需要大量采样，训练成本未详细讨论
- 对抗奖励的训练稳定性在大规模模型（如SD-XL、FLUX）上是否仍然成立有待验证
- 目前仅处理推理侧调度参数（如CFG scale、timestep），未涉及更深层的模型架构自适应（如层跳过、token选择等）
- 与最新的蒸馏方法（consistency distillation等）的结合/对比不够充分
- 对应idea拓展: [RL自适应策略配置：从NAT图像生成到通用生成模型推理优化](../../../ideas/image_generation/20260317_rl_adaptive_inference_generation.md)

## 与相关工作的对比

- **vs AdaNAT (ECCV 2024)**：AdaGen是AdaNAT的直接扩展版本——AdaNAT仅在MaskGIT上验证，AdaGen推广到四种范式且新增action smoothing、推理时精化和保真-多样性控制机制
- **vs DDPO/DRaFT等Diffusion RLHF方法**：这些方法用RL微调生成器本身的参数，而AdaGen冻结生成器仅学旁路策略网络，计算成本低得多且不影响原始模型能力；另外AdaGen的对抗奖励解决了预训练奖励模型导致的多样性丧失问题
- **vs 自动采样器搜索（AutoDiffusion/USF等）**：这些方法学到的是全局共享的静态调度，不具备样本自适应能力；且大多仅针对扩散模型

## 启发与关联

- 与 [RL自适应策略配置](../../../ideas/image_generation/20260317_rl_adaptive_inference_generation.md) 直接相关——该idea正是从AdaNAT出发提出将框架推广到扩散模型推理优化，而AdaGen已部分实现了这一设想
- 对抗奖励的设计可迁移到 [过程感知的在线偏好对齐](../../../ideas/image_generation/20260316_process_aware_alignment.md) ——用动态奖励替代静态奖励模型可能缓解过程级优化中的奖励过拟合
- "不改模型只改推理调度"的思路可与 [DPCache](../../CVPR2026/image_generation/dpcache_denoising_path_planning_diffusion_accel.md) 等token/cache层面的推理加速方法正交组合

## 评分

- 新颖性: ⭐⭐⭐⭐ 对抗奖励和统一MDP建模有新意，但核心RL+调度的思路在AdaNAT中已建立
- 实验充分度: ⭐⭐⭐⭐⭐ 四种范式×五个数据集×大量消融，实验极其充分
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，Table I/II的统一建模可视化很赞，方法讲解循序渐进
- 价值: ⭐⭐⭐⭐ 实用价值高（0.07%开销换显著性能提升），但需验证在更大模型上的效果
