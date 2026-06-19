---
title: >-
  [论文解读] RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies
description: >-
  [ICML 2026][机器人][记忆增强] RoboMME 首次把人类认知里的"时序/空间/物体/程序"四类记忆系统性映射到 16 个长时机器人操控任务（770k 高质量时间步），并在 π0.5 底座上系统消融 14 种"记忆表征 × 集成方式"，得出"感知记忆 + AdaLN 调制器"是当前最佳综合权衡的结论。
tags:
  - "ICML 2026"
  - "机器人"
  - "记忆增强"
  - "VLA"
  - "机器人基准"
  - "长时操控"
  - "π0.5"
---

# RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies

**会议**: ICML 2026  
**arXiv**: [2603.04639](https://arxiv.org/abs/2603.04639)  
**代码**: https://robomme.github.io/  
**领域**: 机器人  
**关键词**: 记忆增强、VLA、机器人基准、长时操控、π0.5

## 一句话总结
RoboMME 首次把人类认知里的"时序/空间/物体/程序"四类记忆系统性映射到 16 个长时机器人操控任务（770k 高质量时间步），并在 π0.5 底座上系统消融 14 种"记忆表征 × 集成方式"，得出"感知记忆 + AdaLN 调制器"是当前最佳综合权衡的结论。

## 研究背景与动机
**领域现状**：LIBERO、CALVIN、RLBench、SimplerEnv 这些主流操控基准虽然涉及时序，但绝大多数任务实际上是"马尔可夫"的——当前帧加指令就够预测下一动作，过去观测可丢。结果是几乎所有 VLA（π0.5、RoboVLM 等）都在这些基准上拿到很高 SR，却没真正考察记忆能力。

**现有痛点**：少数有意识考察记忆的工作各走各路：MemoryBench 只覆盖三个近乎被解决的空间任务，MIKASA-Robo 任务太短且缺高质量示范，HistRISE / MemoryVLA / ContextVLA / RoboMamba 等记忆增强模型各自用不同 backbone、不同评测协议自卖自夸——无法横向比较哪种"记忆设计"真的更好。

**核心矛盾**：(1) 缺一个"任务确实非马尔可夫且足够大规模"的基准，(2) 缺一个固定 backbone、固定数据预算下系统对比所有主流记忆架构的实验框架；这两个空缺让"记忆增强"领域处于"看似在进步但不知道哪种思路真的赢"的状态。

**本文目标**：(i) 设计一个认知理论驱动、显式非马尔可夫、覆盖四类记忆需求的大规模操控基准；(ii) 在同一个 π0.5 底座、同一数据预算下，把符号/感知/循环三大记忆表征与三种集成方式做完整正交消融。

**切入角度**：Atkinson-Shiffrin 经典记忆模型把长时记忆分成程序性和陈述性，陈述性又分情节（含时序、空间、物体三维）和程序性。作者直接把这四维当作任务设计的轴，每维造一个 task suite，保证基准本身覆盖了认知合理的"记忆刺激空间"。

**核心 idea**：用认知维度组织任务（when/where/what/how），再用"记忆表征 × 集成机制"二维矩阵组织模型，得到的不是"又一个 SOTA"而是"哪种设计在哪类任务上有效"的可解释结论。

## 方法详解
全文有两个独立但耦合的产物：一个是 RoboMME 基准（16 任务、4 suite、1600 示范、770k 时间步），一个是 MME-VLA 模型套件（14 个建在 π0.5 上的记忆变体）。基准给出严格非马尔可夫的评测环境，模型套件给出可控正交消融。

### 整体框架
基准侧用 ManiSkill 仿真 + 7-DoF Franka Panda，前置/腕部双相机 256×256、关节-末端执行器双动作空间；每个任务 100 个 episode、轨迹回放生成；轨迹注入 5% 关键路点噪声并恢复，提升失败-恢复行为。模型侧固定 π0.5 backbone、512 token 的记忆预算、80k step 训练，统一在 50 episodes × 3 seeds × last-3 ckpt 下评测，把所有变量收敛到"记忆设计"本身。

### 关键设计

**1. 认知导向的 4 维任务分类法（Counting / Permanence / Reference / Imitation）：把"记忆"拆成 when/where/what/how 四个可独立评测的维度**

以前的记忆基准要么单测一类、要么把多类混在一起，结果是没法定位"模型到底缺哪种记忆"。RoboMME 直接借 Atkinson-Shiffrin 记忆模型，按四个认知维度各造一个 task suite，每维 4 个难度递增的任务，且全部设计成马尔可夫策略一定会失败的场景。Counting suite 测时序记忆（PickXTimes 要按指定次数重复抓取、StopCube 要在特定时刻按按钮）；Permanence suite 测空间记忆（VideoUnmask/ButtonUnmask 在方块全被遮挡时要靠颜色记忆找目标，Swap 变体还会动态交换容器位置）；Reference suite 测物体记忆（PickHighlight 要拿起短暂高亮过的方块、VideoPlaceOrder 要按语言时序/序数引用执行）；Imitation suite 测程序记忆（MoveCube/InsertPeg/RouteStick 要在看完一段视频示范后复现同样的抓握模式、插入方向、轨迹）。

按认知维度切开后，每个模型的强弱画像可以直接读出来——后续任何记忆方法都能在这四维上画一张雷达图，便于诊断和改进，而不是只在一个混合任务上报一个总分。

**2. MME-VLA 模型套件：三种记忆表征 × 三种集成方式的正交消融矩阵**

之前的记忆方法各用各的 backbone 自报家门，根本不可比。RoboMME 把所有变量锁死在 π0.5 + 512 token 记忆预算下，只留"记忆怎么存 × 记忆怎么用"两个轴变化，做成 14 个变体。表征侧分三路：符号记忆把历史压成自然语言子目标（SimpleSG 仅描述、GroundSG 还带前视图坐标 $[x,y]$，可由 Gemini-2.5-Pro / 微调 Qwen3-VL-4B / Oracle 真值生成）；感知记忆把历史保留为视觉 token，用 token dropping（按 RGB 差异去冗余）或 frame sampling（均匀降采样）压进预算；循环记忆用 TTT（在线更新快权重）或 RMT（学习记忆 slot）把序列压成定长隐状态。集成侧也三路：Memory-as-Context 直接拼到 VLM 输入，Memory-as-Modulator 用 AdaLN 把记忆经多头注意力转成 scale/shift 调制动作特征，Memory-as-Expert 加一条记忆专家通路与主网络做 blockwise causal attention。

锁死 backbone 和预算这一步是整个工作的科学性所在——剩下的性能差异可以干净地归因到"记忆设计"本身，而不是混进了 backbone 强弱或数据量差异。

**3. 严格非马尔可夫的数据构造与统一评测协议：从源头切断 shortcut，再锁住所有混淆变量**

很多"记忆基准"的任务其实当前帧也能解，导致无记忆 baseline 也能拿高分。RoboMME 要求每个任务"相同观测可能对应不同历史 → 不同正确动作"（如同样看到红色按钮，前面已按 2 次还是 5 次会触发不同子动作），视频条件任务只在初始步给历史帧 + 配对 proprioception、执行中只给当前帧，从根上逼模型必须用历史。评测端同样把混淆变量锁死：动作 chunk 训练长度 20、执行长度 16，每任务 50 episodes、最大 1300 步，结果在最后 3 ckpt × 3 seeds 共 9 次运行下取平均，并把 π0.5、π0.5 w/ past actions、SAM2Act+、MemER 作为外部对照。

切断 shortcut 是评测记忆的必要前提，统一协议则让 14 个变体的横向对比真正可信——这正是该领域此前最缺的东西。

### 损失函数 / 训练策略
所有模型用 π0.5 原生 flow matching 动作扩散损失做多任务联合训练；非循环记忆变体 batch=64，循环记忆变体（TTT/RMT）因显存大降到 batch=16；统一 80k 步；符号记忆的 QwenVL 在 1600 个示范的子目标标注上做监督微调，Gemini 仅靠 prompt 工程。

## 实验关键数据

### 主实验（部分代表性变体在 16 任务上的平均 SR%）

| 模型类别 | 变体（记忆+集成） | 平均 SR (%) |
|------|------|------|
| 无记忆基线 | π0.5 | 17.93 |
| 无记忆基线 | π0.5 + past actions | 19.73 |
| 外部 SOTA | SAM2Act+ | 21.37 |
| 外部 SOTA | MemER | 42.38 |
| 符号 (oracle 上界) | GroundSG + Oracle | **84.08** |
| 符号 (真实 VLM) | GroundSG + QwenVL | 32.70 |
| 感知 (本文最佳) | **FrameSamp + Modul** | **44.51** |
| 感知 | TokenDrop + Modul | 38.04 |
| 感知 | FrameSamp + Expert | 36.25 |
| 循环 | TTT + Context | 22.28 |
| 循环 | RMT + Context | 19.46 |
| 人类参考 | Human | 90.50 |

### 消融实验（按 suite 看 FrameSamp+Modul vs π0.5 基线）

| Suite（代表任务） | π0.5 基线 | FrameSamp+Modul | 提升 |
|------|---------|---------|---------|
| Counting (StopCube) | 6.67 | 42.00 | +35.3 |
| Permanence (VideoUnmaskSwap) | 18.67 | 24.44 | +5.8 |
| Reference (VideoRepick) | 0.44 | 30.44 | +30.0 |
| Imitation (RouteStick) | 4.67 | 66.67 | +62.0 |

### 关键发现
- **没有银弹**：14 个变体里没有任何一种能在所有 4 类记忆上同时领先；符号记忆在 Counting/Permanence 这种"高层逻辑离散"任务上强，但在 Imitation 这种连续动作模仿上弱；感知记忆反之
- **感知 + AdaLN 调制器是综合最佳**：FrameSamp+Modul 平均 44.51 超过所有其他可训练变体，比无记忆 π0.5 涨 26.6 个点，验证了"动作通路直接条件化记忆"比"塞进 prompt"更高效
- **循环记忆全面落后**：TTT/RMT 把历史压成定长隐状态会损失太多视觉细节，平均 SR 不到 23，反不如直接保留视觉 token；提示当前 SSM 风格的循环表征还不够强
- **GroundSG+Oracle 84.08 显示上界很高**：只要子目标信息足够准（含 [x, y] 坐标），简单子目标拼接就能逼近人类水平；瓶颈完全在子目标预测器，不在 VLA 本身——把 VLM 当作"记忆生成器"是有前景的方向
- **MemER 42.38 vs SAM2Act+ 21.37**：混合"感知关键帧 + 符号子目标"的 MemER 比纯感知记忆 bank 的 SAM2Act+ 翻倍，再次说明记忆设计上"语言-视觉混合"比单模态更鲁棒

## 亮点与洞察
- 把"记忆"从一个被滥用的形容词回归成可解构的认知维度（when/where/what/how），是该领域少见的科学化尝试——后续任何记忆方法都可以在这四维上画雷达图
- "表征 × 集成"二维消融矩阵把模型设计空间正交化，是个通用方法论：任何"加一个新模块到已有模型"的工作都该按这种网格做对比，而不是只在一个组合下报数
- 发现 FrameSamp+Modul 优于 Context 和 Expert，间接说明在 flow matching 动作头里 AdaLN 调制比 prompt concat 更适合"非语义型条件信号"，这对 DiT 风格的扩散动作模型设计有迁移价值
- 把人类表现（90.5）和 Oracle 上界（84.08）都标出来——前者揭示当前最佳 VLA 与人类还有 46 分 gap，后者揭示"如果子目标预测器完美则差距只剩 6 分"，把后续研究重心清晰指向"子目标 VLM 的精度提升"

## 局限与展望
- 全在 ManiSkill 仿真里，缺真实机器人验证；尤其 Permanence 类任务在真实遮挡和传感器噪声下表现可能大幅下滑
- 16 任务都是桌面单臂场景，缺移动操控、双臂协作、人机交互等更复杂的记忆需求
- 512 token 的记忆预算是固定的，没研究"预算-性能"曲线；不同任务可能需要不同预算
- 循环记忆的实现选择（TTT、RMT）相对保守，没尝试更强的 Mamba-2/Griffin 等近期 SSM 变体；它们的潜力可能被低估
- 子目标标注依赖仿真器真值，移植到真实场景需要更通用的标注流程

## 相关工作与启发
- **vs MemoryBench**: 后者只 3 个空间任务且接近 saturate；RoboMME 覆盖 4 类记忆共 16 个非马尔可夫任务、规模大一个量级
- **vs MIKASA-Robo**: MIKASA 任务短、示范少，主要针对 RL；RoboMME 提供 770k 高质量示范支持大规模模仿学习
- **vs MemoryVLA / SAM2Act / MemER**: 这些方法在自定义任务上各报各的；RoboMME 第一次把它们和 14 个新变体放在同一基准、同一 backbone 下横向比较
- **vs ContextVLA / UniVLA**: 它们用过去帧或过去动作做简单拼接，本工作进一步把"集成方式"细化为 Context/Modulator/Expert 三类并系统对比，发现 Modulator 显著更优

## 评分
- 新颖性: ⭐⭐⭐⭐ 基准设计的认知导向 + 消融矩阵的正交化思路都很有方法论价值，单点技术不算新但组合罕见
- 实验充分度: ⭐⭐⭐⭐⭐ 14 个变体 × 16 任务 × 9 次运行 + 4 个外部 baseline，几乎是"该测的都测了"
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰、表格信息密度高；缺一些失败案例的可视化分析
- 价值: ⭐⭐⭐⭐⭐ 给"记忆增强机器人策略"提供了第一个真正可用的标准化评测平台，影响力会随时间增长

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](../../ICLR2026/robotics/robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [\[ICML 2026\] TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation](tapsampling_inference-time_sampling_with_a_task-progress-understanding_verifier_.md)
- [\[CVPR 2026\] FM-Steer: Enhance Generalist Policies with Value-Guided Cascaded Denoising](../../CVPR2026/robotics/fm-steer_enhance_generalist_policies_with_value-guided_cascaded_denoising.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)
- [\[CVPR 2026\] OctoNav: Towards Generalist Embodied Navigation](../../CVPR2026/robotics/octonav_towards_generalist_embodied_navigation.md)

</div>

<!-- RELATED:END -->
