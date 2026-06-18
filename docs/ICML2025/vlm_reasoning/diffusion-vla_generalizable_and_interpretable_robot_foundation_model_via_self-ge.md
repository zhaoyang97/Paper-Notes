---
title: >-
  [论文解读] Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning
description: >-
  [ICML 2025][多模态VLM][视觉语言] 提出 DiffusionVLA (DiVLA)，将自回归 VLM 的推理能力与扩散模型的动作生成能力统一到一个端到端框架中，通过推理注入模块（Reasoning Injection Module）将自生成的语言推理直接嵌入策略学习过程，实现了对未见物体的泛化分类、可解释的动作决策以及高速推理（2B 模型 82Hz）。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "视觉语言"
  - "扩散策略"
  - "自回归推理"
  - "机器人操控"
  - "多模态基础模型"
---

# Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning

**会议**: ICML 2025  
**arXiv**: [2412.03293](https://arxiv.org/abs/2412.03293)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: Vision-Language-Action, 扩散策略, 自回归推理, 机器人操控, 多模态基础模型

## 一句话总结

提出 DiffusionVLA (DiVLA)，将自回归 VLM 的推理能力与扩散模型的动作生成能力统一到一个端到端框架中，通过推理注入模块（Reasoning Injection Module）将自生成的语言推理直接嵌入策略学习过程，实现了对未见物体的泛化分类、可解释的动作决策以及高速推理（2B 模型 82Hz）。

## 研究背景与动机

现有的机器人基础模型存在两条路线的根本矛盾：

**自回归 VLA 模型**（如 RT-2、OpenVLA）：将动作预测建模为 next-token prediction，继承了 LLM 的推理能力，但将连续动作离散化为固定大小的 token 会破坏动作的连贯性和精度，且逐步生成 token 的方式在实时控制场景中推理效率低下。

**扩散策略模型**（如 Diffusion Policy）：通过噪声-去噪过程建模动作序列，能更好地捕捉机器人动作的多模态分布，生成速度也更快，但天然缺乏推理能力，难以处理需要语义理解的复杂任务。

核心问题：**能否将自回归模型的推理能力与扩散模型的鲁棒高频动作生成优势结合起来？** 简单拼接两者并不能充分利用推理潜力，因为逻辑推理与可执行的机器人策略之间存在隐式鸿沟。

## 方法详解

### 整体框架

DiVLA 构建于预训练的视觉语言模型（VLM）之上，整体分为三个核心组件：

1. **视觉编码器**：使用 SigLIP 将多视角图像（外部相机 + 腕部相机）编码为稠密视觉特征，再通过 Transformer 映射为固定数量 $N$ 个视觉 embedding。对多视角输入，共享 SigLIP 骨干编码各视角后拼接 token。

2. **VLM 骨干**：采用 Qwen2-VL（2B/8B/72B 三种规格），保留其自回归文本生成能力用于推理。使用公开预训练权重初始化。框架设计解耦了视觉语言理解与动作生成，因此可灵活替换为其他 VLM。

3. **扩散动作头**：VLM 最后一层 embedding 生成固定数量的 action token，经投影模块（两层 MLP + LayerNorm，类似 LLaVA 的 projector 设计）转换后送入标准 Diffusion Policy 解码器，随机初始化权重。最底层附加一个 MLP 预测机器人关节空间动作。对于多形态机器人，只需初始化新的 MLP 层即可快速适配。

### 关键设计

#### 推理注入模块 (Reasoning Injection Module)

这是本文最核心的贡献。不同于大多数自回归 VLA 需要递归地将推理输出转化为下一轮输入的方式，DiVLA 提出更高效的直接嵌入策略：

- **推理生成**：VLM 自回归地生成任务分解和解释文本（如"抓取蓝色玩具车"、"放入玩具区域"）
- **注入机制**：取推理输出的 tokenized 最终 embedding，通过 **Feature-wise Linear Modulation (FiLM)** 直接注入策略网络各层。FiLM 通过学习仿射变换 $\gamma$ 和 $\beta$ 来调制策略网络特征：

$$h_{out} = \gamma(r) \odot h_{in} + \beta(r)$$

其中 $r$ 是推理 embedding，$h_{in}$ 是策略网络的中间特征。

- **设计哲学**：策略网络主要处理动作相关 token，推理模块作为辅助增强，提供语义上下文而不主导主决策流。这种"注入"而非"串联"的设计避免了迭代输入输出循环带来的计算和操作复杂性。

#### 推理数据增强

原始 Droid 预训练数据仅包含机器人动作和部分观测/语言指令，缺乏推理信息。作者利用 GPT-4o 自动将原始数据转化为包含推理标注的形式，使网络架构在预训练和微调阶段保持一致。

#### 多形态适配

当需要部署到不同机器人形态时，不需要像 Octo 那样复制独立的动作解码器，仅需初始化新的 MLP 层进行训练评估，保留预训练数据的知识实现快速适配。

### 损失函数 / 训练策略

**联合训练目标**：

$$L = L_{diff} + \alpha \cdot L_{ntp}$$

- $L_{diff}$：扩散损失，用于动作生成
- $L_{ntp}$：下一个 token 预测损失，用于推理文本生成
- 实验观察 $L_{ntp}$ 的量级始终约为 $L_{diff}$ 的十分之一，因此设 $\alpha = 10$ 以平衡两项贡献

**训练细节**：
- 预训练数据：DiVLA-2B/7B 用 Droid（39K 轨迹），DiVLA-72B 用 OXE + Droid
- 微调方式：视觉编码器和 VLM 冻结，使用 LoRA 微调 VLM
- 学习率：固定 2e-5
- 微调数据：同一形态的任务数据合并训练

## 实验关键数据

### 主实验

**多任务学习（5项任务，Franka 机器人）**：

| 模型 | 预训练轨迹数 | 分布内平均成功率 | 视觉泛化平均成功率 |
|------|------------|----------------|-----------------|
| Diffusion Policy | - | 27.9% | 8.9% |
| TinyVLA | - | 45.5% | 28.9% |
| Octo | 970K | 24.3% | 17.8% |
| OpenVLA-7B | 970K | 39.4% | 26.7% |
| **DiVLA-2B** | **39K** | **83.6%** | **57.8%** |

**零样本 Bin Picking（102 个未见物体）**：

| 模型 | 成功率 |
|------|-------|
| Diffusion Policy | 8.9% |
| Octo | 19.6% |
| TinyVLA | 23.5% |
| OpenVLA | 28.4% |
| **DiVLA** | **63.7%** |

**双臂桌面清理（AgileX 双臂机器人）**：

| 场景 | Diffusion Policy | OpenVLA | DiVLA-2B |
|------|-----------------|---------|----------|
| 已见物体 | 45.8% | 0% | **72.9%** |
| 混合物体 | 31.2% | 0% | **70.8%** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅扩散策略（无推理） | 27.9% (多任务) | 缺乏语义理解导致泛化差 |
| 仅自回归 VLA | 39.4% (OpenVLA) | 动作精度和速度受限 |
| DiVLA 无推理注入 | 未单独报告 | 推理与动作存在隐式鸿沟 |
| DiVLA 完整方案 | 83.6% (多任务) | 推理注入显著提升泛化能力 |
| DiVLA-2B 推理速度 | 82 Hz | 单卡 A6000 |
| DiVLA-7B 推理速度 | 42 Hz | 单卡 A6000 |

### 关键发现

1. **数据效率极高**：DiVLA-2B 仅用 39K 预训练轨迹就超越了用 970K 轨迹预训练的 Octo 和 OpenVLA，分布内成功率提升 44.2%（绝对值）。

2. **视觉鲁棒性**：在添加干扰物、更换背景、改变光照三种视觉变化下，所有方法都会性能下降，但 DiVLA 始终保持最高成功率，无需任何数据增强技术。

3. **工厂分拣表现**：在复杂场景（6-11 个物体随机堆叠）中，其他方法性能急剧下降（DP 降至 9.2%），DiVLA 仍保持约 60% 成功率，整体平均 66.2%，超越第二名 OpenVLA 达 20.9%。

4. **推理实现可解释性**：模型能自适应调整推理——当目标物被替换时，推理文本从"抓取蓝色玩具车"即时切换为"抓取六角扳手"，表明模型在进行上下文感知的决策而非执行预编程动作。

5. **未见物体的类比泛化**：模型能将螺丝刀类比归类为六角扳手（基于视觉相似性），将绿色手套识别为"绿色手套"，表现出超越直接识别的语义泛化能力。

6. **模型可扩展性**：DiVLA 家族从 2B 扩展到 72B，泛化和性能随模型规模提升，符合 scaling law。

## 亮点与洞察

1. **架构设计的优雅性**：通过 FiLM 注入而非递归串联来融合推理与动作，既保留了推理对策略的增强作用，又避免了额外推理开销——推理注入不增加推理时的计算量。

2. **"会说又会做"的统一模型**：DiVLA 同时保留了对话问答能力和机器人控制能力，这在之前的单一模型中很难实现。

3. **实用价值**：82Hz 的推理速度满足实时控制需求；通过观察自生成推理文本可以直接诊断失败原因，这对实际部署至关重要。

4. **轻量级跨形态迁移**：仅更换最后一层 MLP 即可适配新机器人形态（如从 Franka 到 AgileX 双臂），知识复用效率高。

## 局限与展望

1. **缺乏仿真实验对比**：论文主要在真实机器人上评估，缺少与更多 baseline 在标准仿真基准（如 CALVIN、RLBench）上的对比。
2. **推理数据依赖 GPT-4o**：预训练阶段的推理标注需要 GPT-4o 生成，成本较高且可能引入噪声；未来可探索自动化或更廉价的推理数据获取方式。
3. **消融不够充分**：推理注入模块的具体贡献（与不注入相比）未以独立消融的形式量化呈现。
4. **单一推理粒度**：当前推理仅产生短语级别的描述，未探索更细粒度或多步骤的推理链对复杂长序列任务的影响。
5. **72B 模型实用性存疑**：虽然展示了 scaling 效果，但 72B 模型的推理速度未报告，其实时控制可行性有待验证。

## 相关工作与启发

- **RT-2 / OpenVLA**：开创了将 NTP 应用于机器人学习的范式，但动作离散化和推理速度是瓶颈。DiVLA 通过将 NTP 限定于推理任务、将动作交给扩散模型来解决。
- **Diffusion Policy**：证明了扩散模型在捕捉多模态动作分布上的优势，但完全缺乏语言推理。DiVLA 补充了这一缺失。
- **π₀ (Physical Intelligence)**：同样探索了将语言模型与动作生成结合，但使用 flow matching 而非 diffusion。DiVLA 的推理注入模块提供了更明确的可解释性。
- **Transfusion / Show-O**：在图像生成领域探索了统一 NTP 与扩散的方法，DiVLA 将这一思路迁移到了机器人领域。
- **FiLM 条件化**：灵感来自 RT-1 和 YAY，将语言条件调制到策略网络的方法已有先例，DiVLA 创新在于将自生成推理（而非人类指令）作为调制信号。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 推理注入模块的设计有新意，但统一自回归和扩散模型的思路在视觉生成领域已有探索
- **实验充分度**: ⭐⭐⭐⭐ — 多种真实机器人多任务评估，但缺少仿真基准和充分的消融实验
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，但部分实验细节需要在附录补充
- **价值**: ⭐⭐⭐⭐⭐ — 高实用价值，82Hz 实时推理 + 可解释决策 + 强泛化能力对机器人部署意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Revisiting Model Stitching in the Foundation Model Era](../../CVPR2025/multimodal_vlm/revisiting_model_stitching_in_the_foundation_model_era.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/aigi_holmes_towards_explainable_and_generalizable_ai_generated_image_detection_via_mllm.md)
- [\[ICLR 2026\] Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences](../../ICLR2026/multimodal_vlm/evaluating_vlms_spatial_reasoning_over_robot_motion_a_step_towards_robot_plannin.md)
- [\[ICCV 2025\] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model](../../ICCV2025/multimodal_vlm/interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)
- [\[ACL 2025\] LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions](../../ACL2025/multimodal_vlm/logicqa_logical_anomaly_detection_with_vision_language_model_generated_questions.md)

</div>

<!-- RELATED:END -->
