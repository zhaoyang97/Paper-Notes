# Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment

**会议**: CVPR 2025  
**arXiv**: [2412.19326](https://arxiv.org/abs/2412.19326)  
**代码**: https://github.com/OpenGVLab/TPO  
**领域**: 多模态VLM  
**关键词**: MLLM, 多任务学习, 任务对齐, 视觉感知头, 视频理解, 目标跟踪, 时序定位

## 一句话总结
提出 Task Preference Optimization（TPO），通过可学习的任务 token 将视觉任务专用头（区域定位/时序定位/分割）接入 MLLM，利用视觉任务标注作为"任务偏好"反向优化 MLLM，在不损害对话能力的前提下大幅提升细粒度视觉理解，VideoChat 基线上平均提升 14.6%。

## 研究背景与动机
1. **领域现状**：当前 MLLM（如 LLaVA、VideoChat）在通用视觉对话上表现出色，但在细粒度视觉任务（如跟踪、时序定位、分割）上能力不足，存在精确感知的短板。
2. **现有痛点**：已有方案要么把视觉任务文本化后自回归预测（P2S 方式，如 Shikra、TimeChat），要么接外部工具（P2E 方式，如 LISA）。但文本化方案会因离散化丢失精度，同时多任务混训常导致原有对话能力下降——这与视觉基础模型中"多任务互利"的经验矛盾。
3. **核心矛盾**：作者观察到冲突根源在于离散文本 token 与密集视觉预测之间的学习差异——用自回归文本 loss 去学 bounding box 坐标或 timestamp 本身就不匹配。解耦二者的表征即可解决。
4. **本文要解决的问题**：如何在不损害 MLLM 对话能力的前提下，端到端地引入多种细粒度视觉任务的精确监督，且不同任务之间能相互促进？
5. **切入角度**：借鉴 DPO 中用偏好信号引导 LLM 的思路——视觉任务的标注可以看作人类对"精确感知"的偏好，通过专用头的可微损失反传来优化 MLLM。
6. **核心idea**：引入可学习的任务 token 作为 MLLM 和视觉任务头之间的桥梁，任务头接受密集视觉监督并将梯度传回 MLLM，实现"用视觉任务监督优化语言模型"。

## 方法详解

### 整体框架
MLLM-TPO 由两部分构成：标准 MLLM $M$（视觉编码器 $E$ + 连接器 $C$ + LLM $G$）和任务偏好模型 TPM $P$（可学习任务 token $\{\mathbf{v}_i\}$ + 任务头 $\{H_i\}$）。

输入：图像/视频 + 用户指令。MLLM 首先识别指令中涉及的任务类型，激活对应的任务 token。任务 token 经过 LLM 得到 embedding $\mathbf{e}_j = G(\mathbf{v}_j)$，送入对应视觉任务头做密集预测。

总优化目标：$\mathcal{L} = \mathcal{L}_{\text{mllm}} + \mathcal{L}_{\text{assign}}(G(\mathbf{T}_q), \mathbf{s}) + \sum_{i=1}^{n} \mathcal{L}_{\text{task}}(\mathbf{A}_i, H_i(G(\mathbf{v}_i)))$

其中 $\mathcal{L}_{\text{mllm}}$ 是标准对话 loss，$\mathcal{L}_{\text{assign}}$ 是任务类型分类 loss，$\mathcal{L}_{\text{task}}$ 是各视觉任务头的监督 loss（回归/分类）。

### 关键设计

1. **三种视觉任务头（Task Preference Model）**：
   - 做什么：分别处理空间定位、时序定位和像素级分割三类核心视觉感知任务
   - **Region Head**：2 层 MLP + ReLU，将 LLM embedding 回归到 bounding box 坐标，用于空间定位（referring expression grounding）
   - **Temporal Head**：基于 CG-DETR 架构，包含视频编码器和文本编码器，接受 temporal task embedding 后预测 moment 的起止时间和 highlight 分数，用于时序定位
   - **Mask Head**：复用 SAM2 的图像编码器和 mask decoder，替换 prompt encoder 为单层 MLP（mask adapter），实现 referring segmentation 和跟踪
   - 设计动机：这三类头覆盖了大部分判别式视觉任务，且各自有成熟的专家模型架构可复用

2. **可学习任务 Token 作为桥梁**：
   - 做什么：解耦视觉任务表征与 MLLM 文本表征
   - 核心思路：每种任务对应一个可学习 token $\mathbf{v}_i \in \mathbb{R}^{1 \times C}$，输入 LLM 后输出 task embedding $\mathbf{e}_i = G(\mathbf{v}_i)$，再送入对应任务头。这样任务的密集视觉监督通过梯度反传从头 → embedding → LLM，间接增强 MLLM 的视觉理解能力
   - 设计动机：避免将视觉任务硬转成文本（导致信息丢失），同时让任务梯度能流回 LLM

3. **三阶段 Local-to-Global 训练**：
   - **Stage 1 (Task Assignment)**：用 LoRA 微调 LLM，学会根据用户指令识别任务类型并生成对应特殊 token（每任务 50k 样本）
   - **Stage 2 (Vision Task Training)**：分别训练各任务头和对应任务 token，使头具备初步功能，冻结视觉编码器和连接器
   - **Stage 3 (Multi-task Training)**：解冻所有模块，混合多任务数据和对话数据联合训练，任务头的梯度反传到整个 MLLM
   - 设计动机：local-to-global 策略避免了一步到位的联合训练导致 MLLM 对话能力退化。分阶段让模型逐步适应

### 损失函数 / 训练策略
- Region Head：MSE loss（回归坐标）
- Temporal Head：CG-DETR 原有 loss（回归 + 分类）
- Mask Head：SAM2 原有 mask loss
- 训练配置：32-64 A100，batch size 128-256，LR 2e-5，LoRA rank=16, alpha=32，DeepSpeed bf16
- 数据量：Stage 1 约 150k，Stage 2 约 700k，Stage 3 约 3.5M（其中大部分为对话数据）

## 实验关键数据

### 主实验
基于 VideoChat2 (Mistral-7B, 16帧) 的多模态视频理解：

| 模型 | MVBench | VideoMME | MLVU |
|------|---------|----------|------|
| VideoChat2 (baseline) | 60.4 | 39.5 | 44.5 |
| **VideoChat-TPO** | **66.8 (+6.4)** | **48.8 (+9.3)** | **54.7 (+10.2)** |
| ST-LLM (64帧) | 54.9 | 37.9 | — |
| ShareGPT4Video | 51.2 | 39.9 | 46.4 |

视觉任务表现（零样本 + 微调）：

| 任务 | 数据集 | VideoChat-TPO | 对比方法 |
|------|--------|---------------|---------|
| 时序定位(零样本) | Charades-STA R@0.5 | **40.2** | ChatVTG: 33.0 |
| 时序定位(微调) | Charades-STA R@0.5 | **65.0** | UniVTG: 60.2 |
| 空间定位 | RefCOCO val | **85.9** | NExT-Chat: 85.5 |
| 跟踪 | LaSOT Success | **69.4** | Merlin: 39.8 |
| 视频分割(零样本) | MeViS J&F | **47.0** | VideoLISA: 44.4 |

### 消融实验
| 配置 | Charades R@0.5 | RefCOCO Acc@0.5 | MeViS J&F | MVBench |
|------|---------------|----------------|-----------|---------|
| 仅 Temporal Head | 30.2 | — | — | — |
| 仅 Region Head | — | 77.3 | — | — |
| 仅 Mask Head | — | — | 55.1 | — |
| Region + Mask | — | 80.2 | 58.3 | — |
| T + R + M (无对话数据) | 36.7 | 81.6 | 61.4 | — |
| **T + R + M + Conv (完整)** | **40.2** | **82.0** | **63.9** | **66.8** |
| 文本化任务数据替代 | 18.6 | — | — | 64.7 |
| 简单MLP替代CG-DETR | 17.8 | — | — | 65.8 |

### 关键发现
- **多任务协同效应显著**：联合训练 3 个任务头 + 对话数据效果远超单独训练，每增加一个头都带来增量收益。这说明不同视觉任务的感知能力可以互相促进。
- **TPO 优于文本化任务数据**：同等数据下 TPO 在 MVBench 上比 textualized task data 高 2.1%，证明密集视觉监督比文本化的自回归监督更有效。
- **强任务头带来更大收益**：CG-DETR 比简单 MLP 在时序任务上高 22.4 个点，且对整体多模态性能（MVBench +1.0%）也有更大提升。
- **跨模型泛化**：在 LLaVA-OneVision 上 TPO 也带来 MVBench +8.1% 的提升，验证普适性。
- **数据扩展有效**：增加时序推理数据后 Charades R@0.5 从 38.3 提升到 40.2，增加对话数据也持续提升各项指标。

## 亮点与洞察
- **"用视觉监督优化语言模型"的范式很有启发**：通过任务头的密集标注梯度反传来增强 MLLM，比传统的文本化或工具调用方案更优雅，避免了信息损失。
- **可学习任务 token 的设计简洁有效**：一个 token 就能架起 MLLM 和专家头之间的桥梁，且可以灵活扩展新的视觉任务，只需加新 token + 新头。
- **多任务协同发现有实际意义**：跟踪和分割的联合训练互相提升，时序定位和空间定位也有协同效应，这对设计多功能视觉 Agent 很有参考价值。

## 局限性 / 可改进方向
- **仅覆盖判别式视觉任务**：作者承认没有涉及生成式任务（如图像生成、视频生成），限制了 TPO 的任务多样性。
- **依赖人工标注数据**：没有探索自监督或对比学习等无标注方案来提供任务偏好信号，扩展性受限。
- **训练成本不低**：3 阶段训练共需 64 A100 约 63.5 小时，比纯对话微调多约 25%。
- **任务头架构相对固定**：当前三种头是手动选择的，如何自动发现有益的视觉任务组合值得研究。

## 相关工作与启发
- **vs Shikra/TimeChat (P2S 方案)**：它们将视觉任务文本化后自回归预测，导致信息丢失且可能损害对话能力。TPO 用独立的任务头接受原始监督信号，效果更好。
- **vs LISA/NExT-Chat (P2E 方案)**：它们也用外部解码器，但通常只关注单一任务且不强调对话能力的同步提升。TPO 的三阶段训练确保多任务和对话能力同步提升。
- **vs DPO/PPO (偏好对齐)**：传统偏好对齐关注文本回复质量，TPO 将"偏好"概念扩展到视觉感知精度，是偏好对齐思想在视觉领域的新颖应用。

## 评分
- 新颖性: ⭐⭐⭐⭐ "任务偏好"概念新颖，通过可微任务头优化MLLM是有意义的范式
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖6+多模态benchmark和7+视觉任务benchmark，消融非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，三阶段训练解释清楚
- 价值: ⭐⭐⭐⭐ 对构建多功能MLLM提供了有效的训练方法论
