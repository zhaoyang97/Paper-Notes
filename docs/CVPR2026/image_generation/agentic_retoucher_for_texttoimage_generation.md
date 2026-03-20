# Agentic Retoucher for Text-To-Image Generation

**会议**: CVPR 2026  
**arXiv**: [2601.02046](https://arxiv.org/abs/2601.02046)  
**代码**: 无  
**领域**: 图像生成 / Agent / 图像质量评估  
**关键词**: T2I后处理, 感知-推理-行动循环, 扭曲检测, 局部修复, GenBlemish-27K  

## 一句话总结
Agentic Retoucher 将 T2I 生成后的缺陷修复重构为"感知→推理→行动"的人类式闭环决策过程，用三个协作 agent 分别做上下文感知的扭曲检测、人类对齐的诊断推理和自适应局部修复，在 GenBlemish-27K 上 plausibility 提升 2.89 分，83.2% 的结果被人类评为优于原图。

## 背景与动机
T2I 扩散模型（SDXL、FLUX 等）虽然能生成高质量图像，但仍然经常出现局部扭曲——手指畸形、面部不对称、文字不可读、肢体错位等。现有修复方案要么需要昂贵的全图重生成，要么依赖 VLM 做自动评估但 VLM 的空间定位能力很弱（六指图片被 VLM 判为正常）。缺乏一个能**自主发现 → 诊断 → 修复**局部缺陷的自动化系统。

## 核心问题
如何让 T2I 模型具备自主感知和修复生成缺陷的能力？如何解决 VLM 在细粒度缺陷检测上的不可靠性（幻觉导致的误判）？

## 方法详解

### 整体框架
Agentic Retoucher 由三个协作 agent 组成闭环：(1) Perception Agent 生成扭曲显著性图定位问题区域；(2) Reasoning Agent 对定位区域做诊断推理（分类 + 文字描述）；(3) Action Agent 根据推理结果选择工具做局部修复。修复后的图片再送回 Perception Agent 检查，迭代 2-3 轮直到无显著扭曲。

### 关键设计
1. **Context-Aware Perception Agent（上下文感知扭曲检测器）**: 采用双编码器架构（ViT 编码图像 + T5 编码 prompt），通过自注意力融合视觉和文本信息，生成扭曲显著性图 $S \in [0,1]^{H \times W}$。用混合损失训练：$\mathcal{L}_{sal} = \alpha \mathcal{L}_{MSE} + (1-\alpha) \mathcal{L}_{KLD}$，其中 KLD 项与人类注视分布对齐。比传统显著性模型和通用 VLM 在 AUC-Judd 上高出 10+ 个百分点。

2. **Human-Aligned Reasoning Agent（人类对齐推理 agent）**: 基于 Qwen2.5-VL-7B + LoRA 微调。两阶段训练：(a) SFT 阶段建立结构化输出格式和扭曲分类能力（LoRA rank=64）；(b) GRPO 阶段用偏好对齐减少幻觉。最终在扭曲类型分类准确率达到 80.10%（vs GPT-5 Zero-Shot 61.31%），语义描述 SimCSE 达 0.8517。

3. **Adaptive Action Agent（自适应修复 agent）**: 从模块化工具库中选择修复方式——VLM-基（Qwen-Edit、Gemini 2.5 Flash Image）或 Mask-基（Flux-Fill、SD-inpainting）。根据推理结果确定修复的空间范围、工具选择和指令，闭环后再验证。

### 损失函数 / 训练策略
- Perception Agent: MSE + KLD 混合损失
- Reasoning Agent: SFT（交叉熵）+ GRPO（偏好优化，奖励基于分类准确率和文本对齐度）

## 实验关键数据
| 数据集 | 条件 | Plausibility | Aesthetics | Alignment | Overall |
|--------|------|------|----------|------|------|
| GenBlemish-27K | Original | 44.21 | 53.69 | 57.89 | 47.15 |
| GenBlemish-27K | Ours w/ Qwen-Edit | **47.10** | **55.75** | **59.54** | **49.27** |
| SynArtifacts-1K | Ours w/ Gemini Flash | 65.96 | 65.27 | 62.94 | **58.43** |

人类评估：83.2% 的修复结果被判为优于原图（48.8% 明显更好 + 34.4% 略好）。

### 消融实验要点
- **Perception Agent**: 去掉注意力机制降低 SIM 和 CC；去掉 KLD 损失降低 NSS 和 AUC-Judd
- **Reasoning Agent**: 仅 GRPO（无 SFT）效果很差（准确率 58.97%）；SFT+GRPO 最优（80.10%）
- **工具选择**: 所有工具（Qwen-Edit、Gemini、Flux-Fill、SD-inpainting）配合 Agentic Retoucher 均有提升，说明框架与工具无关
- **GPT-5 和 Gemini 2.5 Pro Zero-Shot** 在扭曲推理上仅 61.31%/60.28%，说明通用 VLM 不擅长此任务

## 亮点
- 首次将 T2I 后处理修复建模为"感知-推理-行动"闭环 agent 系统，而非简单的一次性修复
- GenBlemish-27K 数据集提供了 27K 个像素级标注的扭曲区域，覆盖 12 类缺陷，是首个大规模 T2I 缺陷标注数据集
- 实验证明 VLM（包括 GPT-5）在零样本设置下无法可靠检测 AI 生成图像的扭曲——这是一个重要发现
- 框架与具体修复工具解耦，可以即插即用不同的编辑模型

## 局限性 / 可改进方向
- 迭代修复引入额外计算开销（2-3 轮推理）
- 当前修复工具是预定义的，无法学习新的修复策略
- 主要针对局部几何扭曲（手指、面部），对风格不一致或全局语义错误的覆盖较弱
- GenBlemish-27K 中手部扭曲占 46.8%，数据分布偏斜

## 与相关工作的对比
- **vs RichHF**: RichHF 做评估但不做修复，且过度关注面部/肢体区域。Agentic Retoucher 不仅评估还能闭环修复
- **vs AgenticIR/JarvisArt**: 这些是通用图像修复/修图 agent。Agentic Retoucher 专门针对 AI 生成图像的特有缺陷类型设计
- **vs Imagic/Step1x-Edit**: 这些需要手动提供 mask 或编辑指令。Agentic Retoucher 全自动定位和修复

## 启发与关联
- "感知-推理-行动"闭环范式对其他需要自动质量控制的生成任务有启发（视频生成、3D 生成）
- GenBlemish-27K 的扭曲分类体系（6 维度 12 类）可以作为评估 T2I 模型质量的标准化工具
- VLM 在细粒度空间定位上的失败案例值得关注——可能需要专门的空间理解训练

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 agent 系统应用于 T2I 后处理修复是新视角，但各组件（显著性检测、VLM 推理、inpainting）本身不新
- 实验充分度: ⭐⭐⭐⭐ 两个数据集 + 多种修复工具 + 消融 + 人类评估，但缺少与端到端修复方法的对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表精美
- 价值: ⭐⭐⭐⭐ 填补了 T2I 自动质量修复的空白，GenBlemish-27K 数据集有独立价值
