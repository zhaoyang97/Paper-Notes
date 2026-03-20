# CodePercept: Code-Grounded Visual STEM Perception for MLLMs

**会议**: CVPR 2026  
**arXiv**: [2603.10757](https://arxiv.org/abs/2603.10757)  
**代码**: [GitHub](https://github.com/TongkunGuan/Qwen-CodePercept)  
**领域**: 多模态大模型 / STEM 视觉推理  
**关键词**: MLLM, STEM推理, 视觉感知, 代码生成, 强化学习  

## 一句话总结
通过感知-推理解耦缩放实验证明 MLLM 在 STEM 任务中的瓶颈是感知而非推理，提出以可执行代码为感知介质的 CodePercept 范式，构建 ICC-1M 数据集和 STEM2Code-Eval 基准，系统性提升 MLLM 的 STEM 视觉感知能力。

## 背景与动机
当 MLLM 在 STEM（数学、物理、化学、工程）视觉推理任务失败时，现有研究大多聚焦于提升推理能力（如 RL、思维链等），但一个根本性问题尚未回答：**失败的根源是感知不足还是推理不够？** 作者通过系统性的缩放分析，将任务解耦为感知（image→caption）和推理（caption→answer）两阶段，独立缩放各组件发现：扩展感知能力的收益始终大于扩展推理能力。此外，现有评估方法仅通过问题解决准确率间接衡量感知，无法直接评估全面的视觉理解能力。知识蒸馏生成描述也面临幻觉和"描述失语"（复杂空间关系难以用自然语言完整表达）的问题。

## 核心问题
1. 感知与推理的瓶颈归因：MLLM 在 STEM 中的失败更多来自感知不足
2. 自然语言在描述复杂 STEM 图形时存在固有的模糊性和不完整性
3. 缺少直接评估 STEM 视觉感知能力的范式

## 方法详解
### 整体框架
CodePercept 以"代码即感知介质"为核心思想：要求模型生成可执行 Python 代码来重建图像，只有完整准确的视觉理解才能实现高保真重建。框架包含数据构建（ICC-1M）、两个代码驱动训练任务、以及 SFT+RL 两阶段训练。

### 关键设计
1. **ICC-1M 数据集构建**: 通过三条并行管线生成 100 万+ 图像-描述-代码三元组：(1) Image Reproduction 从现有 STEM 图直接生成代码；(2) Image Diversity 提取种子图的科学原理并重新实例化生成多样变体；(3) Solid Geometry Synthesis 用参数化模板生成立体几何。经过图像质量、代码质量、图-代码一致性三重质控。
2. **Code-Grounded Caption Generation**: 利用代码执行日志（execution tracer）提取验证过的视觉事实，与直接描述融合，消除数值、空间关系上的幻觉。关键在于代码本身作为 ground truth，结合执行追踪器解决复杂代码分析困难。
3. **STEM Image-to-Code Translation**: 训练模型直接生成带解释性注释的重建代码，先由 MLLM 生成解释性草稿，再用 ground-truth 代码纠正事实错误，保留教学结构。
4. **STEM2Code-Eval 基准**: 1000 张人工标注图像，要求模型生成可执行代码重建原图，三阶段管线（代码智能体生成、候选筛选、人工标注）确保高质量。

### 损失函数 / 训练策略
- **Stage 1 (SFT)**: 基于 Qwen3-VL 同时优化 image-caption 和 image-to-code 两个任务，用 SWIFT 在 32 A100 上训练 1 epoch，学习率 3e-6 cosine 衰减
- **Stage 2 (RL)**: 使用 GRPO 仅对代码生成进行强化学习，10k 样本，奖励由格式奖励（代码格式验证）+ 内容奖励（执行成功性 + GPT-4o 代码语义评分 + GPT-4o 图像相似度评分）组成

## 实验关键数据
| 评估类型 | 模型 | 指标 | 本文 | 基线 | 提升 |
|---|---|---|---|---|---|
| 感知评估(captioner-solver) | CodePercept-8B-S1 vs Qwen3-VL-8B | 6 STEM benchmark 平均 | 63.32% | 60.36% | +3.0% |
| 感知评估(captioner-solver) | CodePercept-32B-S1 vs Qwen3-VL-32B | 6 STEM benchmark 平均 | 67.30% | 64.63% | +2.7% |
| STEM2Code-Eval | CodePercept-8B-R1 | Avg Score | 48.65 | 28.41(基线) | +20.2 |
| STEM2Code-Eval | CodePercept-32B-R1 | Avg Score | 65.75 | 38.42(基线) | +27.3 |
| STEM2Code-Eval | CodePercept-8B-R1 | Exec Rate | 93.4% | 85.3%(基线) | +8.1% |

### 消融实验要点
- 三条数据管线逐步叠加效果：IR→ID→SG 平均分从 60.91→62.15→62.75
- Code-Grounded Caption 比 Native Caption（直接用 Gemini 描述）高 2.0%，验证了代码消除幻觉的有效性
- 同时训练 caption+code 比单独训练 caption 再高 0.6%，说明两种模态互补

## 亮点
- "感知优先于推理"的发现改变了 MLLM 改进的思路方向，提供了清晰的实证
- 用代码作为感知评估介质既巧妙又严谨：代码是确定性的、可执行验证的
- Code-Grounded Caption 利用代码执行追踪器消除幻觉是一个可迁移的通用思路
- ICC-1M 中 Image Diversity 管线的"原理抽取→多样实例化"策略具有数据扩展的方法论价值

## 局限性 / 可改进方向
- 仅基于 matplotlib 生成代码，无法覆盖需要交互式或 3D 渲染的 STEM 图形
- 感知-推理解耦分析依赖于当前的流水线架构，端到端模型中边界可能不同
- RL 阶段依赖 GPT-4o 作为奖励模型，成本高且引入了外部模型偏差
- STEM2Code-Eval 仅覆盖 matplotlib 可重建的图形类型

## 与相关工作的对比
- 与 Perception-R1、Vision-R1 等聚焦推理的工作形成互补：CodePercept 证明感知才是瓶颈
- 与 OmniCaptioner 的 captioner-solver 评估范式不同：OmniCaptioner 仅衡量问题相关信息，CodePercept 要求全面视觉理解
- 与 UI/Chart/SVG 等领域特定代码生成不同，CodePercept 面向通用 STEM 领域，且代码同时服务于评估和训练

## 启发与关联
- 代码作为感知介质的思路可推广到其他需要精确空间理解的领域（如 CAD、建筑平面图）
- 数据合成中"原理抽取→多样化实例化"的策略可用于其他数据增强场景
- 缩放分析方法论（独立缩放各组件）可用于诊断其他多模块系统的瓶颈

## 评分
- 新颖性: ⭐⭐⭐⭐ 感知瓶颈的实证发现和代码作为感知介质的范式都有创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个 STEM benchmark + 自建评测 + 详尽消融，4B/8B/32B 三个尺度
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，motivation 到方法到实验链条完整
- 价值: ⭐⭐⭐⭐ 对 MLLM 感知能力的改进方向有重要指导意义
