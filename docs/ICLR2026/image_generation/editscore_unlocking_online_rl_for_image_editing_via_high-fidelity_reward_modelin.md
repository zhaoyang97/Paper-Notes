# EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling

**会议**: ICLR 2026  
**arXiv**: [2509.23909](https://arxiv.org/abs/2509.23909)  
**代码**: [GitHub](https://github.com/VectorSpaceLab/EditScore)  
**领域**: 扩散模型 / 图像编辑  
**关键词**: Reward Model, Reinforcement Learning, Image Editing, Online RL, Flow-GRPO  

## 一句话总结
提出首个系统性的"基准评测→奖励模型→强化学习训练"图像编辑 RL 管线：构建 EditReward-Bench 基准，训练 EditScore 系列奖励模型（7B-72B，超过 GPT-5），并成功将其用于 Online RL 训练显著提升编辑模型性能。

## 研究背景与动机
1. **领域现状**：强化学习在 LLM 和 T2I 领域已展现巨大价值（如 FlowGRPO），但在图像编辑领域的应用几乎空白。RL 理论上可通过试错-反馈过程发现超越静态数据集的编辑策略。
2. **现有痛点**：Online RL 的核心瓶颈在于缺乏高保真、高效、可扩展的奖励信号。GPT-5 等大 VLM 成本过高无法大规模查询；开源 VLM（即使 Qwen2.5-VL-72B）作为奖励信号也不够准确，导致训练不稳定或策略崩溃。
3. **核心矛盾**：参数规模无法替代领域对齐的准确性——通用 VLM 评估精细编辑质量时表现不佳（一致性判断甚至不如随机），特别是在一致性（Consistency）维度。
4. **本文要解决什么？** 构建一个高保真、领域专用的奖励模型来解锁图像编辑的在线 RL。
5. **切入角度**：全栈系统——benchmark 驱动奖励模型开发，奖励模型驱动 RL 训练。
6. **核心idea一句话**：高保真的领域专用奖励模型是解锁图像编辑在线 RL 的关键。

## 方法详解
### 整体框架
三大组件：(1) EditReward-Bench——系统评估奖励模型的基准；(2) EditScore——基于 Qwen2.5-VL 微调的专用奖励模型系列；(3) 基于 EditScore 的 Online RL 训练管线（Flow-GRPO）。

### 关键设计
1. **EditReward-Bench 基准**：
   - 覆盖 4 大类 13 项编辑任务（Subject/Appearance/Scene/Advanced）
   - 11 个异构编辑模型生成候选（含 GPT-4o-Image、Gemini-2.5等 SOTA 专有模型）
   - 三维评估：Prompt Following (PF)、Consistency (C)、Overall Quality (O)
   - **双专家讨论标注协议**：两位 AI 专家实时讨论达成共识（非独立标注），一致性率 >97%
   - 3072 个偏好对（PF:944, C:890, O:1238），使用 pairwise accuracy 评估

2. **EditScore 奖励模型（7B-72B）**：
   - 将奖励建模为条件文本生成任务，在 Qwen2.5-VL 上用 LoRA 微调
   - 输入：(Instruction, Input Image, Output Image)；输出：(Reasoning, Scalar Score)
   - 遵循 VIEScore 框架评估两个正交方面——语义一致性 $S_{SC}$ 和感知质量 $S_{PQ}$
   - 最终分数：$S_{final} = \sqrt{S_{SC} \cdot S_{PQ}}$
   - **推理时自集成策略**：$K$ 次独立随机前向传播后平均分数：
   $$S_{final}(\mathbf{z}) = \frac{1}{K} \sum_{i=1}^{K} s_i$$
   直觉：$K$ 个推理路径可视为不同判断视角，聚合后更准确。实验显示推理时计算扩展的效率远高于参数扩展。

3. **数据构建管线**：
   - 图像选取 + 指令创建（Qwen2.5-VL-72B 生成 + K-center greedy 采样）
   - 5 个编辑模型生成候选输出
   - GPT-4.1 标注 SC/PQ 分数 + 推理
   - 双维度过滤：最大分数过滤（去除不可实现的编辑）+ 标准差过滤（去除低区分度样本）
   - 奖励模型 70K 样本，RL 训练 60K 样本

### 损失函数 / 训练策略
- 奖励模型：标准自回归目标 + LoRA 微调，评分范围扩展至 [0,25]（实验显示优于 [0,10] 或 [0,30]）
- RL：Flow-GRPO，采样步 $T=20$，组大小 $G=12$，噪声水平 $\sigma=0.9$，KL 权重 $\beta=0.04$
- 推理时 reasoning-before-scoring 格式比直接打分提升 +0.038 准确率

## 实验关键数据
### 奖励模型评估（EditReward-Bench Overall Accuracy）

| 模型 | PF | C | O |
|------|-----|-----|-----|
| GPT-4.1 | 0.673 | 0.602 | 0.705 |
| GPT-5 | 0.777 | 0.669 | 0.755 |
| Qwen2.5-VL-72B | 0.540 | 0.435 | 0.621 |
| EditScore-7B (Avg@4) | 0.722 | 0.720 | 0.727 |
| EditScore-72B (Avg@4) | **0.755** | **0.735** | **0.763** |

*EditScore-7B 即超越 10 倍大的 Qwen2.5-VL-72B；EditScore-72B (Avg@4) 超过 GPT-5。*

### RL 训练效果（OmniGen2 Base）

| 奖励信号 | GEdit SC | GEdit PQ | GEdit O | ImgEdit O |
|----------|---------|---------|---------|-----------|
| No RL | 6.72 | 7.20 | 6.28 | 3.40 |
| Qwen2.5-VL-72B | 6.89 | 7.21 | 6.42 | 3.60 |
| GPT-4.1 | 7.24 | 7.41 | 6.73 | 3.66 |
| **EditScore-7B (Avg@4)** | **7.20** | **7.46** | **6.68** | **3.63** |

*EditScore-7B 即可匹敌 GPT-4.1 作为奖励信号的效果，而 Qwen2.5-VL-72B 几乎无法提供有效引导。*

### 关键发现
- 通用开源 VLM 即使 72B 参数也无法作为有效奖励信号（训练不稳定），参数规模 ≠ 领域准确性
- 推理时自集成比参数扩展更高效：EditScore-7B (K=4) > EditScore-32B (K=1)，且延迟亚线性增长
- Score range [0,25] 最优；reasoning + score 比直接 score 提升 0.038
- GPT-4.1 标注的数据训练出的奖励模型在 RL 中表现优于 GPT-5 标注的，因为 GPT-4.1 数据方差更大（3.309 vs 2.942），更强的区分度有助于策略学习
- TempFlow-GRPO（时间感知损失权重）+ EditScore 可进一步提升至 Overall 7.21

## 亮点与洞察
- **全栈贡献**：从 benchmark 到奖励模型到 RL 训练的完整管线，填补领域空白
- **反直觉发现**：标注方差越大反而 RL 训练效果越好，指出了奖励模型设计的新维度
- **推理时扩展效率极高**：利用 shared KV-cache prefilling，自集成延迟亚线性增长
- **跨模型/跨算法泛化**：EditScore 在 OmniGen2 和 FLUX-Kontext-dev 上均有效，兼容 Flow-GRPO 和 TempFlow-GRPO
- **双专家讨论标注**：显著提升标注一致性（Consistency 维度一致性率提升 12.12%）

## 局限性 / 可改进方向
- 数据构建依赖 GPT-4.1 标注，成本不低且可能引入偏差
- EditScore 基于 Qwen2.5-VL 微调，随 VLM 迭代需持续更新（已验证 Qwen3-VL-8B 更优）
- Text Change 等需要 OCR 能力的任务评估可能不够充分
- RL 训练的计算开销（多采样 + 奖励评估）限制了实际部署规模
- 仅验证了 Flow-GRPO 及其变体，未探索 PPO/DPO 等其他算法

## 相关工作与启发
- **FlowGRPO / DanceGRPO**：T2I 领域的 RL 成功案例，EditScore 将此范式扩展到编辑
- **VIEScore**：编辑评估框架，EditScore 在其基础上增加了 reasoning 和评分范围优化
- **Adjoint Matching**：奖励对齐的模型训练方法；EditScore 关注的是推理时奖励模型
- 启发：在 RL 应用中，**领域专用奖励模型**比通用大模型更有价值——这一结论可能推广到其他视觉生成任务

## 评分
- 新颖性: ⭐⭐⭐⭐ — 全栈管线首创，但各组件设计相对标准
- 实验充分度: ⭐⭐⭐⭐⭐ — benchmark 构建严谨 + 消融全面 + 跨模型/算法验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，但论文篇幅较长需要仔细跟踪大量表格
- 价值: ⭐⭐⭐⭐⭐ — 为图像编辑领域的 RL 训练铺平道路，代码/模型/数据全开源
