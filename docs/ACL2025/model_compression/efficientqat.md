# EfficientQAT: Efficient Quantization-Aware Training for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2407.11062](https://arxiv.org/abs/2407.11062)  
**代码**: [https://github.com/OpenGVLab/EfficientQAT](https://github.com/OpenGVLab/EfficientQAT)  
**领域**: 模型压缩 / LLM效率  
**关键词**: quantization-aware training, LLM compression, block-wise training, low-bit quantization, step size  

## 一句话总结
EfficientQAT 提出两阶段 QAT 框架——先逐块训练所有参数（Block-AP）提供良好初始化，再端到端训练量化参数（E2E-QP）捕获跨块交互，在单张 A100 上 41 小时完成 Llama-2-70B 的 2-bit 量化，精度仅降 3 点。

## 研究背景与动机

1. **领域现状**：LLM 量化方法分三类：(1) PTQ（GPTQ/AWQ/OmniQuant）通过逐块重构快速量化，但低比特下精度损失大；(2) QAT（BitNet b1.58）端到端训练所有参数，精度最好但资源需求极高（需从头训练）；(3) Q-PEFT（QLoRA/PEQA）冻结量化权重只训练少量参数，低比特下恢复不了精度损失。
2. **现有痛点**：(1) PTQ 限制了优化空间——只训练 rounding/clipping 参数，且忽略跨块交互；(2) 原生 QAT 需要完整训练数据和多卡 GPU，对 70B 模型不可行；(3) Q-PEFT 可训练参数太少（step size 仅占 ~1.6%），低比特场景下恢复不了量化损失。
3. **核心矛盾**：全参数训练（精度好但开销大）vs 有限参数训练（效率高但精度差）的矛盾；逐块训练（内存友好但忽略跨块交互）vs 端到端训练（能捕获交互但内存爆炸）的矛盾。
4. **本文要解决什么？** 在单 GPU 上实现接近原生 QAT 的量化精度，特别是 2-bit/3-bit 极低比特场景。
5. **切入角度**：将 QAT 分解为两个互补阶段——块级全参数训练提供好的初始化 + 端到端只训练 step size 捕获跨块交互。
6. **核心idea一句话**：通过将 QAT 分为逐块全参数训练（Block-AP）和端到端量化参数微调（E2E-QP）两阶段，兼顾优化空间充足和内存高效。

## 方法详解

### 整体框架
阶段一（Block-AP）：按 transformer block 顺序，每个 block 内训练所有参数（权重 $W$、step size $s$、zero point $z$），使用重构损失 → 输出量化模型 $W_q, s, z$ → 阶段二（E2E-QP）：固定量化权重 $W_q$，端到端只训练 step size $s$，仅需 ~1.6% 参数的梯度。

### 关键设计

1. **Block-AP（逐块全参数训练）**:
   - 做什么：在每个 transformer block 内同时训练权重、step size 和 zero point
   - 核心思路：标准均匀量化 $W_{int} = \text{clamp}(\lfloor W/s \rceil + z, 0, 2^N-1)$，反量化 $\hat{W} = (W_{int} - z) \cdot s$。将量化/反量化嵌入计算图，用 STE 通过梯度下降优化所有参数
   - 设计动机：之前逐块方法（OmniQuant/BRECQ/AutoRound）只训练部分参数（clipping/rounding/LoRA），限制了优化空间。Block-AP 直接训练所有参数，无需复杂设计，在 2-bit 场景优势尤其明显
   - 与之前方法的区别：首个在逐块重构范式中训练所有参数的方法；之前方法限制更新范围到 $(-1, +1)$ 防止过拟合，Block-AP 无此限制

2. **E2E-QP（端到端量化参数训练）**:
   - 做什么：固定 Block-AP 输出的量化权重 $W_q$，仅端到端训练 step size $s$
   - 核心思路：只做反量化（Eq.2），不做量化（Eq.1），梯度 $\partial\hat{w}/\partial s = w_q - z$ 计算简单；可训练参数仅约 1.6%（group size=64 时），内存需求极低
   - 设计动机：Block-AP 忽略了跨块交互，E2E-QP 通过端到端目标函数让所有 block 的 step size 协同优化。同时内存极低——70B 模型 2-bit E2E-QP 仅需 34.2GB
   - 灵活性：可直接在目标数据集上训练（continual pre-training 或 instruction-tuning）

3. **两阶段的互补性**:
   - Block-AP 提供高质量初始化（大优化空间 → 低量化误差），但不考虑跨块
   - E2E-QP 在此基础上做轻量微调（小参数量 → 不过拟合），捕获跨块交互
   - 两阶段结合同时获得了 QAT 的精度和 PTQ 的效率

## 实验关键数据

### 主实验：Llama-2 零样本推理（5 任务平均准确率）

| 方法 | 比特 | Llama-2-7B | Llama-2-13B | Llama-2-70B |
|------|------|-----------|------------|------------|
| FP16 | 16 | 64.86 | 67.81 | 72.41 |
| GPTQ | 3 | 62.48 | 66.18 | 71.47 |
| AWQ | 3 | 62.82 | 66.14 | 71.41 |
| OmniQuant | 3 | 62.42 | 66.18 | 71.07 |
| **EfficientQAT** | **3** | **64.02** | **67.28** | **71.76** |
| OmniQuant | 2 | 46.98 | 53.56 | 54.87 |
| AutoRound | 2 | 54.50 | 60.72 | 67.70 |
| **EfficientQAT** | **2** | **59.50** | **63.88** | **68.93** |
| AQLM (VQ) | 2 | 57.61 | 62.22 | 69.85 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Block-AP + E2E-QP | 最佳 | 完整 EfficientQAT |
| Block-AP only | 较好 | 缺少跨块交互 |
| E2E-QP only (RTN 初始化) | 差 | RTN 初始化太差难恢复 |
| Block-AP 训练 rounding only | 较差 | 限制优化空间 |
| Block-AP 训练 clipping only | 较差 | 同上 |
| E2E 训练 s + z | ≈ E2E 训 s | z 转全精度有额外开销 |

### 关键发现
- **2-bit 场景优势突出**：EfficientQAT 在 2-bit 下比 OmniQuant 高约 12-14 点，比 AutoRound 高约 5 点
- **3-bit 几乎无损**：Llama-2-70B 3-bit（71.76）vs FP16（72.41），仅降 0.65 点
- **训练效率极高**：70B 模型 2-bit 量化仅需单 A100、41 小时、34.2GB 内存
- **Block-AP 中全参数训练 > 部分参数训练**：简单粗暴地训所有参数比精心设计 rounding/clipping 参数更有效，推翻了"需要限制优化空间防过拟合"的常规认知
- **跨模态通用**：在 instruction-tuned LLM 和多模态 LLM（LLaVA）上同样有效

## 亮点与洞察
- **"大道至简"的设计哲学**：Block-AP 最大的创新恰恰是"什么都不设计"——直接训所有参数而非像前人那样精心设计 rounding/clipping 参数。这说明在 LLM 量化中，优化空间比正则化更重要
- **两阶段分解的优雅性**：将 QAT 的"全参数+端到端"分解为"全参数+逐块"和"少参数+端到端"，每个阶段都处于高效区间，组合后逼近原生 QAT 效果
- **E2E-QP 仅训 step size 的简洁性**：step size 仅占 ~1.6% 参数但能有效捕获跨块交互，说明量化参数的端到端调整比权重微调更高杠杆
- **可迁移到 Q-PEFT 场景**：EfficientQAT 的 E2E-QP 阶段可直接在指令微调数据上训练，统一了压缩和微调

## 局限性 / 可改进方向
- 仅探索了均匀量化，未与向量量化（AQLM/QuIP#）结合——AQLM 在 2-bit 上仍有竞争力
- Block-AP 仍需要逐块的全精度前向传播，对超大模型（700B+）可能内存仍不够
- 未探索激活量化（WAQ），仅做权重量化
- 校准数据选择和训练超参数的敏感性分析不够充分

## 相关工作与启发
- **vs OmniQuant (Shao et al., 2023)**: OmniQuant 逐块训练 clipping 参数，优化空间受限。EfficientQAT 的 Block-AP 训练所有参数，2-bit 下高 12+ 点
- **vs BitNet b1.58 (Ma et al., 2024)**: BitNet 是原生 QAT 从头训练，EfficientQAT 是对已有模型的高效 QAT，适用范围更广
- **vs PEQA (Kim et al., 2023)**: PEQA 仅端到端训练 step size（无 Block-AP 初始化），从 RTN 起步恢复困难。EfficientQAT 的 Block-AP 提供了关键的高质量初始化

## 评分
- 新颖性: ⭐⭐⭐⭐ 两阶段分解思路清晰，Block-AP 的"全参数训练"虽简单但首次提出且有效
- 实验充分度: ⭐⭐⭐⭐⭐ 7B-70B 规模、2/3/4-bit、基座+指令+多模态场景、完整消融
- 写作质量: ⭐⭐⭐⭐ 明确清晰，方法描述详细
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高，单 GPU 完成 70B 模型 2-bit QAT 是切实可部署的方案
