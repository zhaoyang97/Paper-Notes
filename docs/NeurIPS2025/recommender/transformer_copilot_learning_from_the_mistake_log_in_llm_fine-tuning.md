# Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning

**会议**: NeurIPS 2025  
**arXiv**: [2505.16270](https://arxiv.org/abs/2505.16270)  
**代码**: [GitHub](https://github.com/jiaruzouu/TransformerCopilot)  
**领域**: LLM 微调 / 推理增强 / 错误纠正  
**关键词**: Mistake Log, Pilot-Copilot, logits rectification, fine-tuning, error-aware

## 一句话总结
提出 Transformer Copilot 框架，在 LLM 微调过程中系统记录"错误日志"(Mistake Log)，训练一个辅助 Copilot 模型学习 Pilot 的错误模式，推理时通过 logits 修正提升生成质量，在 12 个基准上最高提升 34.5%。

## 研究背景与动机

1. **领域现状**：监督微调（SFT）是适配 LLM 到特定领域的标准方法。但微调后模型在推理时仍存在训练-测试不对齐问题——模型无法完全捕获任务特定细微差异，或过拟合训练数据中的某些模式。

2. **现有痛点**：
   - 标准 SFT 仅用损失梯度更新参数，每个错误被消费后立即丢弃，模型不保留"在哪里、如何、为什么犯错"的显式记忆
   - 数据侧干预（self-refinement）和外部反馈（RLHF、Reflexion）需要额外数据或人类标注
   - 微调后的最终参数 $\theta_T^P$ 不包含训练轨迹中的错误信息——这些有价值的学习信号被浪费

3. **核心矛盾**：SFT 优化参数使 loss 最小化，但参数更新是"即用即丢"的——模型可能在同类问题上反复犯类似错误，因为它没有显式的错误反思机制。

4. **本文要解决什么**：不改变 Pilot 模型的训练过程，而是通过记录和利用训练中的中间信号（错误日志），在推理时辅助纠正。

5. **切入角度**：类比人类学习中"错题本"的反思机制——记录错误、分析原因、在考试时提醒自己注意类似错误。

6. **核心idea一句话**：在微调过程中系统记录模型的错误模式（输入、内部状态、token级误差），训练辅助 Copilot 学习这些模式并在推理时修正 Pilot 的 logits。

## 方法详解

### 整体框架
三部分：(1) Mistake Log 定义与收集；(2) Copilot 模型设计与联合训练；(3) 推理时 logits 融合。Pilot 正常微调，Copilot 并行学习 Pilot 的错误模式，推理时两者协作生成。

### 关键设计

1. **Mistake Log（错误日志）**：
   - 做什么：系统记录微调全过程中的三类信息
   - 三个组件：
     - **Question** $\tilde{X}_t$：输入表示（encoder 输出或 embedding 层输出）
     - **Rationale** $h_t$：每个 token 在所有 decoder 层的隐藏状态 $\{h_{t,i,l}\}_{l=1}^{L^P}$，反映模型的内部"推理过程"
     - **Mistake** $\ell_t$：token 级预测误差 $\ell_t(p_{t,i}, \hat{p}_{t,i}) = p_{t,i} - \hat{p}_{t,i}$，精确量化每个 token 的错误方向和程度
   - 完整 Mistake Log：$M_T = \{(\tilde{X}_t, h_t, \ell_t)\}_{t=1}^T$

2. **Copilot 模型设计**：
   - 做什么：从 Pilot 的 decoder 初始化，学习预测 Pilot 的 token 级误差
   - **Encoder-Decoder Copilot**：输入是 token 级误差序列 $\ell_{t,<i}$（投影到 hidden dim）。使用修改的 cross-attention：Query 来自 Copilot 自身隐藏状态，Key/Value 来自 Pilot 的输入表示和池化后的隐藏状态的拼接
   - **Decoder-only Copilot**：奇数层用标准 self-attention，偶数层用修改的 cross-attention 关注 Pilot 信息
   - 损失函数：$\mathcal{L}_t^C = \sqrt{\sum_i \|f_{t,i}^C - \ell_t(p_{t,i}, \hat{p}_{t,i})\|^2}$（RMSE 避免梯度过度平滑）

3. **联合训练范式**：
   - 每轮：(a) Pilot 前向传播并更新参数；(b) 收集 Mistake Log 条目；(c) 从 Mistake Log 采样训练 Copilot
   - Copilot 持续跟随 Pilot 的演化，学习最新的错误模式

4. **推理时 logits 修正**：
   - 核心公式：$\tilde{p}_{t,i} = \hat{p}_{t,i} + \lambda f_{t,i}^C$
   - Copilot 自回归生成误差预测，加回 Pilot 的 logits 上
   - $\lambda$ 是修正强度超参（默认 1），理论保证存在 $\lambda_0 > 0$ 使修正后更接近真实分布

### 理论保证（定理4.1）
在 Copilot 误差 $\epsilon_C < \sqrt{\epsilon_P^2 + \sigma_P^2}$ 的温和条件下，修正后的 $\tilde{p}_{t,i}$ 严格比 $\hat{p}_{t,i}$ 更接近真实分布 $p_{t,i}$。注意 Copilot 可以比 Pilot 有更大的偏差（$\epsilon_C > \epsilon_P$），仍然有效——这解释了为什么可以用较小的 Copilot。

## 实验关键数据

### 主实验
12 个基准涵盖常识推理、算术推理和推荐任务。

| Pilot 模型 | 任务类型 | 无 Copilot | + Copilot | 提升 |
|-----------|---------|-----------|----------|------|
| T5 系列 | 常识推理 | baseline | +2-15% | 显著 |
| LLaMA-3.2-3B | 常识推理 | baseline | +5-34.5% | 最高 34.5% |
| Qwen2.5-7B + 1B Copilot | 综合 | 低于 Qwen2.5-14B | **超越 Qwen2.5-14B** | 少用 4B 参数 |
| 各种 Pilot | 算术推理 | baseline | +2-20% | 稳定提升 |

### 消融与分析

| 分析维度 | 发现 |
|---------|------|
| Copilot 大小 | 1B Copilot 即可有效辅助 3B-7B Pilot |
| 计算开销 | 边际增量——Copilot 的推理成本远低于增大 Pilot |
| 可迁移性 | 训练好的 Copilot 可直接迁移到新 Pilot（无需重训） |
| 可扩展性 | 随 Pilot 规模增大持续有效 |
| Logits 修正可视化 | Copilot 的修正方向与正确答案一致 |

### 关键发现
- **Copilot 是 "错误矫正器" 而非 "独立推理器"**：它学习的是 Pilot 的错误模式而非独立的任务知识
- **小 Copilot 足矣**：1B Copilot 辅助 7B Pilot 可超越 14B 的单模型——比直接增大模型更参数高效
- **跨模型可迁移**：在一个 Pilot 上训练的 Copilot 可迁移到同族其他 Pilot，暗示错误模式跨模型共享
- **Token 级修正精准**：可视化显示 Copilot 在 Pilot 犯格式或事实错误的精确位置施加修正

## 亮点与洞察
- **"错题本"隐喻非常直观且有效**：将人类学习中的反思机制形式化为 Mistake Log，概念自然且实用
- **利用已被丢弃的训练信号**：标准 SFT 中中间隐藏状态和 token 级误差在参数更新后就被丢弃，Copilot 将这些"废料"变为有价值的监督信号
- **理论保证的宽松条件**：Copilot 不需要比 Pilot 更准确，只需要满足温和条件就能保证改进——这使得用小模型做 Copilot 变得可行
- **参数效率极高**：不修改 Pilot 的任何部分，仅通过外挂小模型实现显著提升

## 局限性 / 可改进方向
- **训练时内存开销**：需要存储 Mistake Log（包含所有训练步的隐藏状态和误差），对大规模训练可能需要策略（如只保留最近 N 步）
- **推理延迟**：虽然 Copilot 小，但额外的前向传播仍增加延迟
- **Copilot 的误差累积**：自回归推理中 Copilot 使用自己生成的误差预测而非真实误差，可能累积偏差
- **对 Pilot 训练过程的依赖**：Copilot 学的是特定训练轨迹的错误模式，换训练数据或超参可能需要重训
- **未与 RLHF/DPO 等对齐方法对比**：这些方法也解决训练-推理不对齐问题

## 相关工作与启发
- **vs Self-Refinement/Reflexion**：这些方法在推理时让模型自我反思，需要多次推理。Copilot 只需一次前向传播，更高效
- **vs 知识蒸馏**：蒸馏是大模型指导小模型，Copilot 是小模型辅助大模型——方向相反
- **vs Speculative Decoding**：都是用小模型辅助大模型推理，但 Speculative Decoding 加速解码，Copilot 提升质量
- **vs Logits Calibration**：温度缩放等后处理方法是全局调整，Copilot 做 token 级条件修正，更精细

## 评分
- 新颖性: ⭐⭐⭐⭐ Mistake Log 概念和 Pilot-Copilot 框架新颖，但 logits 修正本身不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 12 个基准 + 3 类任务 + 多种 Pilot + 理论分析 + 可视化 + 迁移性/可扩展性分析，非常全面
