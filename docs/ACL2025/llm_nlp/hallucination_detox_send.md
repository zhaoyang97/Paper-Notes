# Hallucination Detox: Sensitivity Dropout (SenD) for Large Language Model Training

**会议**: ACL 2025
**arXiv**: [2410.15460](https://arxiv.org/abs/2410.15460)
**领域**: LLM NLP
**关键词**: 幻觉缓解, 训练协议, Sensitivity Dropout, EigenScore, 嵌入振荡

## 一句话总结

本文揭示了 LLM 训练过程中幻觉行为的振荡现象，提出 Sensitivity Dropout（SenD）训练协议——通过识别并确定性丢弃高变异敏感嵌入索引来降低训练中的幻觉方差，同时提出计算高效的 Efficient EigenScore（EES）近似方法，在 Pythia 和 Llama 模型上实现高达 17% 的测试时可靠性提升。

## 研究背景与动机

- LLM 的幻觉（hallucination）问题已成为可靠性的核心挑战，现有方法大多聚焦于推理阶段的后处理（如 RAG、RLHF），训练过程中的幻觉产生机制研究不足
- 本文关注"混淆幻觉"（confabulations）——模型对相同/相似输入产生不同且可能事实不正确的回复
- 实验发现训练损失收敛并不保证幻觉减少：多种幻觉检测指标（SelfCheckGPT、HaluEval、XSum）在训练过程中呈现持续**振荡行为**
- 模型规模增大对 confabulation 问题几乎无帮助——SelfCheckGPT 的自一致性在不同规模模型间振荡幅度相似

## 方法详解

### 整体框架

SenD 是一个训练时幻觉缓解框架，包含三个关键组件：
1. **敏感嵌入索引（SEI）识别**：通过追踪倒数第二层句子嵌入在检查点间的变化，定位高变异维度
2. **确定性丢弃**：与随机 Dropout 不同，SenD 基于变异分析确定性地丢弃 top-K% 的 SEI
3. **EES 停止准则**：用高效 EigenScore 近似实时监控幻觉水平，提供训练停止信号

### 关键设计

**敏感嵌入索引（SEI）**：
- 利用 Su et al. 的句子嵌入提取方法，将倒数第二层的 n×m 激活矩阵压缩为 n 维向量
- 定义净变化公式 Δeᵢᵗ = |eᵢᵗ - eᵢᵗ⁻¹| 捕获连续检查点间的嵌入变化
- SEI 选取标准：综合考虑方差和累计净变化，取 top-20% 高变异索引
- 实验证明 SEI dropout 比随机 dropout 更显著降低 EigenScore，且对事实正确输出影响小

**Efficient EigenScore（EES）**：
- 传统 EigenScore 需要 O(n³) 的特征值分解，计算代价随模型规模增长不可接受
- EES 利用 Chebyshev 多项式和密度态（DOS）近似谱属性，将复杂度降至 O(n²)
- 在矩阵大小 10^8 时，EES 将计算时间从 ~7 秒减半至 ~4 秒
- EES 输出范围 [-1,1]（vs. EigenScore 的 [0,∞)），可作为独立幻觉检测指标

**SenD 训练流程**：
- 将数据集分为 α% 训练集和 (100-α)% 追踪集
- 每 T 个检查点窗口内：正常训练，记录倒数第二层表征，计算SEI，执行确定性丢弃
- 双重收敛准则：Loss < ε **且** EES < δ
- 额外计算开销仅约 11%（Llama 8B / HELM 数据集：61 分钟 vs. 55 分钟/epoch）

## 实验关键数据

### 主实验

**Llama 3.1 8B 幻觉指标**（HELM 数据集训练）：

| 指标 | SenD | Normal |
|------|------|--------|
| FactScore | 0.44 | 0.39 |
| FactScore + RAG | 0.50 | 0.40 |
| HaluEval Accuracy | 0.74 | 0.74 |

**下游任务保持**（Llama 3.1 8B）：

| 指标 | SenD (HELM) | Normal (HELM) | SenD (MedHalt) | Normal (MedHalt) |
|------|-------------|---------------|----------------|------------------|
| HellaSwag | 0.73 | 0.74 | 0.73 | 0.75 |
| MMLU | 0.67 | 0.65 | 0.42 | 0.64 |
| Token Entropy | 0.79 | 0.95 | 0.32 | 0.33 |

### 关键发现

1. **幻觉振荡普遍存在**：从 70M 到 12B 参数，所有 Pythia 模型在训练中持续展现幻觉指标振荡
2. **SEI dropout 优于随机 dropout**：在所有模型规模和输出类型上，SEI dropout 更显著降低 EigenScore
3. **SenD 提升测试时信心**：Token Entropy 降低高达 17%（如 HELM 数据集上从 0.95 降至 0.79）
4. **FactScore 提升**：SenD 比标准训练提升 11%（无 RAG），SenD+RAG 比 Normal+RAG 提升 10%
5. **四个领域有效**：Wikipedia（HELM）、医疗（MedHalt）、法律（LegalBench）、代码（CodeSearchNet）均有效
6. **与后处理互补**：SenD 不替代 RAG 等后处理方法，SenD+RAG 效果优于单独 RAG

## 亮点与洞察

- 首次从训练动态角度系统研究幻觉成因，发现训练时嵌入维度的振荡与幻觉的关联
- SenD 的确定性丢弃思路新颖——不同于随机 Dropout 的正则化，而是通过精确定位高变异维度来提升确定性
- EES 对 EigenScore 的高效近似有独立价值，可广泛用于训练监控场景
- 从"损失收敛≠幻觉减少"的实验观察出发，提供了双重收敛准则的理论依据

## 局限性

- 当前仅应用于持续训练（continual training），未在预训练阶段验证
- 实验规模有限——最大到 Llama 3.1 8B，未在 70B+ 大模型上验证
- 为减少遗忘效应，冻结了大部分层（Llama 8B 冻结 24 层），可能低估了 SenD 的完整潜力
- EES 作为近似方法，在尺度和精度上与原始 EigenScore 有差异
- MMLU 在 MedHalt 领域训练后下降明显（0.42 vs 0.64），领域适应性存在取舍

## 相关工作

- **幻觉检测**：SelfCheckGPT（自一致性）、EigenScore（语义变异性）、Semantic Entropy（语义熵）
- **正则化技术**：随机 Dropout、自适应 Dropout（Ba & Frey 2013）、确定性 Dropout（Santra et al. 2020）
- **训练动态分析**：Pythia 套件（Biderman et al. 2023）提供全检查点访问
- **后处理方法**：RAG、RLHF 等推理阶段幻觉缓解技术

## 评分

- 新颖性：⭐⭐⭐⭐⭐（首次在训练过程中系统性解决幻觉问题）
- 实用性：⭐⭐⭐⭐（额外计算开销小，可与现有方法互补）
- 实验充分度：⭐⭐⭐⭐（多模型、多领域、多指标验证）
- 写作质量：⭐⭐⭐⭐（研究问题清晰，但符号系统略繁复）
