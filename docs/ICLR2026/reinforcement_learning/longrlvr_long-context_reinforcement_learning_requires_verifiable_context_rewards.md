# LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards

**会议**: ICLR 2026  
**arXiv**: [2603.02146](https://arxiv.org/abs/2603.02146)  
**代码**: [real-absolute-AI/LongRLVR](https://github.com/real-absolute-AI/LongRLVR)  
**领域**: llm_efficiency  
**关键词**: RLVR, 长上下文推理, 上下文定位, 可验证奖励, 梯度消失, GRPO  

## 一句话总结

提出 LongRLVR，通过在 RLVR 训练中引入可验证的上下文奖励（context reward），解决长上下文场景下仅靠最终答案奖励导致的上下文定位（grounding）梯度消失问题，显著提升 LLM 长上下文推理能力。

## 研究背景与动机

1. **RLVR 在长上下文中失效**：RLVR（如 DeepSeek-R1）在数学/编程等依赖参数化知识的推理任务上表现优异，但在长上下文场景（需要从外部文档中检索和推理）中效果不佳
2. **上下文定位是核心瓶颈**：长上下文推理需要先准确定位相关证据（contextual grounding），再基于证据生成答案；仅靠最终答案奖励的信号过于稀疏，无法有效引导定位过程
3. **梯度消失的理论证明**：作者从理论上证明，outcome-only reward 导致 grounding head 的梯度被"激活事件"概率 Pr(ε_j) 缩放——即只有当其他所有必要证据已被选中时，选中某个证据 chunk 才能获得正梯度信号，这在训练初期几乎不可能发生
4. **实验验证**：naive RLVR 训练时，上下文召回率（contextual recall）快速停滞，直接限制了答案准确率的提升上限（Figure 1）

## 方法详解

### 整体框架

将长上下文 RLVR 策略显式分解为两阶段：
- **Grounding Head** $\pi_\theta^{gnd}(Z|X,Q)$：从上下文 X 中选择相关证据子集 Z
- **Answer Head** $\pi_\theta^{ans}(y|X,Q,Z)$：基于选中证据生成最终答案 y

训练时，模型先生成 chunk 标识符列表（grounding），再生成最终答案。

### 可验证上下文奖励

总奖励 = 答案奖励 + 上下文奖励：

$$r_{total}(y,Z) = r_{ans}(y) + r_{ctx}(y,Z,G)$$

上下文奖励采用调制 F-score 设计：

$$r_{ctx}(y,Z,G) = \eta \cdot F_\beta(Z,G) + (1-\eta) \cdot r_{ans}(y) \cdot F_\beta(Z,G)$$

- **无条件定位奖励** $\eta \cdot F_\beta$：为 grounding 提供稳定的密集学习信号
- **协同成功奖励** $(1-\eta) \cdot r_{ans} \cdot F_\beta$：只有答案正确时才解锁完整的定位奖励，防止定位与最终目标脱钩
- 超参数：$\eta=0.1$，$\beta=2$（偏重召回）

### 理论保证（Proposition 2）

上下文奖励为每个 ground-truth chunk $c_j$ 提供的梯度包含 $\alpha_j \cdot Var(z_j)$ 项，该项不依赖稀有的"激活事件"概率，从而消除梯度消失。

### 合成数据流水线

- 从 book/arXiv/code 领域采集 8K-64K token 长文档
- 语义聚类→每个聚类用 Qwen3-235B 生成候选 QA 对并标注 grounding chunks
- 两阶段拒绝采样（簇内最优→文档最优），质量评分 > 9/10
- 最终生成 46K 高质量长上下文 QA 数据

## 实验

### 主实验（Table 1）

| 模型 | RULER-QA (AVG) | LongBench v2 | LongReason (AVG) |
|------|:-:|:-:|:-:|
| Qwen2.5-14B-1M (base) | 75.20 | 40.2 | 73.55 |
| +RLVR | 73.17 | 39.8 | 72.33 |
| **+LongRLVR** | **88.90** | **46.5** | **78.42** |
| Qwen2.5-7B-1M (base) | 65.00 | 33.0 | 66.45 |
| +RLVR | 66.90 | 32.4 | 69.27 |
| **+LongRLVR** | **78.67** | **38.6** | **79.22** |
| LLaMA-3.1-8B (base) | 62.77 | 30.4 | 49.31 |
| +RLVR | 67.80 | 32.4 | 49.62 |
| **+LongRLVR** | **80.33** | **36.2** | **53.23** |

- Qwen2.5-14B-LongRLVR 超越 Qwen3-14B（RULER-QA 88.90 vs 87.60）和 QwenLong-L1-32B
- Qwen2.5-7B-LongRLVR 在 LongReason 上大幅超越 LLaMA-3.1-70B（79.22 vs 57.59）

### 消融实验

| 消融维度 | 关键发现 |
|---------|---------|
| **奖励组件**（Figure 3）| answer-only 召回停滞→性能天花板；context-only 召回高但答案不准；两者协同最优 |
| **数据质量**（Figure 4）| 拒绝采样 best > median > worst（38.6 vs 36.6 vs 34.8）；过滤简单题有效，过滤难题有害 |
| **η 混合因子**（Figure 5a）| η=0.1 最优；η=0 初始信号太稀疏；η=1 定位与答案脱耦 |
| **F-score β**（Figure 5b）| β=2 最优；偏重召回对多证据推理至关重要 |
| **chunk 数量**（Figure 5c）| 16-128 chunks 性能稳健，模型学到语义级定位而非依赖分块策略 |

## 亮点

- 从理论（梯度消失证明）和实验双重角度揭示 outcome-only RLVR 在长上下文中的根本缺陷，分析严谨
- 上下文奖励的设计巧妙：调制 F-score 同时兼顾密集信号和目标对齐，避免 reward hacking
- 7B/14B 小模型训练后超越 70B+ 大模型甚至专用推理模型（Qwen3-14B），参数效率极高
- 对 chunk 数量的鲁棒性说明模型学到了真正的语义定位能力

## 局限性

- 需要 ground-truth grounding chunks 标注，依赖高质量合成数据流水线，泛化到无标注场景未验证
- 仅在 QA 任务上验证，对摘要、信息抽取等其他长上下文任务的效果未知
- 训练数据长度限于 8K-64K tokens，对更长上下文（如 256K+）的可扩展性未探讨
- F-score 奖励假设 chunk 粒度的标注是可用的，实际应用中获取此类标注可能代价高昂
- 理论分析基于独立 chunk 选择假设，实际 LLM 的自回归生成中 chunk 选择存在依赖

## 相关工作

- **RLVR 推理增强**: DeepSeek-R1, Kimi, DAPO 等——本文指出它们在长上下文中的局限
- **长上下文对齐**: RoPE 扩展 (YaRN, LongRoPE)、长上下文 SFT/DPO——本文是 RLVR 路线的改进
- **QwenLong-L1-32B**: 基于推理模型的长上下文 RLVR——LongRLVR 用更小模型达到可比性能
- **长上下文 Agent**: 分块-多轮协作方案——与本文正交，可结合使用

## 评分

⭐⭐⭐⭐ (4/5)

- **新颖性**: ⭐⭐⭐⭐ — 理论驱动的奖励设计思路清晰，梯度消失分析是核心贡献
- **实验充分度**: ⭐⭐⭐⭐ — 多模型、多基准、丰富消融，数据覆盖全面
- **写作质量**: ⭐⭐⭐⭐⭐ — 理论与实验衔接紧密，论述逻辑清晰
- **实用价值**: ⭐⭐⭐⭐ — 对长上下文 RLVR 训练有直接指导意义，但需合成标注数据
