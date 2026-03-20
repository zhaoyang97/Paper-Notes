# Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts

**会议**: ACL 2025  
**arXiv**: [2501.02009](https://arxiv.org/abs/2501.02009)  
**代码**: 无  
**领域**: llm_nlp  
**关键词**: steering vectors, cross-model transfer, representation alignment, linear transformation, LLM interpretability  

## 一句话总结
提出 L-Cross Modulation 方法，通过简单线性变换将一个 LLM 的概念方向向量（steering vectors）迁移到另一个 LLM 中实现行为控制，发现三个关键结论：(1) 跨模型 SV 迁移有效；(2) 不同概念共享同一变换矩阵；(3) 小模型的 SV 可以控制大模型（弱到强迁移）。

## 研究背景与动机
1. **领域现状**：Steering vectors（SVs）是 LLM 内部概念的方向表示，可用于控制生成行为（如引导生成有害/无害内容），但研究局限于单个 LLM 内部。
2. **现有痛点**：不同 LLM 有各自的表示空间，SV 不能直接跨模型使用——如从 Llama 提取的 SV 无法直接用于 Qwen。
3. **核心矛盾**：如果不同 LLM 学到了同一概念的不同表示，跨模型控制就不可行；但 Platonic Representation Hypothesis 认为不同网络会趋向共享的现实统计模型。
4. **本文要解决什么？** 验证概念表示在不同 LLM 间是否共享一个底层结构，以及能否通过简单变换实现跨模型迁移。
5. **切入角度**：类比柏拉图洞穴寓言——不同 LLM 是不同"囚徒"，看到的是同一现实的不同"影子"，线性变换是"影子"之间的桥梁。
6. **核心 idea 一句话**：不同 LLM 的概念表示存在线性可变换的共享结构，支持弱到强的跨模型控制。

## 方法详解

### 整体框架
(1) 在共享语料库上用源/目标 LLM 分别编码句子 → (2) 最小二乘法求解线性变换矩阵 $\mathbf{T}$ 使源表示映射到目标表示 → (3) 用 $\mathbf{T}$ 变换源模型的 SV，注入目标模型的隐状态进行行为控制。

### 关键设计
1. **线性变换矩阵 $\mathbf{T}$ 的学习**:
   - 做什么：用 OLS 最小化 $\|\lambda_\mathcal{D}^{m_t} - \lambda_\mathcal{D}^{m_s} \mathbf{T}'\|$
   - 闭式解：$\mathbf{T} = (\lambda^{m_s\top}\lambda^{m_s})^\dagger \lambda^{m_s\top}\lambda^{m_t}$
   - 设计动机：线性变换保持概念间的基本关系（仅旋转和缩放），有利于验证表示的普遍性

2. **语料库选择**:
   - 可用概念相关对比文本 $Y_W$（更精确）
   - 也可用概念无关通用文本（更通用）——实验发现通用语料库的 $\mathbf{T}$ 也能有效跨概念迁移

3. **弱到强迁移**:
   - 从 Qwen-0.5B 提取的 SV 经变换后可以有效控制 Qwen-7B 的行为
   - 意味着小模型已捕捉到概念的核心方向

## 实验关键数据

### RQ1: 跨模型迁移有效性（11 个概念）
| 概念 | No Mod | Self Mod | L-Cross (Qwen→Llama2) |
|------|--------|----------|----------------------|
| Harmfulness (↑) | 0% | 90%+ | ~90%（接近 Self Mod） |
| Happiness (↑) | 低 | 高 | 接近 Self |
| Refusal (↓) | 基线 | 大幅变化 | 有效变化 |

### RQ2: 变换矩阵跨概念泛化
- 用概念 W1 的数据学到的 $\mathbf{T}_{Y_{W_1}}$ 可以有效迁移概念 W2 的 SV
- 甚至用通用语料库（概念无关文本）学到的 $\mathbf{T}$ 也有效
- 暗示不同概念的 SV 共享底层的跨模型变换结构

### RQ3: 弱到强迁移
| 源模型 | 目标模型 | 效果 |
|--------|---------|------|
| Qwen-0.5B | Qwen-7B | 有效（接近同模型 Self Mod） |

### 关键发现
- 线性变换足以对齐不同 LLM 的概念空间（不需要复杂非线性映射）
- 同一变换矩阵对多个概念都有效 → 概念间的关系结构跨模型一致
- 小模型到大模型的迁移有效 → 概念的核心方向在不同规模模型中共享

## 亮点与洞察
- 用柏拉图洞穴寓言类比非常生动——不同 LLM 看到同一"现实"的不同"影子"，但底层结构一致
- 线性可迁移性支持了 Platonic Representation Hypothesis 在 LLM 概念层面的成立
- 弱到强迁移对 AI 安全有直接意义：可以用小模型作为"概念探测器"来控制大模型

## 局限性 / 可改进方向
- 仅在 3 个 LLM 系列（Llama2/3.1、Qwen2）上验证
- 线性变换可能不适用于差异极大的架构
- 缩放因子 $\beta$ 仍需手动调整
- 未验证在多轮对话等复杂场景中的效果

## 相关工作与启发
- **vs Activation Steering (CAA/RepE)**: 从单模型扩展到跨模型，是首次系统研究
- **vs Platonic Representation Hypothesis**: 提供了概念层面的实证支持
- **vs 弱到强泛化 (Burns et al.)**: 在概念控制维度证明弱到强可行

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 跨模型概念迁移+弱到强控制是全新视角
- 实验充分度: ⭐⭐⭐⭐ 11 个概念、3 个 RQ 递进、多模型对比
- 写作质量: ⭐⭐⭐⭐⭐ 柏拉图寓言的类比优美，叙述逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 可解释性和安全研究有深远影响
