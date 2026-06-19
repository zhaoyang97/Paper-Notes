---
title: >-
  [论文解读] Whispering Agents: An Event-Driven Covert Communication Protocol for the Internet of Agents
description: >-
  [AAAI 2026][LLM 其他][隐蔽通信] 首次形式化定义了智能体互联网（IoA）中的"隐蔽事件通道"（Covert Event Channel），并设计了 ΠCCAP 协议，通过在智能体对话的存储、时序和行为维度中嵌入秘密数据，实现了高容量、高鲁棒性且对 LLM 审查员不可感知的隐蔽通信。
tags:
  - "AAAI 2026"
  - "LLM 其他"
  - "隐蔽通信"
  - "隐写术"
  - "智能体互联网"
  - "多智能体系统"
  - "事件驱动协议"
---

# Whispering Agents: An Event-Driven Covert Communication Protocol for the Internet of Agents

**会议**: AAAI 2026  
**arXiv**: [2508.02188](https://arxiv.org/abs/2508.02188)  
**代码**: [https://github.com/haha1128/a2a-stego-project](https://github.com/haha1128/a2a-stego-project)  
**领域**: 其他 (AI 安全 / 隐蔽通信)  
**关键词**: 隐蔽通信, 隐写术, 智能体互联网, 多智能体系统, 事件驱动协议

## 一句话总结

首次形式化定义了智能体互联网（IoA）中的"隐蔽事件通道"（Covert Event Channel），并设计了 ΠCCAP 协议，通过在智能体对话的存储、时序和行为维度中嵌入秘密数据，实现了高容量、高鲁棒性且对 LLM 审查员不可感知的隐蔽通信。

## 研究背景与动机

1. **领域现状**：大语言模型催生了智能体互联网（IoA）的兴起——用户将高层目标委托给主代理，主代理发现、协商和编排专业代理网络完成复杂任务。Agent-to-Agent（A2A）通信成为未来数字社会的基础设施。

2. **现有痛点**：标准 A2A 协议（如 Google A2A）通过加密保护消息内容，但无法保护"通信行为本身"。即使消息不可读，某些代理之间的交互模式也可能泄露关键战略信息（如物流代理与海军舰艇的加密通信暴露资产重新部署）。现有隐写术针对静态载体（文本、图像）或刚性协议字段设计，不适用于 IoA 的动态、事件驱动、多轮交互场景。

3. **核心矛盾**：IoA 的丰富事件驱动环境提供了强大的隐蔽通信媒介，但缺少正式的威胁模型、实用的多轮隐蔽通信协议和综合评估框架。

4. **本文目标**：为 IoA 建立隐蔽通信的形式化模型、设计工程级可靠协议，并全面评估其容量、鲁棒性和隐蔽性。

5. **切入角度**：将智能体对话的原子事件 $e = (ag, act, t)$ 分解为三个可调制的组件——数据载荷（存储通道）、时间戳（时序通道）和行为类型（行为通道），构建混合隐蔽通道。

6. **核心 idea**：利用代理对话的自然行为模式（内容、时间和行为选择）作为三维隐蔽载体，通过可证明安全的生成式隐写术嵌入数据，并引入意图不可区分性（IND-INT）安全标准。

## 方法详解

### 整体框架

ΠCCAP 协议分三个阶段运行：(1) 初始化与握手——通过 ECDH 密钥交换建立会话密钥，基于密钥相关时间戳触发器激活隐蔽通道；(2) 迭代嵌入与传输——发送方构造每个原子事件，分别在数据载荷、行为类型和时间戳中嵌入/编码信息；(3) 重组与隐式确认——接收方按时间戳过滤、校验完整性、重组消息并发送隐式 ACK。

### 关键设计

1. **隐蔽事件通道的形式化定义**：
    - 功能：首次提出统一的 IoA 隐蔽通信模型
    - 核心思路：将原子事件 $e = (ag, (a_{\text{type}}, a_{\text{data}}), t)$ 的三个组件分别对应三种通道：(1) 存储通道——调制 $a_{\text{data}}$ 嵌入高容量数据；(2) 时序通道——调制时间戳 $t$ 或间隔 $\Delta t$ 进行信令；(3) 行为通道——从上下文合理的行为类型集合中策略性选择 $a_{\text{type}}$ 信号化数据
    - 设计动机：单一通道各有局限，混合架构可满足分层安全目标——存储通道提供 IND-STAT 安全基础，时序和行为通道补充实现 IND-INT 安全

2. **双层安全标准**：
    - 功能：定义了比现有隐写安全性更强的安全框架
    - 核心思路：(1) **IND-STAT**（统计不可区分性）：整个通信轨迹（trace）与自然事件分布计算上不可区分；(2) **IND-INT**（意图不可区分性）：任何单个事件即使在完整上下文下也无法被判定为隐蔽意图生成。文章正式证明了 IND-INT 是严格强于 IND-STAT 的安全概念（通过归约反证法）
    - 设计动机：IND-STAT 可能仍存在语义异常的单个事件（整体分布正常但局部奇怪），IND-INT 消除了这种漏洞

3. **统一隐蔽头（Unified Covert Header）**：
    - 功能：管理数据分片和保证完整性
    - 核心思路：每个数据分片前附加动态头部，包含 12-bit 总数据大小（TDS）、6-bit 序列号（SN）、1-bit 终止标志（FIN）、4-bit 校验和（CS）。首个分片用 23-bit 全头，后续用 11-bit 紧凑头（省略 TDS）。最终消息附加哈希值 $\text{Hash}(M)$ 进行端到端完整性验证
    - 设计动机：工程级可靠传输——错误检测（CRC-4）、消息完整性（Hash）、传输确认（隐式 ACK），是隐蔽通信中首次集成这些机制

### 损失函数 / 训练策略

协议不涉及模型训练，但关键算法配置包括：
- 隐写算法：Discop 或 Differential-based，精度 52 bits
- 文本生成参数：top-k=50, top-p=1.0, temperature=1.0, 最大 256 token
- 发送方模型：Llama-3.2-3B-Instruct, Qwen2-7B-Instruct
- ECDH 密钥交换用于会话密钥生成
- 对抗评估使用 fine-tuned BERT 分类器和 GPT-4o 作为审查员

## 实验关键数据

### 主实验

| 指标 | 本文 ΠCCAP | 传统隐写术 | Secret Collusion |
|------|------|----------|------|
| 隐蔽容量 | ~4 kbits/对话 | ~56.68 bits/句 | 不稳定 |
| 错误检测 | ✓ | ✗ | ✗ |
| 完整性校验 | ✓ | ✗ | ✗ |
| 传输确认 | ✓ | ✗ | ✗ |
| 正式安全证明 | ✓ | ✓ | ✗ |

### 容量与鲁棒性实验

| 主题 | 算法/模型/载荷 | EBPR (bit) | BPR (bit) | 熵 (bit/token) |
|------|---------|------|------|------|
| Philosophy | Discop/Llama3/512 | 84.59±11.00 | 104.15±12.63 | 1.72±0.07 |
| Art | Discop/Llama3/1024 | 80.03±3.32 | 97.97±3.77 | 1.71±0.14 |
| QA | Discop/Qwen2/1024 | 72.69±5.05 | 89.63±5.74 | 1.81±0.07 |
| Philosophy | Discop/Qwen2/1024 | 89.72±6.61 | 108.99±7.51 | 1.72±0.06 |

### 隐蔽性评估

| 评估方式 | 指标 | 结果 | 说明 |
|------|---------|------|------|
| BERT 分类器 | AUC | 0.503 | ≈随机猜测(0.5) |
| BERT 分类器 | Precision | 0.250 | 完全无法识别隐写文本 |
| GPT-4o (Discop) | 自然度 | 4.95-5.00/5.0 | 与正常对话无差异 |
| GPT-4o (Discop) | 连贯性 | 4.95-5.00/5.0 | 逻辑完全合理 |
| GPT-4o (Discop) | 可疑度 | 1.00-1.06/5.0 | 几乎完全不可疑 |

### 关键发现

- **100% 消息完整性**：所有实验配置下消息均成功解码并通过校验和和哈希验证
- 对话主题与容量高度相关——哲学主题词汇丰富、熵更高，因此 EBPR 最高
- 协议开销（BPR 与 EBPR 差距）非常小，头部和哈希的工程成本可忽略
- BERT 隐写分析器 AUC=0.503，统计上等同于随机猜测——彻底通过 IND-STAT 验证
- GPT-4o 审查员评分与正常对话无统计差异——通过 IND-INT 验证

## 亮点与洞察

- **首次形式化 IoA 隐蔽通信**：建立了完整的威胁模型、安全定义和协议规范，填补了重要空白
- **三维混合通道**设计巧妙——存储、时序、行为三个维度互为补充，联合实现分层安全
- **IND-INT 安全标准**的提出及其严格强于 IND-STAT 的正式证明具有理论贡献
- 协议具有工程级可靠性——首次将错误检测、完整性校验和传输确认集成到隐蔽通道中
- 容量比传统隐写术高两个数量级（4kbits vs 56.68bits）

## 局限与展望

- 当前仅在对话文本场景下评估，未扩展到多模态（图像、代码等）
- 假设自然行为分布 $\mathcal{D}$ 稳定且对手被动观察，未考虑自适应/主动攻击者
- IND-INT 的实验验证虽强但缺乏严格的组合式证明（跨存储、时序、行为通道）
- 在实际部署中密钥分发的安全性取决于 PKI 基础设施的可靠性
- 伦理层面——虽声明为防御性研究，但技术本身可能被滥用于 AI 代理间的恶意串谋

## 相关工作与启发

- **vs 传统隐写术（Discop 等）**：针对静态单条消息载体设计，无法利用多轮对话的动态性和事件结构；ΠCCAP 将整个对话轨迹作为隐蔽载体
- **vs Secret Collusion（Motwani et al.）**：依赖脆弱的 prompt injection，缺乏隐蔽性和稳定性保证；ΠCCAP 有形式化安全证明
- **vs A2A 协议（Google）**：仅保护明文通信的可靠性，不提供隐蔽通信能力；ΠCCAP 将可靠性机制嵌入隐蔽通道本身

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首创性地形式化了 IoA 隐蔽通信模型和 IND-INT 安全标准，混合三维通道设计新颖
- 实验充分度: ⭐⭐⭐⭐ 容量/鲁棒性/隐蔽性多维评估完整，但缺乏消融实验和更强对手测试
- 写作质量: ⭐⭐⭐⭐⭐ 形式化定义严谨，协议描述清晰，安全证明完整
- 价值: ⭐⭐⭐⭐ 对 AI 安全社区有重要警示和防御研究价值，但应用场景相对小众

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Blue Teaming Function-Calling Agents](blue_teaming_function-calling_agents.md)
- [\[AAAI 2026\] Scaling Equitable Reflection Assessment in Education via Large Language Models and Role-Based Feedback Agents](scaling_equitable_reflection_assessment_in_education_via_large_language_models_a.md)
- [\[ICML 2026\] Preregistration for Experiments with AI Agents](../../ICML2026/llm_nlp/preregistration_for_experiments_with_ai_agents.md)
- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](../../ICLR2026/llm_nlp/ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[ICLR 2026\] Speculative Actions: A Lossless Framework for Faster AI Agents](../../ICLR2026/llm_nlp/speculative_actions_faster_ai_agents.md)

</div>

<!-- RELATED:END -->
