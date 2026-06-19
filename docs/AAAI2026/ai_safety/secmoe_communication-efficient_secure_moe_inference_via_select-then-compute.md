---
title: >-
  [论文解读] SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute
description: >-
  [AAAI 2026][AI安全][MoE] 提出 SecMoE 框架，通过 Select-Then-Compute 范式在两方安全计算中高效实现稀疏 MoE 推理，避免冗余专家计算，通信量降低最高 29.8 倍，端到端加速最高 16.1 倍。 核心问题 Transformer 模型的隐私保护推理（Privacy-Prese…
tags:
  - "AAAI 2026"
  - "AI安全"
  - "MoE"
  - "隐私推理"
  - "安全多方计算"
  - "同态加密"
  - "Select-Then-Compute"
---

# SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute

**会议**: AAAI 2026  
**arXiv**: [2601.06790](https://arxiv.org/abs/2601.06790)  
**代码**: 未公开  
**领域**: AI Safety / 隐私保护机器学习  
**关键词**: MoE, 隐私推理, 安全多方计算, 同态加密, Select-Then-Compute

## 一句话总结

提出 SecMoE 框架，通过 Select-Then-Compute 范式在两方安全计算中高效实现稀疏 MoE 推理，避免冗余专家计算，通信量降低最高 29.8 倍，端到端加速最高 16.1 倍。

## 背景与动机

### 核心问题
Transformer 模型的隐私保护推理（Privacy-Preserving Inference）日益重要，但现有安全两方计算（2-PC）框架主要针对 BERT、GPT-2 等小模型，与实际大模型规模存在百倍差距。MoE（Mixture of Experts）架构通过稀疏激活可在低计算开销下扩展模型容量，是弥补这一差距的潜在方案。

### 隐私泄露风险
在标准 2-PC 协议中，服务器使用明文权重通过同态加密计算 FFN 层。但在 MoE 场景下，如果服务器知道哪个专家被激活，就能推断出客户端输入的 token 级隐私信息。这是一个此前未被充分解决的新型隐私威胁。

### 朴素方案的缺陷
最直接的保护方式是先评估所有专家再选择，但这完全抵消了稀疏 MoE 的核心优势——计算效率。例如对 128 专家模型，朴素方案需要计算全部 128 个专家的 FFN，开销巨大。

## 方法详解

### 核心范式：Select-Then-Compute

SecMoE 的核心思想是将安全计算分为**选择阶段（Selection Phase）**和**计算阶段（Compute Phase）**两步：

1. **选择阶段**：将多个计算入口（entry）统一为相同电路结构，提取各入口的参数作为候选项，通过密文向量进行不经意选择（oblivious selection）
2. **计算阶段**：仅对选中的单个入口执行加密计算

该范式同时应用于安全稀疏 MoE 层和安全分段多项式评估两个场景。

### 设计一：安全稀疏 MoE 协议

威胁模型为半诚实（semi-honest）的两方设置：客户端 $C$ 持有私有输入，服务器 $S$ 持有模型权重。

**选择阶段**：
- 客户端和服务器通过 $\Pi_{\text{Topk}}$ 协议得到 top-k 专家的秘密共享索引
- 通过 $\Pi_{\text{onehot}}$ 生成 $N_{\text{exp}}$ 长度的 one-hot 布尔向量 $t^b$
- 经 $\Pi_{\text{B2A}}$ 转换为算术形式 $t^a$，客户端加密后发送给服务器
- 服务器利用同态加密的局部无通信优势，计算选中专家的加密权重：

$$[\![W_r^1]\!] = \sum_{i=0}^{N_{\text{exp}}-1} W_i^1 \cdot [\![t^a]\!]$$

此操作对 $V_i$ 和 $W_i^2$ 同理执行，仅需传输一个 $N_{\text{exp}}$ 长度的选择向量。

**计算阶段**：
- 客户端加密输入份额 $[\![\langle x \rangle_c]\!]$ 发送至服务器
- 服务器执行密文-密文矩阵乘法 $[\![W_r^1]\!] \cdot [\![x]\!]$ 和 $[\![V_r]\!] \cdot [\![x]\!]$
- 完成 GeLU 激活和 GLU 门控后，再做一次密文乘法 $[\![W_r^2]\!] \cdot [\![\text{GLU}]\!]$
- 通过随机掩码 $R$ 保护中间结果，最终双方各得输出份额

关键优势：从 32 专家到 128 专家，SecMoE 计算仅增长 24%，而 Iron/BumbleBee 增长 178%。

### 设计二：安全分段多项式选择（Secure GeLU）

GeLU 函数通过分段二次多项式近似：

$$\text{GeLU}(x) = \begin{cases} 0 & x \in (-\infty, -5] \\ P_1(x) & x \in (-5, -3] \\ P_2(x) & x \in (-3, -1] \\ P_3(x) & x \in (-1, 1] \\ P_4(x) & x \in (1, 3] \\ x & x \in (3, \infty) \end{cases}$$

**选择阶段**：
- 将所有分段多项式系数收集为矩阵，行索引 $i$ 表示段，列索引 $j$ 表示系数（从最高次到常数项）
- 低次多项式通过零填充统一至最高次幂
- 通过安全比较 $\Pi_{\text{comp}}\{x < b_i\}$ 生成 one-hot 段选择器
- 单次掩码矩阵-向量积即可检索出目标系数行

**计算阶段**：
- 计算输入自乘 $\langle x^2 \rangle := \Pi_{\text{Mul}}(x, x)$
- 用选中系数执行二次多项式求值：$\langle y \rangle = \Pi_{\text{Mul}}(\langle x^2 \rangle, \langle c_r \rangle_0) + \Pi_{\text{Mul}}(\langle x \rangle, \langle c_r \rangle_1) + \langle c_r \rangle_2$
- 最大绝对误差 $1.2 \times 10^{-2}$，平均绝对误差 $1.7 \times 10^{-3}$

**进一步优化**：统一断点比较并复用结果，减少安全比较通信轮次；利用系数矩阵的零项跳过 $\Pi_{\text{MUX}}$ 操作。

## 实验结果

### 实验设置
- 环境：$\mathbb{Z}_{2^{64}}$ 环，定点精度 $s=18$，双节点（64 vCPU + 128GB RAM）
- 网络：LAN（1Gbps, 0.5ms）和 WAN（400Mbps, 4ms）
- 基线：Iron (NeurIPS 2022)、BumbleBee (NDSS 2025)
- 模型：MoE-Small（124M, 8 专家）、Switch-Base（0.62B-7B, 8-128 专家）

### 表1：运行时间对比（分钟，128 专家设置）

| 方法 | MoE-Small LAN | MoE-Small WAN | Switch-Base LAN | Switch-Base WAN |
|------|---------|---------|----------|----------|
| Iron | 12.07 (4.7×) | 59.14 (16.1×) | 35.5 (2.9×) | 143.78 (9.7×) |
| BumbleBee | 9.76 (3.8×) | 13.88 (3.8×) | 32.3 (2.6×) | 34.89 (2.3×) |
| **SecMoE** | **2.52** | **3.68** | **12.1** | **14.73** |

### 表2：通信量对比（GB）

| 方法 | 16 专家 | 32 专家 | 64 专家 | 128 专家 |
|------|--------|--------|--------|---------|
| Iron | 7.13 (8.9×) | 9.44 (11.2×) | 17.19 (21.2×) | 24.17 (29.4×) |
| BumbleBee | 1.42 (1.8×) | 2.04 (2.4×) | 3.37 (4.2×) | 5.81 (7.1×) |
| **SecMoE** | **0.81** | **0.84** | **0.81** | **0.82** |

SecMoE 通信量几乎不随专家数增长（16→128 专家仅从 0.81GB 变为 0.82GB），而 Iron 增长 3.4 倍。

### 精度验证（MoE-Small on GLUE）

| 数据集 | 指标 | 明文基线 | SecMoE |
|--------|------|---------|--------|
| CoLA | MCC | 41.0 | 41.0 |
| QNLI | ACC | 90.3 | 90.2 |
| RTE | ACC | 69.9 | 70.0 |

精度损失 ≤0.1%，可忽略不计。

## 主要发现

1. **通信近乎恒定**：SecMoE 的通信量与专家数量几乎无关，这是 Select-Then-Compute 范式的直接结果——只传输一个选择向量和执行单个专家的计算
2. **扩展性极优**：模型参数扩大 63 倍时，端到端运行时间仅增加 15.2 倍
3. **WAN 优势更大**：在带宽受限的 WAN 环境下，SecMoE 的通信节省转化为更显著的加速（最高 16.1×），适合实际部署
4. **GeLU 优化效果**：Switch-Base 128 专家下，SecMoE 的 GeLU 协议比 BumbleBee 快 7.1 倍，通信减少 81%

## 亮点

- **首个实用的安全 MoE 推理协议**：填补了 2-PC 安全推理在 MoE 架构上的空白
- **优雅的统一抽象**：Select-Then-Compute 将 MoE 专家选择和分段多项式评估统一为相同范式，设计简洁
- **精度无损**：在 GLUE 基准上与明文推理精度几乎一致
- **通信量关于专家数恒定**：打破了现有方法通信随专家数线性增长的瓶颈

## 局限性

1. **仅支持半诚实模型**：假设双方按协议执行但试图窥探信息，未考虑恶意对手场景
2. **内存瓶颈**：256+ 专家因加载模型参数和存储 Beaver Triple 导致内存不足
3. **Top-1 专家限制**：实验仅验证 $K_{\text{exp}}=1$，未充分探索 Top-2 等多专家激活场景
4. **模型规模有限**：最大测试 Switch-Base 7B，未验证百亿级以上模型
5. **Softmax 未优化**：指数函数的高阶 Taylor 展开不适合 Select-Then-Compute，保留了原始方案

## 相关工作

- **神经网络安全推理**：MiniONN、Gazelle、CrypTFlow2 等奠定了 2-PC 安全 NN 推理基础
- **Transformer 安全推理**：Iron 首次将 HE 引入 Transformer 线性层；BumbleBee 优化了格基加法 HE；BOLT 和 SHAFT 分别改进了非线性层和预处理阶段
- **MoE 架构**：稀疏 MoE（Shazeer 2017）和 Switch Transformer（Fedus 2022）是本工作的模型基础

## 评分

⭐⭐⭐⭐ (4/5)

- 创新性：⭐⭐⭐⭐ — Select-Then-Compute 范式新颖，统一解决 MoE 隐私和效率问题
- 实验：⭐⭐⭐⭐ — 多模型、多专家数、LAN/WAN 设置完整，但缺少更大模型和真实部署评估
- 写作：⭐⭐⭐⭐ — 协议描述严谨，图示清晰
- 实用性：⭐⭐⭐ — 半诚实假设和内存限制制约了实际应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing](../../ICML2026/ai_safety/fusefss_efficient_secure_llm_inference_with_function_secret_sharing.md)
- [\[ACL 2026\] On the (In-)Security of the Shuffling Defense in the Transformer Secure Inference](../../ACL2026/ai_safety/on_the_in-security_of_the_shuffling_defense_in_the_transformer_secure_inference.md)
- [\[CVPR 2026\] Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression](../../CVPR2026/ai_safety/computation_and_communication_efficient_federated_unlearning_via_on-server_gradi.md)
- [\[AAAI 2026\] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)
- [\[AAAI 2026\] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)

</div>

<!-- RELATED:END -->
