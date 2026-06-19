---
title: >-
  [论文解读] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment
description: >-
  [NeurIPS 2025][LLM安全][model protection] 提出 CoreGuard，通过行置换（row permutation）锁定 Transformer 线性层权重 + 列置换传播协议（propagation protocol）将 TEE 授权次数降至 1 次，以极低计算和通信开销保护边缘部署 LLM 的基础能力不被模型窃取攻击利用。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "model protection"
  - "edge deployment"
  - "TEE"
  - "permutation"
  - "LLM safety"
  - "model stealing"
---

# CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment

**会议**: NeurIPS 2025  
**arXiv**: [2410.13903](https://arxiv.org/abs/2410.13903)  
**代码**: 未公开  
**领域**: AI安全  
**关键词**: model protection, edge deployment, TEE, permutation, LLM safety, model stealing

## 一句话总结

提出 CoreGuard，通过行置换（row permutation）锁定 Transformer 线性层权重 + 列置换传播协议（propagation protocol）将 TEE 授权次数降至 1 次，以极低计算和通信开销保护边缘部署 LLM 的基础能力不被模型窃取攻击利用。

## 研究背景与动机

**领域现状**：专有 LLM（如 ChatGPT、Claude）正越来越多地部署到边缘设备上以实现低延迟和隐私保护（如 Apple Intelligence 在 iOS 上部署 3B 参数 LLM）。

**现有痛点**：
   - **水印方法**（被动防护）：仅证明所有权，攻击者仍可在无监督环境下滥用模型
   - **模型加密**：运行时解密后仍可被逆向工程攻击
   - **TEE 全模型保护**（TPTE）：CPU 飞地执行速度约降低 50×，不可行
   - **部分 TEE 执行**（PPTE, 如 DarkneTZ）：保护参数太少，攻击者仅用 1% 数据即可重建
   - **参数混洗**（PSP, 如 ShadowNet）：每个线性层都需 TEE↔GPU 传输，LLaMA3-8B 单 token 生成就需 ~1.3GB 传输、~1.3 秒延迟

**核心矛盾**：现有方案在安全性和效率之间存在根本矛盾——足够安全的保护带来不可接受的计算/通信开销，而高效方案的安全性不充分。尤其是"基础能力窃取"威胁：攻击者可微调锁定模型以利用其泛化能力执行新任务。

**本文目标** 设计一个即插即用的保护方案，使锁定模型仅在 TEE 授权下正常工作，且计算和通信开销可忽略不计。

**切入角度**：利用置换矩阵的数学性质（$\pi\pi^T = I$）实现"锁+钥匙"机制，并通过列置换传播使授权信号自动流过所有后续层，避免重复 TEE 调用。

**核心 idea**：行置换锁定权重 + 列置换传播授权，将 LLM 保护的 TEE 交互从每层 2 次压缩到全模型仅 5 轮。

## 方法详解

### 整体框架

CoreGuard 分为两个阶段：

**模型锁定（部署前）**：对训练好的模型的线性层权重矩阵进行行置换（protection protocol），使未授权输入无法正确计算；同时对输出处理层进行列置换（propagation protocol），使授权信号自动传播。

**推理授权（部署后）**：TEE 对首个被置换层的输入进行一次性列置换授权，之后所有后续层自动获得正确的置换输入。授权过程中用 OTP（一次性密码本）加密隐藏置换操作。

### 关键设计

1. **保护协议（Protection Protocol）**:

    - 功能：对输入处理层（QKV 投影、FFN 输入层）进行行置换使其"锁定"
    - 核心思路：设置换矩阵 $\pi \in \{0,1\}^{d \times d}$，对权重做行置换 $W'_q = \pi^T W_q$。只有列置换的输入 $x\pi$ 才能正确计算：$x\pi \cdot \pi^T W_q = xW_q = Q$，否则输出完全错误
    - 设计动机：置换不改变参数功能，只是将参数映射到新域，猜测正确 $\pi$ 的概率为 $1/(d!)$（$d=4096$ 时远超可枚举范围）

2. **传播协议（Propagation Protocol）**:

    - 功能：使授权信号自动从一层传播到下一层，避免每层都需 TEE 授权
    - 核心思路：对输出处理层（$W_o$、$W_n$、Add-Norm 参数）进行列置换——$W'_n = W_n\pi$, $b'_n = b_n\pi$, $\gamma'_2 = \gamma_2\pi$, $\beta'_2 = \beta_2\pi$。这使层的输出自动变为列置换形式 $n' = mW_n\pi + b_n\pi = n\pi$，整个 Transformer 层的功能变为 $f_{w'}(x\pi) = f_w(x)\pi$，输出 $z\pi$ 直接作为下一层的授权输入
    - 设计动机：ShadowNet 每个线性层需 2 次 TEE 传输（LLaMA3-8B 共 448 次）；传播协议将其压缩到全模型仅需初始 1 次授权

3. **推理授权与 OTP 加密**:

    - 功能：安全地生成首次授权的列置换特征，防止攻击者通过输入-输出对比推导 $\pi$
    - 核心思路：分 4 步——(a) FFN 输入层正常计算得到 $m$；(b) TEE 内用 OTP 加密并置换 $m' = (m + p)\pi$；(c) GPU 用预对齐的权重 $W'_n = \pi^T W_n$ 处理加密特征 $n' = n + pW_n$；(d) TEE 内解密 $n'' = n' - pW_n = n$，然后做 Add-Norm 和列置换输出 $z\pi$
    - 设计动机：OTP 保证每次加密结果不同（即使相同输入），置换和加密相互掩护；TEE 仅执行轻量线性运算，全过程 5 轮 TEE-GPU 传输

4. **授权位置选择**:

    - 功能：决定从哪一层开始锁定
    - 核心思路：设在模型中间层，攻击者至少需要恢复一半参数才能获得完整模型
    - 设计动机：若在首层或末层，攻击者只需重训一个层（类似 prompt tuning 或训练分类头）；中间位置最大化攻击难度

### 损失函数 / 训练策略

CoreGuard 是即插即用方案，不需要额外训练。锁定操作是确定性的数学变换（置换矩阵操作），保证授权后的模型功能与原模型完全一致，零精度损失。

## 实验关键数据

### 主实验 — 安全性对比（模型窃取攻击准确率 ↓）

| 模型 | 任务 | No-shield | DarkneTZ | ShadowNet | DTE | CoreGuard | Black-box |
|------|------|:---------:|:--------:|:---------:|:---:|:---------:|:---------:|
| Qwen2-0.5B | GSM8k | 21.53 | 16.81 | 1.34 | 2.36 | **2.41** | 1.29 |
| Gemma2-2B | GSM8k | 40.94 | 37.07 | 10.81 | 4.56 | **3.91** | 1.74 |
| ChatGLM3-6B | GSM8k | 55.95 | 54.91 | 0.43 | 0.93 | **1.04** | 0.23 |
| LLaMA3-8B | GSM8k | 53.07 | 51.31 | 4.15 | 6.09 | **6.22** | 4.05 |
| 相对平均 | 全部 | 9.58× | 8.43× | 1.09× | 1.18× | **1.17×** | 1.00× |

CoreGuard 安全性与黑盒上界几乎持平（1.17× vs 1.00×），远优于 PPTE 方案（DarkneTZ 8.43×）。所有模型的未授权直接推理准确率均为 **0.00%**。

### 消融实验 — 效率对比

| 方案 | TEE 传输次数/token | 额外 FLOPs | 实用性 |
|------|:-----------------:|:----------:|:------:|
| ShadowNet | 448 (LLaMA3-8B) | 高 | 单 token ~1.3s |
| DTE (半模型TEE) | 0 | 极高 (~50×) | 不可行 |
| CoreGuard | **5** | **可忽略** | ✓ |

### 关键发现
- **安全性等价于 TEE 全保护**：CoreGuard 的 1.17× 相对精度与 DTE（直接用 TEE 保护同等参数）的 1.18× 几乎相同，说明置换保护效力等价于 TEE 直接保护
- **零精度损失**：授权模型与原始模型在四个任务上精度完全一致（GSM8k、Spider、PubMedQA、SQuAD）
- **攻击者面临 NP-hard 问题**：即使有辅助信息，恢复 $\pi$ 的问题归约为 Learning With Errors (LWE) 问题
- **PPTE 方案普遍不安全**：DarkneTZ（8.43×）、SOTER 等仅保护少量参数的方案在 100% 数据集攻击下几乎无防御效果
- **TPTE 完全无效**：NPLO（9.59×）几乎等同于无防护（9.58×），仅保护任务适配器对 LLM 毫无用处

## 亮点与洞察
- **传播协议的数学优雅性**：利用置换矩阵的正交性（$\pi\pi^T = I$），使授权信号在 Transformer 各层间"自传播"。锁（行置换）和钥匙（列置换）在数学上精确对消，零误差
- **OTP + 置换的安全嵌套**：OTP 掩护置换、置换掩护 OTP，单独观察任一操作都无法推导另一个，安全性来自两层互相掩护
- **即插即用设计**：不需要重新训练模型，不改变模型架构，仅对权重做数学变换。可直接应用于任何 Transformer 模型

## 局限与展望
- 仅考虑了微调攻击（fine-tuning attack），未评估蒸馏攻击、侧信道攻击等其他威胁向量
- 假设 TEE 完全安全，但实际 TEE 实现（如 Intel SGX）已被多次攻破
- 置换粒度为通道级（$d$ 维），更细粒度的保护（如子通道级）是否能进一步提升安全性未探讨
- OTP 噪声 $pW_n$ 的预计算需要在离线阶段完成，存储和更新开销未充分讨论
- 仅实验了最大 8B 参数的模型，超大规模模型（>70B）的适用性未验证
- 对非标准 Transformer 架构（如 MoE、SSM）的适配性未讨论

## 相关工作与启发
- **vs ShadowNet**: 都用置换保护，但 ShadowNet 每层需 TEE 授权（448 次），CoreGuard 仅 1 次授权 + 传播，通信开销降低两个数量级
- **vs DarkneTZ/SOTER (PPTE)**: 将最后/部分层放 TEE，安全性不足（8.43×）；CoreGuard 保护同样多的参数但用置换代替 TEE 执行
- **vs 模型加密**: 加密保护静态参数但无法防御运行时逆向工程；CoreGuard 的置换使运行时参数也无法直接使用
- 对边缘AI安全部署有直接实践价值，传播协议的思路可推广到其他安全计算场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 传播协议将 TEE 交互降至 1 次是巧妙的工程创新
- 实验充分度: ⭐⭐⭐⭐ 4 模型 × 4 任务 × 8 基线，安全性和效率评估全面
- 写作质量: ⭐⭐⭐⭐ 威胁模型定义清晰，安全分析严谨
- 价值: ⭐⭐⭐⭐ 边缘 LLM 保护是实际工业需求，方案即插即用可落地

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction](masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)
- [\[NeurIPS 2025\] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)
- [\[ACL 2026\] CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks](../../ACL2026/llm_safety/crossguard_safeguarding_mllms_against_joint-modal_implicit_malicious_attacks.md)
- [\[ACL 2025\] Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs](../../ACL2025/llm_safety/are_the_hidden_states_hiding_something_testing_the_limits_of_factuality-encoding.md)
- [\[NeurIPS 2025\] Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs](bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)

</div>

<!-- RELATED:END -->
