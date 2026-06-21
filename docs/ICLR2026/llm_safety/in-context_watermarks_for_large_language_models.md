---
title: >-
  [论文解读] In-Context Watermarks for Large Language Models
description: >-
  [ICLR 2026][LLM安全][In-Context Watermark] 本文提出 In-Context Watermark (ICW)，**只靠精心设计的提示词**就能让任意黑盒 LLM 在输出中嵌入可检测的隐形水印，无需访问解码过程，并以"学术同行评审中检测 AI 代写评审"为典型场景展示其实用价值。
tags:
  - "ICLR 2026"
  - "LLM安全"
  - "In-Context Watermark"
  - "黑盒水印"
  - "提示工程"
  - "间接提示注入"
  - "AI 生成检测"
---

# In-Context Watermarks for Large Language Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=fD9YRHazW3](https://openreview.net/forum?id=fD9YRHazW3)  
**代码**: [https://github.com/yepengliu/In-Context-Watermarks](https://github.com/yepengliu/In-Context-Watermarks)  
**领域**: LLM 安全 / 文本水印 / 内容溯源  
**关键词**: In-Context Watermark, 黑盒水印, 提示工程, 间接提示注入, AI 生成检测  

## 一句话总结
本文提出 In-Context Watermark (ICW)，**只靠精心设计的提示词**就能让任意黑盒 LLM 在输出中嵌入可检测的隐形水印，无需访问解码过程，并以"学术同行评审中检测 AI 代写评审"为典型场景展示其实用价值。

## 研究背景与动机
**领域现状**：LLM 水印是给 AI 生成文本盖"溯源戳"的主流手段，但绝大多数方法（Kirchenbauer 的绿/红 token 列表、Aaronson 的 Gumbel-Max 伪随机采样等）都把嵌入与检测的控制权交给**模型所有者**，需要在解码阶段干预 next-token 分布或采样过程。

**现有痛点**：这种"in-process"范式有一个硬性前提——你得能碰到模型的解码过程。可现实中大量场景里检测方根本拿不到模型。论文给的标志性例子是学术会议组织方想识别"懒惰评审人把论文丢给 LLM 生成评审"这种违规行为：组织方既不知道评审人用了哪个模型，也无法干预其解码；而 DetectGPT、GPTZero 这类事后检测工具准确率低、误报率高，主流商业 LLM 也没公开部署水印。

**核心矛盾**：水印要么需要模型所有者配合（in-process），要么需要事后改写已生成文本（post-hoc），但都无法覆盖"检测方既无模型访问权、又想主动埋下溯源信号"的广阔中间地带。

**本文目标**：探索一个全新问题——能否**仅凭提示工程**，在不需要任何模型特权访问的前提下嵌入水印？

**核心 idea（黑盒提示即水印）**：利用现代 LLM 强大的**上下文学习**与**指令遵循**能力，把水印逻辑写成一条自然语言"水印指令"塞进提示（system prompt 或文档正文），让模型在后续所有回答里自动携带隐形水印。更进一步，借**间接提示注入 (IPI)** 把水印指令用"白色字体/零字号"藏进论文 PDF——评审人一旦把全文喂给 LLM 生成评审，输出就会带上可检测的水印。

## 方法详解

### 整体框架
ICW 把水印从"改解码"搬到了"改提示"：检测方与生成方共享一个密钥 $k$ 和水印方案 $\tau$，水印指令 $\text{Instruction}(k,\tau)$ 作为提示前缀。给定正常查询 $Q$，水印化响应为 $y \leftarrow M(\text{Instruction}(k,\tau) \oplus Q)$，其中 $\oplus$ 表示拼接、$M$ 是任意黑盒 LLM。检测器 $D(\cdot|k,\tau)$ 把判断 $y$ 是否含水印形式化为假设检验：当 $D(y|k,\tau) \ge \eta$ 时判为含水印（拒绝零假设）。论文区分两种部署设定——**DTS（Direct Text Stamp）**把水印指令直接放进 system prompt 实现全程水印；**IPI（Indirect Prompt Injection）**把指令隐蔽嵌入长文档，靠用户主动把文档喂给 LLM 来触发，对应同行评审反作弊场景。

```mermaid
flowchart LR
    A[水印指令 Instruction k,τ] --> C[黑盒 LLM M]
    B[正常查询 Q / 文档] --> C
    C --> D[水印化响应 y]
    D --> E[检测器 D·|k,τ]
    E --> F{D ≥ η ?}
    F -->|是| G[判定含水印]
    F -->|否| H[判定无水印]
```

围绕"自然语言的不同粒度"，论文设计了四种 ICW 策略，分别覆盖字符级、单词级、词表级与句子级，并为每种配套定制检测器。

### 关键设计
**1. Unicode ICW：字符级隐形戳，最低能力门槛**。最朴素的一档——指令让模型在每个词后插入一个零宽空格 $\text{U+}200\text{B}$，输出形如 $\{y^{(1)},\text{U+}200\text{B},\dots,y^{(n)},\text{U+}200\text{B}\}$。检测器直接数隐形字符密度 $D(y|k_u,\tau_u)=|y|_{k_u}/N$。它对模型指令遵循能力要求极低、对人类完全不可见、对复制粘贴和增删词几乎完美鲁棒；但只活在数字文本里（打印/扫描即失效），且遇到 LLM 改写会被整批清除，属于"易部署但脆弱"的一极。

**2. Initials ICW：首字母偏置 + z 统计量检测**。用密钥 $k_c$ 随机抽一组"绿字母" $A_G$，指令让模型**多用以绿字母开头的词**。检测时把文本的绿首字母占比与人类基线对比，算 z 统计量 $D(y|k_c,\tau_c)=(|y|_G-\gamma|y|)/\sqrt{\gamma(1-\gamma)|y|}$，其中 $|y|_G$ 是绿首字母词数、$\gamma$ 是人类文本中绿首字母词的期望比例（用 Canterbury 语料库估计首字母分布 $P_A$ 后求 $\gamma=\sum_i P_A(a^{(i)}\in A_G)$）。它对指令遵循能力要求较高，但在强模型上检测性与鲁棒性都不错；代价是会引入可被对手反推出 $A_G$ 的统计偏置，存在被伪造（spoofing）的风险。

**3. Lexical ICW：词表分绿/红，对齐经典绿名单思想**。受 Kirchenbauer 绿/红 token 列表启发，但改用**完整单词**而非 token：用密钥 $k_L$ 把词表 $V$ 切成绿词表 $V_G$（占比 $\gamma|V|$）和红词表 $V_R$，并把 $V$ 限制为形容词/副词/动词这些更承载文体特征、与主题无关的词类以压缩规模。指令让模型尽量多用绿词，检测沿用 Initials 的 z 统计量框架，只是把 $|y|_G$ 换成绿词命中数、$\gamma=|V_G|/|V|$。它对模型的**长上下文检索能力**要求最高——模型得在长绿名单里记住并适时选用这些词，对当前模型是不小的挑战。

**4. Acrostics ICW：句首藏密文，离散编辑下最稳**。句子级策略——用密钥 $k_s$ 采一段水印密钥序列 $\zeta=\{\zeta^{(1)},\dots,\zeta^{(m)}\}$，指令让每个句子的首字母依次对齐 $\zeta$（即生成的句首字母序列 $\ell$ 满足 $\ell^{(i)}=\zeta^{(i)}$），把秘密信息编进句首（图 1 中让回答成为 "MHAB" 的藏头诗）。检测用 Levenshtein 距离衡量 $\ell$ 与 $\zeta$ 的接近度，并标准化为 z 统计量 $D(y|k_s,\tau_s)=(\mu-d(\ell,\zeta))/\sigma$，其中 $\mu,\sigma$ 通过从嫌疑文本重采样 $N$ 条句首序列估计。由于只约束句首字母、不限制句子其余内容，它对增删、替换、改写都高度冗余鲁棒，只要大部分句首字母序列保留即可；用固定 $\zeta$ 会暴露模式，故建议采很长的 $\zeta$、每次生成只用一段短子序列以提升隐蔽性。

## 实验关键数据
**设置**：用两个黑盒商业模型 gpt-4o-mini 与能力更强的 gpt-o3-mini；DTS 用长问答数据集 ELI5（原答案当人类文本），IPI 用 2020–2023 ICLR 论文整篇作为输入；每次评测 500 条水印文本 + 500 条人类文本，各 300 词。Baseline 为两个黑盒后处理方法 PostMark、YCZ+23 与事后检测器 GPTZero。指标含 ROC-AUC、低误报率下的 T@1%F / T@10%F、鲁棒性（删 30%、换 30%、改写）与文本质量（困惑度 + LLM-as-a-Judge）。

### 主实验表格（检测性能 ROC-AUC，节选 T@1%F）

| 模型 | 方法 | DTS AUC ↑ | DTS T@1%F ↑ | IPI AUC ↑ | IPI T@1%F ↑ |
|------|------|-----------|-------------|-----------|-------------|
| GPT-4o-mini | Unicode ICW | 1.000 | 1.000 | 0.857 | 0.714 |
| GPT-4o-mini | Initials ICW | 0.572 | 0.006 | 0.620 | 0.006 |
| GPT-4o-mini | Lexical ICW | 0.910 | 0.320 | 0.889 | 0.054 |
| GPT-4o-mini | Acrostics ICW | 0.590 | 0.036 | 0.592 | 0.002 |
| GPT-o3-mini | Unicode ICW | 1.000 | 1.000 | 1.000 | 1.000 |
| GPT-o3-mini | Initials ICW | 0.999 | 0.990 | 0.997 | 0.910 |
| GPT-o3-mini | Lexical ICW | 0.995 | 0.930 | 0.997 | 0.974 |
| GPT-o3-mini | Acrostics ICW | 1.000 | 1.000 | 0.997 | 0.982 |
| — | PostMark (DTS) | 0.977 | 0.802 | — | — |
| — | YCZ+23 (DTS) | 0.998 | 0.992 | — | — |

核心结论一目了然：弱模型 GPT-4o-mini 只有 Unicode（指令门槛最低）能用，其余三种近乎失效；换到 GPT-o3-mini 后四种 ICW 全部逼近满分，且 PostMark/YCZ+23 这类后处理方法**根本无法用于 IPI**（评审人没动机给自己加水印），凸显 ICW 在黑盒+无访问权场景的独占优势。

### 消融/鲁棒性表格（DTS, gpt-o3-mini, AUC）

| 攻击 | Initials | Lexical | Acrostics | YCZ+23 | PostMark |
|------|----------|---------|-----------|--------|----------|
| 删词 30% | 0.999 | 0.857 | 0.881 | 0.980 | 0.908 |
| 换词 30% | 0.999 | 0.758 | 1.000 | 0.982 | 0.956 |
| 改写 | 0.887 | 0.924 | 0.922 | 0.557 | 0.841 |

文本质量上（LLM-as-a-Judge Overall），Unicode 4.810、Lexical 4.808、Acrostics 4.813，均接近人类 4.235 之上、远胜 PostMark 的 2.997。

### 关键发现
- **ICW 效果与模型能力强相关**：四种策略的可用性随模型上下文学习/指令遵循/长上下文检索能力提升而水涨船高，作者据此推断"模型越强，ICW 越好用"。
- **改写攻击下 ICW 反而比 baseline 更稳**：YCZ+23 在改写下崩到 0.557，而 Initials/Lexical/Acrostics 都维持在 0.88–0.92。
- **Lexical 在换词攻击下偏弱**（0.758），因为它依赖的绿词正好是名词/动词/形容词/副词，恰是同义词替换的主要目标。
- **IPI 长上下文可行**：把水印指令藏进整篇论文后，强模型仍能在长文档里可靠遵循指令。

## 亮点与洞察
- **范式转移**：把水印控制权从"模型所有者改解码"挪到"任何第三方改提示"，第一次让黑盒、无访问权场景下的主动溯源成为可能。
- **威胁模型反转很巧**：IPI 通常是攻击者往文档里塞恶意指令，本文把它反过来当防御手段——善意组织方埋水印指令，可能违规的用户反而是触发者。
- **四策略覆盖完整 trade-off 谱系**：从"低门槛易碎"（Unicode）到"高门槛强鲁棒"（Acrostics），并配 Table 1 直观对比 LLM 要求/检测性/鲁棒性/质量，工程选型友好。
- **顺势而为的乐观论断**：把方法局限（弱模型不灵）转化为正向趋势预测（模型升级 = 水印免费变强），叙事有说服力。

## 局限与展望
- **强依赖模型能力**：除 Unicode 外，三种策略在 GPT-4o-mini 级别的模型上基本失效，对中小模型不通用。
- **攻防尚浅**：论文坦承 IPI 下的攻击/防御（如检测并删除嵌入指令、或在评审提示前加"ignore prior prompts"绕过）只做了初步探讨，系统性攻防留作未来工作。
- **各策略各有死穴**：Unicode 遇改写/打印即废；Initials 的统计偏置易被反推导致 spoofing；Lexical 长名单检索难、换词攻击弱；固定 $\zeta$ 的 Acrostics 模式显眼。
- **伦理边界**：IPI 本质是把"提示注入攻击"工具化用于监管，水印指令藏进他人提交的论文涉及知情同意与公平性问题（论文也讨论了应由组织方而非作者来埋戳以避免利益冲突）。

## 相关工作与启发
本文位于 LLM 水印的第三条路：传统分 **post-hoc**（格式/词汇/句法变换、LLM 改写重生成，如 YCZ+23、PostMark）与 **in-process**（解码期 logits 扰动或伪随机采样，如 Kirchenbauer、Aaronson、Bahri 黑盒 n-gram 评分），ICW 则两者都不属于——它既不改已生成文本、也不碰解码，纯靠提示。技术上它把**提示注入攻击**的研究成果（零字号/透明文本等混淆手段）创造性地转用于防御。启发在于：随着指令遵循成为 LLM 的核心能力，"用自然语言指令实现以往需要底层访问才能做的事"是一条会持续变强的通用思路，水印只是其中一例；同时它也提醒社区，提示注入这把双刃剑既能被攻击者用、也能被监管者用。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 把水印从"改解码"彻底搬到"改提示"，开辟黑盒无访问权下的主动溯源新范式，并反转 IPI 威胁模型用于反作弊，问题设定与思路都很新。
- **实验充分度**: ⭐⭐⭐⭐ 两种模型 × 两种设定 × 四种策略，检测/鲁棒/质量三维度齐全，与黑盒+事后 baseline 对比扎实；扣分在攻防探讨偏浅、只测了两个商业模型、缺开源/中小模型的横向验证。
- **写作质量**: ⭐⭐⭐⭐⭐ 动机用同行评审场景讲得极具画面感，四策略按粒度组织清晰，Table 1 trade-off 概览友好，叙事流畅。
- **价值**: ⭐⭐⭐⭐ 直击"无模型访问权也要溯源 AI 文本"的真实刚需，对学术诚信、内容平台都有现实意义；随模型升级价值还会上升，但当前对弱模型不通用限制了即时落地面。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](../../ACL2026/llm_safety/topic-based_watermarks_for_large_language_models.md)
- [\[ICLR 2026\] Do LLMs Forget What They Should? Evaluating In-Context Forgetting in Large Language Models](do_llms_forget_what_they_should_evaluating_in-context_forgetting_in_large_langua.md)
- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](../../ACL2025/llm_safety/ensemble_watermarks_llm.md)
- [\[ICLR 2026\] LLM Fingerprinting via Semantically Conditioned Watermarks](llm_fingerprinting_via_semantically_conditioned_watermarks.md)
- [\[ICLR 2026\] PRISON: Unmasking the Criminal Potential of Large Language Models](prison_unmasking_the_criminal_potential_of_large_language_models.md)

</div>

<!-- RELATED:END -->
