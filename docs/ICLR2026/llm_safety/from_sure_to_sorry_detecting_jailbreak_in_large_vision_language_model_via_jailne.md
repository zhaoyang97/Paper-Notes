---
title: >-
  [论文解读] From "Sure" to "Sorry": Detecting Jailbreak in Large Vision Language Model via JailNeurons
description: >-
  [ICLR 2026][LLM安全][越狱检测] JDJN 把 LVLM 越狱检测落到"神经元级"——用因果消融在每层定位出一小撮专门被越狱输入激活的 JailNeurons：，再跨层聚合它们的激活训练一个轻量 SVM，从而以近乎零误报、超低延迟实现对未见攻击类型的强泛化检测。 领域现状：大型视觉语言模型（LVLM）继承了…
tags:
  - "ICLR 2026"
  - "LLM安全"
  - "越狱检测"
  - "LVLM 安全"
  - "JailNeurons"
  - "神经元定位"
  - "泛化检测"
---

# From "Sure" to "Sorry": Detecting Jailbreak in Large Vision Language Model via JailNeurons

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ofJaimOZBF](https://openreview.net/forum?id=ofJaimOZBF)  
**代码**: 待确认  
**领域**: LLM 安全 / 多模态大模型越狱检测  
**关键词**: 越狱检测, LVLM 安全, JailNeurons, 神经元定位, 泛化检测  

## 一句话总结
JDJN 把 LVLM 越狱检测落到"神经元级"——用因果消融在每层定位出一小撮专门被越狱输入激活的 **JailNeurons**，再跨层聚合它们的激活训练一个轻量 SVM，从而以近乎零误报、超低延迟实现对未见攻击类型的强泛化检测。

## 研究背景与动机
**领域现状**：大型视觉语言模型（LVLM）继承了 LLM 的强能力，也放大了安全风险。视觉模态扩大了攻击面，越狱攻击主要有三类：往图像注入梯度优化的对抗扰动、把恶意文本渲染成图像字符（如 FigStep）、用语义相关图像放大有害性。现有防御要么是训练期对齐（算力/标注昂贵），要么是推理期检测（输入预处理、输出判别、中间表示语义检查）。

**现有痛点**：推理期检测普遍存在两个问题——要么只能识别特定攻击类型，要么需要让模型跑完整次生成（如用裁判 LLM 评估输出）导致延迟高，难以真正部署。更关键的是，已有"安全机制"研究关注的是 **SafeNeurons**（解释对齐模型为何能拒绝普通有害查询），却没人去刻画越狱是"如何绕过"安全机制的。

**核心矛盾**：越狱相关信号确实藏在模型的高维隐状态里（单层线性分类器在同分布数据上几乎 100% 准确），但**没有任何单层能泛化到所有 OOD 攻击**——作者的预备实验显示，任意单层在最差分布上的准确率都跌破 80%。高维激活里塞满了与越狱无关的噪声，单层也不足以覆盖多样攻击的特征。

**本文目标**：提出一个高效、可泛化、非侵入（不改目标模型）的越狱检测器，对未见攻击类型和 OOD 良性数据都保持高 TPR、极低 FPR。

**核心 idea**：**【神经元稀疏假设】** 类比 SafeNeurons 的稀疏性，作者假设只有极少数神经元（JailNeurons，实测占比 <2%）编码越狱相关信号；**【因果定位 + 跨层聚合】** 用"从 Sure 到 Sorry"的因果消融把这撮神经元揪出来，再"从顶到底"采样多层聚合，从噪声中隔离出稳健可迁移的判别特征。

## 方法详解

### 整体框架
JDJN（Jailbreak Detection via JailNeurons）把检测拆成两个子问题——每层如何定位 JailNeurons、如何挑选最有信息量的层——并对应三阶段流水线：① **JailNeuron 定位**：对每层训练一个稀疏 mask，找出"屏蔽后能把模型输出从有害翻转为拒绝"的关键神经元；② **检测器训练**：从顶到底等间隔选若干层，拼接这些层 JailNeurons 的激活，训练一个轻量 SVM；③ **部署**：推理时只前向一遍，按 mask 切片拼接激活喂给 SVM 判别，一旦判为越狱立即触发拒绝、跳过逐 token 生成。

```mermaid
flowchart LR
    A[越狱+良性样本] --> B[逐层训练稀疏 mask<br/>因果消融: Sure→Sorry]
    B --> C[阈值 τ 筛出 JailNeurons]
    C --> D[从顶到底每隔 k 层采样]
    D --> E[拼接多层 JailNeuron 激活]
    E --> F[训练线性 SVM 检测器]
    F --> G[推理: 一次前向→切片→SVM 判别]
```

### 关键设计
**1. 从 Sure 到 Sorry：单层 JailNeuron 的因果定位** —— 这是方法的灵魂。给定一个本会触发有害回复（"Sure, here is..."）的越狱输入，作者要找出"屏蔽哪些神经元能把输出翻转成拒绝（'Sorry, I cannot...'）"的那一小撮神经元，因为它们才是越狱在因果上依赖的部件。具体做法是为第 $i$ 层注册前向 hook，用一个可学习的 mask $m\in[0,1]^d$ 改写该层输出：$h(o_i, m) = (1-m)\odot o_i$。然后求解一个带稀疏正则的优化问题，让被改写后的模型输出朝拒绝词的目标嵌入 $e_s$（如 "Sorry"）靠拢：

$$m^* = \arg\min_{m\in[0,1]^d} \lambda\|m\|_1 + L_{CE}(f_i(m,x), e_s)$$

其中 $\|m\|_1$ 的 L1 正则促使 mask 稀疏（最小干预），$L_{CE}$ 是交叉熵。为保证 $m\in[0,1]^d$ 的约束，把 $m$ 重参数化为 $\mathrm{sig}(\delta)$（sigmoid），最终对无约束参数 $\delta\in\mathbb{R}^d$ 优化。mask 值高于阈值 $\tau$（如 0.4）的神经元即被标定为 JailNeurons。这个"因果翻转"视角是它区别于以往只做表示相似度比对方法的根本——它定位的是**导致**越狱成功的神经元，而非仅仅相关的信号。

**2. 从顶到底：多层选择克服单层不可泛化** —— 既然单层信息不足且无法泛化，JDJN 用跨层聚合来覆盖越狱行为的多样特征。它采用等差采样：对 $l$ 层的模型从第一层起每隔 $k$ 层取一层，共选 $l_j=\lceil l/k\rceil$ 层，以兼顾不同抽象层级的表示。把这些层上 JailNeurons（mask 值 $>\tau$ 对应的隐状态分量）切出来拼接，作为判别特征。这一步直接回应了预备实验的核心发现——"不同攻击类型影响模型不同部位"，所以必须多层取证而非押注单层。

**3. 轻量线性 SVM：抗过拟合的刻意选择** —— 聚合后的 JailNeuron 激活喂给一个线性 SVM 做二分类。作者特意对比了 MLP 和非线性 SVM，发现更复杂的模型并没更好：MLP 在 ID 数据上准确率极高但严重过拟合，OOD 泛化显著低于线性 SVM。配合 JailNeuron 已经把高维噪声压到 <2%，线性分类器既数据高效（几百样本即可）又泛化稳健，这是"特征选得好就不需要复杂分类器"的典型体现。

## 实验关键数据
四个 LVLM（MiniGPT4-7B、LLaVA-v1.5-7B、Qwen2-VL-7B、Janus-pro-7B）× 三类攻击（梯度型 JAMLLM、排版型 FigStep、JailbreakV benchmark）× 多个良性数据集（MM-Bench、MM-Vet、Normal、ScreenSpots、AndroidControl）。训练只用 80% 单一攻击 + 单一良性集，在同分布 ID 与未见攻击/未见良性的 OOD 上分别评估。

### 主实验：检测成功率（TPR@FPR≤0.05，LLaVA / Janus-pro）

| 方法 | JailBreakV | FigStep | JAMLLM (LLaVA) | JailBreakV | FigStep | JAMLLM (Janus) |
|------|-----------|---------|----------------|-----------|---------|----------------|
| **JDJN1**（训练于 JailBreakV） | **0.997** | **1.0** | **0.942** | **0.996** | **1.0** | 0.853 |
| JDJN2（训练于 FigStep） | 0.732 | 1.0 | 0.524 | 0.838 | 1.0 | 0.776 |
| JailGuard | 0.676 | 0.532 | 0.71 | 0.573 | 0.566 | 0.71 |
| GradSafe | 0.862 | 0.742 | 0.534 | 0.844 | 0.728 | 0.454 |
| JailDAM | 0.913 | 0.926 | 0.342 | 0.917 | 0.932 | 0.433 |
| AdaShield | 0.675 | 0.786 | 0.213 | 0.774 | 0.812 | 0.353 |

JDJN1 在 ID 上 TPR >99%，对完全未见的 JAMLLM 仍保持 94.2% TPR，全面碾压 7 个基线；用更多样的 JailBreakV 训练比用 FigStep 泛化明显更好。

### 效率对比（单条 FigStep 处理耗时）

| 方法 | LLaVA | Janus-pro |
|------|-------|-----------|
| **JDJN1** | **1.02s** | **0.26s** |
| JailGuard | 84.27s | 31.25s |
| ECSO | 15.12s | 5.36s |
| CIDER | 5.42s | 3.02s |
| 无防御（完整生成） | 12.08s | 4.29s |

JDJN 只需一次前向、无需生成完整回复，**比无防御的原始 LVLM 还快**——因为检出有害即刻拒绝，绕过了昂贵的逐 token 生成。

### 误报率（FPR，LLaVA，训练攻击固定 JailBreakV）

| 训练良性集 | MM-Vet | MM-Bench | Normal | ScreenSpots | AndroidControl |
|-----------|--------|----------|--------|-------------|----------------|
| **JDJN1（MM-Vet）** | **0.0** | **0.0** | **0.019** | **0.022** | **0.012** |
| JDJN3（MM-Bench） | 0.168 | 0.0 | 0.346 | 0.343 | 0.212 |
| JDJN4（Normal） | 0.285 | 0.21 | 0.0 | 0.198 | 0.272 |

### 关键发现
- **良性训练集的"通用性"决定泛化**：用开放图文集 MM-Vet 训练，FPR 全线 <5% 且多数为 0%；而 MM-Bench（只输出 ABCD）或 Normal（纯文本）会让检测器学到"有没有图/输出格式"这类表面捷径，泛化崩坏。
- **mask 引导至关重要且对 τ 鲁棒**：任意 k 下用 mask（τ=0.3）都超过无 mask 基线（τ=0），τ>0 时性能稳健，LLaVA 最优 τ=0.4、Janus-pro 为 0.2。
- **JailNeurons 极稀疏**：λ≥0.1 时各模型 JailNeurons 占比 <2%；λ=0.1 时六数据集准确率均 >94%，但 λ=0.5 信息丢太多、Normal 上掉到 73%。
- **拒绝词的选择有讲究**：用 "sorry" 定位优于 "unfortunately"（Normal 上 0.956 vs 0.722），因为 "sorry" 作为更常见的拒绝表达编码了更丰富的越狱相关信息。

## 亮点与洞察
- **从"相关"到"因果"的视角升级**：以往内部表示类检测多是比对隐状态相似度（相关性），JDJN 用"屏蔽即翻转输出"的消融把神经元锚定为越狱的因果责任方，定位更精准、可解释性更强。
- **SafeNeurons 的对偶补全**：明确区分"为何能拒绝"（SafeNeurons）与"如何被绕过"（JailNeurons），填补了 LVLM 安全机制研究中长期缺失的一半。
- **效率反超无防御**：把"检出即拒绝、跳过生成"做成了天然加速，安全检测不再是部署负担而是性能红利——这对真实落地极有说服力。
- **稀疏 + 线性的工程美学**：先用稀疏 mask 把信噪比拉满，再用线性 SVM 抗过拟合，少量样本即可训练，非侵入零改模型。

## 局限与展望
- **白盒假设**：方法需要访问模型内部激活与梯度来定位 JailNeurons，对闭源 API 模型不适用。
- **依赖成功越狱样本**：定位过程要拿"本会触发有害回复"的样本做因果翻转，冷启动需要先有可用的越狱样本集。
- **良性训练集敏感**：泛化强弱高度依赖良性数据是否足够通用（MM-Vet 远好于受限的 MM-Bench/Normal），实践中需谨慎选数据，否则易学到表面捷径。
- **规模与模态广度**：实验集中在 7B 量级、三类攻击；面对更大模型、自适应攻击者（知道 JailNeurons 机制后的对抗）以及更多模态组合的鲁棒性仍待验证。

## 相关工作与启发
- **LVLM 越狱检测三流派**：输入预处理（JailGuard、CIDER）、输出分析（ECSO、用裁判 LLM）、内部激活异常（HiddenDetect 用 logit lens 比对拒绝片段）。JDJN 属第三类但首次落到神经元级因果定位。
- **LLM/LVLM 安全机制**：高维表示分析（logit lens、steering vector）与内部结构分析（用 SNIP 定位安全神经元，即 SafeNeurons 系列）。本文是首个通过神经元激活值研究 LVLM 越狱机制并给出检测算法的工作。
- **启发**：稀疏 mask + 因果消融定位"功能神经元"是一个可迁移的范式，可推广到幻觉抑制、有害概念擦除、能力定位等其他可解释性安全任务；"检出即早停"的思路也提示安全模块可以反向优化推理效率。

## 评分
- 新颖性: ⭐⭐⭐⭐ JailNeurons 概念与"Sure→Sorry"因果定位是对 SafeNeurons 研究有意义的对偶补全，神经元级越狱检测在 LVLM 上属首创。
- 实验充分度: ⭐⭐⭐⭐ 4 模型 × 3 攻击 × 5 良性集 + 7 基线 + 多超参消融，TPR/FPR/效率三维度都覆盖；扣分在自适应攻击与更大模型缺席。
- 写作质量: ⭐⭐⭐⭐ 动机递进清晰（单层可分但不可泛化→稀疏假设→跨层聚合），图表与 RQ 组织得当。
- 价值: ⭐⭐⭐⭐ 近零误报 + 反超无防御的速度 + 非侵入，工程落地价值突出，对 LVLM 安全部署有直接参考意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] STAR: Strategy-driven Automatic Jailbreak Red-teaming for Large Language Model](star_strategy-driven_automatic_jailbreak_red-teaming_for_large_language_model.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](../../ACL2026/llm_safety/rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)
- [\[ICLR 2026\] Unlearning Isn't Invisible: Detecting Unlearning Traces in LLMs from Model Outputs](unlearning_isnt_invisible_detecting_unlearning_traces_in_llms_from_model_outputs.md)
- [\[ICLR 2026\] VEAttack: Downstream-Agnostic Vision Encoder Attack Against Large Vision Language Models](veattack_downstream-agnostic_vision_encoder_attack_against_large_vision_language.md)
- [\[ICLR 2026\] TAO-Attack: Toward Advanced Optimization-based Jailbreak Attacks for Large Language Models](tao-attack_toward_advanced_optimization-based_jailbreak_attacks_for_large_langua.md)

</div>

<!-- RELATED:END -->
