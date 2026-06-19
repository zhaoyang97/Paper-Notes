---
title: >-
  [论文解读] Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks
description: >-
  [ICML 2025][LLM安全][文本水印] 提出 SIRA（Self-Information Rewrite Attack），利用自信息识别水印嵌入的高熵 token 并进行定向替换，在 7 种主流水印方法上实现近 100% 攻击成功率，成本仅 $0.88/百万 token，且完全黑盒、可迁移至任意 LLM 甚至移动端模型。
tags:
  - "ICML 2025"
  - "LLM安全"
  - "文本水印"
  - "水印鲁棒性"
  - "自信息"
  - "改写攻击"
---

# Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks

**会议**: ICML 2025  
**arXiv**: [2505.05190](https://arxiv.org/abs/2505.05190)  
**代码**: [SIRA](https://github.com/YixinCheng/SIRA)  
**领域**: AI安全  
**关键词**: 文本水印, 水印鲁棒性, 自信息, 改写攻击, LLM安全

## 一句话总结

提出 SIRA（Self-Information Rewrite Attack），利用自信息识别水印嵌入的高熵 token 并进行定向替换，在 7 种主流水印方法上实现近 100% 攻击成功率，成本仅 $0.88/百万 token，且完全黑盒、可迁移至任意 LLM 甚至移动端模型。

## 研究背景与动机

### 1. 文本水印的重要性

LLM（ChatGPT、Claude）生成文本能力日益强大，但也带来虚假信息传播和学术诚信等风险。文本水印通过在生成过程中嵌入不可见统计信号，使检测器能验证文本是否由特定模型生成。

### 2. 现有攻击的不足

- **文本操纵攻击**（删词、插入表情等）：简单粗暴，被过滤器轻易识别，语义还会受损
- **知情攻击**（watermark-stealing）：需大量访问被水印的 LLM 甚至检测器，假设过强
- **通用改写攻击**（DIPPER、GPT paraphraser）：非定向暴力改写，效率低且对新水印算法（如 SIR）已失效

### 3. 关键漏洞的发现

水印算法为保持文本质量，选择在高熵 token（不确定性大的位置）嵌入水印。然而高熵 token 同时具有高自信息。SIRA 利用这一"看似无害但可被利用"的设计缺陷：
- 用任意 LLM 计算每个 token 的自信息
- 自信息高的 token 很可能是水印 green list token
- 遮蔽这些 token 后，将非定向改写转化为定向填空任务

### 4. 核心优势

完全黑盒（不访问水印算法、密钥、检测器），可迁移到任何 LLM，甚至 3B 移动端模型即可执行。

## 方法详解

### 整体框架：两步流程

**Step 1：生成遮蔽模板**
- 对水印文本 $y_w$ 用任意 LLM 计算每个 token $t_k$ 的自信息 $I(t_k) = -\log P(t_k | t_{<k})$
- 设定阈值，将自信息超过阈值的 token 遮蔽为占位符 → 得到遮蔽文本
- 同时让 LLM 对原文做通用改写 → 得到参考文本

**Step 2：定向填空**
- 将遮蔽文本和参考文本一起输入 LLM
- 要求 LLM 补全遮蔽位置，同时保持参考文本的信息完整性
- 输出的文本 $y_p$ 在遮蔽位置被替换为非水印 token，水印被有效移除

### 关键设计

#### 1. 自信息作为水印定位信号

- **功能**：无需知道水印算法就能精准定位可能嵌入水印的 token
- **核心思路**：水印嵌入在高熵位置 → 高熵 token 对应高自信息 → 自信息可由任意 LLM 计算
- **设计动机**：将非定向改写（LLM 随机决定修改哪些词）转化为定向填空（精准替换 green token）

#### 2. 两步改写策略

- **为什么分两步**：单步改写可能在保持语义的同时保留部分水印 token。两步策略先识别后替换，确保高覆盖率
- **参考文本的作用**：为填空提供语义约束，避免补全后语义偏移

#### 3. 可迁移性

- 攻击模型可以是任何 LLM（GPT-4、Llama-3、甚至 3B 模型）
- 不需要与被水印 LLM 相同——因为自信息计算是通用的语言建模能力

## 实验关键数据

### 主实验：攻击成功率

| 水印方法 | 类别 | DIPPER 攻击 | GPT 改写 | **SIRA** |
|----------|------|------------|----------|---------|
| KGW | KGW家族 | 62.3% | 71.5% | **99.2%** |
| Unigram | KGW家族 | 58.7% | 65.2% | **98.8%** |
| EXP | Christ家族 | 45.2% | 52.1% | **99.5%** |
| SIR | KGW家族(新) | 23.4% | 31.8% | **97.6%** |
| EWD | KGW家族 | 55.1% | 63.4% | **99.1%** |
| DIP | Christ家族 | 41.3% | 48.9% | **98.3%** |
| UW | KGW家族 | 51.8% | 59.7% | **99.0%** |

- SIRA 在所有 7 种方法上接近 100%，尤其对 SIR（DIPPER 仅 23.4%）提升巨大
- 成本仅 $0.88/百万 token，远低于 DIPPER 的硬件需求

### 消融与分析

| 配置 | 攻击成功率 | 语义保持 | 说明 |
|------|-----------|---------|------|
| SIRA 完整 | ~99% | 高 | 定向遮蔽 + 参考填空 |
| 无自信息遮蔽（纯改写） | ~55% | 中 | 退化为非定向攻击 |
| 无参考文本（只填空） | ~92% | 中低 | 缺少语义约束 |
| 3B 移动端模型执行 | ~95% | 中高 | 证明可迁移至小模型 |
| 不同阈值 | 阈值↑成功率略↓ | 阈值↑保持↑ | 权衡精度与保守性 |

### 关键发现

- 自信息遮蔽是攻击成功的核心驱动力（消融从 ~55% 提升到 ~99%）
- 参考文本主要保障语义质量而非攻击成功率
- 即使用 3B 小模型，攻击效果仍 >95%，说明漏洞不依赖攻击模型能力
- 对采用动态密钥的水印（SIR）同样有效，因为自信息定位不依赖密钥知识

## 亮点与洞察

- **揭示根本性漏洞**：水印必须嵌入高熵 token 以保质量，但这一特征恰好是最易被攻击的信号——这是水印方案的内在矛盾
- **攻击范式的升级**：从非定向暴力改写升级到定向填空，是方法论的质变
- **极低门槛**：$0.88/百万 token + 3B 小模型即可发动，意味着任何人都能破解当前水印
- **对未来水印设计的警示**：水印算法不能继续将高熵位置作为唯一嵌入策略

## 局限与展望

- 攻击假设攻击者能获取水印文本的完整文本——在流式输出场景需要适配
- 自信息阈值需要人工设定，不同领域文本可能需不同阈值
- 目前只测试了解码阶段水印，对编码阶段水印或 semantic watermark 的效果待验证
- 防御对策更紧迫：如何设计不依赖高熵集中嵌入的水印方案？

## 相关工作与启发

- **vs DIPPER (Krishna et al. 2024)**：依赖特定微调模型，非定向，对 SIR 等新水印已失效
- **vs GPT Paraphraser**：通用改写，效率低成功率低
- **vs Watermark-stealing**：需海量访问被水印LLM，假设过强
- **本文位置**：首个定向黑盒改写攻击，兼具低成本、高成功率、强可迁移性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次利用自信息定位水印token，攻击范式质变
- 实验充分度: ⭐⭐⭐⭐⭐ 7种水印方法全面覆盖，消融严谨
- 写作质量: ⭐⭐⭐⭐⭐ 威胁模型清晰、技术路线直观
- 价值: ⭐⭐⭐⭐⭐ 对水印研究社区有重大警示价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs](../../NeurIPS2025/llm_safety/bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)
- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](../../NeurIPS2025/llm_safety/toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[ICML 2025\] X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP](x-transfer_attacks_towards_super_transferable_adversarial_attacks_on_clip.md)
- [\[ICML 2025\] Robust Multi-bit Text Watermark with LLM-based Paraphrasers](robust_multi-bit_text_watermark_with_llm-based_paraphrasers.md)
- [\[ACL 2025\] Mamba Knockout for Unraveling Factual Information Flow](../../ACL2025/llm_safety/mamba_knockout_for_unraveling_factual_information_flow.md)

</div>

<!-- RELATED:END -->
