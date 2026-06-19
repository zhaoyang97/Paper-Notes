---
title: >-
  [论文解读] On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection
description: >-
  [NeurIPS 2025 Spotlight][LLM安全][文本水印检测] 系统性地评估了八种经典拟合优度（GoF）检验在 LLM 文本水印检测中的效果，发现 GoF 检验在检测功效和鲁棒性上均显著优于现有基线方法。 LLM 生成的文本引发了内容真实性和版权问题，文本水印通过在生成文本中嵌入可检测的统计信号来验证内容来源…
tags:
  - "NeurIPS 2025 Spotlight"
  - "LLM安全"
  - "文本水印检测"
  - "拟合优度检验"
  - "LLM水印"
  - "统计检验"
  - "鲁棒性"
---

# On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.03944](https://arxiv.org/abs/2510.03944)  
**代码**: [GitHub](https://github.com/hwq0726/GoF-for-Watermark-Detection)  
**领域**: AI Safety / LLM Watermarking  
**关键词**: 文本水印检测, 拟合优度检验, LLM水印, 统计检验, 鲁棒性

## 一句话总结

系统性地评估了八种经典拟合优度（GoF）检验在 LLM 文本水印检测中的效果，发现 GoF 检验在检测功效和鲁棒性上均显著优于现有基线方法。

## 研究背景与动机

LLM 生成的文本引发了内容真实性和版权问题，文本水印通过在生成文本中嵌入可检测的统计信号来验证内容来源。水印检测的核心是假设检验问题：在 $H_0$ 下，枢轴统计量 $Y_t = Y(w_t, \zeta_t)$ 独立同分布于已知分布 $\mu_0$；在 $H_1$ 下偏离该分布。

这天然是一个**拟合优度检验**（GoF）问题：判断 i.i.d. 样本是否来自给定分布。然而，现有文献主要聚焦于设计新水印方案，而非提升检测效能。Li et al. 提出了截断 $\phi$-散度 GoF 检验，但仅分析了 Gumbel-max 水印，且依赖渐近假设。

核心问题：经典 GoF 检验在现代水印检测中的表现如何？

## 方法详解

### 整体框架

将水印检测统一为 GoF 检验框架。检测流程：（1）从文本序列计算枢轴统计量 $Y_1, ..., Y_n$；（2）计算 p-值 $p_t = 1 - F_0(Y_t)$；（3）用 GoF 检验统计量评估偏离程度；（4）根据临界值判定是否拒绝 $H_0$。

### 关键设计

1. **八种 GoF 检验的统一评估**: 包括 Kolmogorov-Smirnov (Kol)、Anderson-Darling (And)、Cramér-von Mises (Cra)、Kuiper (Kui)、Watson (Wat)、Neyman 光滑检验 (Ney)、Chi-squared (Chi) 和截断散度检验 (Phi)。每种检验用不同方式度量经验 CDF 与零假设 CDF 的偏离。

2. **三种水印方案的适配**: Gumbel-max（$\mu_0 = U(0,1)$）、逆变换水印（$\mu_0(Y \leq r) = r^2$）、Google SynthID（$\mu_0$ 为 Irwin-Hall 分布）。绿红列表水印被排除，因为其二值枢轴统计量使 GoF 退化为原始检测规则。

3. **低温优势分析**: 低温下水印信号减弱但文本重复增多，引入结构化模式使经验 CDF 偏离零假设 CDF，GoF 检验能独特地利用这一效应——这是现有方法未开发的优势。

### 损失函数 / 训练策略

本文为评估性工作，无需训练。关键技术细节：
- 第一类错误控制在 $\alpha = 0.01$，通过理论分布或 Monte Carlo 模拟调整临界值
- 大多数 GoF 检验的零分布无闭式解，使用大样本渐近近似
- 所有 GoF 检验对枢轴统计量的排列不变，检测结果不受顺序影响

## 实验关键数据

### 主实验

| 水印方案 | 温度 | 长度 | Baseline | Chi | And | Kol | Phi |
|----------|------|------|----------|-----|-----|-----|-----|
| Gumbel-max | T=0.3 | n=400 | 15.1% | **2.9%** | 4.9% | 4.7% | 5.7% |
| Gumbel-max | T=0.7 | n=200 | 0.6% | **0.3%** | 0.5% | 0.6% | 0.3% |
| Inverse-tran | T=0.3 | n=400 | 27.1% | - | 9.3% | 7.4% | 12.1% |

（表中数据为 Type II 错误率×100，越低越好）

### 消融实验

| 配置 | 指标 | 说明 |
|------|------|------|
| 不同温度 T∈{0.1,0.3,0.7,1.0} | Type II 错误率 | GoF 在所有温度下优于 baseline |
| 不同文本长度 n | 检测功效 | 长文本优势更明显 |
| 删除编辑 r=0.1,0.2 | 鲁棒性 | GoF 保持高检测功效 |
| 同义词替换 r=0.1,0.2 | 鲁棒性 | GoF 表现稳健 |
| 信息丰富编辑 r=0.3,0.5 | 鲁棒性 | GoF 在强攻击下仍有优势 |

### 关键发现

- GoF 检验在几乎所有配置下都优于 baseline 检测方法
- Chi-squared 检验在多个场景中表现最优，But And 和 Kol 也很有竞争力
- 低温（T=0.1）下 GoF 的优势最为显著，因为能利用文本重复模式
- Type I 错误在所有 GoF 检验中都接近目标水平 0.01，控制良好
- 三种 LLM（OPT-1.3B, OPT-13B, Llama 3.1-8B）上结果一致

## 亮点与洞察

- GoF 检验是"简单但强大且被低估的工具"——这一结论具有很强的实践指导意义
- 低温下文本重复给 GoF 带来独特优势的发现很有趣，解释了为何在不同温度下都能保持强检测力
- 统一框架使得不同水印方案的检测方法可以标准化
- 排除绿红列表水印的分析严谨，说明 GoF 在该场景下退化

## 局限与展望

- 实验主要在开源 LLM 上进行，未测试闭源商业模型（GPT-4 等）
- 仅考虑了文本完成和长文QA两种任务
- GoF 检验的最优选择依赖于具体水印方案和场景，缺乏自动选择指南
- 未分析 GoF 检验的计算效率与延迟
- 理论方面缺乏渐近以外的有限样本分析

## 相关工作与启发

- Li et al. 的截断散度 GoF 检验是直接前驱工作，但本文大幅扩展了评估范围
- 水印检测与统计检验之间的联系为两个领域的交叉提供了桥梁
- 对未来水印设计的启示：应结合 GoF 检验的特性来优化水印方案的可检测性
- 信息丰富编辑（已知密钥的攻击者）场景对安全性评估很有参考价值

## 评分

- **新颖性**: ⭐⭐⭐ 主要贡献在于系统评估而非方法创新，但发现有价值
- **实验充分度**: ⭐⭐⭐⭐⭐ 8种GoF检验 × 3种水印 × 3种LLM × 4种温度 × 多种编辑，非常全面
- **写作质量**: ⭐⭐⭐⭐ 条理清晰，统计背景解释充分
- **价值**: ⭐⭐⭐⭐ 为水印检测提供了即插即用的检测工具箱，实用性强

## 补充细节

- 绿红列表水印被排除的原因：枢轴统计量是二值的（是否为绿色token），GoF 退化为计数检验
- 对 Neyman 检验使用 k=3 阶 Legendre 正交多项式
- 信息丰富编辑模拟了知道密钥的攻击者能选择性修改高信号 token 的最坏情况
- 所有 GoF 检验都是排列不变的，与 token 顺序无关，这是核心优势

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ORBIT -- Open Recommendation Benchmark for Reproducible Research with Hidden Tests](orbit_--_open_recommendation_benchmark_for_reproducible_research_with_hidden_tes.md)
- [\[ACL 2026\] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](../../ACL2026/llm_safety/rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)
- [\[ICML 2025\] Empirical Privacy Variance](../../ICML2025/llm_safety/empirical_privacy_variance.md)
- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[ICML 2025\] Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective](../../ICML2025/llm_safety/unlocking_the_power_of_rehearsal_in_continual_learning_a_theoretical_perspective.md)

</div>

<!-- RELATED:END -->
