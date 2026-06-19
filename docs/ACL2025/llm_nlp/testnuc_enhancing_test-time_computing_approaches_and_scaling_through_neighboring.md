---
title: >-
  [论文解读] TestNUC: Enhancing Test-Time Computing Approaches and Scaling through Neighboring Unlabeled Data Consistency
description: >-
  [ACL 2025][LLM 其他][测试时计算] TestNUC 提出了一种线性扩展的测试时推理增强方法，通过检索测试样本的近邻无标注数据，让 LLM 同时预测测试样本及其邻居，再通过加权多数投票聚合，稳定提升分类准确率。 测试时计算（test-time computing）通过在推理阶段投入更多计算资源来提升 LLM 性…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "测试时计算"
  - "邻域一致性"
  - "无标注数据"
  - "多数投票"
  - "LLM推理增强"
---

# TestNUC: Enhancing Test-Time Computing Approaches and Scaling through Neighboring Unlabeled Data Consistency

**会议**: ACL 2025  
**arXiv**: [2502.19163](https://arxiv.org/abs/2502.19163)  
**代码**: 有 ([https://github.com/HenryPengZou/TestNUC](https://github.com/HenryPengZou/TestNUC))  
**领域**: 其他  
**关键词**: 测试时计算, 邻域一致性, 无标注数据, 多数投票, LLM推理增强

## 一句话总结

TestNUC 提出了一种线性扩展的测试时推理增强方法，通过检索测试样本的近邻无标注数据，让 LLM 同时预测测试样本及其邻居，再通过加权多数投票聚合，稳定提升分类准确率。

## 研究背景与动机

测试时计算（test-time computing）通过在推理阶段投入更多计算资源来提升 LLM 性能，已成为热门方向。现有策略分两类：

**输入级**（如 few-shot ICL）：增加 prompt 中的 token → 计算成本随 token 数**平方增长**

**输出级**（如 self-consistency、best-of-N）：采样多个答案并聚合 → 忽略了现实场景中大量可用的**无标注数据**

核心问题：**如何高效利用无标注数据来增强测试时推理？**

作者注意到一个"嵌入空间中的局部一致性"现象：语义相似的实例很可能共享相同标签。初步分析表明，在 K=20 的最邻近中，即使最差的情况（GoEmotion 150 类）纯度也达到 ~30%，大多数数据集远高于此。如果用多数投票聚合邻域的真实标签，可以得到非常准确且稳定的预测。

## 方法详解

### 整体框架

TestNUC 包含两个步骤：
1. **Neighbor Retrieval**：基于嵌入相似度检索测试样本的 top-K 近邻无标注数据
2. **Collaborative Prediction**：LLM 分别对测试样本和 K 个近邻生成预测，通过设计的聚合策略得出最终答案

### 关键设计

1. **邻域纯度分析（Preliminary Analysis）**

    - 定义邻域纯度 $\phi(\mathcal{N}) = \frac{1}{KN} \sum_{i=1}^N \sum_{j \in \mathcal{N}} \mathbf{1}(y_i = y_j)$
    - 实证发现：近邻samples 具有高标签一致性，多数投票准确率随 K 增加保持稳定
    - 加权投票进一步提高了大 K 下的稳定性

2. **三种聚合策略**

    - **朴素多数投票**：直接取 K 个预测中最频繁的类别
    - **加权多数投票**：以余弦相似度为权重进行投票，降低远距离邻居的噪声影响
    - **过滤加权多数投票**（完整版）：额外利用 LLM 的言语化置信度过滤低质量预测
        - 对每个邻居，LLM 同时输出预测和置信度
        - 仅保留置信度 ≥ 阈值 θ 的预测参与投票

3. **与现有方法的无缝集成**

    - 与 **Self-Consistency** 集成：在每个邻居上也做 self-consistency，再聚合
    - 与 **TopK-ICL** 集成：先用 ICL 增强每个邻居的预测，再聚合
    - 与 **Best-of-N** 集成：在 TestNUC 的聚合结果上再做 best-of-N
    - 所有集成均带来额外性能提升

4. **计算复杂度分析**

    - 嵌入预计算成本为 O(N)（离线完成），检索成本为 O(N)
    - LLM 推理成本为 O(K)，与 self-consistency（O(M)）相当
    - 总体线性扩展，远优于 ICL 的二次扩展

## 实验关键数据

### 主实验——4 个 LLM × 8 个数据集（Table 1 摘要）

| 模型 | 方法 | 平均准确率 |
|------|------|-----------|
| GPT-4o-mini | Standard Prompting | 0.613 |
| GPT-4o-mini | Self-Consistency | 0.625 |
| GPT-4o-mini | TestNUC | 0.660 |
| GPT-4o-mini | TestNUC† (K=50) | **0.676** |
| Llama-3.1-8B | Standard Prompting | 0.572 |
| Llama-3.1-8B | TestNUC† | **0.652** |
| GPT-4o | Standard Prompting | 0.715 |
| GPT-4o | TestNUC† | **0.754** |

### 与现有测试时方法集成（Table 2 摘要）

| 基础方法 | +TestNUC 提升 |
|----------|-------------|
| KNN-ICL | +7.51% |
| KNN-ICL-P | +5.98% |
| Self-Consistency | **+9.56%** |
| Best-of-N | +6.24% |

### 消融实验

| 维度 | 结论 |
|------|------|
| K 值敏感性 | K=10-50 性能稳步提升，对 K 值鲁棒 |
| 嵌入模型 | 不同嵌入模型（SFR、GTE、NV-Embed 等）均有效 |
| 无标注数据量 | 随数据量增加性能单调提升，数据越多效果越好 |
| 聚合策略 | 过滤加权投票 > 加权投票 > 朴素投票 |

### 关键发现

1. **TestNUC 在所有 LLM 和数据集上一致优于基线**，平均提升 4-8 个百分点
2. **与现有方法完美互补**：集成后均带来额外提升，Self-Consistency 集成提升最大（+9.56%）
3. **随无标注数据量线性扩展**：数据越多效果越好，适用于数据丰富的实际场景
4. **对嵌入模型不敏感**：不同大小和来源的嵌入模型均表现良好
5. 在意图检测（BANKING/CLINC）上提升最大，在情感检测（GoEmotion）上提升最小——与邻域纯度分析一致

## 亮点与洞察

1. **思路简洁优雅**："相似样本应有相同标签"的直觉简单但有效，且有理论支撑（邻域纯度分析）
2. **即插即用**：与现有测试时方法正交，可直接叠加使用
3. **无需训练**：利用现成嵌入模型和 LLM，无需任何额外训练
4. **实用性强**：线性扩展 + 对超参数鲁棒 + 跨模型有效 = 容易部署

## 局限与展望

- 需要有一定量的无标注数据（任务相关的），不适用于完全冷启动场景
- 增加了 K 次额外 LLM 调用的成本，当 K 较大时可能影响延迟敏感场景
- 仅在分类任务上验证，对生成任务（如文本摘要、翻译）的适用性未知
- 邻域纯度在细粒度任务（150 类）上下降较快，限制了在极细粒度分类上的效果
- 未探讨更复杂的聚合策略（如学习型聚合）

## 相关工作与启发

- kNN-LM (Khandelwal et al., 2020) 利用邻近样本改善语言模型泛化，思路相近但在生成层面操作
- Self-Consistency (Wang et al., 2023b) 通过输出采样聚合提升推理，TestNUC 从输入空间互补
- 半监督学习中利用无标注数据的思想（如 FixMatch）与 TestNUC 的邻域一致性假设相通

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次在测试时计算框架中系统利用无标注数据的邻域一致性
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 个 LLM × 8 个数据集 × 多种集成方式 × 丰富消融
- **写作质量**: ⭐⭐⭐⭐ — 前期分析（纯度、多数投票准确率）为方法奠定了坚实的动机基础
- **价值**: ⭐⭐⭐⭐ — 简单有效的推理增强方法，实际意义显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation](qg-sms_enhancing_test_item_analysis_via_student_modeling_and_simulation.md)
- [\[ACL 2025\] Growing Through Experience: Scaling Episodic Grounding in Language Models](episodic_grounding_experience.md)
- [\[CVPR 2025\] Test-Time Visual In-Context Tuning](../../CVPR2025/llm_nlp/test-time_visual_in-context_tuning.md)
- [\[ICML 2025\] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](../../ICML2025/llm_nlp/best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)
- [\[ACL 2025\] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](contrastive_prompting_embeddings.md)

</div>

<!-- RELATED:END -->
