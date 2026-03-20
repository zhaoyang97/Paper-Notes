# EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models

**会议**: ACL 2025  
**arXiv**: [2506.00479](https://arxiv.org/abs/2506.00479)  
**代码**: [https://effivlm-bench.github.io/](https://effivlm-bench.github.io/)  
**领域**: LLM效率  
**关键词**: VLM acceleration, token compression, KV cache, parameter compression, benchmark

## 一句话总结
提出 EffiVLM-Bench，首个系统评估大型视觉语言模型（LVLM）训练免加速方法的统一框架，覆盖 17 个 benchmark、3 个前沿模型，引入泛化性和忠诚度等新指标，揭示了 token 压缩与参数压缩在不同场景下的性能-效率权衡。

## 研究背景与动机
1. **领域现状**：LVLM 取得了显著成功，但计算和内存开销巨大，限制了实际部署。训练免加速方法（token 压缩、参数压缩）因无需重训练而受到关注
2. **现有痛点**：
   - 评估停留在过时模型（LLaVA/LLaVA-v1.5），未考虑动态分辨率等新架构
   - Benchmark 局限于通用 VQA，忽视 OCR、长文本生成等更挑战性任务
   - 评估只看绝对性能，忽略泛化性（跨模型/任务）和忠诚度（是否保持原模型行为）
   - 缺乏性能-效率 Pareto 最优的系统分析
3. **核心矛盾**：各加速方法各自为政地评估，缺乏统一框架和全面指标，无法指导实际部署选择
4. **核心idea一句话**：构建统一的评估框架，从性能、泛化性、忠诚度、效率四个维度系统对比 LVLM 加速方法

## 方法详解

### 整体框架
EffiVLM-Bench 评估两类训练免加速方法：(1) Token 压缩 — 包括 token pruning（FastV、VisionZip、PruMerge+）和 KV cache 压缩（StreamingLLM、H2O、SnapKV、PyramidKV、LOOK-M、VL-Cache）；(2) 参数压缩 — 包括权重剪枝（Wanda、SparseGPT）和量化（GPTQ、AWQ、QuIP#）。评估覆盖 17 个 benchmark、3 个模型（LLaVA-OV-7B、Qwen2-VL-7B、InternVL2.5-38B），在 1%/10%/40% 等多种压缩预算下测试。

### 关键设计

1. **四维评估指标体系**:
   - **Performance (OP)**：压缩后与原始模型在各 benchmark 上的性能比值的均方根
   - **Generalization (OG)**：性能比值在不同 benchmark 和模型间的变异系数（越低越好）
   - **Loyalty (OL)**：压缩后模型的预测与原模型预测的一致率（越高越好）
   - **Efficiency (OE)**：实际推理时间的加速比（TTFT + 解码时间）

2. **Token 压缩分类**:
   - **Token pruning**：在前向过程中裁剪冗余视觉 token。分为视觉编码器内裁剪（VisionZip、PruMerge+）和 LLM backbone 内裁剪（FastV）
   - **KV cache 压缩**：利用注意力稀疏性选留更少的 key-value 对，减少内存开销。分为通用 LLM 方法（H2O、SnapKV）和专为 LVLM 设计的方法（VL-Cache、LOOK-M）

3. **参数压缩分类**:
   - **权重剪枝**：SparseGPT、Wanda，在 50% 稀疏率下评估
   - **量化**：GPTQ、AWQ（INT4/INT8）、QuIP#（FP8/FP4/FP2）

## 实验关键数据

### 主实验 — Token 压缩
| 方法类别 | 方法 | Budget | LLaVA-OV-7B OP | Qwen2-VL-7B OP | InternVL2.5-38B OP |
|---------|------|--------|----------------|----------------|---------------------|
| Token Pruning | FastV | 1% | 0.48 | 0.51 | 0.47 |
| Token Pruning | VisionZip | 1% | 0.75 | 0.70 | 0.55 |
| Token Pruning | VisionZip | 40% | 0.93 | 0.93 | 0.93 |
| KV Cache | H2O | 40% | - | - | - (best OG) |
| KV Cache | PyramidKV | 10% | - | - | - (best OG) |

### 参数压缩
| 方法 | 精度 | LLaVA-OV OP | Qwen2-VL OP | InternVL2.5 OP |
|------|------|------------|------------|----------------|
| AWQ | INT8 | 0.99 | 1.00 | 0.98 |
| AWQ | INT4 | 0.91 | 0.98 | 0.90 |
| SparseGPT | 50% | 0.73 | 0.80 | 0.76 |
| Wanda | 50% | 0.56 | 0.56 | 0.63 |

### 消融/关键发现
| 观察 | 结论 |
|------|------|
| Obs 1 | Token 压缩性能高度依赖任务；1% 预算下视觉编码器内裁剪（VisionZip）远优于 LLM 内裁剪（FastV） |
| Obs 2 | KV cache 压缩在泛化性和忠诚度上优于 token pruning |
| Obs 3 | Token pruning 大幅降低 TTFT（3.2x at 1%），KV cache 在长输出任务更优 |
| Obs 4 | Token 压缩在单图/多图/视频任务上展现一致趋势 |
| Obs 5 | 量化是最稳健选择（INT8 几乎无损），剪枝在 50% 时性能损失大 |

### 关键发现
- **1% 预算是分水岭**：大多数方法在 1% 预算下性能急剧下降，尤其在 OCR 和长文本生成任务上
- **视觉编码器内裁剪 > LLM 内裁剪**：在极端压缩下（1%），VisionZip/PruMerge+ 保留 70-75% 性能，FastV 仅保留约 48%
- **KV cache vs Token pruning 各有优势**：Token pruning 对 TTFT 贡献大（适合短回答 VQA），KV cache 对解码加速和长输出更优
- **量化是最安全策略**：INT8 量化几乎无损（OP≈1.0），INT4 仍保留 90%+ 性能

## 亮点与洞察
- **四维评估框架**（性能+泛化性+忠诚度+效率）比单看准确率更全面。特别是 Loyalty 指标，揭示了压缩后模型行为可能发生意外改变的问题，这在安全关键应用中至关重要
- **TTFT vs 解码时间的分解分析**很有洞察力：Token pruning 主要加速 TTFT（prefill），KV cache 主要加速解码。这意味着实际部署应根据任务特点（短回答 vs 长文本）选择不同压缩策略
- **Pareto 最优分析**直接指导部署决策，比仅报告单一指标实用得多

## 局限性 / 可改进方向
- 参数压缩只评估了 LLM backbone，未涉及视觉编码器的压缩
- 未评估 token 压缩与参数压缩的**组合使用**效果
- 模型以 7B-38B 为主，未覆盖更小（1-3B）或更大（70B+）的模型
- Loyalty 指标在开放式生成任务上的定义（exact match）可能过于严格

## 相关工作与启发
- **vs 纯 LLM 压缩评估**：EffiVLM-Bench 专注多模态场景，发现视觉 token 的处理策略与纯文本有本质不同
- **vs LLM-Viewer**：LLM-Viewer 侧重单一量化方法的深度分析，EffiVLM-Bench 横向对比多种压缩类型
- **启发**：在构建效率评估框架时，应同时考虑绝对性能和相对保持率，并区分不同推理阶段的加速效果

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个全面的 LVLM 加速评估框架，引入泛化性和忠诚度指标
- 实验充分度: ⭐⭐⭐⭐⭐ 17 个 benchmark × 3 个模型 × 9 个压缩方法 × 多种预算，极其全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，观察总结有条理
- 价值: ⭐⭐⭐⭐ 对 LVLM 部署优化有重要参考价值
