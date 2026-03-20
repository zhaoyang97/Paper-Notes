---
title: "EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models"
authors: "Zekun Wang, Minghua Ma, Zexin Wang, Rongchuan Mu, Liping Shan, Ming Liu, Bing Qin"
venue: "ACL 2025"
arxiv_id: "2506.00479"
date: 2025-06-01
tags: [benchmark, LVLM, efficiency, token-compression, parameter-compression, KV-cache]
---

# EffiVLM-Bench: 大型视觉语言模型免训练加速的综合评测基准

## 一句话总结

提出 EffiVLM-Bench，一个统一评估框架，系统性地评估大型视觉语言模型(LVLM)的免训练加速方法，涵盖 token 压缩和参数压缩两大类，从性能、泛化性、忠实度和效率四个维度进行全面对比分析。

## 研究动机

现有 LVLM 加速方法的评估存在三个关键不足：
1. **模型架构过时**：评估常停留在 LLaVA/LLaVA-v1.5 等旧模型，未考虑具有动态分辨率处理机制的最新 LVLM
2. **基准有限**：通常仅使用通用 VQA 任务，忽略了 OCR、长文本生成等更具挑战性的任务
3. **指标单一**：仅关注绝对性能，忽视了泛化性和忠实度等关键维度，也缺乏对性能-效率 Pareto 最优权衡的系统探索

## 方法详解

### 评估框架设计

EffiVLM-Bench 定义了四个核心维度的评估指标：

- **整体性能 (OP)**：压缩模型在各基准上评估指标与原始模型的比值，取均方根平均，衡量压缩后的绝对表现
- **泛化性 (OG)**：跨基准和模型的性能变异系数，值越低表示泛化越好
- **忠实度 (OL)**：压缩模型与原始模型预测的一致程度，衡量压缩是否引入新偏差
- **效率 (OE)**：基于实际推理时间的加速比，而非 FLOPs 等理论指标

### 评估覆盖范围

- **17 个基准任务**：涵盖文档理解(DocVQA)、图表解读(ChartQA)、OCR(OCRBench)、通用 VQA(GQA)、数学推理(MathVista)等，覆盖单图、多图和视频场景
- **3 个前沿模型**：LLaVA-OneVision-7B、Qwen2-VL-7B、InternVL2.5-38B
- **两大类方法**：
  - Token 压缩：token 剪枝(FastV, VisionZip, PruMerge+)和 KV cache 压缩(StreamingLLM, H2O, SnapKV, PyramidKV, LOOK-M, VL-Cache)
  - 参数压缩：权重剪枝(EcoFLAP, Wanda, SparseGPT)和量化(AWQ, GPTQ)

## 核心发现

### 发现1：Token 压缩性能高度依赖任务和模型

- 高预算时大多方法稳定，但在 1% 极低预算下性能急剧下降，尤其在需要细粒度视觉信息的任务(OCRBench)或长输出任务(LLaVA-Wilder)上
- 在视觉编码器中剪枝(VisionZip, PruMerge+)一致优于在 LLM 骨干网络中剪枝(FastV)：1% 预算下 FastV 仅保留 48% 性能，VisionZip 保留 75%

### 发现2：KV cache 压缩的泛化性和忠实度优于 token 剪枝

- H2O 和 PyramidKV 在综合指标上领先
- KV cache 方法在忠实度上显著优于 token 剪枝(40% 预算下 H2O 忠实度 94.57% vs FastV 80.57%)
- 当泛化性和忠实度为关键需求时，应优先选择 KV cache 压缩

### 发现3：根据任务特性选择压缩策略

- **TTFT(首 token 时间)**：token 剪枝在预填充阶段移除视觉 token，大幅降低 TTFT(1% 预算下可达 3.2× 加速)；KV cache 方法因需重计算注意力权重，TTFT 加速有限
- **解码延迟**：两类方法在相同预算下加速比相近，但 KV cache 方法在低预算长输出任务上更优
- **实践建议**：短回答任务(如 VQA)用 token 剪枝，长输出任务用 KV cache 压缩

### 发现4：参数压缩对性能保持效果更好

- 即使在 50% 或 2:4 稀疏度下性能仍相对稳定
- 量化(AWQ)比剪枝保留更高性能
- 两类压缩方法正交，可有效组合使用

## 深入分析

### 层自适应机制的再审视

- PyramidKV 在低预算下反而不如 SnapKV，因为层自适应策略将过多预算分配给早期层(第 0 层分配了近 7 倍平均预算)，导致后续层"饥饿"
- 提出混合分配策略：80% 预算均匀分配 + 20% 自适应分配，效果最优

### 头自适应机制的价值

- 允许同层不同注意力头选择不同 token 能显著提升性能
- 不同头捕获不同信息模式，头自适应选择在紧预算下更好地保留关键信息

### 注意力汇聚 Token 的重要性

- 视觉模态同样存在注意力汇聚(attention sink)现象
- FastV 使用文本引导的指标可能遗漏关键的视觉汇聚 token
- 强制 FastV 优先选择 top-10% 关键视觉 token 后性能显著提升(ChartQA 10% 预算：31.04→45.56)

### 跨模态合并的陷阱

- LOOK-M 在低预算下性能骤降，因为其将被淘汰的视觉 token 合并到文本 token 中，破坏了关键文本特征
- 修改为模态内合并后性能一致提升(1% 预算 DocVQA：38→44)

## 个人评价

- 工作非常扎实，覆盖面广(17 个基准、3 个模型、9 种方法)，定义了四个清晰的评估维度
- 对层自适应、头自适应、注意力汇聚、跨模态合并等机制的深入分析具有实践指导价值
- 发现3中根据 TTFT vs 解码延迟选择压缩策略的建议非常实用
- 缺少对最新高效架构(如 MiniCPM-V、Phi-3-Vision)的覆盖

## 局限性

1. 仅覆盖部分代表性 LVLM 模型和任务，更多架构和专业领域待探索
2. 仅考虑免训练方法，纳入训练方法可提供更深入洞见
3. 极低预算下的分析可能存在其他未发掘的影响因素

## 相关资源

- 项目主页：https://effivlm-bench.github.io/
- 代码和评测配方已开源
