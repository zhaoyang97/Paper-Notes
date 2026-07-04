---
title: >-
  [论文解读] Domain-Adapted Granger Causality for Real-Time Cross-Slice Attack Attribution in 6G Networks
description: >-
  [NEURIPS2025][因果推理][Granger causality] 提出一种面向6G网络切片的域适应Granger因果框架，将增强型Granger因果检验与网络资源争用建模相结合，实现实时跨切片攻击归因，在1100个攻击场景上达到89.2%准确率和87ms响应时间，显著超越现有统计、深度学习和因果发现方法。
tags:
  - "NEURIPS2025"
  - "因果推理"
  - "Granger causality"
  - "6G network slicing"
  - "cross-slice attack attribution"
  - "resource contention"
  - "real-time security"
---

# Domain-Adapted Granger Causality for Real-Time Cross-Slice Attack Attribution in 6G Networks

**会议**: NEURIPS2025  
**arXiv**: [2510.05165](https://arxiv.org/abs/2510.05165)  
**代码**: 待确认（作者声明将以 Apache 2.0 开源）  
**领域**: 因果推理  
**关键词**: Granger causality, 6G network slicing, cross-slice attack attribution, resource contention, real-time security  

## 一句话总结

提出一种面向6G网络切片的域适应Granger因果框架，将增强型Granger因果检验与网络资源争用建模相结合，实现实时跨切片攻击归因，在1100个攻击场景上达到89.2%准确率和87ms响应时间，显著超越现有统计、深度学习和因果发现方法。

## 研究背景与动机

**核心问题**：6G网络通过网络切片（network slicing）将共享物理基础设施划分为多个逻辑网络服务不同业务（eMBB/URLLC/mMTC），但资源共享使得攻击可以跨切片传播，溯源归因（attribution）极为困难

**现有方案局限**：当前方法存在三大不足——(a) 相关性方法（如Pearson相关）产生高假阳性率（30.6%），无法区分因果与虚假相关；(b) 深度学习方法（如Transformer-XL）虽准确率较高但响应时间超过200ms，不满足实时需求；(c) 纯统计因果方法（如VAR-Granger）忽略了网络资源争用这一领域特有的因果路径

**Granger因果的机遇与限制**：经典Granger因果基于时间序列预测改善来判断因果，是时间因果推断的经典框架，但直接应用于6G多切片场景时，共享资源（CPU/内存/网络）是重要的混淆因子，不加控制会产生大量虚假因果关系

**实时性约束**：6G安全编排要求亚100ms响应时间，排除了计算复杂度过高的方法（PC算法156ms，Transformer-XL 234ms）

**可解释性需求**：自动安全编排系统需要可解释的因果解释来指导响应策略，黑盒深度学习方法难以提供

**工业场景迫切性**：IoT/工业自动化场景中，跨切片攻击可导致真实的生产安全威胁（如制造控制系统延迟超限触发紧急停机）

## 方法详解

### 整体框架

Domain-Adapted Granger Causality 框架接收 N 个切片的安全遥测流 {x_t^(i)} 和资源分配数据 A(t)，通过三个核心组件协同工作：(1) 增强型Granger因果检验消除资源混淆；(2) 资源争用建模捕获领域特有因果路径；(3) 集成因果强度评分+Viterbi算法提取最优攻击路径。整体算法复杂度 O(N²·W·(p+q+K) + N³·logN)，在典型部署（N≤50, W=300）下可保证亚100ms执行。

### 模块一：增强型 Granger 因果检验（Enhanced Granger Causality）

- **功能**：对每对切片 (s_i, s_j) 检测 s_i 的遥测时间序列是否对 s_j 有统计意义上的因果预测能力，同时控制共享资源的混淆效应
- **核心思路**：在标准Granger回归中引入共享资源控制项 Z_t = [Z_{1,t}, ..., Z_{K,t}]ᵀ。无限制模型：Y_t = Σα_i·Y_{t-i} + Σβ_j·X_{t-j} + Σγ_k·Z_{k,t} + ε_t；限制模型（H_0: β_j=0）：Y_t = Σα_i·Y_{t-i} + Σγ_k·Z_{k,t} + η_t。增强F统计量为 F = ((RSS_R - RSS_U)/q) / (RSS_U/(T-p-q-K-1))，在 H_0 下服从 F(q, T-p-q-K-1) 分布
- **设计动机**：共享资源是6G场景中最主要的混淆源——两个切片可能仅因共享同一CPU而在遥测上呈现相关性。资源条件化项 γ_k·Z_{k,t} 显式阻断了 X ← R → Y 的混淆路径。与标准Granger相比，资源条件化使准确率提升8.2个百分点（74.1%→82.3%）

### 模块二：资源争用建模（Resource Contention Modeling）

- **功能**：捕获纯统计方法遗漏的、通过共享资源机制介导的因果路径，建模切片间的资源争用强度
- **核心思路**：争用强度定义为 ρ_{ij}(t) = Σ_k w_k · A_{ik}(t) · A_{jk}(t) · σ(U_{k,t} - τ_k)，其中 A_{ik}(t) 是归一化资源分配比例，U_{k,t} 是资源利用率，w_k 是可学习的关键性权重，τ_k 是争用阈值，σ(·) 是 sigmoid 函数
- **设计动机**：乘积项 A_{ik}·A_{jk} 作为逻辑AND门——只有两个切片同时使用同一资源才产生争用；sigmoid 项实现阈值效应——资源利用率超过阈值后争用才显著；线性加权求和让模型自动学习不同资源（CPU/内存/网络）的相对关键程度。实验学到的权重（CPU 0.45, 内存 0.31, 网络 0.24）符合直觉。该模块额外贡献4.7个百分点准确率提升（82.3%→87.0%）

### 模块三：集成因果强度与攻击路径提取

- **功能**：融合统计因果证据和领域因果证据为统一评分，并从因果图中提取最优攻击路径
- **核心思路**：集成因果强度 Γ_{ij}(t) = ω_1·φ(F_{ij}(t)) + ω_2·ρ_{ij}(t)，其中 φ 是 min-max 归一化，{ω_1, ω_2} 通过MLE学习（最优值 ω_1=0.67, ω_2=0.33）。通过Benjamini-Hochberg校正控制FDR（阈值0.05），满足阈值 τ_{causal} 的边构成因果图 G，Viterbi算法求解最优路径 C* = argmax Π Γ_{ij}(T)
- **设计动机**：线性融合在实时系统中算法稳定性和可解释性优于非线性融合；BH校正控制了 N(N-1) 次成对检验的多重比较问题。敏感性分析显示 ω_1∈[0.55, 0.80] 范围内准确率波动仅±1.5个百分点，证明融合不敏感于参数微调

### 损失函数与参数学习

- 参数 θ = {w_1,...,w_K, τ_1,...,τ_K, ω_1, ω_2} 通过正则化对数似然最大化学习：L(θ) = Σ_m log P(C^(m) | X^(m), A^(m), θ) - λ‖θ‖₂²
- 超参数通过5折交叉验证选择（最优 λ*=10⁻³）
- 理论保证：Theorem 1 证明增强F统计量在 H_0 下的渐近 F 分布；Theorem 2 证明在DAG假设下以概率 ≥ 1-N(N-1)α 唯一识别因果关系

## 实验关键数据

### 表1：主实验性能对比（N=1100 攻击场景，5折交叉验证）

| 方法 | 准确率(%) | 精确率(%) | 召回率(%) | FDR(%) | 响应时间(ms) |
|------|----------|----------|----------|--------|-------------|
| Pearson Correlation | 72.9±1.8 | 69.4±2.1 | 76.2±1.9 | 30.6±2.1 | 21±3 |
| Transfer Entropy | 78.4±1.5 | 75.8±1.7 | 81.3±1.6 | 24.2±1.7 | 58±7 |
| VAR-Granger | 74.1±1.7 | 71.2±1.9 | 77.6±1.8 | 28.8±1.9 | 43±5 |
| PC Algorithm | 76.2±1.6 | 73.5±1.8 | 79.4±1.7 | 26.5±1.8 | 156±20 |
| GraphSAGE | 76.8±1.6 | 74.3±1.8 | 79.9±1.7 | 25.7±1.8 | 142±18 |
| LSTM-Attention | 79.1±1.4 | 76.7±1.6 | 82.2±1.5 | 23.3±1.6 | 167±22 |
| Transformer-XL | 81.3±1.3 | 78.9±1.5 | 84.1±1.4 | 21.1±1.5 | 234±31 |
| **Ours** | **89.2±0.9** | **87.6±1.1** | **91.1±1.0** | **12.4±1.1** | **87±9** |

### 表2：扩展指标对比

| 方法 | AUC-ROC | AUC-PR | F1 | MCC | 内存(MB) |
|------|---------|--------|-----|------|---------|
| Correlation | 0.742±0.018 | 0.689±0.021 | 0.726±0.019 | 0.461±0.024 | 12±2 |
| Transfer Entropy | 0.798±0.015 | 0.751±0.017 | 0.784±0.016 | 0.572±0.019 | 28±4 |
| Transformer-XL | 0.827±0.013 | 0.789±0.015 | 0.815±0.014 | 0.634±0.017 | 756±48 |
| **Ours** | **0.921±0.009** | **0.876±0.011** | **0.892±0.010** | **0.785±0.012** | **67±8** |

### 关键发现

1. 本方法比最强基线 Transformer-XL 高出7.9个百分点准确率（p<0.001, Cohen's d>1.5），同时响应时间快2.7倍（87ms vs 234ms），内存仅为其1/11
2. 消融实验量化了各组件贡献：资源条件化 +8.2pp、资源争用建模 +4.7pp、集成学习 +2.2pp，均达到 p<0.001 显著水平
3. 框架在60%部分可观测性下仍保持84.3%准确率，展现出优雅的性能退化特性
4. 工业IoT攻击案例中，完整重建5跳攻击链（96.3%准确率、零假阳性、73ms响应），传统相关性方法产生11个误报

## 亮点与洞察

1. **统计因果与领域知识的有机融合**：不是简单叠加而是通过MLE学习最优融合权重，且理论证明了各组件的必要性（消融实验定量支持）
2. **实时与准确的罕见兼得**：87ms 的响应时间满足6G安全编排的严格实时约束，且准确率反超重量级的 Transformer-XL
3. **可解释性输出**：每条因果边都有F统计量p值和资源争用分数的双重解释，适合自动化安全决策
4. **资源争用建模的巧妙设计**：乘积-sigmoid-加权的三层结构既紧凑又富有物理直觉

## 局限性

1. 弱平稳性假设在快速非线性攻击演化期间可能被违反——虽然资源条件化部分缓解，但根本性的非平稳场景（如攻击模式突变）仍可能失效
2. 需要约2秒遥测数据才能可靠归因，对超快速攻击可能不足
3. O(N³logN) 复杂度在超大规模部署（N>50）时需要近似算法
4. 资源争用模型采用乘积-加权的线性结构，对更复杂的非线性干扰模式可能建模不充分
5. 实验在单一生产级测试床上进行，缺乏跨运营商/跨地域的泛化性验证

## 相关工作与启发

- **经典Granger因果 [Granger, 1969; Shojaie & Fox, 2024]**：本文的核心方法论基础，增强点在于资源条件化和领域融合
- **Transfer Entropy [Schreiber, 2000]**：信息论视角的时间因果方法，本实验中78.4%准确率，不如本方法的资源条件化策略
- **图神经网络方法 [GraphSAGE, Hamilton et al., 2017]**：学习切片关系的空间结构，但缺乏时间因果判断和资源争用建模
- **IoT安全中的Granger因果 [Begum et al., 2025; Lv et al., 2024]**：在IoT场景初步应用Granger因果，但未针对6G多切片的资源争用特点做领域适配
- **因果发现方法 [PC, DirectLiNGAM]**：基于条件独立性检验的结构学习，计算开销大且不适合实时场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 将Granger因果与网络资源争用建模创新性地结合，在6G安全领域开辟了新方向
- 实验充分度: ⭐⭐⭐⭐ 1100场景、多基线对比、消融分析、敏感性分析、工业案例研究，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，算法伪代码完整，理论证明规范，图表丰富
- 价值: ⭐⭐⭐⭐ 提供了6G网络安全归因的实用框架，理论保证+实时性能使其具有部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](revealing_multimodal_causality_with_large_language_models.md)
- [\[NeurIPS 2025\] Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](causality-induced_positional_encoding_for_transformer-based_representation_learn.md)
- [\[NeurIPS 2025\] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)
- [\[ICML 2025\] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](../../ICML2025/causal_inference/learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [\[ICML 2025\] E-LDA: Toward Interpretable LDA Topic Models with Strong Guarantees in Logarithmic Parallel Time](../../ICML2025/causal_inference/e-lda_toward_interpretable_lda_topic_models_with_strong_guarantees_in_logarithmi.md)

</div>

<!-- RELATED:END -->
