---
title: >-
  [论文解读] Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration
description: >-
  [NeurIPS 2025][可解释性][扰动解释方法] 揭示了不确定性校准（模型置信度与实际准确率的对齐）与扰动式可解释性方法质量之间的根本联系，证明模型在扰动输入下的误校准直接损害全局和局部解释质量，并提出 ReCalX 通过扰动级别自适应温度缩放显著改善解释的鲁棒性和保真度。 扰动式解释方法（如 SHAP、LIME）是…
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "扰动解释方法"
  - "不确定性校准"
  - "ReCalX"
  - "Shapley值"
  - "LIME"
  - "温度缩放"
---

# Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration

**会议**: NeurIPS 2025  
**arXiv**: [2511.10439](https://arxiv.org/abs/2511.10439)  
**代码**: [GitHub](https://github.com/thomdeck/recalx)  
**领域**: 可解释性  
**关键词**: 可解释性, 扰动解释方法, 不确定性校准, ReCalX, Shapley值, LIME, 温度缩放

## 一句话总结

揭示了不确定性校准（模型置信度与实际准确率的对齐）与扰动式可解释性方法质量之间的根本联系，证明模型在扰动输入下的误校准直接损害全局和局部解释质量，并提出 ReCalX 通过扰动级别自适应温度缩放显著改善解释的鲁棒性和保真度。

## 研究背景与动机

扰动式解释方法（如 SHAP、LIME）是实践中最常用的模型可解释性技术，其原理是系统性地修改输入特征，观察模型输出变化来量化特征重要性。然而存在一个根本性问题：

**分布外问题**：扰动产生的输入与训练分布差异显著，模型在这些"假"输入上的预测往往不可靠

**误导性概率**：模型对扰动样本可能输出高置信但错误的概率，聚合这些误导性预测来构建解释会严重扭曲结果

**解释不稳定**：这也可能导致扰动式解释的常见不稳定性问题

核心问题：**如果用来构建解释的底层预测本身就不可靠，那解释又怎能可靠？**

现有校准方法（如标准温度缩放）仅在原始无扰动数据上优化，忽略了解释过程中遇到的特定扰动场景。初步证据表明校准可能有助于可解释性，但缺乏严格的理论分析。

## 方法详解

### 整体框架

三步走：(1) 理论证明校准误差如何损害解释质量；(2) 实证验证模型在扰动下确实严重误校准；(3) 提出 ReCalX 进行扰动特定的重校准。

### 理论分析一：对全局解释的影响

定义特征子集 $S$ 的预测能力为模型在仅观察 $S$ 中特征时的性能提升：

$$v_f^\pi(S) = \mathbb{E}[\mathcal{L}(f_\emptyset^\pi(X), Y)] - \mathbb{E}[\mathcal{L}(f_S^\pi(X), Y)]$$

**定理 3.2**（预测能力分解）：对于交叉熵损失，有：

$$v_f^\pi(S) = \underbrace{D_{\text{KL}}(P_Y \| f_\emptyset^\pi(X))}_{\text{扰动基线偏差}} + \underbrace{I(f_S^\pi(X), Y)}_{\text{信息量}} - \underbrace{CE_{\text{KL}}(f_S^\pi)}_{\text{校准误差}}$$

三个成分的含义：
- 第一项：扰动策略引入的基线偏差
- 第二项：模型在只看 $S$ 特征时对 $Y$ 的互信息——理想预测能力
- 第三项：**校准误差直接削减预测能力**，导致特征重要性被低估或高估

**推论 3.3**：如果模型在所有子集扰动下都完美校准，则 $v_f^\pi(S) = I(f_S^\pi(X), Y)$。

### 理论分析二：对局部解释的影响

**定理 3.4**：局部解释 $\phi(x)$ 与完美校准下的理想解释 $\phi^*(x)$ 之间的差距有上界：

$$\frac{1}{d} \|\phi(x) - \phi^*(x)\|_2^2 \leq 2 \cdot CE_{\text{KL}}^{\max_S} + \sqrt{8 \log(1/\delta)}$$

以概率至少 $1-\delta$ 成立。$CE_{\text{KL}}^{\max_S}$ 是所有扰动子集中的最大校准误差。这表明要改善局部解释，必须降低**所有**扰动级别下的校准误差，仅校准原始数据不够。

### 关键设计：ReCalX

标准温度缩放只用一个全局温度 $T$，无法适应不同扰动强度。ReCalX 的核心思想是**按扰动级别自适应温度**。

**扰动级别定义**：$\lambda(S) = (d - |S|) / d \in [0, 1]$，即被扰动特征的比例。

**分箱温度学习**：将 $[0,1]$ 分成 $B$ 个等宽的 bin，对每个 bin $b$ 学习独立的温度 $T_b$。给定验证集，对每个 bin 内的扰动样本最小化交叉熵损失优化 $T_b$。

**推理时应用**：

$$f_{\text{ReCalX}}^\pi(x, S; \{T_b\}_{b=1}^B)_k = \frac{\exp(z_k(\pi(x,S)) / T(S))}{\sum_{j=1}^K \exp(z_j(\pi(x,S)) / T(S))}$$

其中 $T(S)$ 根据 $\lambda(S)$ 所在 bin 选择对应温度。

**信息保持性**：温度缩放是分量严格单调函数，满足信息保持性（Proposition 4.2），即不改变预测排序和互信息 $I(f_S^\pi(X), Y)$，确保解释针对的仍是原始模型行为。

### 实现细节

- 验证集：每数据集随机选 200 样本，每个扰动级别生成 10 个扰动实例，共 2000 样本/bin
- 评估：使用一致且渐近无偏的 KL 校准误差估计器，每个设置至少 5000 测试样本
- bin 数 $B$：默认 10，更多 bin 单调改善但边际递减

## 实验关键数据

### 主实验一：表格数据校准误差（均值替换扰动）

| 数据集 | 模型 | 未校准 $CE^{\max}$ | 温度缩放 $CE^{\max}$ | **ReCalX** $CE^{\max}$ | 改善↓ |
|--------|------|-------------------|---------------------|----------------------|------|
| Electricity | MLP | 0.1534 | 0.1664 | **0.0163** | 89.4% |
| Covertype | MLP | 0.0797 | 0.1115 | **0.0061** | 92.3% |
| Credit | MLP | 0.4763 | 0.5961 | **0.0533** | 88.8% |
| Pol | MLP | 0.6735 | 0.6521 | **0.1679** | 75.1% |
| Covertype | ResNet | 0.0963 | 0.1413 | **0.0080** | 91.7% |
| Pol | ResNet | 0.8633 | 1.0173 | **0.0910** | 89.5% |

标准温度缩放往往**加剧**扰动下误校准（如 MLP+Credit：0.4763→0.5961），而 ReCalX 改善 75-92%。

### 主实验二：图像模型校准误差（ImageNet）

| 模型 | 扰动方式 | 未校准 $CE^{\max}$ | 温度缩放 $CE^{\max}$ | **ReCalX** $CE^{\max}$ | 改善↓ |
|------|---------|-------------------|---------------------|----------------------|------|
| ResNet50 | Zero | 0.4177 | 0.1810 | **0.0128** | 96.9% |
| DenseNet121 | Zero | 0.3769 | 0.2640 | **0.0098** | 97.4% |
| ViT | Zero | 0.2618 | 0.3057 | **0.0078** | 97.0% |
| SigLIP | Zero | 0.2013 | 0.1476 | **0.0300** | 85.1% |
| ResNet50 | Blur | 0.4158 | 0.1659 | **0.0139** | 96.7% |
| ViT | Blur | 0.0365 | 0.0559 | **0.0072** | 80.3% |

图像模型上 ReCalX 最高达 97.4% 校准误差降幅，ViT 的温度缩放反而恶化（0.2618→0.3057）。

### 解释鲁棒性（ImageNet，平均敏感度 $S_{\text{AVG}}$↓）

| 模型 | LIME(原)→LIME(ReCalX) | KernelSHAP(原)→KernelSHAP(ReCalX) | FeatureAblation(原)→FeatureAblation(ReCalX) |
|------|---------------------|----------------------------------|-------------------------------------------|
| ResNet50 | 1.349→**1.190** | 1.434→**1.364** | 0.965→**0.825** |
| DenseNet121 | 1.174→**0.952** | 1.465→**1.125** | 0.716→**0.602** |
| ViT | 1.498→**1.155** | 1.399→**1.279** | 1.041→**0.880** |
| SigLIP | 1.215→**0.963** | 1.434→**1.222** | 1.140→**0.922** |

ReCalX 在所有模型 × 所有解释方法 × 两种扰动类型上一致提升解释鲁棒性。

### 消融实验与关键发现

1. **Remove-and-Retrain 保真度**：按 ReCalX 增强后的 Shapley 重要性排序移除特征，性能下降更陡峭（如 Electricity 数据集移除 top-3 特征：33% loss 增加 vs 未校准的 24%），说明校准后的解释更准确地识别了真正重要的特征
2. **误校准随扰动量增加**：表格数据上误校准与扰动级别近乎单调递增；图像模型表现更多样（ResNet50 在低扰动时反而更差）
3. **少量验证样本即可**：仅几百个验证样本 ReCalX 就能达到大部分校准改善
4. **跨扰动类型相关性**：不同扰动类型的误校准模式之间存在较强相关

## 亮点与洞察

1. **填补理论空白**：首次严格证明校准误差如何直接损害全局和局部扰动式解释的质量，建立了校准与可解释性之间的定量联系
2. **揭示反直觉现象**：标准温度缩放（在原始数据上优化）可能在扰动场景下反而**加剧**误校准，这解释了为何简单校准对解释帮助有限
3. **方法极简但有效**：ReCalX 本质上就是对不同扰动级别用不同温度，实现简单、推理开销几乎为零（毫秒级选择温度）
4. **信息保持性保证**：从理论上确保 ReCalX 不会改变原始模型的预测行为（排序不变），这点对"解释原始模型"至关重要
5. **跨领域适用**：从表格数据（MLP、ResNet）到图像（ResNet50、DenseNet121、ViT、SigLIP），甚至零样本模型都有效

## 局限性

1. **仅适用于扰动式解释**：对梯度式（Integrated Gradients）或反事实式解释方法不直接适用，尽管原理可能推广
2. **离散 bin 近似**：将扰动级别离散化为 bin 可能遗漏细粒度变化，但实验表明 10 bin 已足够
3. **需要验证数据和标签**：校准需要有标签的验证集，纯无监督场景不适用
4. **温度缩放局限**：温度缩放假设校准误差可通过 logit 全局缩放解决，对严重病态模型可能不够
5. **大规模 LLM 验证缺失**：实验集中在 CV 分类模型和表格模型，NLP/LLM 场景未测试

## 相关工作与启发

- 与 SHAP（Lundberg & Lee, 2017）和 LIME（Ribeiro et al., 2016）的关系：ReCalX 是这些方法的后处理增强，不修改解释算法本身
- 校准文献（Guo et al., 2017）首次报告了深度网络的系统性过自信问题，本文将其扩展到扰动场景
- 分布外检测和校准（Ovadia et al., 2019）的工作指出模型在 OOD 输入上校准崩溃，而扰动正是一种特殊的 OOD
- 启发：类似的"按条件分组校准"思路可推广到对抗样本检测、模型监控等场景

## 评分

- **创新性**: ⭐⭐⭐⭐ — 理论贡献扎实，建立了新的"校准-解释质量"联系，方法虽简单但有理论指导
- **实验充分性**: ⭐⭐⭐⭐⭐ — 覆盖表格+图像、多模型架构、多解释方法、多扰动类型，消融细致
- **实用性**: ⭐⭐⭐⭐⭐ — 几乎零额外推理成本，即插即用，对任何使用 SHAP/LIME 的实践者都有价值
- **写作质量**: ⭐⭐⭐⭐⭐ — 理论-实证-方法三位一体，逻辑链条完整，图表美观
- **总体评价**: ⭐⭐⭐⭐ — 在可解释性和校准的交叉领域做出重要贡献，理论深度和实用价值兼备

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS](llm_probing_with_contrastive_eigenproblems_improving_understanding_and_applicabi.md)
- [\[NeurIPS 2025\] Minimizing False-Positive Attributions in Explanations of Non-Linear Models](minimizing_false-positive_attributions_in_explanations_of_non-linear_models.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[NeurIPS 2025\] An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)
- [\[ICML 2025\] On the Effect of Uncertainty on Layer-wise Inference Dynamics](../../ICML2025/interpretability/on_the_effect_of_uncertainty_on_layer-wise_inference_dynamics.md)

</div>

<!-- RELATED:END -->
