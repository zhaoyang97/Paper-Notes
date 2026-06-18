---
title: >-
  [论文解读] Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models
description: >-
  [ACL 2025][多模态VLM][模型剪枝] 提出层次化安全重对齐方法HSR，通过先识别安全关键注意力头、再在这些头中定位并恢复被剪枝的安全关键神经元，以极低参数开销（万分之几）显著恢复被剪枝LVLM丢失的安全性能。 问题背景 大视觉语言模型（LVLM）参数规模庞大，部署到资源受限环境时常需网络剪枝压缩。然而…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "模型剪枝"
  - "安全对齐"
  - "大视觉语言模型"
  - "注意力头"
  - "神经元恢复"
---

# Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models

**会议**: ACL 2025  
**arXiv**: [2505.16104](https://arxiv.org/abs/2505.16104)  
**作者**: Yue Li, Xin Yi, Dongsheng Shi, Gerard de Melo, Xiaoling Wang, Linlin Wang (华东师范大学, HPI/波茨坦大学)
**代码**: [TheShineyue/HSR](https://github.com/TheShineyue/HSR)  
**领域**: 多模态VLM  
**关键词**: 模型剪枝, 安全对齐, 大视觉语言模型, 注意力头, 神经元恢复

## 一句话总结

提出层次化安全重对齐方法HSR，通过先识别安全关键注意力头、再在这些头中定位并恢复被剪枝的安全关键神经元，以极低参数开销（万分之几）显著恢复被剪枝LVLM丢失的安全性能。

## 研究背景与动机

### 问题背景
大视觉语言模型（LVLM）参数规模庞大，部署到资源受限环境时常需网络剪枝压缩。然而，模型安全性相关的神经元区域与通用能力区域存在分离且呈稀疏分布，以"效用重要性"为指标的剪枝方法天然倾向于移除这些对效用贡献低但对安全至关重要的神经元。

### 已有工作的不足
- 现有剪枝研究（Wanda、SparseGPT、SNIP等）主要关注压缩后的效用保持，忽略安全性退化问题
- 安全对齐研究集中在防御越狱攻击和安全训练，未关注剪枝导致的安全损失
- 已有观察（Hasan et al. 2024）仅在低稀疏度下发现安全改善，高稀疏度下的安全退化被忽视

### 核心动机
作者对6个主流LVLM进行50%稀疏度的Wanda剪枝实验，发现所有模型均出现安全性退化：最严重的ASR上升15.4%，最轻的也有2.8%。这是首个专门针对剪枝后LVLM进行安全恢复的工作。

## 方法详解

### 整体框架
HSR采用从粗到细的两级层次化策略：先在注意力头层面定位安全关键头，再在神经元层面识别并恢复安全关键神经元。

### 关键设计1：安全关键头识别（Ships指标）

借鉴Zhou et al.（2025）的Safety Head Importance Score（Ships），量化每个注意力头对安全的贡献：

1. 对每个注意力头$h_i^l$，通过将其Q/K/V矩阵乘以极小系数$\epsilon$来消除其贡献
2. 计算消融前后模型在有害输入上输出分布的KL散度，作为该头的安全贡献分数
3. 针对GQA机制（现代LVLM常用），推导了适配的掩码方程
4. 在数据集层面，通过SVD分解网络激活矩阵，用主角度量化安全表示的偏离程度：

$$\text{Ships}(D, h_i^l) = \sum_{r=1}^{r_{\max}} \cos^{-1}(\sigma_r(U_\theta^{(r)}, U_A^{(r)}))$$

5. 选取Ships分数最高的top-h个注意力头，作为安全关键头

### 关键设计2：安全关键神经元定位与恢复

在安全关键头内，进一步识别被剪枝但对安全至关重要的神经元：

1. **双重重要性评估**：分别在安全数据$D^s$（有害指令+拒绝回复）和效用数据$D^u$（安全指令+正常回复）上计算每个权重的重要性分数$\mathbf{I}^s$和$\mathbf{I}^u$
2. **三种评分方法**：Wanda Score（权重绝对值×输入激活$\ell_2$范数）、SparseGPT Score（基于Hessian矩阵）、SNIP Score（一阶Taylor近似）
3. **集合运算选取安全关键神经元**：

$$S(p, q, p_{\max}) = (S^s(q) \cap S^u(p_{\max})) - S^u(p)$$

其中$S^s(q)$为安全重要性top-q%的权重集合，$S^u(p_{\max})$为效用重要性top-$p_{\max}$%的集合，减去已保留的$S^u(p)$。这确保恢复的神经元：安全重要性高、效用重要性适中（不会严重影响效用）、且确实被剪掉了。

4. 将这些被识别的安全关键神经元在剪枝模型中恢复原始权重值

### 数据构造
- 安全数据集：VLGuard训练集中的Unsafe-Unsafe对（不安全图像+不安全指令）
- 效用数据集：VLGuard训练集中的Safe-Safe对（安全图像+安全指令）

## 实验关键数据

### 实验1：不同剪枝方法下的HSR效果（Qwen2.5-VL, 50%稀疏度）

| 方法 | SafeBench ASR↓ | Ch3Ef ASR↓ | 平均ASR↓ | 安全恢复率RSR | MMBench↑ | DocVQA↑ | 参数恢复比 |
|------|------|------|------|------|------|------|------|
| Full Model | 1.40 | 2.35 | 1.88 | - | 87.02 | 94.51 | - |
| SNIP | 4.60 | 8.12 | 6.36 | - | 84.55 | 92.93 | - |
| SNIP + HSR | 3.00 | 5.34 | 4.17 | 48.88% | 84.62 | 92.90 | 0.150‱ |
| Wanda | 11.20 | 17.74 | 14.47 | - | 85.15 | 91.97 | - |
| Wanda + HSR | 9.00 | 13.03 | 11.02 | 27.40% | 85.01 | 92.13 | 0.020‱ |
| SparseGPT | 3.00 | 3.21 | 3.10 | - | 83.88 | 90.64 | - |
| SparseGPT + HSR | 2.80 | 2.56 | 2.68 | 34.43% | 83.88 | 90.63 | 0.133‱ |

HSR在三种剪枝方法下均有效恢复安全性，安全恢复率超27%，恢复参数量仅占剪枝参数的万分之几。

### 实验2：不同LVLM上的HSR效果（Wanda 50%稀疏度）

| 模型 | 剪枝后平均ASR | HSR后平均ASR | 安全恢复率RSR | 效用变化 | 参数恢复比 |
|------|------|------|------|------|------|
| Qwen2-VL | 22.24 | 16.78 | 35.29% | +1.21 | 0.016‱ |
| LLaVA-NeXT-Mistral | 17.60 | 14.57 | **104.12%** | -0.32 | 0.385‱ |
| LLaVA-NeXT-Vicuna | 18.12 | 17.09 | 36.52% | -0.33 | 1.803‱ |
| LLaVA-NeXT-Llama3 | 17.71 | 16.99 | 14.81% | -0.19 | 0.799‱ |
| Llama3.2-Vision | 8.98 | 7.93 | 16.69% | -1.94 | 0.065‱ |

LLaVA-NeXT-Mistral实现了超100%的安全恢复率（HSR后ASR甚至低于未剪枝模型）。Qwen系列模型的效用反而因HSR略有提升。

### 实验3：不同稀疏度的影响（Qwen2-VL + Wanda）

| 稀疏度 | 剪枝后Safety/Utility | HSR后Safety/Utility |
|------|------|------|
| 40% | 10.69 / 82.79 | 10.01 / 82.65 |
| 50% | 22.24 / 76.10 | 16.78 / 77.31 |
| 60% | 27.05 / 48.17 | 25.61 / 63.37 |

50%稀疏度安全恢复最显著；60%稀疏度时效用提升最大（+15.2），说明被恢复的安全神经元也具有效用贡献。

## 关键发现

- **少量神经元决定安全性**：仅top 0.35%的安全重要神经元起关键作用，超过此范围的神经元反而对安全产生负面影响
- **安全与效用神经元存在纠缠**：对安全贡献最大的神经元往往也对效用有显著贡献
- **存在"有害安全"神经元**：直接恢复整个注意力头（HSR-a）反而导致ASR上升，因为头内部分神经元对安全有负面作用
- **Ships总分与安全退化强相关**：6个模型的Ships总分排名与剪枝后安全退化排名的Spearman相关系数达0.8857
- **GQA机制影响恢复效率**：Qwen系列采用GQA且每组query数最多、head数最少，使得单个神经元影响范围更广，恢复效率更高

## 亮点

- **首创性**：首个专门针对LVLM剪枝后安全恢复的工作，填补了剪枝安全性研究的空白
- **极低开销**：恢复参数量仅占剪枝参数的万分之几（最低0.016‱），几乎不影响模型稀疏度
- **层次化设计**：从注意力头到神经元的两级筛选，既保持轻量又避免恢复有害神经元
- **广泛验证**：覆盖6种主流LVLM、3种剪枝方法、结构化/非结构化剪枝、多种稀疏度
- **可与多种剪枝方法兼容**：Wanda、SparseGPT、SNIP均可适配

## 局限性

- **效用可能轻微下降**：在部分模型上HSR导致效用略有损失（如Llama3.2-Vision下降约2%）
- **Llama3系列恢复效果有限**：基于Llama3的LVLM安全恢复率仅14-17%，远低于其他模型
- **仍需恢复部分神经元**：虽然量级极小，但可能存在零参数恢复的更优方案
- **依赖安全标注数据**：需要有害指令-拒绝回复的配对数据集构建安全重要性评估
- **仅验证50%附近稀疏度**：极高稀疏度（>60%）下效果未充分探索

## 与相关工作的对比

- **Wei et al. (2024)**：发现安全区域与效用区域分离且稀疏，本文在此基础上设计了恢复策略
- **Zhou et al. (2025)**：提出Ships指标评估注意力头的安全贡献，本文将其用于剪枝后的安全恢复场景
- **Arditi et al. (2024)**：发现LLM中存在单一的拒绝方向，本文从注意力头和神经元粒度进行更细致的安全定位
- **Hasan et al. (2024)**：观察到低稀疏度剪枝可改善安全性，本文关注高稀疏度下的安全退化并提出修复方案
- **AdaShield (Wang et al. 2024b)**：通过输入端添加防御提示增强安全，本文从模型内部参数恢复角度解决问题

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统研究剪枝导致的LVLM安全退化并提出修复方案
- 实验充分度: ⭐⭐⭐⭐ — 覆盖6个模型、3种剪枝方法、详细消融和超参分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题动机充分，分析细致
- 价值: ⭐⭐⭐⭐ — 为模型压缩部署的安全性提供了实用且低开销的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] VLSBench: Unveiling Visual Leakage in Multimodal Safety](vlsbench_unveiling_visual_leakage_in_multimodal_safety.md)
- [\[CVPR 2025\] SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models](../../CVPR2025/multimodal_vlm/spa-vl_a_comprehensive_safety_preference_alignment_dataset_for_vision_language_m.md)
- [\[NeurIPS 2025\] JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/jailbound_jailbreaking_internal_safety_boundaries_of_vision-language_models.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](cant_see_the_forest_for_the.md)
- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
