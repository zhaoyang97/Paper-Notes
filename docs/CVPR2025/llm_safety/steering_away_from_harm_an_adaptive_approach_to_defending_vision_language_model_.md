---
title: >-
  [论文解读] Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks
description: >-
  [CVPR 2025][LLM安全][VLM越狱防御] 提出ASTRA，通过**图像归因**定位对抗图像中与越狱最相关的视觉token，构建**转向向量**表征有害响应方向，并在推理时进行**自适应激活转向**将模型远离有害方向，实现了比JailGuard低12%毒性分数、低18% ASR且快9倍的SOTA防御效果。
tags:
  - "CVPR 2025"
  - "LLM安全"
  - "VLM越狱防御"
  - "激活转向"
  - "图像归因"
  - "对抗攻击"
  - "推理时安全"
---

# Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks

**会议**: CVPR 2025  
**arXiv**: [2411.16721](https://arxiv.org/abs/2411.16721)  
**代码**: [https://github.com/ASTRAL-Group/ASTRA](https://github.com/ASTRAL-Group/ASTRA)  
**领域**: 多模态VLM  
**关键词**: VLM越狱防御, 激活转向, 图像归因, 对抗攻击, 推理时安全

## 一句话总结

提出ASTRA，通过**图像归因**定位对抗图像中与越狱最相关的视觉token，构建**转向向量**表征有害响应方向，并在推理时进行**自适应激活转向**将模型远离有害方向，实现了比JailGuard低12%毒性分数、低18% ASR且快9倍的SOTA防御效果。

## 研究背景与动机

VLM（如MiniGPT-4、LLaVA、Qwen2-VL）在部署时面临严重的安全威胁：视觉输入为越狱攻击提供了新的攻击面。现有越狱攻击分为两类：(1) **扰动型攻击**——通过PGD等方法生成对抗图像绕过安全对齐；(2) **结构型攻击**——通过排版将恶意文本嵌入图像。

现有防御方法各有致命缺陷：
- **输入预处理/对抗训练**：需要大量计算资源去净化图像或微调模型，训练成本极高
- **响应评估型方法**（如JailGuard、ECSO）：需要多次推理生成响应来判断是否有害，推理成本9x以上
- **LLM激活转向方法**：直接借用文本域的转向向量到VLM效果差，因为文本和视觉域的表征存在不匹配

核心矛盾：**高效性和有效性不可兼得**——有效的防御要么训练代价高，要么推理要多次生成。本文的切入角度是：利用激活转向（activation steering）的思路，但解决两大关键问题：(1) 如何在视觉模态中构建有效的转向向量；(2) 如何让转向对良性输入无影响、对对抗输入强效。

## 方法详解

### 整体框架

ASTRA分两步：(1) **离线阶段**——通过图像归因识别对抗图像中与越狱相关的视觉token，构建转向向量；(2) **在线推理阶段**——自适应激活转向，根据校准后激活与转向向量的投影关系，有条件地修改激活方向。整个防御仅需少量推理即可构建，部署时不增加推理时间。

### 关键设计

1. **图像归因构建转向向量（Image Attribution for Steering Vectors）**
    - **功能**：定位对抗图像中真正"有毒"的视觉token，从中提取表征有害响应方向的转向向量
    - **核心思路**：对PGD生成的对抗图像的视觉token进行随机消融（ablation），观察每个token的存在/缺失对越狱概率的影响。具体步骤：
     - 随机采样多个消融向量 $g \sim \{0,1\}^m$，计算每次消融后模型生成"Sure,..."（越狱标志）的对数概率 $f(g)$
     - 用Lasso拟合线性代理模型 $\hat{f}$，权重即为各token的归因分数
     - 选取Top-k归因分数的token，用空查询模板配对，计算有/无这些token时激活的差值作为转向向量 $v^l$
     - 对多个对抗样本取平均得到最终转向向量
    - **设计动机**：不是所有对抗视觉token都与越狱有关，需要精确定位"有毒"token才能构建有效且可迁移的转向向量。直接用文本对比对构建的LLM转向向量在VLM上效果差，因此必须从视觉域入手

2. **自适应激活转向（Adaptive Activation Steering）**
    - **功能**：推理时根据输入是否包含有害方向分量，自适应决定是否以及多大程度进行转向
    - **核心思路**：传统线性转向 $h^l = h^l - \alpha \cdot v^l/\|v^l\|$ 对所有输入无差别修改，导致良性输入性能大幅下降。ASTRA引入条件投影：
     $$h^l = h^l - \alpha \cdot \max\left(\frac{(h^l - h_0^l)^\top v^l}{\|h^l - h_0^l\| \|v^l\|} \cdot \|h^l\|, 0\right) \cdot \frac{v^l}{\|v^l\|}$$
     当校准后激活 $h^l - h_0^l$ 与转向向量 $v^l$ 方向不一致时，$\max$ 项为0，激活不变；方向一致时才按投影大小进行转向
    - **设计动机**：良性输入的激活不应被修改。通过投影和max操作实现"有害输入强转向、良性输入零影响"的自适应效果

3. **激活校准（Activation Calibration）**
    - **功能**：解决不同输入激活聚集在远离原点的区域、导致角度区分失效的问题
    - **核心思路**：收集大量测试输入，计算各层激活的均值 $h_0^l$ 作为校准点。投影时使用 $(h^l - h_0^l)$ 替代 $h^l$，将激活"中心化"后再计算与转向向量的角度
    - **设计动机**：原始激活空间中，不同输入的向量可能角度相似（都指向相似方向），校准后能更好区分有害和良性激活的方向差异

### 损失函数 / 训练策略

转向向量构建阶段使用Lasso回归（L1正则化）拟合代理模型，鼓励稀疏解以定位少数关键token。整个框架不需要微调VLM本身，仅需少量前向推理（几十次消融+前向）即可完成防御构建。

## 实验关键数据

### 主实验（MiniGPT-4上扰动型攻击防御对比）

| 方法 | 毒性分数(ε=16/255) ↓ | 毒性分数(ε=64/255) ↓ | ASR(ε=16/255) ↓ | ASR(unconstrained) ↓ |
|------|---------------------|---------------------|-----------------|---------------------|
| w/o defense | 39.73 | 54.70 | 44.55 | 53.64 |
| JailGuard | 16.51 | 20.93 | 30.00 | 28.18 |
| ECSO | 34.59 | 38.54 | 40.91 | 37.27 |
| **ASTRA** | **11.30** | **4.51** | **9.09** | **9.09** |

### Qwen2-VL上防御效果

| 方法 | 毒性分数(ε=64/255) ↓ | ASR(unconstrained) ↓ |
|------|---------------------|---------------------|
| w/o defense | 55.59 | 76.36 |
| JailGuard | 28.74 | 15.45 |
| **ASTRA** | **2.39** | **15.45** |

### 推理效率对比（每token推理时间，MiniGPT-4）

| 方法 | 推理时间(ms) | 是否单次推理 |
|------|-------------|-------------|
| w/o defense | 173.19 | ✓ |
| JailGuard | 1557.98 (9x) | ✗ |
| ECSO | 457.55 (2.6x) | ✗ |
| **ASTRA** | **173.77** (~1x) | ✓ |

### OOD迁移性（MiniGPT-4，用ε=16/255的转向向量）

| 攻击类型 | w/o defense ASR | ASTRA ASR |
|---------|-----------------|-----------|
| SD（结构型） | 13.75 | 3.75 |
| TYPO（排版型） | 43.75 | 11.25 |
| PGD ε=32/255 | 78.18 | 12.73 |
| MI-FGSM ε=32/255 | 79.09 | 13.64 |
| GCG（纯文本） | 58.18 | 9.09 |

### 自适应攻击抵抗（攻击者知道ASTRA的所有细节）

| 模型 | 攻击无防御ASR(ε=16) | 自适应攻击有防御ASR(ε=16) |
|------|-------------------|------------------------|
| MiniGPT-4 | 44.55 | 13.64 |
| Qwen2-VL | 67.27 | 58.16 |

### 关键发现

1. ASTRA在所有扰动强度下均大幅超越所有基线，尤其在高扰动下优势更明显（ε=64/255时毒性仅4.51%）
2. 推理时间几乎不增加（+0.3%），是JailGuard的1/9
3. 转向向量具有良好的跨攻击迁移性，用一种攻击构建的向量可防御结构型、PGD变体、甚至纯文本攻击
4. 即使面对自适应攻击（攻击者完全了解防御）仍有显著效果

## 亮点与洞察

1. **图像归因是关键创新**——通过Lasso回归精确定位"有毒"视觉token，而非盲目使用所有对抗信号，使转向向量更具迁移性
2. **自适应转向设计精巧**——将投影+max的简单数学操作转化为"良性零影响、恶意强干预"的自适应机制，比传统固定系数转向好得多
3. **效率极高**——防御构建仅需几十次前向推理（几分钟），部署后不增加推理时间，适合实际部署
4. **OOD迁移性出色**——用PGD对抗图像构建的转向向量能防御排版攻击和纯文本攻击，说明不同攻击在激活空间共享有害方向

## 局限性

1. 在Qwen2-VL上面对自适应攻击时ASR仍然较高（58%），说明对更强模型的防御效果有限
2. 转向向量需要针对特定模型构建，不能跨模型迁移
3. 校准激活需要额外收集一批测试数据，增加了一定的准备工作量
4. 对结构型攻击（typography）的防御效果虽有提升但不如对扰动型攻击显著

## 相关工作与启发

- **Activation Engineering (LLM)**: Rimsky et al.通过对比对构建转向向量，Ball et al.研究不同越狱模板的表征机制，本文将这一思路扩展到VLM的视觉域
- **VLM安全**: 与JailGuard（多次推理检测）和ECSO（生成-评估-再生成）不同，ASTRA是首个单次推理的激活层防御方法
- **启发**: 激活空间中有害方向的可分性为VLM安全提供了新视角——安全不一定需要重训模型或多次推理，干预表征空间即可

## 评分

⭐⭐⭐⭐ — 方法简洁高效、实验扎实（3个模型×多种攻击×迁移性×自适应攻击），推理零开销是最大卖点，但面对强自适应攻击的鲁棒性仍有提升空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Hyperbolic Safety-Aware Vision-Language Models](hyperbolic_safety-aware_vision-language_models.md)
- [\[ACL 2026\] ATAAT: Adaptive Threat-Aware Adversarial Tuning Framework against Backdoor Attacks on Vision-Language-Action Models](../../ACL2026/llm_safety/ataat_adaptive_threat-aware_adversarial_tuning_framework_against_backdoor_attack.md)
- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](../../NeurIPS2025/llm_safety/steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)
- [\[NeurIPS 2025\] Attention! Your Vision Language Model Could Be Maliciously Manipulated](../../NeurIPS2025/llm_safety/attention_your_vision_language_model_could_be_maliciously_manipulated.md)
- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](../../ACL2026/llm_safety/robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)

</div>

<!-- RELATED:END -->
