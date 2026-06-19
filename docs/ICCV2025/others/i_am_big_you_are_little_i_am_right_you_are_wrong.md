---
title: >-
  [论文解读] I Am Big, You Are Little; I Am Right, You Are Wrong
description: >-
  [ICCV 2025][Minimal Pixel Sets] 利用因果推理 XAI 工具 rex 提取图像分类模型的最小充分像素集（MPS），系统比较 5 种架构 15 个模型的"注意力集中度"，发现大型模型（EVA/ConvNext）仅用图像 5% 像素即可做出分类，且不同架构的 MPS 在大小和位置上存在统计显著差异。
tags:
  - "ICCV 2025"
  - "Minimal Pixel Sets"
  - "Image Classification"
  - "Model Comparison"
  - "Explainable AI"
  - "Causal Reasoning"
---

# I Am Big, You Are Little; I Am Right, You Are Wrong

**会议**: ICCV 2025  
**arXiv**: [2507.23509](https://arxiv.org/abs/2507.23509)  
**代码**: [ReX-XAI/ReX](https://github.com/ReX-XAI/ReX)  
**领域**: Others (Explainable AI / Model Analysis)  
**关键词**: Minimal Pixel Sets, Image Classification, Model Comparison, Explainable AI, Causal Reasoning

## 一句话总结

利用因果推理 XAI 工具 rex 提取图像分类模型的最小充分像素集（MPS），系统比较 5 种架构 15 个模型的"注意力集中度"，发现大型模型（EVA/ConvNext）仅用图像 5% 像素即可做出分类，且不同架构的 MPS 在大小和位置上存在统计显著差异。

## 研究背景与动机

随着视觉分类模型种类（CNN、ViT、混合架构）和规模的激增，我们对模型"如何做决策"的理解仍然有限。现有比较主要集中在准确率和鲁棒性上，而对模型到底依赖"哪些像素"做分类缺乏系统研究。

先前工作（Jiang et al.）使用 SAG 工具提取 patch 级别的"最小充分解释"，但存在两大问题：
1. SAG 的"充分性"定义过于宽松——使用置信度阈值而非强制匹配原始分类
2. SAG 基于固定大小 patch，无法达到像素级精度

本文提出使用 rex 工具提取 **Minimal Pixel Sets (MPS)**，有两个关键优势：
- 不受 patch 尺寸限制，达到像素级最小化
- 充分性定义严格：MPS 必须使模型复现原始 top-1 分类

## 方法详解

### 整体框架

研究框架：
1. 选取 5 种架构的 15 个预训练 ImageNet-1k 模型
2. 使用 rex 对 1000 张图像（500 验证集 + 500 测试集）提取 MPS
3. 通过统计分析（Kruskal-Wallis H 检验、Friedman 检验）比较 MPS 的大小和位置差异
4. 分析正确/错误分类与 MPS 大小的关系

### 关键设计

1. **Minimal Pixel Sets (MPS) 提取**：rex 采用因果推理框架，将模型视为黑盒因果模型。核心算法：

    - 将图像分为 4 个 superpixel 区域
    - 通过组合遮盖（baseline=0）生成 mutant，测试模型分类结果
    - 按因果 responsibility 排序像素（高 responsibility = 对分类贡献大）
    - 迭代细化 superpixel 划分（默认 20 次迭代），生成 responsibility landscape
    - 按 responsibility 排名逐步添加像素，直到复现原始 top-1 分类
    - 最终得到的像素集即为 MPS（近似最小但保证充分）

2. **模型架构选择**：覆盖从小型 CNN 到超大 Transformer 的完整谱系：

    - **Inception** (CNN): V3, V4, ResNet-V2（宽网络架构）
    - **ResNet** (CNN): 152-B A1, A2, D（残差网络）
    - **ConvNext** (现代 CNN): V2 Large, V2 Huge v1/v2（现代化 ResNet）
    - **ViT** (Transformer): Large, Huge V1/V2（标准视觉 Transformer）
    - **EVA** (Transformer): 02 Large V1/V2, Giant（大规模预训练 ViT，高达 10 亿参数）

3. **统计分析方法**：

    - **Kruskal-Wallis H 检验**：跨架构 MPS 大小差异（非参数检验）
    - **Friedman 检验**：架构内部模型间 MPS 大小差异（配对数据检验）
    - **Sørensen-Dice 系数**：衡量不同模型 MPS 的重叠度
    - **Hausdorff 距离**：衡量 MPS 空间位置差异
    - **Bonferroni 校正**：控制多重比较的 I 类错误
    - **混合线性模型**：控制模型准确率混淆因素，分析正确/错误分类对 MPS 大小的影响

### 损失函数 / 训练策略

本文无训练过程。所有模型使用 Timm 库的预训练 IN-1k 权重（2024.02），转换为 ONNX 格式在 ONNX Runtime 上推理。rex 对所有模型使用相同超参数和随机种子。

## 实验关键数据

### 主实验 (表格)

**MPS 大小（占图像面积比例）**

| 模型 | 总平均 | 正确分类 | 错误分类 | 模型准确率 |
|------|--------|----------|----------|-----------|
| ConvNext-V2 Huge v1 | **0.052** | 0.048 | 0.061 | 0.890 |
| EVA Giant | **0.056** | 0.052 | 0.065 | 0.894 |
| ConvNext-V2 Huge v2 | 0.068 | 0.063 | 0.082 | 0.894 |
| ConvNext-V2 Large | 0.089 | 0.075 | 0.122 | 0.880 |
| ViT Large | 0.099 | 0.098 | 0.111 | 0.900 |
| ViT Huge V2 | 0.103 | 0.102 | 0.113 | 0.872 |
| ResNet152-D | 0.137 | 0.130 | 0.155 | 0.828 |
| ViT Huge V1 | 0.158 | 0.154 | 0.170 | 0.882 |
| Inception V4 | 0.239 | 0.224 | 0.261 | 0.840 |
| Inception V3 | 0.247 | 0.231 | 0.271 | 0.800 |
| Inception-ResNet V2 | **0.254** | 0.246 | 0.265 | 0.814 |

Inception 模型的 MPS 是 ConvNext 的 **3.6 倍**，差异显著。

### 消融实验 (表格)

**跨架构 MPS 重叠度（Dice 系数，最佳模型间）**

| | EVA Giant | ConvNext | ViT Large | ResNet152 | Inception |
|--|-----------|----------|-----------|-----------|-----------|
| EVA Giant | 1.0 | 0.287 | 0.253 | 0.165 | 0.141 |
| ConvNext | 0.287 | 1.0 | 0.304 | 0.162 | 0.163 |
| ViT Large | 0.253 | 0.304 | 1.0 | 0.232 | 0.225 |
| ResNet152 | 0.165 | 0.162 | 0.232 | 1.0 | 0.282 |
| Inception | 0.141 | 0.163 | 0.225 | 0.282 | 1.0 |

MPS 的重叠度普遍很低（多数 < 0.3），说明不同架构关注的图像区域差异极大。

### 关键发现

1. **跨架构差异显著**：Kruskal-Wallis 检验 $H(4)=1176.134, p<0.001$，拒绝"各架构 MPS 大小无差异"的原假设
2. **架构内部也有差异**：Friedman 检验在 ConvNext、EVA、ResNet、ViT 上均 $p<0.01$，仅 Inception 模型内部无显著差异（$p=0.36$）
3. **错误分类 → 更大 MPS**：混合线性模型显示错误分类平均增加 2.6% 面积（$p<0.01$）
4. **EVA Giant 仅用 5.4% 像素**即可做出分类，暗示大模型可能存在过拟合或"短视"倾向
5. **不同模型看不同区域**：同一图像上 ResNet 的 MPS 可能与其他模型完全不重叠（Dice=0）

## 亮点与洞察

1. **因果推理视角的模型分析**：MPS 基于严格的因果充分性定义，比 GradCAM/SHAP 等归因方法更具可操作性
2. **大模型的"短视"现象**：10 亿参数的 EVA-Giant 仅依赖 5% 像素做分类，引发对模型安全性的担忧，尤其在医疗、自动驾驶等高风险场景
3. **MPS 作为模型选择依据**：除准确率/鲁棒性外，MPS 特征可作为模型选型的新维度
4. **错误分类的 MPS 偏大**：可作为后验检查手段——如果一个样本的 MPS 异常大，flags 可能的错误预测

## 局限与展望

- rex 使用 baseline=0 遮盖，产生大量 OOD 图像；不同 baseline（如 blur）可能改变结论
- MPS 的计算本身是 NP 难问题（DP-complete），rex 只是近似最小
- 仅分析了 top-1 MPS，未深入研究多重解释（一张图可能有多个不相交的 MPS）
- 未量化分析 MPS 大小与模型鲁棒性的关系
- CalTech-256 验证集较小（50 张），统计效力有限
- ImageNet-1k 标签存在噪声，可能影响正确/错误分类分析

## 相关工作与启发

- **rex**：基于实际因果性（Halpern-Pearl 框架）的黑盒 XAI 工具
- **SAG (Jiang et al.)**：使用 patch 级别的组合解释，但充分性定义较弱
- **GradCAM / SHAP / LIME**：白盒/黑盒归因方法，提供像素重要度排序但不保证充分性
- 启发：在模型部署前，用 MPS 分析"模型到底看了多少"是一种简洁有效的审计手段

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次大规模系统性用因果 MPS 比较 15 个视觉模型
- **实验充分度**: ⭐⭐⭐⭐ 5 架构 15 模型 1000 图像，多种统计检验，CalTech 交叉验证
- **写作质量**: ⭐⭐⭐⭐ 研究问题清晰、统计方法严谨、案例分析生动
- **价值**: ⭐⭐⭐ 开创性分析视角，但实际应用指导有限（未提出改进方案）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ResNets Are Deeper Than You Think](../../NeurIPS2025/others/resnets_are_deeper_than_you_think.md)
- [\[ICCV 2025\] You Share Beliefs, I Adapt: Progressive Heterogeneous Collaborative Perception](you_share_beliefs_i_adapt_progressive_heterogeneous_collaborative_perception.md)
- [\[ACL 2025\] You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with Multi-Agent Conversations](../../ACL2025/others/you_need_to_mimic_to_get_fame_solving_meeting_transcript_scarcity_with_a_multi-a.md)
- [\[ACL 2025\] A Little Human Data Goes A Long Way](../../ACL2025/others/a_little_human_data_goes_a_long_way.md)
- [\[NeurIPS 2025\] Improving Forecasts of Suicide Attempts for Patients with Little Data](../../NeurIPS2025/others/improving_forecasts_of_suicide_attempts_for_patients_with_little_data.md)

</div>

<!-- RELATED:END -->
