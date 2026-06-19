---
title: >-
  [论文解读] ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition
description: >-
  [ICCV 2025][多模态VLM][3D视觉定位] 提出 ViewSRD 框架，将 3D 视觉定位建模为结构化多视角分解过程：通过 SRD 模块将复杂多锚点查询解耦为简单单锚点查询，并引入跨模态一致视角 token (CCVT) 解决视角变化导致的空间描述不一致问题。 领域现状 领域现状：3D 视觉定位 (3DVG) 旨…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "3D视觉定位"
  - "多视角"
  - "查询分解"
  - "跨模态视角token"
  - "空间推理"
---

# ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition

**会议**: ICCV 2025  
**arXiv**: [2507.11261](https://arxiv.org/abs/2507.11261)  
**代码**: [https://github.com/visualjason/ViewSRD](https://github.com/visualjason/ViewSRD)  
**领域**: 多模态VLM  
**关键词**: 3D视觉定位, 多视角, 查询分解, 跨模态视角token, 空间推理

## 一句话总结

提出 ViewSRD 框架，将 3D 视觉定位建模为结构化多视角分解过程：通过 SRD 模块将复杂多锚点查询解耦为简单单锚点查询，并引入跨模态一致视角 token (CCVT) 解决视角变化导致的空间描述不一致问题。

## 研究背景与动机

### 领域现状

**领域现状**：3D 视觉定位 (3DVG) 旨在根据自然语言描述在 3D 空间中定位目标物体。现有方法面临两大核心挑战：

**1. 多锚点查询的歧义性**

真实的指代表达常涉及多个锚点物体，如"最靠近桌子右边且在沙发旁边的枕头"。现有模型（包括 LLM）在解析此类复杂查询时难以正确区分目标与锚点之间的关系。

**2. 视角变化导致空间关系不一致**

同一物体在不同视角下的空间关系描述会发生变化：从正面看"床头柜在床的右边"，从对面看则变成"在左边"。这种视角依赖的不一致性使模型难以建立准确的文本-视觉对应。

现有方法通常只解决其中一个问题：有的做多视角但不处理复杂查询，有的简化查询但不考虑视角。ViewSRD 是首个同时解决两个问题的统一框架。

### 解决思路

**本文目标**：### 整体框架

ViewSRD 包含三个关键模块：
1. **SRD 模块**：将多锚点查询分解为多个单锚点查询
2. **Multi-TSI 模块**：通过 CCVT 融合文本和场景的多视角特征
3. **文本-场景推理模块**：聚合多视角预测得到最终定位结果

### 关键设计

**1. 简单关系解耦 (SRD) 模块**

- 预训练分类器 Clas 识别句子中的目标词（Target）。


## 方法详解

### 整体框架

ViewSRD 包含三个关键模块：
1. **SRD 模块**：将多锚点查询分解为多个单锚点查询
2. **Multi-TSI 模块**：通过 CCVT 融合文本和场景的多视角特征
3. **文本-场景推理模块**：聚合多视角预测得到最终定位结果

### 关键设计

**1. 简单关系解耦 (SRD) 模块**

- 预训练分类器 Clas 识别句子中的目标词（Target）
- 根据数据集的锚点标签集匹配锚点词（Anchor）
- 设计结构化提示模板，利用 LLM 将复杂查询分解为 I+1 个句子（I 个锚点各一个 + 原始查询）
- 句子匹配算法：基于标签一致性和语义一致性的加权平均打分，筛选最相关的简化查询

**2. 文本聚合策略**

使用 BERT 编码 I+1 个句子特征后：
- 随机选一个作为主特征 F_main
- 其余作为辅助特征 F_aux
- 加权聚合：F_agg = alpha * F_main + (1-alpha) * mean(F_aux)
- alpha 训练时在 {0, 0.1, 0.3, 0.5} 中随机采样，验证时固定为 0.5

**3. 跨模态一致视角 Token (CCVT)**

引入 N 个可学习的视角 token V = {V_1, ..., V_N}，同时嵌入到文本和场景模块中：

*多视角文本模块*：
- 计算每个视角 token 与各句子 [CLS] 特征的归一化点积
- Softmax 加权后重调视角 token 的贡献（与描述匹配的视角增强，不匹配的减弱）
- 通过交叉注意力将视角信息编码到文本特征

*多视角场景模块*：
- 使用 PointNet++ 编码各视角的 3D 场景
- 将 CCVT 拼接到场景特征序列末尾
- 经 Transformer 层处理后，仅保留物体 token（丢弃视角 token）

**4. 文本-场景推理模块**

- 场景特征作为 Query，文本特征作为 Key/Value
- 视角聚合：对多视角输出取平均和最大值的组合
- 预测头投射到结果空间

### 损失函数 / 训练策略

总损失：L = lambda_Obj * L_Object + lambda_Ref * L_Ref^P + lambda_Sent * L_Sent

- L_Object: 物体形状和中心的回归损失
- L_Ref^P: 并行指代损失（同时定位目标和锚点）
- L_Sent: 句子级损失，识别目标和锚点短语

训练细节：单卡 RTX 4090，AdamW 优化器，PyTorch 实现。

## 实验关键数据

### 主实验

在 Nr3D 和 Sr3D 数据集上的定位准确率，以及 ScanRefer 的 Acc@0.25 和 Acc@0.5。ViewSRD 在所有基准上达到 SOTA，特别是在需要精确空间分辨的复杂查询上优势明显。

从论文描述可知：
- Nr3D 包含 45,503 个标注，76 个物体类别，挑战在于同类干扰物多
- Sr3D 包含 83,572 个模板化描述
- ScanRefer 包含 51,583 个自由描述

### 消融实验

各模块的贡献：

| 模块 | 效果 |
|------|------|
| SRD | 将复杂查询分解为简单查询，显著提升多锚点场景准确率 |
| CCVT | 编码视角信息，解决空间描述不一致问题 |
| 文本聚合 | 随机采样 alpha 增强训练鲁棒性 |
| 并行指代损失 | 同时定位目标和锚点，提供更强监督 |

CCVT 作为共享 token 同时嵌入文本和场景模块的设计优于仅嵌入单一模态。

### 关键发现

- LLM 在解析多锚点查询时效果有限，SRD 的结构化分解比直接让 LLM 处理更有效
- 共享视角 token 比独立视角 token 更能保持跨模态一致性
- 随机采样主/辅特征的聚合策略有效增强了鲁棒性
- 在复杂查询（多锚点、视角依赖）上的提升远大于简单查询

## 亮点与洞察

1. **问题定义精准**：将 3DVG 的两大挑战（多锚点歧义 + 视角不一致）统一到一个框架解决
2. **SRD 的实用性**：利用 LLM 做结构化分解而非端到端处理，保证了可控性和可解释性
3. **CCVT 的跨模态设计**：共享 token 同时引导文本和场景模块，比独立处理更优雅
4. **轻量高效**：单 RTX 4090 即可训练，适合学术研究

## 局限与展望

- SRD 模块依赖 LLM（如 GPT）做查询分解，引入额外延迟和 API 成本
- 预训练目标分类器 Clas 需要额外标注数据
- 视角数 N 是超参数，不同场景可能需要不同设置
- 仅在室内场景数据集 (ScanNet) 上验证，对室外大场景的泛化性未知
- 缓存中实验表格被截断，具体数值有待完整论文确认

## 相关工作与启发

- 与 MVT 的多视角方法相比，ViewSRD 引入了显式的视角权重学习机制
- 与 ViewRefer 相比，CCVT 提供了训练时的明确引导信号
- SRD 的查询分解思路可推广到其他需要处理复杂指代的任务（如视觉对话、导航）
- 跨模态一致 token 的设计可推广到 2D 多视角理解

## 评分

- 新颖性: ⭐⭐⭐⭐ （SRD + CCVT 的组合解决了明确的痛点，设计优雅）
- 实验充分度: ⭐⭐⭐⭐ （主流数据集全覆盖，消融完整）
- 写作质量: ⭐⭐⭐⭐⭐ （图示清晰，动机阐述充分，方法描述详细）
- 价值: ⭐⭐⭐⭐ （对 3DVG 有实质推进，但应用范围局限于室内场景）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](../../ACL2025/multimodal_vlm/vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](../../NeurIPS2025/multimodal_vlm/mmwalk_towards_multi-modal_multi-view_walking_assistance.md)
- [\[ICCV 2025\] Visual Intention Grounding for Egocentric Assistants](visual_intention_grounding_for_egocentric_assistants.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](dogr_towards_versatile_visual_document_grounding_and_referring.md)

</div>

<!-- RELATED:END -->
