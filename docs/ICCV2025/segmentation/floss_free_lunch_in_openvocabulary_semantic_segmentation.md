# FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation

**会议**: ICCV 2025  
**arXiv**: [2504.10487](https://arxiv.org/abs/2504.10487)  
**代码**: [https://github.com/yasserben/FLOSS](https://github.com/yasserben/FLOSS)  
**领域**: 分割 / 开放词汇 / 文本提示  
**关键词**: open-vocabulary segmentation, template selection, class-expert, entropy, plug-and-play, training-free  

## 一句话总结
挑战OVSS中"平均80个模板"的默认做法，发现每个类别存在特定的"专家模板"（class-expert）远优于平均分类器，提出用预测熵无监督选择专家模板+融合专家预测的FLOSS方法，在不需要标签和训练的情况下一致提升现有OVSS方法。

## 背景与动机
OVSS方法（如MaskCLIP/NACLIP/CLIP-DINOiser）默认使用CLIP原始的80个ImageNet模板（如"a photo of \<class\>"、"a sketch of \<class\>"等）来构建文本分类器，对所有模板的嵌入取平均。这种做法从CLIP的zero-shot分类沿用至今，但从未有人在分割场景下系统研究每个单独模板的表现。核心发现：对于每个类别，存在某些单一模板构建的分类器比80模板平均分类器效果更好——这些就是"class-experts"。

## 核心问题
(1) 如何在无标签、无训练的条件下识别每个类别的专家模板？(2) 识别后如何有效融合多个专家的预测来输出最终分割？

## 方法详解

### 整体框架
FLOSS完全是plug-and-play的后处理方法。给定一组无标签图像和一个现有的OVSS模型：(1) 对每个单一模板构建分类器执行分割 → (2) 用类别级预测熵选择top-N低熵模板作为class-expert → (3) 每个class-expert产生一张分割图 → (4) 通过"最高置信度投票"融合K个专家预测为最终分割图。

### 关键设计
1. **核心发现——class-expert的存在**：通过系统实验发现，对于Cityscapes的每个类别，80个模板中都有若干单一模板的IoU超过平均模板分类器（Figure 1）。且不同类别的expert集合不同——"a photo of a car"可能是car的expert但不是sky的expert。这暗示平均所有模板是一种次优选择。

2. **基于预测熵的无监督专家识别**：对每个模板$\mathcal{T}_m$和每个类别$k$，计算被该模板分类为类$k$的所有像素的平均softmax熵。选择熵最低的Top-N个模板作为该类的专家（$N=4$）。低熵意味着分类器对其预测更确信，经验上与高IoU强相关。不需要任何标签。

3. **专家预测融合**：K个class-expert各生成一张完整的分割图。对于每个像素，检查哪些expert在该位置预测了自己专长的类别（即expert-k预测该像素为类k）。如果有多个这样的expert，取softmax概率最高的那个作为最终预测；如果没有expert预测自己的类别（约2%的像素），回退到默认的$W_{\text{CLIP}}$平均分类器。

### 损失函数 / 训练策略
完全training-free、label-free。唯一超参数N=4。

## 实验关键数据
| 模型 | 方法 | CS | VOC20 | PC59 | ADE | Stuff | Avg |
|------|------|-----|-------|------|-----|-------|-----|
| CLIP-DINOiser | baseline | 31.3 | 80.8 | 36.0 | 17.5 | 24.6 | 38.0 |
| | +FLOSS | **34.6** | **82.2** | **36.3** | **18.0** | **24.7** | **39.2** |
| NACLIP | baseline | 35.5 | 83.0 | 35.2 | 19.1 | 22.4 | 39.0 |
| | +FLOSS | **37.0** | **83.5** | **35.9** | **19.6** | **22.7** | **39.7** |
| MaskCLIP | baseline | 25.0 | 61.8 | 25.5 | 14.2 | 17.5 | 28.8 |
| | +FLOSS | **25.8** | **61.8** | **26.2** | **14.4** | **17.8** | **29.2** |

- 所有3种OVSS模型×5个数据集均一致提升
- Cityscapes上CLIP-DINOiser提升最大：+3.3 mIoU
- 跨域泛化：CS上选的expert在ACDC Fog上+4.9 mIoU
- 低数据场景：仅1张无标签Cityscapes图就能超越baseline
- ViT-B/16和ViT-L/14两种backbone均有效
- 预测的expert有~50%是真正的expert（通过与GT对比的quality metric验证）

### 消融实验要点
- 融合策略："Highest"（最高置信度投票）> "Average" > "Default"
- 熵是最有效的无监督expert选择指标，Avg. Probability接近
- N=4最优（~80模板中选4个），N过多包含非expert反而降低效果
- 只需50%是true expert就能超越baseline（oracle实验）
- Oracle上限（用GT选best expert）：sky类可提升30+ IoU

## 亮点
- **洞察新颖且意外**：80模板平均的默认做法被挑战——"不是所有模板都对每个类有用"，这是社区忽视的重要发现
- **方法极简但有效**：只是改变了用哪些模板，完全不改视觉encoder或模型架构
- **plug-and-play的"免费午餐"**：任何OVSS方法都能直接叠加使用，跨模型跨数据集一致有效
- **低数据可用性极好**：仅1张无标签图就能选出有用的expert——这对部署到新领域极其实用
- **跨域泛化**：CS选的expert可以迁移到ACDC/BDD/Mapillary等不同域的数据集

## 局限性 / 可改进方向
- 类别数大时计算开销增加显著（ADE 150类时推理时间从23ms增到339ms）
- 仅限80个ImageNet模板池——更多或更好的模板生成（如用LLM生成）可能进一步提升
- 提升幅度在某些简单数据集上较小（VOC20/PC59接近ImageNet分布）
- 尚未探索非CLIP backbone（如SigLIP）或更复杂的OVSS方法

## 与相关工作的对比
- **vs. CorrCLIP**：CorrCLIP从视觉端修复CLIP（注意力范围重建），FLOSS从文本端优化CLIP（模板选择）——两者完全正交，可以叠加！
- **vs. ProxyCLIP**：ProxyCLIP用DINO辅助视觉特征，FLOSS优化文本分类器——同样正交
- **vs. Prompt engineering**：传统prompt工程通过LLM生成更好的类名描述；FLOSS从已有的80模板中选择最佳子集，是一种"模板选择"而非"模板生成"

## 启发与关联
- **idea灵感**：将class-expert选择扩展到prompt生成——对每个类用LLM生成N个候选描述，然后用熵选择最优描述。这结合了prompt engineering和expert selection
- 与CorrCLIP正交的特性意味着可以**CorrCLIP + FLOSS**联合使用获得进一步提升
- 低数据可用性使其非常适合部署到新的专业领域（医学、遥感等）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "不是所有模板都平等"的发现虽然简单但社区完全忽视，entropy-based expert selection无需标签
- 实验充分度: ⭐⭐⭐⭐⭐ 5个标准benchmark + 跨域泛化 + 低数据 + 跨数据集迁移 + 多种融合策略/无监督指标消融
- 写作质量: ⭐⭐⭐⭐⭐ Figure 1/2极其直观，问题定义→发现→解法→验证的逻辑完美
- 价值: ⭐⭐⭐⭐⭐ 真正的"免费午餐"——零训练零标签一致提升，与任何OVSS方法兼容
