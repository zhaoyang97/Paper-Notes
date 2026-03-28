# OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps

**会议**: NeurIPS 2025  
**arXiv**: [2509.19282](https://arxiv.org/abs/2509.19282)  
**代码**: https://mlpc-ucsd.github.io/OverLayBench  
**领域**: 图像生成 / 布局生成  
**关键词**: Layout-to-Image, 重叠布局, Amodal Mask, 扩散模型, 基准评测

## 一句话总结
OverLayBench 构建了首个聚焦密集重叠场景的 Layout-to-Image 基准（4052 样本 + OverLayScore 难度指标），揭示 SOTA 方法在复杂重叠下 mIoU 从 71%→54% 急剧退化，提出 Amodal Mask 监督在重叠 IoU 上提升 15.9%。

## 研究背景与动机

1. **领域现状**：L2I 方法（GLIGEN、InstanceDiffusion、CreatiLayout）在简单布局上效果好，但现有基准 80%+ 集中在低重叠场景。
2. **现有痛点**：多个物体大面积重叠且语义相似时（如两只同色猫），模型容易合并或丢失实例，但缺乏系统评测。
3. **核心矛盾**：重叠是真实场景的常态，但现有方法和基准都回避了这个困难场景。
4. **本文要解决什么？** 量化重叠难度 + 构建分层评测基准 + 探索改善重叠生成的方法。
5. **切入角度**：OverLayScore = Σ IoU(Bi,Bj)·cos(pi,pj)（空间重叠×语义相似度），用 Amodal Mask 提供完整物体轮廓监督。
6. **核心 idea 一句话**：OverLayScore 量化重叠难度 + 分层基准 + Amodal Mask 监督提升重叠生成质量。

## 方法详解

### 整体框架
4052 个样本（2052 简单 + 1000 常规 + 1000 复杂），Qwen2.5-VL-32B 标注实例描述和关系。CreatiLayout-AM 在 CreatiLayout 基础上加 amodal mask token/pixel-level loss。

### 关键设计

1. **OverLayScore 指标**: $\text{Score} = \sum \text{IoU}(B_i,B_j) \cdot \cos(p_i,p_j)$——空间重叠×语义相似度
2. **Amodal Mask 监督**: $\mathcal{L} = \mathcal{L}_{LDM} + \lambda\mathcal{L}_{token} + \beta\mathcal{L}_{pixel}$——用完整物体 mask（含被遮挡部分）监督
3. **O-mIoU 指标**: 仅在重叠区域计算 IoU，更精确评估遮挡处理能力

### 损失函数 / 训练策略
- 基于 FLUX DiT 架构，amodal mask 额外标注

## 实验关键数据

### 主实验

| 方法 | 简单 mIoU | 复杂 mIoU | O-mIoU |
|------|----------|----------|--------|
| CreatiLayout-FLUX | **71.17%** | 54% | 49.80% |
| **CreatiLayout-AM** | — | — | **65.70%** (+15.9%) |

### 关键发现
- 所有方法在复杂重叠下急剧退化（71%→54%）
- Amodal mask 监督显著改善重叠区域生成（+15.9% O-mIoU）
- DiT 模型普遍优于 U-Net 模型

## 亮点与洞察
- **OverLayScore 将生成难度量化**：空间×语义的乘积捕捉了"困难重叠"的本质
- **Amodal Mask 是自然的解决方案**：告诉模型被遮挡部分也应存在

## 局限性 / 可改进方向
- 仅边界框级重叠评估
- 数据集规模较小（4052 样本）
- 复杂重叠上改善仍有限

## 相关工作与启发
- **vs GLIGEN**: 固定注意力 mask 在重叠场景下失效
- **vs InstanceDiffusion**: 实例控制生成但缺乏重叠评测

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个聚焦重叠的 L2I 基准
- 实验充分度: ⭐⭐⭐⭐ 8+ 方法评测 + amodal 消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐⭐ 揭示 L2I 的关键弱点
- 新颖性: ⭐⭐⭐⭐ 新问题+新度量+新基准
- 实验充分度: ⭐⭐⭐⭐ 多模型对比
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ L2I评估重要补充
### 深入分析
- 密集重叠是现有布局生成方法的系统性盲区——OverLayScore填补了评估空白
- OverLayScore与人类感知高度相关，CreatiLayout-AM在密集重叠场景显著优于基线
- 该方法的核心创新在于设计思路的简洁性和有效性
- 实验结果充分验证了核心假设

