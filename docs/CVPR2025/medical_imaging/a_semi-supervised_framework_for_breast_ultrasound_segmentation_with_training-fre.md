# A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement

**会议**: CVPR2025  
**arXiv**: [2603.06167](https://arxiv.org/abs/2603.06167)  
**代码**: 待确认  
**领域**: medical_imaging  
**关键词**: semi-supervised learning, breast ultrasound segmentation, pseudo label, vision-language model, contrastive learning

## 一句话总结

提出结合 VLM 无训练伪标签生成（外观描述 prompt 驱动 Grounding DINO + SAM）和双教师不确定性融合精炼的半监督乳腺超声分割框架，仅用 2.5% 标注数据即达到接近全监督的性能。

## 研究背景与动机

1. **标注成本高昂**：乳腺超声（BUS）图像的像素级标注需要放射科专家，耗时且昂贵，限制了全监督方法的实用性
2. **现有 SSL 方法局限**：伪标签初期易出错导致 confirmation bias；极少标注下教师模型欠训练产生噪声伪标签；针对 RGB 自然图像设计的强弱增强不适合灰度、散斑噪声的 BUS 数据
3. **VLM 直接迁移失败**：医学术语（如"tumor"、"lesion"）作为 prompt 时，通用 VLM 缺乏领域语义，在 BUS 上定位不稳定
4. **微调方案不切实际**：微调大模型需要边界框标注、大量标注数据和定制 vision-text 对，与极少标注的临床场景矛盾
5. **对比学习忽视困难区域**：现有 SSL 中对比学习采样全局或"可靠"像素特征，忽略了不确定但信息丰富的边界区域
6. **BUS 病灶的外观一致性**：乳腺肿瘤通常呈现"暗椭圆/暗圆形"等一致性外观特征，可用简单外观描述实现跨域结构迁移

## 方法详解

### 整体框架

两阶段流程：(1) 外观 prompt 驱动的无训练伪标签生成 (APPG)；(2) 双教师半监督伪标签精炼（静态教师预热 + 不确定性加权融合 + 自适应反向对比学习）。

### Step 1: APPG — 外观 Prompt 驱动的无训练伪标签生成

- **Prompt 设计**：利用 LLM (GPT-5) 将医学领域通用特征（形状、边界、亮度对比）转化为简洁外观描述："dark oval"、"dark round"、"dark lobulated"
- **两阶段生成**：Grounding DINO 根据外观 prompt 产生 bounding box $b_i^u$，再送入 SAM 生成分割 mask $\hat{y}_i^0$
- **关键洞察**：外观描述实现了自然图像到医学图像的跨域结构迁移，无需任何训练或微调

### Step 2: 静态教师预热训练

- 过滤无效伪标签（前景面积 < 1% 的 mask 丢弃）
- 用 APPG 生成的有效伪标签训练静态教师 $T^A$（BCE + Dice loss），捕获粗粒度结构先验
- 训练完成后冻结 $T^A$ 参数

### Step 3: 不确定性驱动的半监督学习

**Uncertainty-Entropy Weighted Fusion (UEWF)**：
- 静态教师 $T^A$（结构可靠，不自适应）和 EMA 动态教师 $T^B$（时序一致，可能有噪声）分别生成软伪标签
- 计算 Shannon 熵 → patch-wise 平均池化平滑（k=14） → 置信度权重为熵的倒数
- 加权融合：$\hat{y}_i^F = \frac{w_A \cdot \hat{y}_i^A + w_B \cdot \hat{y}_i^B}{w_A + w_B + \epsilon}$

**Adaptive Uncertainty-Guided Reverse Contrastive Learning (AURCL)**：
- 动态 top-K 阈值选择低置信度像素（自适应百分位 + 固定下界 $\tau_{fix}=0.2$）
- 对低置信度像素执行概率反转：$\tilde{p}(u,v) = 1 - p(u,v)$
- patch 级别特征提取 → 同位置原始/反转特征为正对，不同位置为负对
- InfoNCE 对比损失增强边界区域特征区分

### 损失函数

$$\mathcal{L} = \mathcal{L}_s + \lambda_u \mathcal{L}_u + \lambda_c \mathcal{L}_c$$

$\mathcal{L}_s$ 和 $\mathcal{L}_u$ 均为 BCE + Dice，$\lambda_u=1, \lambda_c=0.5$

## 实验关键数据

### BUSI 数据集（2.5% 标注 = 12 张标注图）

| 方法 | Dice(%) | IoU(%) | Acc(%) |
|------|---------|--------|--------|
| U-Net (全监督, 100%) | 81.68 | 73.74 | 96.65 |
| BCP (CVPR'23) | 58.93 | 49.48 | 93.89 |
| CSC-PA (CVPR'25) | 58.78 | 45.97 | 93.68 |
| Text-semiseg (MICCAI'25) | 56.85 | 45.35 | 93.13 |
| **Ours** | **72.72** | **63.11** | **95.08** |

### UBB 跨数据集（2.5% 标注 = 13 张标注图）

| 方法 | Dice(%) | IoU(%) | Acc(%) |
|------|---------|--------|--------|
| U-Net (全监督, 100%) | 74.81 | 65.56 | 97.29 |
| Text-semiseg (MICCAI'25) | 59.76 | 46.30 | 93.97 |
| **Ours** | **75.75** | **65.86** | **96.67** |

**关键发现**：
- 2.5% 标注下 Dice 提升 +13.79%（BUSI）和 +15.99%（UBB）相比前 SOTA
- UBB 上仅用 13 张标注图即超越全监督 U-Net（75.75% vs 74.81%），证明框架在极少标注下的强大能力
- 消融实验：APPG 贡献最大（Dice +14.09%），双教师 +3.83%，AURCL +0.47%，UEWF +0.52%
- 外观 prompt（"dark oval"等）远优于医学术语 prompt（"tumor"等）用于跨域定位

## 亮点

1. **跨域外观迁移的巧妙设计**：用简单外观描述（"dark oval"）代替医学术语 prompt，实现自然图像到医学图像的有效零样本迁移，思路简洁优雅
2. **极少标注下的突破性表现**：2.5% 标注即接近甚至超越全监督，临床实用价值极高
3. **双教师互补精炼**：静态教师提供结构先验，动态教师捕捉训练进展，不确定性加权融合两者优势
4. **范式可扩展性**：对其他成像模态或疾病，仅需一条全局外观描述即可获取可靠伪监督

## 局限性

1. 外观描述策略对形态高度一致的病灶（如 BUS 暗椭圆）效果好，但对形态复杂多变的病灶（如浸润性肿瘤）可能失效
2. Grounding DINO+SAM 管线在部分图像上仍会失败（需过滤无效 mask），伪标签覆盖率非 100%
3. 基于 ResNet-34 骨干网络，未探索更强的分割架构
4. 4 个 BUS 数据集的数据规模较小（BUSI 647 张），大规模验证缺乏
5. 无数据增强策略（作者有意为之），可能限制了与增强型方法的公平比较

## 相关工作

- **半监督分割**：Mean Teacher、U2PL（熵基置信度）、BCP（双向 copy-paste）、MCF（多级特征一致性）、PH-Net（困难区域学习）
- **VLM 辅助分割**：UniSeg（通用分割）、SAM-MediCLIPV2（医学域对齐）、CLIP-style SSL（但需大规模域内预训练）
- **BUS 特定方法**：PGCL（伪标签引导对比学习）、AAU（解剖感知不确定性）、Text-semiseg（CLIP 增强 SSL）
- 本工作的核心区别：无训练、无域内微调，仅靠外观 prompt 实现跨域迁移

## 评分

- 新颖性: ⭐⭐⭐⭐ — 外观 prompt 跨域迁移思路简洁有效，AURCL 反向对比设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 多数据集、多标注比例、全面消融、VLM baseline 对比
- 写作质量: ⭐⭐⭐⭐ — 方法描述清晰系统，动机论证充分
- 价值: ⭐⭐⭐⭐⭐ — 极少标注下的突破性性能具有重要临床意义，范式可扩展到其他医学成像模态
