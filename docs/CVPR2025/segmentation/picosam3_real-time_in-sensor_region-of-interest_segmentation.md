# PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation

**会议**: CVPR 2025  
**arXiv**: [2603.11917](https://arxiv.org/abs/2603.11917)  
**代码**: [GitHub](https://github.com/pbonazzi/picosam3)  
**领域**: 分割 / 边缘部署  
**关键词**: 传感器内计算, SAM 蒸馏, INT8 量化, 轻量分割, Sony IMX500, ROI 提示

## 一句话总结
PicoSAM3 是一个 1.3M 参数的超轻量可提示分割模型，通过 ROI 隐式提示编码、密集 CNN 架构（无 Transformer）、SAM3 知识蒸馏和 INT8 量化，在 COCO 上达 65.45% mIoU，并实现在 Sony IMX500 视觉传感器上 11.82ms 实时推理。

## 研究背景与动机
1. **领域现状**：SAM/SAM2/SAM3 系列在可提示分割取得突破性进展，但其 Transformer 架构的计算量和内存占用使其无法部署在极端边缘设备上。
2. **现有痛点**：（1）TinySAM、EdgeSAM 等压缩方案仍保留 Transformer 结构，内存超 8MB 限制且依赖不受支持的算子（Softmax、LayerNorm）；（2）传感器内计算（In-sensor computing）要求模型完全在 CMOS 传感器内运行，对算子类型、整数精度、内存有极严格限制；（3）现有轻量 SAM 忽略 ROI 的空间灵活性。
3. **核心矛盾**：如何在 <8MB SRAM、仅支持量化友好算子的传感器芯片上，实现高质量可提示分割？
4. **切入角度**：完全放弃 Transformer，采用纯 CNN 架构 + ROI 隐式提示 + SAM3 蒸馏 + INT8 量化，专门为传感器内部署设计。

## 方法详解

### 整体框架
训练：COCO 数据 → 对每个标注实例提取 ROI 裁剪（10% padding, 正方形, 96×96）→ SAM3 teacher 对同一 bbox 生成 soft mask（离线缓存）→ PicoSAM3 student 仅看裁剪 RGB 图学习分割 → 多损失蒸馏训练。推理：用户在 IMX500 上拖选 ROI → 传感器硬件裁剪 → INT8 模型推理 → 返回分割掩码。

### 关键设计

1. **隐式提示编码（Implicit Prompt via Centered Cropping）**:
   - 做什么：将 bbox 提示隐式地通过裁剪编码到 RGB 输入中
   - 核心思路：训练时根据 bbox 提取带 10% padding 的正方形裁剪，resize 到 96×96。目标物体始终近似居中，网络学习分割输入中心附近的主要物体。推理时利用 IMX500 的硬件 ROI 功能实现同样的裁剪
   - 设计动机：IMX500 仅支持 RGB 输入，无法传入额外的提示张量，因此只能通过空间裁剪隐式编码提示

2. **模型架构（Dense CNN U-Net）**:
   - 做什么：1.37M 参数的纯 CNN 对称编解码器
   - 核心思路：基于 PicoSAM2 的 U-Net 结构（深度可分离卷积，通道 48→96→160→256→320），新增三项改进：（1）带膨胀深度卷积的增强 bottleneck（dilation=2 扩大感受野）；（2）输出头前的 Efficient Channel Attention（ECA）块实现自适应特征重校准；（3）带深度卷积的细化头改善边界
   - 设计动机：完全避免 Transformer 的 Self-Attention 算子，确保所有操作兼容 INT8 量化和 IMX500 的有限算子集

3. **SAM3 知识蒸馏**:
   - 做什么：从 1.2GB 的 SAM3 teacher 蒸馏到 5.26MB 的 student
   - 核心思路：离线缓存 teacher 对所有 COCO 标注的 soft mask → 三损失联合训练：（1）teacher 损失 $\mathcal{L}_{teacher}$ = MSE + Dice（温度缩放 $\tau=5$）；（2）GT 损失 $\mathcal{L}_{gt}$ = BCE + Dice；（3）面积保持损失 $\mathcal{L}_{area}$（防止预测掩码面积塌缩到 GT 的 40% 以下）
   - 自适应权重：总损失中 teacher 和 GT 的权重由 teacher 置信度自动调节——高置信时依赖 teacher soft label，低置信时回退到 GT

4. **INT8 后训练量化**:
   - 做什么：4× 模型压缩（5.26MB → 1.31MB），精度损失 <0.2%
   - 核心思路：用 Sony MCT 工具链进行对称逐通道权重量化 + 逐张量激活量化，10 batch 校准
   - 设计动机：深度可分离卷积对量化噪声天然鲁棒，无需复杂的 outlier 抑制算法

## 实验关键数据

### 主实验（COCO / LVIS）

| 模型 | 参数量 | 大小 | COCO mIoU | LVIS mIoU | IMX500 延迟 |
|------|--------|------|-----------|-----------|------------|
| SAM-H | 635M | 2420MB | 53.6% | 60.5% | — |
| TinySAM | 9.7M | 37MB | 50.9% | 52.1% | — |
| EdgeSAM | 9.7M | 37MB | 48.0% | 53.7% | — |
| PicoSAM2 | 1.3M | 4.87MB | 51.9% | 44.9% | — |
| Q-PicoSAM2 | 1.3M | 1.22MB | 50.5% | 45.1% | 14.3ms |
| **PicoSAM3** | **1.37M** | **5.26MB** | **65.45%** | **64.01%** | — |
| **Q-PicoSAM3** | **1.37M** | **1.31MB** | **65.34%** | **63.98%** | **11.82ms** |

- PicoSAM3 参数量仅为 SAM-H 的 1/460，COCO mIoU 反超 +11.85%
- 相比 PicoSAM2，COCO 提升 +13.55%，LVIS 提升 +19.11%
- INT8 量化几乎无损（-0.11% mIoU），IMX500 上约 84 FPS

### 消融实验

| 模型 | Distillation | ROI | COCO mIoU | LVIS mIoU |
|------|--------------|-----|-----------|-----------|
| PicoSAM2 | ✗ | ✗ | 53.00% | 41.40% |
| PicoSAM2 | SAM2 | ✗ | 51.93% | 44.88% |
| PicoSAM2 | SAM2 | ✓ | 63.11% | 61.80% |
| PicoSAM2 | SAM3 | ✓ | 63.51% | 62.31% |
| PicoSAM3 | SAM3 | ✓ | 65.45% | 64.01% |

- ROI 裁剪是最关键的提升因素（mAP +10.56%），证实空间聚焦对低分辨率模型至关重要
- SAM3 蒸馏优于 SAM2（+0.4% mIoU），PicoSAM3 架构改进再获 +1.9%

### 关键发现
- 大模型（SAM2.1 Large）在 96×96 输入上反而性能崩溃（~25% mAP），因为分辨率不匹配
- 纯 CNN 架构的激活分布紧凑近高斯，对 INT8 量化天然友好
- ROI 裁剪 + 居中隐式提示可完美对接 IMX500 的硬件 ROI 功能

## 亮点与洞察
- **传感器级 SAM**：首次在 CMOS 传感器内实现可提示分割，终端延迟 11.82ms（~84FPS），具有极高的实用价值
- **逆直觉发现**：1.3M 参数模型在 96×96 输入上超越 635M 的 SAM-H，说明模型-分辨率匹配比绝对模型大小更重要
- **隐式提示设计**：通过裁剪实现提示功能，无需额外网络或输入通道，优雅地绕过了硬件限制

## 局限性 / 可改进方向
- 96×96 输入分辨率限制了精细边界质量，对大物体或复杂形状可能不够
- 隐式 ROI 提示仅支持矩形框区域，不支持点/文本等提示类型
- 仅在 COCO/LVIS 验证，缺少医疗/遥感等领域迁移实验
- 训练仅 1 epoch on COCO，更长训练可能进一步提升

## 相关工作与启发
- **vs TinySAM/EdgeSAM**: 参数量少 7×，mIoU 高 14+%，差距巨大
- **vs PicoSAM2**: 架构改进（ECA + 膨胀 bottleneck）+ SAM3 蒸馏 + ROI 提示三者共同贡献 +13.55%
- **vs FastSAM**: mIoU +13.85%，模型仅 1.9% 大小

## 评分
- 新颖性: ⭐⭐⭐⭐ 传感器内分割是新颖且有实际意义的方向，ROI 隐式提示设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 消融全面（蒸馏/ROI/架构/量化），有真实硬件部署验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，硬件背景介绍充分
- 价值: ⭐⭐⭐⭐⭐ 极端边缘部署方向有高实用价值，方法完整可复现
