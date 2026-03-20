# ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation

**会议**: CVPR 2025  
**arXiv**: [2503.15185](https://arxiv.org/abs/2503.15185)  
**代码**: [https://kuai-lab.github.io/cvpr2025protoocc](https://kuai-lab.github.io/cvpr2025protoocc)  
**领域**: 自动驾驶 / 3D占用预测 / 视角变换  
**关键词**: Occupancy Prediction, Prototype, View Transformation, Low-Resolution, Multi-Perspective Decoding  

## 一句话总结
提出ProtoOcc，通过**原型感知视角变换**将2D图像聚类原型映射到3D体素查询空间来增强低分辨率体素的上下文信息，配合**多视角占用解码**策略从增强的体素中重建高分辨率3D占用场景，用75%更小的体素分辨率仍能达到与高分辨率方法竞争的性能（Occ3D mIoU 37.80 vs PanoOcc 38.11）。

## 背景与动机
基于相机的3D占用预测（3DOP）中，体素查询的分辨率对视角变换质量至关重要。高分辨率体素查询性能好但计算量大（不利于实时部署），低分辨率查询省算力但信息损失严重。现有方法在视角变换时仅使用低级图像特征做2D-3D交叉注意力，当查询分辨率降低时，这些低级特征不足以支撑精确的3D场景重建。亟需一种在低分辨率查询上仍能编码丰富视觉上下文的方法。

## 核心问题
如何在低分辨率体素查询的约束下，保持高质量的2D到3D视角变换？即：如何用更少的参数/更小的空间编码并保留精确的3D语义占用信息？

## 方法详解

### 整体框架
多视角图像 → 图像骨干(ResNet50+FPN)提取多尺度特征 → **原型感知视角变换**（聚类图像为原型 → 映射到3D体素空间 → 对比学习优化原型质量）→ 得到原型增强的低分辨率体素查询 → **多视角占用解码**（体素增强 → 上采样 → 一致性正则化）→ 高分辨率3D占用预测

### 关键设计
1. **原型映射（Prototype Mapping）**：用超像素迭代聚类将图像特征分为$M$个原型（高级视觉结构表示，如布局和边界）。通过Feature Aggregation & Dispatch将2D原型投影到3D体素空间——计算原型-体素亲和度矩阵$\mathbf{A}$，用sigmoid归一化后做聚合（原型→体素）和分发（体素→原型感知查询）。这使得低分辨率体素查询携带了高级2D结构信息。

2. **原型优化（Prototype Optimization）**：标准3DOP损失不直接优化聚类质量。设计了基于伪掩码的对比学习——用SEEDS超像素或SAM生成伪语义掩码，计算原型感知像素特征与掩码质心的对比损失$\mathcal{L}_{cls}$，增强原型间的判别性。

3. **多视角占用解码（Multi-Perspective Occupancy Decoding）**：低分辨率→高分辨率是病态问题。用体素增强（特征级：Random Dropout/高斯噪声；空间级：转置/翻转）生成多个"视角"，共享权重的转置3D卷积上采样后，对不同增强版本的预测做一致性正则化$\mathcal{L}_{cons}$（分布接近锐化均值）。

### 损失函数 / 训练策略
$$\mathcal{L}_{total} = \sum_{p=0}^{P}(\lambda_1\mathcal{L}_{occ}^{(p)} + \lambda_2\mathcal{L}_{Lov}^{(p)}) + \lambda_3\mathcal{L}_{cls} + \lambda_4\mathcal{L}_{cons}$$
- ResNet50 backbone，12 epochs，输入432×800
- Base: 100×100×16, Small: 50×50×16, Tiny: 50×50×4

## 实验关键数据

### Occ3D-nuScenes验证集
| 方法 | 查询大小 | mIoU |
|------|---------|------|
| PanoOcc (Base) | 100×100×16 | 38.11 |
| **ProtoOcc (Base)** | 100×100×16 | **39.01** |
| PanoOcc (Small) | 50×50×16 | 35.78 |
| **ProtoOcc (Small)** | 50×50×16 | **37.80** |
| PanoOcc (Tiny) | 50×50×4 | 33.99 |
| **ProtoOcc (Tiny)** | 50×50×4 | **35.68** |

关键发现：ProtoOcc-Small (37.80) 与 PanoOcc-Base (38.11) 几乎持平，用75%更少的体素！

### 推理效率（Small查询）
| 方法 | 推理时间 | FLOPs(G) | 参数量 | mIoU |
|------|---------|----------|--------|------|
| PanoOcc-Base | 266ms | 1310 | 46.24M | 38.11 |
| ProtoOcc-Small | **105ms** | 378 | 16.11M | 37.80 |

推理时间节省60%，FLOPs节省71%，参数量减少65%。

### SemanticKITTI验证集
| 基线 | +ProtoOcc后 | IoU提升 | mIoU提升 |
|------|------------|--------|---------|
| VoxFormer-S | +ProtoOcc | +0.35 | +0.88 |
| VoxFormer-B | +ProtoOcc | +0.85 | +1.22 |
| Symphonies-S | +ProtoOcc | +1.35 | +0.86 |

ProtoOcc作为即插即用模块一致性提升所有基线。

### 消融实验要点
- **仅原型映射**：mIoU +0.02（映射本身帮助小）
- **+ 原型优化**：mIoU +0.77（对比学习至关重要）
- **仅多视角解码**：mIoU +1.47（体素增强+一致性正则化贡献大）
- **完整ProtoOcc**：mIoU +2.02
- **原型数量$M$**：350最优（太粗或太细都不好）
- **增强组合**：Random Dropout + 一致性正则化效果最好（37.25）

## 亮点
- **原型→体素的跨空间映射**：首次将2D图像原型表示引入3D占用预测的视角变换，在低分辨率约束下编码高级几何结构
- **计算效率显著**：75%更小的体素 → ~60%更快推理，性能几乎无损。这对自动驾驶实时部署极有价值
- **即插即用**：在SemanticKITTI上两种基线（VoxFormer、Symphonies）均有一致提升
- **多视角解码策略**：体素增强+一致性正则化是解决低分辨率→高分辨率病态问题的巧妙方案
- **注意力图可视化**：ProtoOcc关注小而关键的物体（行人、摩托车），baseline则被视觉主导区域吸引

## 局限性 / 可改进方向
- 原型聚类使用简单的超像素方法，更先进的聚类（如学习型）可能进一步提升
- 多视角解码中的增强组合需要手动实验选择
- 仅在nuScenes和SemanticKITTI验证，其他(如Waymo)未测试
- 时序信息利用有限

## 与相关工作的对比
- **PanoOcc**：标准体素查询+deformable attention。ProtoOcc在相同查询大小下mIoU高~2%，在75%更小查询下性能持平
- **COTR**：先用大查询再下采样。ProtoOcc直接用小查询，更高效
- **DFA3D**：增强视角变换的另一方法。ProtoOcc-Small比DFA3D更好（37.80 vs 36.27），且推理时间更短（105ms vs 153ms）

## 启发与关联
- "原型映射"的思路可推广到其他需要2D-3D转换的任务（如3D检测、3D分割）
- 多视角解码的增强+一致性思路类似于自监督学习中的multi-crop+consistency，可推广到其他3D预测任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 原型→体素映射和多视角体素解码均为新颖设计，但每个组件本身借鉴了已有技术
- 实验充分度: ⭐⭐⭐⭐⭐ 两个基准+多分辨率设定+效率分析+丰富消融+可视化，非常全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰、方法描述系统，图示质量高
- 价值: ⭐⭐⭐⭐⭐ 解决了3DOP实时部署的核心效率问题，75%减少计算几乎无损性能
