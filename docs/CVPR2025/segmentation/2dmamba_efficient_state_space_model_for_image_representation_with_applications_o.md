# 2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification

**会议**: CVPR 2025  
**arXiv**: [2412.00678](https://arxiv.org/abs/2412.00678)  
**代码**: [https://github.com/AtlasAnalyticsLab/2DMamba](https://github.com/AtlasAnalyticsLab/2DMamba)  
**领域**: 医学图像 / 状态空间模型 / 全切片图像分类  
**关键词**: 2D SSM, Mamba, Whole Slide Image, MIL, Hardware-Aware  

## 一句话总结
提出2DMamba，首个具有高效并行算法的**原生2D选择性状态空间模型**，通过保持2D空间连续性（而非展平为1D序列）来建模WSI中的patch间关系，在10个公共病理数据集上全面超越1D Mamba方法，并在ImageNet分类和ADE20K分割上也有提升。

## 背景与动机
Giga-pixel全切片图像（WSI）分析是计算病理学的核心任务。传统MIL方法将WSI分割为patch独立聚合，忽略了空间关系。Transformer引入了patch间交互但面临二次复杂度问题。Mamba（选择性SSM）以线性复杂度+高并行性成为有力替代，但现有Vision Mamba方法都将2D图像展平为1D序列处理，导致"空间不连续"问题——相邻的patch在1D序列中可能距离很远，信息在传播中被遗忘（$\bar{A}$的高阶衰减）。另一方面，已有的2D SSM虽然保持2D结构，但缺乏高效并行算法，训练极慢。

## 核心问题
如何设计一个**既保持2D空间连续性、又具有高效并行实现**的选择性状态空间模型？现有方法要么展平为1D（快但丢失空间结构），要么用原生2D递归（保持结构但慢到不实用）。

## 方法详解

### 整体框架
2DMambaMIL的pipeline：输入WSI → 切分为patch → 用预训练特征提取器（UNI, ViT-L/16）提取patch特征 → 将非组织区域用可学习token填充形成2D特征图 → 送入$U$层2DMamba block → attention聚合器 → 分类/生存分析输出。

### 关键设计
1. **2D选择性SSM架构**：核心创新是将1D选择性扫描扩展为真正的2D扫描。先对每行独立做水平扫描（等价于1D Mamba），得到$h^{hor}_{i,j}$；再对水平扫描的结果做垂直扫描，得到最终状态$h_{i,j}$。展开后发现$h_{i,j} = \sum_{i'\leq i}\sum_{j'\leq j} \bar{A}^{(i-i'+j-j')} \bar{B} x_{i',j'}$，其中$\bar{A}$的阶数是曼哈顿距离而非1D展平距离，保持了2D空间连续性。例如对3×3特征图，(1,1)到(3,3)在2D中距离为4（$\bar{A}^4$），而1D展平中距离为8（$\bar{A}^8$），遗忘更严重。

2. **硬件感知2D并行扫描算子**：朴素的2D扫描需要在HBM上显式存储$N$个中间状态维度的特征图，内存复杂度$O(NL)$。2DMamba用2D tiling策略：将特征图分为2D网格块，每块加载到SRAM中完成水平+垂直扫描并在SRAM内完成$N$个状态维度的归约，只写回聚合结果到HBM。总内存访问复杂度保持$O(L)$，与1D Mamba相同。

3. **SegmentedBlockScan算法**：NVIDIA CUB的BlockScan只支持1D序列且要求大小为32的倍数。2DMamba提出SegmentedBlockScan，在多行多列间分配GPU线程（只要求$H \times W$是32的倍数而非各自），大幅减少padding浪费（14×14特征图从56%降至~1%）。

4. **可学习非组织区域填充**：WSI中大量区域是非组织背景。2DMamba用可学习token $p$代替固定零填充来表示这些区域，使模型能在训练中自适应学习背景表示。

### 损失函数 / 训练策略
- 使用AdamW优化器，初始学习率0.0001，cosine退火
- 训练20个epoch，batch size 1
- Patch大小512×512，20x放大倍率
- 特征提取器UNI（冻结），SSM维度128，状态维度$N=16$

## 实验关键数据

### WSI分类（5个数据集平均）
| 数据集 | 指标 | 2DMambaMIL | MambaMIL | 最佳非Mamba |
|--------|------|------------|----------|------------|
| PANDA | Acc | **0.5075** | 0.4679 | 0.5047 (S4-MIL) |
| PANDA | AUC | **0.8184** | 0.7781 | 0.7986 (S4-MIL) |
| TCGA-BRCA | Acc | **0.9458** | 0.9333 | 0.9375 (DSMIL) |
| TCGA-BRCA | AUC | **0.9782** | 0.9657 | 0.9770 (S4-MIL) |
| BRACS | F1 | **0.7045** | 0.6832 | 0.6131 (DTFD) |

### WSI生存分析（C-index）
| 数据集 | 2DMambaMIL | MambaMIL | 最佳baseline |
|--------|------------|----------|-------------|
| KIRC | **0.7311** | 0.7096 | 0.7271 (DTFD) |
| LUAD | **0.6198** | 0.5952 | 0.6157 (ABMIL) |
| STAD | **0.6428** | 0.6244 | 0.6244 (MambaMIL) |

### 自然图像
| 任务 | 方法 | 指标 |
|------|------|------|
| ImageNet-1K | 2DVMamba-T | **82.8%** (vs VMamba-T 82.6%) |
| ADE20K SS | 2DVMamba-T | **48.6 mIoU** (vs VMamba-T 47.9) |
| ADE20K MS | 2DVMamba-T | **49.3 mIoU** (vs VMamba-T 48.8) |

### 消融实验要点
- **可学习填充 vs 零填充**：可学习填充在PANDA上Acc提升1.56%，AUC提升0.62%
- **多方向1D扫描 vs 2D扫描**：4方向交叉扫描（MambaMIL最佳变体）Acc=0.4939, AUC=0.8006；2DMamba Acc=0.5075, AUC=0.8184，说明多方向1D终究不如原生2D
- **位置编码**：加PE反而降低2DMamba性能（Acc -1.0%, AUC -1.0%），说明2D结构已隐式编码位置
- **扫描顺序**：水平-垂直 vs 垂直-水平差异小（±0.5%），不敏感
- **块数$U$**：1层最优，多层略有下降
- **效率**：CUDA 2D扫描吞吐量约为1D的70%-90%，GPU内存与1D相当（线性），远优于Python实现（90%内存节省）

## 亮点
- **首个高效并行的原生2D Mamba**：通过数学推导将2D扫描分解为行扫描+列扫描的级联，巧妙保持了与1D相同的并行性
- **硬件感知设计**：2D tiling + SRAM内归约避免HBM上的中间状态存储，从算子层面解决效率问题
- **空间连续性的数学证明**：$\bar{A}$的阶数等于曼哈顿距离而非1D序列距离，优雅地解释了为什么2D扫描优于1D
- **通用性**：不仅适用于WSI，还能无缝集成到VMamba中提升自然图像任务

## 局限性 / 可改进方向
- 2D扫描只建模左上方到当前位置的信息流，需要附加反向扫描才能捕获完整上下文
- 吞吐量仍有10-30%的损失相比1D Mamba，对超大规模特征图（如高分辨率遥感）可能仍有瓶颈
- 仅验证了Tiny和Small规模的VMamba集成，更大规模（Base/Large）的效果未知
- 2D SSM的理论最优性（扫描顺序、$\bar{A}$共享策略等）缺乏理论保证

## 与相关工作的对比
- **MambaMIL / S4-MIL**：都是1D SSM，将WSI展平为序列。2DMamba通过原生2D处理避免空间断裂，在所有数据集上一致性地优于它们
- **2D-SSM (Baron et al.)**：早期的2D SSM工作，但缺乏选择性机制和高效并行算法，实际不可用。2DMamba同时解决了选择性和效率问题
- **VMamba**：用4方向1D扫描近似2D结构。2DMamba用原生2D替换其扫描模块后在分割任务上提升0.5-0.7 mIoU，说明多方向1D仍有缺陷

## 启发与关联
- 2D SSM的硬件感知设计思路可以推广到3D体积数据（CT/MRI），类似地分解为三个方向的级联扫描
- 与 `ideas/medical_imaging/20260317_pathology_wsol_debias.md` 有间接关联：2DMamba的注意力热图显示更好的肿瘤区域聚焦，可作为更好的弱监督定位基线

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个高效并行的原生2D选择性SSM，但核心思路（行扫描+列扫描分解）比较直觉
- 实验充分度: ⭐⭐⭐⭐⭐ 10个数据集 + 自然图像 + 丰富消融 + 效率分析 + 可视化，非常全面
- 写作质量: ⭐⭐⭐⭐ 清晰的motivation和数学推导，硬件部分略重但必要
- 价值: ⭐⭐⭐⭐ 解决了实际问题，代码开源，计算病理学社区有用
