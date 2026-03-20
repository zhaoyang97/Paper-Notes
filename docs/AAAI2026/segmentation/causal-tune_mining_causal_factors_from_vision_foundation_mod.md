# Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation

**会议**: AAAI 2026  
**arXiv**: [2512.16567](https://arxiv.org/abs/2512.16567)  
**代码**: [https://github.com/zhangyin1996/Causal-Tune](https://github.com/zhangyin1996/Causal-Tune)  
**领域**: 语义分割  
**关键词**: 域泛化语义分割, 因果推断, 频域分析, VFM微调, DCT带通滤波  

## 一句话总结
提出Causal-Tune，从因果视角分析VFM特征中的artifacts，利用DCT频域分解+高斯带通滤波分离因果/非因果因素，结合因果感知可学习token在频域精化特征，在Cityscapes→ACDC跨域分割中平均提升+2.4% mIoU（Snow场景+4.8%），仅需单卡RTX3090/14GB训练。

## 背景与动机
VFM（DINOv2、CLIP等）经过大规模预训练后，在DGSS中通过PEFT微调表现优异。但这些VFM的特征中存在明显的artifacts（如feature map中的条纹/异常高亮），这源于长期预训练引入的冗余特征。现有adapter-based PEFT方法不加区分地微调所有特征，无法有效抑制这些artifacts，限制了泛化性能。作者发现这些artifacts与非因果因素相关——在DCT频域中，非因果因素（显式的如雨雪雾夜、隐式的如亮度/模糊/噪声）集中在极高频和极低频分量中。

## 核心问题
如何在VFM微调过程中有效识别和分离因果因素（域不变的结构/纹理信息）与非因果因素（域特定的style/weather信息），以增强域泛化能力？现有频域方法（FFT、HWT）不能很好地分离隐式非因果因素，需要更有效的频域工具。

## 方法详解

### 整体框架
冻结VFM（DINOv2），在每一层的特征输出上插入Causal-Tune模块：特征→DCT转到频域→高斯带通滤波分离因果/非因果分量→丢弃非因果分量→因果感知token通过注意力精化因果分量→iDCT回空间域→添加到原特征作为残差。使用Mask2Former作为分割头。

### 关键设计
1. **Causal & Non-causal Factors Filter**: 对每层特征$f_i$做2D DCT转换到频域，然后用高斯带通滤波器$G(u,v) = \exp(-\frac{u^2+v^2}{2R_H^2}) - \exp(-\frac{u^2+v^2}{2R_L^2})$分离频谱。低频（<$R_L$=0.2）和高频（>$R_H$=0.7）被视为非因果因素直接丢弃，中间频段保留为因果因素。实验验证：雾/雨主要在高频，夜景主要在低频，雪同时在高低频，带通滤波是最优策略。

2. **Causal Factors Tune**: 引入因果感知可学习token $T_i^{cau} = B_i A_i$（低秩分解，参数量极小），通过注意力机制与因果特征$F_i^{cau}$交互。因果特征作为Query，token作为Key/Value，计算注意力权重后通过MLP投影并加残差连接精化因果分量，最后iDCT转回空间域。

3. **DCT vs FFT vs HWT**: 选择DCT而非FFT/HWT的原因是DCT对因果/非因果因素的分离效果更好——实验显示DCT的ACDC平均72.0% vs FFT 69.5% vs HWT 69.2%。

### 损失函数 / 训练策略
标准交叉熵分割损失。AdamW优化器，lr=1e-4，batch=4，40k iterations，单卡RTX3090（14GB显存）。$R_L$=0.2, $R_H$=0.7。

## 实验关键数据

| 设置 | 指标(mIoU) | Causal-Tune | Rein(baseline) | SET | FADA |
|--------|-----------|-------------|----------------|-----|------|
| C→ACDC Night | mIoU | 56.2 | 55.9 | 57.3 | 57.4 |
| C→ACDC Snow | mIoU | **75.4** | 70.6 | 73.6 | 73.5 |
| C→ACDC Fog | mIoU | **81.3** | 79.5 | 80.1 | 80.2 |
| C→ACDC Rain | mIoU | **75.2** | 72.5 | 74.8 | 75.0 |
| C→ACDC Avg | mIoU | **72.0** | 69.6 | 71.5 | 71.5 |
| C→BDD100K | mIoU | **66.28** | 63.54 | 65.07 | 65.12 |
| C→Mapillary | mIoU | **76.05** | 74.03 | 75.67 | 75.86 |
| G→Mapillary | mIoU | **68.21** | 66.10 | 67.68 | 68.09 |

Snow场景提升最大达+4.8%，因为雪的非因果因素同时存在于高低频，带通滤波精确去除。

### 消融实验要点
- 频域变换对比：DCT(72.0 avg) >> FFT(69.5) ≈ HWT(69.2)
- 滤波方式：只去低频(68.4) < 只去高频(70.4) < 带通滤波(72.0)，证明高低频非因果因素都需要去除
- 截止频率敏感性：$R_L \leq 0.2$, $R_H \leq 0.8$范围内表现较好，超出则退化
- 局限：合成→真实(G→C)方向提升有限(66.22 vs FADA 68.23)

## 亮点
- **因果视角解释VFM artifacts**很有洞察力——将artifacts与非因果因素建立联系，在频域验证它们集中在极端频段
- **方法极其简洁优雅**：DCT+带通滤波+learnable token，概念清晰，实现简单
- **训练资源友好**：单卡RTX3090/14GB即可，对比同类VFM-based方法非常经济
- Snow场景+4.8%的提升很impressive，因为带通滤波同时去除了高低频的非因果因素
- DCT优于FFT/HWT的实验发现有参考价值——DCT的实数域表示可能更适合CV中的因素分离

## 局限性 / 可改进方向
- 带通滤波的截止频率$R_L, R_H$是固定的，不同天气/域偏移可能需要不同参数→作者也提到要探索动态截止频率
- 合成→真实场景(G→C)效果有限，说明方法对极大域偏移的适用性需要加强
- 仅在驾驶场景数据集上验证，未扩展到室内/医学等其他域泛化场景
- 没有与LoRA等主流PEFT方法的组合探索

## 与相关工作的对比
- **vs Rein(baseline)**: Causal-Tune在Rein基础上增加因果频域处理，ACDC平均+2.4%，BDD100K+2.74%
- **vs SET**: SET也在频域做learnable token但用FFT，未区分因果/非因果，Causal-Tune通过DCT+带通滤波全面超越
- **vs FADA**: FADA用HWT解耦style信息，但忽略了隐式非因果因素（如亮度、模糊），Causal-Tune通过因果分析更全面
- **vs MAD**: MAD用数据增强去除非因果因素，Causal-Tune直接在频域操作VFM特征，更直接

## 启发与关联
- DCT频域的因果/非因果分离思路可以迁移到目标检测、实例分割的域泛化中
- "极端频段=非因果"这个洞察在其他VFM微调场景也可能成立
- 可以与InfoCLIP的思路结合——信息论+因果论双重视角优化VFM微调

## 评分
- 新颖性: ⭐⭐⭐⭐ 因果+频域+VFM PEFT的组合是新颖的，但各组件本身不新
- 实验充分度: ⭐⭐⭐⭐ 多个跨域设置+频域变换对比+滤波方式消融+可视化，但只有驾驶场景
- 写作质量: ⭐⭐⭐⭐ motivation的可视化说服力强，方法描述清晰
- 价值: ⭐⭐⭐⭐ 对恶劣天气下的域泛化分割有显著价值，方法简洁易复现
