# Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.12547](https://arxiv.org/abs/2603.12547)  
**代码**: 待发布 (upon acceptance)  
**领域**: 医学图像分割 / 状态空间模型 / 解码器设计  
**关键词**: 医学图像分割, Mamba, 解码器中心, 深度监督, Co-Attention Gate  

## 一句话总结
提出以解码器为核心的 Deco-Mamba 网络，用 Co-Attention Gate 双向融合编解码器特征、视觉状态空间模块（VSSM）建模长程依赖、可变形卷积恢复细节，并引入窗口化分布感知 KL 散度深度监督，在 7 个医学分割基准上以中等复杂度达到 SOTA。

## 背景与动机
医学图像分割领域存在两个未解决的瓶颈：(1) 大多数方法针对单一数据集/模态优化，跨模态泛化能力差；(2) 研究重心过度集中在编码器（用大型预训练 backbone），解码器设计长期被忽视。现有 Mamba 方法（Mamba-UNet, U-Mamba, Swin-UMamba 等）主要用 Mamba 增强编码器，没有充分发挥其在解码阶段的长程建模优势。传统深度监督通过 resize 中间输出到全分辨率，造成信息损失。

## 核心问题
如何设计一个计算高效且跨模态泛化的解码器，以低参数量实现精细的多尺度特征重建和边界恢复？

## 方法详解

### 整体框架
类 U-Net 结构，编码器双分支（7×7 CNN 保留高分辨细节 + PVT-V2 Transformer 捕捉全局依赖），解码器六阶段由 Co-Attention Gate → VSSM Block → 可变形残差块（DRB）级联组成，配合多尺度分布感知监督。两个版本：V0 (PVT-B0, 9.67M) 和 V1 (PVT-B2, 46.93M)。

### 关键设计
1. **Co-Attention Gate (CAG)**: 改进传统 Attention Gate 的单向门控——将编码器特征和解码器特征互为输入和门控信号，得到两路注意力输出后拼接，再经通道注意力（自适应最大+平均池化 → 双 1×1 卷积 → sigmoid）选择最具信息量的通道。公式：$D_i' = CA[AG(x=X_i, g=D_{i+1}), AG(x=D_{i+1}, g=X_i)]$
2. **视觉状态空间 Mamba 块（VSSMB）**: 采用连续时间 SSM，沿水平、垂直及其逆方向进行选择性扫描，以线性复杂度建模全局上下文。瓶颈层用 2 个 VSSMB，第 2-5 解码阶段各 1 个，最后阶段省略
3. **可变形残差块（DRB）**: 标准 3×3 卷积 + 可变形卷积，预测像素级偏移和调制掩码（sigmoid 约束到 [0,2]），恢复 SSM 可能平滑的局部细节和边界
4. **多尺度分布感知深度监督（MSDA）**: 不将中间输出 resize 到全分辨率，而是在每个解码器原生分辨率上计算窗口内类频率分布 $\tilde{P}^{(s)}$，与预测的 softmax 分布用 KL 散度对齐。边界加权：$W_{h,w}^{(s)} = (1 - \max_n \tilde{P}_{h,w,n}^{(s)})^\alpha$，混合类别区域（即边界附近）获得更高权重

### 损失函数 / 训练策略
$\mathcal{L}_{total} = \mathcal{L}_{dice} + \sum_s \lambda_s \mathcal{L}_{dist}^{(s)}$，各阶段权重递增 $\lambda_1 < \lambda_2 < ... < \lambda_S$。AdamW + cosine 学习率调度（warm restart $T=2$），224×224 输入，lr 1e-4 batch 16（主数据集），A5000 24GB GPU 训练。

## 实验关键数据
| 数据集 | 指标 | Deco-Mamba-V1 | 之前SOTA | 提升 |
|---|---|---|---|---|
| Synapse (8类) | DSC/HD95 | 85.07/14.72 | 83.59/15.99 (Cascaded-MERIT) | +1.48/+1.27 |
| BTCV (13类) | DSC/HD95 | 78.45/11.77 | 75.87/17.02 (PAG-TransYnet) | +2.58/+5.25 |
| ACDC (心脏) | DSC | 92.35 | 92.12 (PVT-EMCAD-B2) | +0.23 |
| MoNuSeg | DSC | 85.14 | 81.45 (Swin-UMamba) | +3.69 |
| GlaS | DSC | 96.91 | 96.91 (Cascaded-MERIT) | 持平 |

Deco-Mamba-V0 (仅 9.67M 参数) 的性能已接近 150M 级 Transformer 方法。

### 消融实验要点
- 去掉 CNN 分支：DSC 84.07（-1.0），去掉 VSSMB：DSC 83.51（-1.56）
- CAG vs 传统 AG：82.98 → 85.07，vs LGAG：82.69 → 85.07，vs CBAM：84.01 → 85.07
- 可变形卷积 vs 标准卷积：84.53 vs 85.07，vs 动态卷积：83.77 vs 85.07
- MSDA 深度监督 vs 传统深度监督：后者 DSC 提升但 HD95 恶化（15.89 vs 14.72），MSDA 两项都改善
- vs 边界感知/距离边界损失：HD95 分别为 21.43/20.64 vs MSDA 的 14.72
- 不同 backbone：PVT-B0 (9.67M) DSC 83.16，Swin-T (70.12M) DSC 83.76，PVT-B2 DSC 85.07

## 亮点
- 令人印象深刻的效率-精度平衡：V0 仅 9.67M 参数即超越 SliceMamba/VM-UNet 等 Mamba 方法，逼近 148M 的 Cascaded-MERIT
- MSDA 损失避免了传统深度监督中 resize 造成的信息损失，直接在原生分辨率操作
- 7 个跨模态基准（CT/MRI/超声/皮肤镜/病理）的全面验证证明泛化性

## 局限性 / 可改进方向
- 仅在 2D 切片上验证，未扩展到 3D 体积分割
- 依赖 PVT 预训练权重，未探索其他预训练策略（如自监督）
- VSSM 的多方向扫描策略选择缺少系统消融

## 与相关工作的对比
- vs EMCAD (CVPR 2024)：同为解码器增强方法，但 EMCAD 不含长程依赖建模，Deco-Mamba-V0 用 PVT-B0 即超越 EMCAD 用 PVT-B2 的结果
- vs Cascaded-MERIT (147.86M)：Deco-Mamba-V1 仅用约 1/3 参数即高 1.48% DSC
- vs Swin-UMamba：在 MoNuSeg 上高 3.69% DSC，参数量也更少

## 启发与关联
- 解码器中心设计的理念值得关注——用轻量编码器+强解码器可能比重编码器+轻解码器更高效
- 分布感知深度监督可推广到其他密集预测任务

## 评分
- 新颖性: ⭐⭐⭐ CAG、VSSM、MSDA 各自有增量，组合有效但单项创新有限
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个数据集、20+ 对比方法、全面消融
- 写作质量: ⭐⭐⭐⭐ 图表清晰，模块描述详细
- 价值: ⭐⭐⭐ 实用性强，对医学分割社区有价值，设计可推广
