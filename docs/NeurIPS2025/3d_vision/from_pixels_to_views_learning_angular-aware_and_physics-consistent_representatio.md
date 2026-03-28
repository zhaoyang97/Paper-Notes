<!-- 由 src/gen_stubs.py 自动生成 -->
# From Pixels to Views: Learning Angular-Aware and Physics-Consistent Representations for Light Field Microscopy

**会议**: NEURIPS2025  
**arXiv**: [2510.22577](https://arxiv.org/abs/2510.22577)  
**代码**: [GitHub](https://github.com/hefengcs/XLFM-Former)  
**领域**: 3d_vision / microscopy  
**关键词**: 光场显微镜, XLFM, 3D重建, Masked View Modeling, 物理一致性  

## 一句话总结
提出XLFM-Former用于扩展光场显微镜(XLFM)的3D重建：构建首个XLFM-Zebrafish标准化基准，设计Masked View Modeling (MVM-LF)自监督预训练学习角度先验，引入光学渲染一致性损失(ORC Loss)确保物理可信性，PSNR较SOTA提升7.7%（54.04 vs 50.16 dB）。

## 背景与动机

1. **领域现状**：XLFM可单次曝光在100Hz实现体积成像，是神经科学大规模活体成像的关键工具（斑马鱼、小鼠）。
2. **核心挑战**：(1) 缺乏标准化数据集和评估协议；(2) XLFM数据编码密集角度采样的3D场景，传统CNN难以建模角度相关性；(3) 高质量体积GT（RL反卷积）计算昂贵。
3. **物理约束缺失**：纯像素级损失训练可能生成视觉合理但光学不一致的重建。

## 方法详解

### 整体框架
Swin Transformer编码器 + CNN解码器，加MVM-LF预训练 + ORC Loss物理约束。

### 关键设计1: Masked View Modeling (MVM-LF)
- XLFM的27个视角中随机遮挡70%，让模型从未遮挡视角重建被遮挡视角
- 以**视角**（而非像素）为掩码单元——匹配XLFM的角度采样物理结构
- 仅用 $\ell_2$ 损失预训练250 epochs，预训练后丢弃解码器，保留编码器初始化

### 关键设计2: 光学渲染一致性损失 (ORC Loss)
- 将预测3D体积通过已知PSF前向卷积得到合成光场图像
- $\mathcal{L}_{ORC} = \|h * \mathcal{V}_{pred} - h * \mathcal{V}_{GT}\|_2^2$
- 确保重建不仅结构匹配GT，还在PSF前向模型下光学一致

### 关键设计3: XLFM-Zebrafish基准数据集
- 22,581张光场图像，3条自由游泳斑马鱼 + 13条固定斑马鱼
- 训练/验证7条 + 测试6条（unseen）
- 双采样率：10fps（高时间分辨）+ 1fps（长期跟踪）

## 实验关键数据

### XLFM-Zebrafish测试集（6个样本平均）

| 方法 | Avg PSNR↑ | Avg SSIM↑ |
|------|-----------|-----------|
| ConvNeXt | 50.16 | 0.9876 |
| ViT | 49.28 | 0.9876 |
| U-Net | 50.60 | 0.9886 |
| ResNet-101 | 50.68 | 0.9893 |
| **XLFM-Former** | **54.04** | **0.9944** |

PSNR提升7.7%（54.04 vs 50.16），在所有6个测试样本上全面超越。

### 消融实验
- 无MVM-LF预训练：-1.2 dB
- 无ORC Loss：-0.8 dB
- 两者均去除：-2.1 dB

## 亮点
1. **首个XLFM标准化基准**：填补了该领域数据集空白
2. **视角级Masked Modeling**：比像素级掩码更匹配光场物理结构
3. **可微渲染物理约束**：ORC Loss桥接数据驱动学习与波光学一致性
4. **全体积重建**：不限于稀疏神经信号，还包含完整形态结构

## 局限性 / 可改进方向
1. 仅在斑马鱼数据上验证，小鼠等更大组织的泛化待测
2. 需要4×A100-80GB训练——计算资源要求高
3. ORC Loss依赖已知PSF——对PSF误差的敏感性未分析

## 与相关工作的对比
- **vs XLFMNet**：仅重建稀疏神经信号，本文做全体积重建
- **vs FNet**：Fourier卷积内存爆炸（需多GPU），Swin Transformer更高效
- **vs MLFM**：像素级随机掩码不如视角级掩码匹配光场结构

## 启发与关联
- 视角级自监督预训练思路可推广到其他多视角成像系统（光场相机、NeRF采集）
- 可微渲染损失适用于任何前向模型已知的逆问题
- XLFM+深度学习的组合对实时全脑成像有重要意义

## 评分
- 新颖性: ⭐⭐⭐⭐ 视角级MAE + 光学一致性损失
- 实验充分度: ⭐⭐⭐⭐ 首个基准+多架构对比+消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法-物理结合紧密
- 价值: ⭐⭐⭐⭐ 计算神经科学的重要基础设施
