# Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians

**会议**: ICCV 2025  
**arXiv**: [2508.01464](https://arxiv.org/abs/2508.01464)  
**代码**: [https://github.com/Zerg-Overmind/Can3Tok](https://github.com/Zerg-Overmind/Can3Tok)  
**领域**: 3D视觉 / 3D生成 / 3DGS  
**关键词**: 场景级3DGS, VAE, 3D tokenization, 潜在空间建模, 扩散生成  

## 一句话总结
提出Can3Tok——首个场景级3DGS VAE：通过cross-attention将大量(40K)无序3D Gaussian压缩到低维canonical token(256×768→64×64×4) + 3DGS归一化解决跨场景尺度不一致 + 语义感知过滤去除floater噪声，在DL3DV-10K上唯一成功的场景级3DGS潜在建模方法(L2=30.1, 失败率2.5%)，支持text-to-3DGS和image-to-3DGS前馈生成。

## 背景与动机
3D生成主要停留在物体级别，场景级前馈生成因缺乏可扩展的3DGS潜在表示学习而受限。核心挑战：(1) 3DGS高度非结构化(异质特征+不规则空间分布+大量floater)，现有VAE(PointNet/L3DG)无法处理；(2) 场景间全局尺度和各Gaussian缩放值不一致(COLMAP非度量化)，使大规模训练无法收敛。

## 核心问题
如何为无序、非结构化、尺度不一致的场景级3DGS表示建立有效的低维潜在空间？

## 方法详解

### 整体框架
输入3DGS(40K Gaussians/场景) → 归一化到单位球 + 语义过滤 → Fourier位置编码 + 最近体素坐标附加 → Cross-attention(learnable canonical query 256×768) → 8层Self-attention → VAE重参数化(μ, logσ²) → 潜在z ∈ 64×64×4 → 16层Self-attention decoder → MLP → 重建3DGS

### 关键设计
1. **Canonical Query + Cross-attention**: 将10K+ Gaussians投影到仅256个learnable token(初始化为规则体素坐标)的低维空间，解决self-attention在大量Gaussian上的OOM问题。canonical query提供结构化几何先验。
2. **3DGS归一化**: 所有场景的Gaussian中心平移到原点 + 缩放到半径r的球内 + 同步缩放Gaussian的scaling参数和相机中心。保证不同场景有统一尺度，关键是使渲染结果在变换前后一致。
3. **语义感知过滤**: 用LangSam("most salient region")在中间帧提取语义区域 → 从mask内选种子Gaussian → KNN扩展到N个 → 去除floater和观测不足区域的噪声Gaussian。
4. **数据增强**: 训练时施加随机SO(3)旋转增强多样性。

### 训练
- 损失: L2重建 + λ·KL散度 (λ=1e-6)
- 潜在空间: z ∈ 64×64×4 (与Stable Diffusion相同大小)
- 8 A100 GPU训练5天
- 编码解码一次仅需~0.06秒

## 实验关键数据

### 场景级3DGS重建(DL3DV-10K测试集)
| 方法 | L2↓ | 失败率↓ |
|------|-----|--------|
| L3DG | 1200.4 | 100% |
| PointNet VAE | 1823.0 | 100% |
| PointTransformer | 230.7 | 70% |
| **Can3Tok** | **30.1** | **2.5%** |

其他所有方法在场景级数据上完全失败(100%或70%失败率)

### 消融实验
| 变体 | L2↓ | 失败率↓ |
|------|-----|--------|
| w/o Learnable Query | - | 100% |
| w/o 归一化 | 1889.7 | 100% |
| w/o 体素坐标附加 | 50.5 | 4.3% |
| w/o 数据过滤 | 73.3 | 6.1% |
| w/o 数据增强 | 53.3 | 4.6% |
| Can3Tok(完整) | 30.1 | 2.5% |

- 归一化和learnable query是必要条件(缺少则100%失败)
- 过滤和增强带来2x+改善

### 生成质量
- Text-to-3DGS FID: 28.32 (PointTransformer: 153.76)
- 训练速度: 1.1 s/iter (L3DG: 11.3 s/iter, 10x更快)

## 亮点 / 我学到了什么
- **尺度归一化是3D表示学习的基础**: 没有归一化，即使最好的模型也100%失败。与2D图像归一化到[-1,1]类似的insight
- **Cross-attention比self-attention更适合非结构化3D数据**: 通过learnable query将无穷维投影到有限维，类似PerceiverIO的设计但用于3DGS
- **潜在空间结构分析**: t-SNE可视化显示同一场景不同旋转在潜在空间形成闭环，证明模型学到了空间结构而非简单记忆
- **与Stable Diffusion兼容**: 潜在空间大小64×64×4完全匹配，可直接接入现有扩散架构

## 局限性 / 可改进方向
- 2.5%失败率来自训练数据中低质量3DGS重建(运动模糊/视角不均)
- 仅限3DGS表示
- 语义过滤可能丢弃重要场景内容

## 与相关工作的对比
- **vs L3DG**: 3D卷积方案+Minkowski Engine，仅适用于物体级；Can3Tok在场景级L3DG完全失败的情况下成功
- **vs Bolt3D**: 并行工作也发现3D CNN在场景级失败
- **vs WonderJourney/LucidDreamer**: 2D扩散驱动的场景生成，逐场景优化慢；Can3Tok支持前馈生成

## 与我的研究方向的关联
- 3DGS潜在建模是3D生成的关键瓶颈
- 归一化+过滤的数据处理pipeline可迁移到其他3D表示学习任务
- 与Stable Diffusion兼容的潜在空间设计为3D生成提供了实用框架

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个场景级3DGS VAE + 系统性解决尺度不一致问题
- 实验充分度: ⭐⭐⭐⭐ DL3DV-10K大规模验证+完整消融+潜在空间分析+两种生成应用
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻(为何现有方法失败)，实验设计合理
- 对我的价值: ⭐⭐⭐⭐ 3DGS生成方向的重要进展，归一化/tokenization策略有实用参考价值
