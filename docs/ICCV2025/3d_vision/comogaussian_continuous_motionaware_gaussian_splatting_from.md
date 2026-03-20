# CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images

**会议**: ICCV 2025  
**arXiv**: [2503.05332](https://arxiv.org/abs/2503.05332)  
**代码**: [https://github.com/Jho-Yonsei/CoMoGaussian](https://github.com/Jho-Yonsei/CoMoGaussian) (项目页面: [https://Jho-Yonsei.github.io/CoMoGaussian/](https://Jho-Yonsei.github.io/CoMoGaussian/))  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, motion blur, neural ODE, camera trajectory, deblurring  

## 一句话总结
用Neural ODE建模曝光时间内的连续相机运动轨迹，结合刚体变换和可学习的连续运动修正(CMR)变换，从运动模糊图像重建清晰3D高斯场景，在所有benchmark上大幅超越SOTA。

## 背景与动机
3DGS依赖清晰图像输入，但现实中小光圈→长曝光→相机运动模糊几乎不可避免。现有去模糊方法（Deblur-NeRF、DP-NeRF、BAD-Gaussians等）虽然尝试估计相机运动轨迹来模拟模糊过程，但都没有真正保证运动的**连续性**——要么用简单的线性插值/样条函数，要么直接预测离散位姿。这导致预测的运动可能出现突变或分段不连续，偏离真实的物理相机轨迹，尤其在复杂非线性运动下效果不佳。

## 核心问题
如何在3DGS框架下准确建模曝光时间内**连续的**相机运动轨迹，使得从运动模糊图像也能重建出高质量清晰场景？难点在于：(1)运动模糊本质上是连续时间积分，离散采样会引入误差；(2)刚体变换虽保几何一致性但在有限采样数下不够精确。

## 方法详解

### 整体框架
输入运动模糊图像集 + COLMAP位姿。对每张图像，以其校准位姿为中心点，用Neural ODE在曝光时间区间内生成N=9个连续相机位姿。从每个位姿通过Mip-Splatting渲染一张清晰图，再用像素级加权求和合成模糊图像，与输入模糊图像做监督。推理时直接从校准位姿渲染即得清晰图。

### 关键设计
1. **连续刚体运动建模**: 将图像索引嵌入为特征→编码器得到初始潜在状态→Neural ODE（4阶Runge-Kutta求解器）在潜在空间生成连续的螺旋轴(screw axis)参数，再通过矩阵指数映射到SE(3)变换矩阵。关键改进：将旋转轴ω̂和旋转角θ解耦独立建模（DP-NeRF中二者是耦合的），旋转轴归一化到单位向量。
2. **连续运动修正(CMR)变换**: 由于数值积分是离散的（只有N个采样），刚体变换不足以完美近似连续运动。CMR通过另一个Neural ODE生成额外的自由度更高的变换矩阵（非严格SO(3)），初始化接近单位矩阵，加正交性正则化确保接近有效旋转。本质上是对刚体运动的"残差修正"。
3. **像素级权重和掩码**: 用浅层CNN为N张渲染图生成像素级softmax权重来合成模糊图。另有标量掩码混合清晰图和模糊图——对于本身不模糊的像素区域直接用清晰渲染。
4. **Neural ODE的优势**: 前向和反向传播共享同一个神经导数函数→保证整个轨迹在同一函数空间内→自然的时间连续性。对比MLP（无序列性）和GRU（前后向需要独立单元导致不连续），Neural ODE生成的轨迹视觉上最平滑。

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c\mathcal{L}_{\text{D-SSIM}} + \lambda_o\mathcal{L}_o + \lambda_\mathcal{M}\mathcal{L}_\mathcal{M}$
- $\lambda_c=0.3$, $\lambda_o=10^{-4}$, $\lambda_\mathcal{M}=10^{-3}$
- 分阶段训练：前1k迭代只训练高斯原语；1k-3k加入运动变换但不加像素权重/掩码；3k后全部组件一起训练
- 总共40k迭代，单卡RTX 4090

## 实验关键数据
| 数据集 | 指标 | CoMoGaussian | Deblurring 3DGS | BAGS | 前SOTA提升 |
|--------|------|-------------|-----------------|------|-----------|
| Deblur-NeRF Synthetic | PSNR/SSIM/LPIPS | **31.02/0.917/0.049** | 28.24/0.858/0.105 | 27.34/0.835/0.112 | +2.78 PSNR |
| Deblur-NeRF Real | PSNR/SSIM/LPIPS | **27.85/0.843/0.082** | 26.61/0.822/0.110 | 26.70/0.824/0.096 | +1.15 PSNR |
| ExbluRF Real | PSNR/SSIM/LPIPS | **30.15/0.756/0.311** | 27.36/0.680/0.399 | 24.70/0.584/0.528 | +2.79 PSNR |

### 消融实验要点
- 刚体变换本身就比baseline(Mip-Splatting)提升5+ PSNR
- CMR在刚体变换基础上进一步提+0.55/+0.02/+0.013(PSNR/SSIM/LPIPS)
- 正交性正则虽数值影响小(+0.11 PSNR)但定性上保留更多细节（防止shearing/scaling畸变）
- Neural ODE > GRU (+0.45 PSNR) > MLP (+0.42 PSNR)，大幅优于物理空间方法(线性插值21.0, B样条21.7)
- N=9时性能接近饱和，CMR使得N=9超过了纯刚体N=13的效果
- 训练速度1.33h（比其他3DGS方法慢，但渲染速度相同）

## 亮点 / 我学到了什么
- **Neural ODE用于相机运动建模是一个非常自然的选择**：连续时间动力学正好对应连续相机运动，且前后向共享参数保证一致性
- **CMR的设计思路可复用**：当离散近似不够时，加一个初始化接近恒等的"残差修正"模块+正则化约束，简洁又有效
- **旋转轴和旋转角解耦的小改进**说明关注数学本质的重要性
- 在sharp图上性能与Mip-Splatting持平（27.56 vs 27.71），说明没有引入退化

## 局限性 / 可改进方向
- 不区分moderate和extreme blur，所有图都用相同的N=9采样——自适应N可以提效率
- 训练时间比其他3DGS方法长（1.33h vs 0.2-0.83h），主要因为每张图要渲N次
- 只处理相机运动模糊，不处理散焦模糊或滚动快门
- 未考虑场景中的动态物体（假设静态场景）

## 与相关工作的对比
- vs **BAD-Gaussians**: 用线性插值/B样条在物理空间插值位姿，PSNR差距巨大(21.69 vs 27.85)，说明简单插值不够
- vs **Deblurring 3DGS**: 通过修改高斯参数生成模糊图训练，不显式建模运动轨迹，PSNR低约1-3 dB
- vs **BAGS**: CNN估计blur-agnostic退化核，在ExbluRF上效果差(24.7 vs 30.15)，说明2D核不能代替3D运动建模
- vs **SMURF**: 也用Neural ODE但仅在2D像素空间warp射线，CoMoGaussian在3D空间建模更完整

## 与我的研究方向的关联
Neural ODE建模连续过程的思路有启发性——在其他需要连续时间建模的任务（视频生成、动态场景重建）中可以借鉴。CMR的"残差修正"设计模式是一个通用技巧。

## 评分
- 新颖性: ⭐⭐⭐⭐ 将Neural ODE引入3DGS去模糊是自然但有效的组合，CMR设计有巧思
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集，多种消融（组件/ODE类型/N值/正则化/sharp图），per-scene结果完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数学推导完整（附录甚至推导了Rodrigues公式），但行文略啰嗦
- 对我的价值: ⭐⭐⭐ 特定任务方法，但Neural ODE和残差修正的设计模式有迁移价值
