# LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction

**会议**: CVPR 2026  
**arXiv**: [2603.12647](https://arxiv.org/abs/2603.12647)  
**代码**: 无（未提供）  
**领域**: 3D视觉 / 自动驾驶 / 3DGS / 场景重建  
**关键词**: 3D Gaussian Splatting, LiDAR reflectance, salient Gaussian, self-driving reconstruction, multi-modal fusion  

## 一句话总结
提出LR-SGS，将LiDAR强度校准为光照不变的反射率通道附加到3D高斯体上，并设计结构感知的Salient Gaussian表示（从LiDAR几何和反射率特征点初始化）配合改进的密度控制和显著变换策略，在Waymo自动驾驶复杂场景中实现优于OmniRe的高保真重建，且高斯体更少、训练更快。

## 背景与动机
3DGS方法已展示了自动驾驶场景重建和新视角合成的潜力，但面临两大困难：(1) 仅用相机的方法在复杂光照和高速自运动下容易出现纹理不一致和优化不稳定；(2) 现有LiDAR+3DGS方法（如StreetGS、OmniRe、PVG）仅用LiDAR做初始化或深度监督，没有充分利用点云中的反射率信息和LiDAR与RGB的互补性。特别是在弱纹理区域和材质边界处，缺乏稳定约束导致重建质量下降。LiDAR的反射率与物体材质相关且近似光照不变，是RGB的天然互补信号。

## 核心问题
如何充分利用LiDAR的几何和反射率信息来增强3DGS自动驾驶场景重建？具体包含三个子问题：(1) 如何用LiDAR的结构信息（边缘点、平面点、反射率梯度边缘点）引导高斯体分布？(2) 如何利用光照不变的反射率信号来稳定复杂光照条件下的重建？(3) 如何确保RGB和反射率在材质边界处的一致性？

## 方法详解
### 整体框架
输入RGB图像序列+LiDAR点云序列。场景用3DGS场景图表示（背景节点+动态物体节点+天空节点）。初始高斯体由两部分组成：从LiDAR特征点提取的Salient Gaussians和从SfM点初始化的Non-Salient Gaussians。每个高斯体除标准属性外还附带一个反射率通道。通过α混合渲染得到Color、Depth、Reflectance三张图像，用Color Loss + LiDAR Loss + Joint Loss联合优化。

### 关键设计
1. **LiDAR强度校准为反射率**: 原始LiDAR强度受距离和入射角影响（$I = \eta \cdot \rho \cdot \cos\alpha / R^2$），通过除以$R^2$并除以$\cos\alpha$（用法线和激光方向夹角估计）得到近似光照不变的反射率$\rho$。将反射率投影到相机平面得到稀疏反射率图像$F_{gt}$，进一步计算反射率梯度图$F'_{gt}$（相邻像素反射率差除以3D距离）。
2. **Salient Gaussian表示**: 针对自动驾驶场景中丰富的边缘和平面结构设计。每个Salient Gaussian有一个主方向$d_{spec}$，另两个方向共享同一尺度参数——边缘型沿最大尺度方向延伸（$\Sigma = R \cdot \text{diag}(\sigma^2_\parallel, \sigma^2_\perp, \sigma^2_\perp) \cdot R^T$），平面型沿最小尺度方向压扁。这减少了优化参数（从3个尺度变为2个）同时保持对边缘和平面的表达能力。
3. **LiDAR特征点初始化**: 从LiDAR提取三类特征点：(a) 几何边缘点——通过计算邻域平滑度$c_j$筛选高曲率点；(b) 几何平面点——低曲率点；(c) 反射率边缘点——通过计算同一扫描环上左右邻域反射率梯度$G_j$筛选。几何边缘+反射率边缘→Edge Salient Gaussians，几何平面→Planar Salient Gaussians，SfM点→Non-Salient Gaussians。
4. **改进密度控制和显著变换（Salient Transform）**: 分裂时Edge Salient Gaussian沿主方向分裂，Planar沿正交平面分裂。引入显著变换策略：定义线性度$L(g)=(s_1-s_2)/s_1$和平面度$P(g)=(s_2-s_3)/s_1$。Non-Salient Gaussian如果$\max\{L,P\} > \tau_{max}$在连续两次评估中成立，则升级为Salient Gaussian；Salient Gaussian如果$\max\{L,P\} < \tau_{min}$持续两次，则降级为Non-Salient。$\tau_{max}=0.5$，$\tau_{min}=0.1$。
5. **RGB-LiDAR反射率联合损失（Joint Loss）**: 将RGB转灰度后与反射率图各经高斯平滑和Scharr算子求梯度。方向一致性损失$\mathcal{L}_{dir} = 1 - \hat{\nabla}F \cdot \hat{\nabla}C_g$（梯度方向点积）；幅值一致性损失$\mathcal{L}_{val} = \|g_F/F - g_{C_g}/C_g\|_1$（归一化幅值差）。在材质边界处强制RGB和反射率的梯度对齐，锐化边界重建。

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_{rgb} + \mathcal{L}_{lidar} + \mathcal{L}_{joint}$：
- $\mathcal{L}_{rgb} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c \mathcal{L}_{D-SSIM}$，$\lambda_c=0.2$
- $\mathcal{L}_{lidar} = \lambda_{depth}\mathcal{L}_{depth} + \lambda_{fle}\mathcal{L}_{fle} + \lambda_{fle'}\mathcal{L}_{fle'}$（深度L1 + 反射率L1 + 反射率梯度L1）
- $\mathcal{L}_{joint} = \lambda_{dir}\mathcal{L}_{dir} + \lambda_{val}\mathcal{L}_{val}$
- 权重：$\lambda_{depth}=\lambda_{fle}=\lambda_{dir}=0.1$，$\lambda_{val}=0.2$，$\lambda_{fle'}=0.05$
- 训练30k iterations，输入分辨率1066×1600，5次运行取平均

## 实验关键数据
| 数据集/场景 | 指标 | LR-SGS (本文) | OmniRe | StreetGS | 提升 |
|--------|------|------|----------|------|------|
| Dense Traffic | PSNR↑ | 28.89 | 28.44 | 27.01 | +0.45 vs OmniRe |
| Dense Traffic | PSNR*(动态) | 24.02 | 23.72 | 21.73 | +0.30 vs OmniRe |
| High-Speed | PSNR↑ | 28.77 | 28.12 | 28.06 | +0.65 vs OmniRe |
| High-Speed | SSIM↑ | 0.896 | 0.871 | 0.878 | +0.025 vs OmniRe |
| Complex Lighting | PSNR↑ | 30.51 | 29.33 | 29.16 | **+1.18 vs OmniRe** |
| Static | PSNR↑ | 28.73 | 28.23 | 28.15 | +0.50 vs OmniRe |
| 效率对比 | Gaussian数 | 2,510,883 | 2,744,275 | 2,929,851 | -8.5% vs OmniRe |
| 效率对比 | 训练时间 | 59m25s | 67m11s | 64m30s | -11.5% vs OmniRe |
| 效率对比 | FPS | 36.95 | 30.55 | 33.61 | +21% vs OmniRe |

### 消融实验要点
- **Salient Gaussian**：去掉SG后PSNR从29.22降至28.74（-0.48），SG使训练时间更短（61m57s vs 70m21s@30k iters）
- **LiDAR特征点初始化**：去掉LF-Init后PSNR从29.22降至28.94（-0.28），7k iters时差距更明显（26.41 vs 25.52）
- **反射率通道**：去掉反射率后PSNR从29.22降至28.87，SSIM从0.850降至0.831，在复杂光照场景效果尤为明显
- **Joint Loss**：去掉后PSNR从30.39降至30.08，反射率RMSE从0.0854升至0.1063，车辆轮廓和斑马线变模糊
- 所有组件都有正向贡献，Salient Gaussian贡献最大（+0.48 PSNR）

## 亮点
- **反射率作为光照不变通道**：巧妙地将LiDAR强度校准为材质相关的反射率，为3DGS提供了除RGB外的稳定监督信号，尤其在夜间和复杂光照场景下效果显著
- **Salient Gaussian的结构感知设计**：通过减少参数（2个尺度替代3个）反而获得了更好地拟合边缘和平面的能力，且减少了冗余高斯体数量
- **显著变换策略**：允许Non-Salient和Salient状态双向转换，使得即使LiDAR未覆盖的区域也能通过SfM点升级产生Salient Gaussian
- **RGB-反射率Joint Loss**：通过梯度方向和幅值对齐实现跨模态一致性，比简单的像素级约束更精确地锐化了材质边界

## 局限性 / 可改进方向
- 仅在Waymo数据集上验证，未涉及nuScenes等其他自动驾驶数据集
- 反射率校准依赖法线估计（需邻域点），在点云稀疏区域可能不准确
- 场景编辑（替换/删除物体）功能仅展示了定性结果，缺乏定量评估
- $\tau_{max}$和$\tau_{min}$为手动设定的超参数，自适应阈值可能更好
- 未讨论实时性能——虽FPS 36.95已较好，但在嵌入式平台上的可行性不明
- 未考虑天气变化（如雨雪雾对LiDAR反射率的影响）

## 与相关工作的对比
- **vs OmniRe**：OmniRe是多类型高斯场景图方法，用LiDAR初始化和深度监督但不利用反射率，在Complex Lighting场景比LR-SGS低1.18dB PSNR。LR-SGS用更少高斯体和更短训练时间达到更好效果
- **vs StreetGS**：StreetGS用LiDAR初始化背景和动态物体的3DGS，但没有结构感知的高斯设计。LR-SGS在所有场景类型上全面超越，尤其Dense Traffic动态物体重建（PSNR* 24.02 vs 21.73）
- **vs TCLC-GS**：TCLC-GS用LiDAR mesh和八叉树特征初始化，pipeline更复杂

## 启发与关联
- Salient Gaussian的思路可与occupancy prediction结合——用结构感知的高斯体作为3D occupancy的几何先验
- 反射率通道可扩展到多光谱（红外等），进一步丰富自动驾驶场景表示
- 与 ideas 中的 [频域安全防御3DGS](../../ideas/3d_vision/20260316_spectral_defense_3dgs.md) 有关——Salient Gaussian的结构化属性可能更利于频域分析

## 评分
- 新颖性: ⭐⭐⭐⭐ 反射率通道+Salient Gaussian+显著变换的组合在自动驾驶3DGS中首次提出
- 实验充分度: ⭐⭐⭐⭐ 4类场景（Dense Traffic/High-Speed/Complex Lighting/Static）、完整消融、效率分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图表丰富（定性对比直观）
- 价值: ⭐⭐⭐⭐ 在自动驾驶3DGS领域贡献了新的模态融合思路，对仿真数据生成有实际价值
