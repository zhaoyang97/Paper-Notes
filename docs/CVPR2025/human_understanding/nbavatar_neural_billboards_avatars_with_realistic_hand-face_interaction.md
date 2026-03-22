# NBAvatar: Neural Billboards Avatars with Realistic Hand-Face Interaction

**会议**: CVPR2025  
**arXiv**: [2603.12063](https://arxiv.org/abs/2603.12063)  
**代码**: [Project Page](https://david-svitov.github.io/NBAvatar_project_page)  
**领域**: human_understanding  
**关键词**: head avatar, hand-face interaction, neural rendering, billboard splatting, deferred neural rendering

## 一句话总结
NBAvatar 提出 Neural Billboard 原语——将可学习平面几何原语与神经纹理延迟渲染结合，实现手脸交互场景下的照片级真实头部 avatar 渲染，在百万像素分辨率下 LPIPS 比 Gaussian 方法降低 30%。

## 研究背景与动机
1. 手脸交互是人类交流的重要信息源，对远程呈现和 VR 应用至关重要
2. 现代方法专注于头部或手部单独渲染，忽略了手脸交互造成的非刚性变形和颜色变化
3. 3DGS 方法虽然质量高但存在固有伪影：面部纹理模糊、Gaussian 在身体边界突出
4. InteractAvatar 是最新手脸交互方法，使用 MLP 预测交互区域偏移，但继承了 3DGS 伪影
5. DNR（延迟神经渲染）可实现实时速度和高保真度，但原始设计基于固定 mesh 参数化
6. 将神经纹理范式适配到可独立变换的平面原语是非平凡的优化挑战

## 方法详解

### 整体框架
NBAvatar 包含三个阶段：(1) 从多视角视频拟合 FLAME/MANO 参数模型 + PBD 物理仿真粗糙面部变形；(2) 将 Neural Billboard 原语锚定到 mesh 表面多边形；(3) 渲染神经特征图后通过 UNet 解码器生成最终 RGB 图像。

### 关键设计

**1. Neural Billboard 原语**
- 参数化：$\{\mu_i, s_i, r_i, T_i^{NT}, T_i^{\alpha}\}$（位置、缩放、旋转、16×16 六通道神经纹理、单通道 alpha 纹理）
- 替换了 Billboard Splatting 的 RGB 纹理为可学习神经特征图
- alpha 纹理从 Gaussian 分布初始化，学习每个平面点的可见性
- 沿相机光线累积神经纹理值：$c(x) = \sum_i T_i^{NT}[\mathbf{u}(x)] T_i^{\alpha}[\mathbf{u}(x)] \prod_{j=1}^{i-1}(1-T_j^{\alpha}[\mathbf{u}(x)])$
- 产生 6 通道特征图 $I_f^{NB}$ 和 1 通道 alpha 图 $I_\alpha^{NB}$

**2. UNet 延迟渲染器**
- 将 6 通道栅格化特征图解码为 RGB 图像 + 透明度图
- 提供高频细节和交互感知归纳偏置
- 手和脸的特征在共享屏幕空间中栅格化，邻近特征自然调制解码器响应
- **不使用显式交互条件模块**，完全依赖空间特征聚合隐式捕获接触动态

**3. 几何与外观解耦训练**
- 核心挑战：billboard 的空间漂移和神经特征会竞争解释轮廓和阴影变化
- 在 billboard 栅格化阶段引入 **中间轮廓监督**：$\mathcal{L}_{NB} = \lambda_{NB} \text{Dice}(I_\alpha^{NB}, GT_\alpha)$
- 确保 billboard 紧贴 GT 轮廓，将刚性几何与视角/姿态相关外观解耦
- KNN 正则化约束相邻 billboard 的旋转/缩放一致性 + 位置偏移正则化

**4. Avatar 动画驱动**
- billboard 锚定到 mesh 多边形，通过多边形的变换 $\{T_i, R_i, k_i\}$ 驱动
- 位置：$\mu_i' = k_i R_i \mu_i + T_i$，缩放：$s_i' = k_i s_i$，旋转：$q_i' = R_i q_i$

### 损失函数
- RGB 训练：MSE + $\lambda_{lpips}$ LPIPS（前 40K iter 全图，之后 256×256 随机裁剪）
- 轮廓监督：$\lambda_{NB} = 0.1$ Dice loss
- 正则化：KNN + 位置偏移 $\lambda_\Delta = 0.001$

## 实验关键数据

### Novel-View 合成（Decaf 数据集，1024×1024）
| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| SplattingAvatar | 25.17 | 0.955 | 0.080 |
| GaussianAvatars | 25.31 | 0.957 | 0.076 |
| **NBAvatar** | **25.65** | **0.958** | **0.056** |

LPIPS 降低 26.3%（vs GA）和 30.0%（vs SA）。

### Self-Reenactment（保留姿态）
| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| SplattingAvatar | 25.82 | 0.962 | 0.066 |
| GaussianAvatars | 25.04 | 0.960 | 0.066 |
| **NBAvatar** | **25.48** | **0.961** | **0.052** |

### 与 InteractAvatar 对比（512×512，IA 评估协议）
| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| InteractAvatar | 29.85 | 0.933 | 0.034 |
| NBAvatar | 24.41 | **0.936** | 0.051 |

SSIM 更高，但 PSNR 低于 IA（预处理差异导致）。定性对比显示 NBAvatar 面部细节更锐利。

### 消融实验（Subject 2 Novel-View）
| 配置 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| w/o $\mathcal{L}_{NB}$ | 26.75 | 0.9711 | 0.039 |
| w/o $\mathcal{L}_{Reg}$ | 28.43 | 0.9748 | 0.033 |
| w/o DNR | 25.88 | 0.9655 | 0.045 |
| **Full** | **28.63** | **0.976** | **0.032** |

## 亮点与洞察
1. **Neural Billboard 原语**：显式几何 + 隐式神经特征的巧妙结合，兼具 billboard 的表面对齐和神经纹理的表达力
2. **隐式交互建模**：不使用显式交互模块，利用 UNet 的空间感受野隐式捕获手脸接触动态，更简洁且泛化更好
3. **轮廓监督解耦**：中间 Dice loss 是稳定联合优化的关键，消融证实其对减少伪影不可或缺
4. **百万像素质量**：在 1024×1024 分辨率下大幅减少 3DGS 典型的边界和模糊伪影
5. **跨主体重演**：支持跨 actor 的手脸姿态迁移

## 局限性
1. UNet 渲染器增加了推理开销，实时性能未报告（推测低于纯 3DGS 方法）
2. 依赖 FLAME/MANO 的 3DMM 拟合质量，拟合误差直接影响渲染
3. 仅在 Decaf 数据集（4 个受试者）验证，泛化性待更多测试
4. 手部表示仍较粗糙，高度铰接的手指细节可能不足

## 相关工作与启发
- Neural Billboard 是 DNR 和 Billboard Splatting 的自然融合，展示了显式+隐式混合表示的潜力
- 隐式交互建模的思路可推广到其他多体交互渲染（如手-物体、人-人交互）
- 轮廓监督的训练策略可用于其他需要几何-外观解耦的任务
- 为远程呈现、VR 社交中的手脸交互渲染提供了新基线

## 评分
- 新颖性: ⭐⭐⭐⭐ (Neural Billboard 原语 + 隐式交互建模)
- 实验充分度: ⭐⭐⭐ (仅 Decaf 数据集 4 个受试者)
- 写作质量: ⭐⭐⭐⭐ (方法动机和消融清晰)
- 价值: ⭐⭐⭐⭐ (推进手脸交互渲染质量)
