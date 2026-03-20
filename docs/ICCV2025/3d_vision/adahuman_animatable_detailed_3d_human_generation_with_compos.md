# AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion

**会议**: ICCV 2025  
**arXiv**: [2505.24877](https://arxiv.org/abs/2505.24877)  
**代码**: [https://nvlabs.github.io/AdaHuman](https://nvlabs.github.io/AdaHuman) (将公开)  
**领域**: 3D人体重建与生成 / 3D Gaussian Splatting / 扩散模型  
**关键词**: 3D人体生成, 多视角扩散, 3DGS, 姿态条件生成, 组合式细化  

## 一句话总结
提出AdaHuman框架，通过姿态条件的联合3D扩散模型（在扩散过程中同步进行多视角图像生成与3DGS重建以保证3D一致性）和组合式3DGS细化模块（利用crop-aware camera ray map融合局部精细细节），从单张野外图片生成高保真可动画的3D人体avatar，在重建和重姿态任务上全面超越现有SOTA。

## 背景与动机
从单张图片生成高质量可动画的3D人体模型是游戏、动画和VR的核心需求。现有方法存在两大痛点：

1. **SDS蒸馏方法**（DreamFusion等）虽灵活，但存在过饱和伪影、生成速度极慢（几十分钟到小时级），不适合大规模avatar创建。
2. **多视角生成+重建的流水线方法**（如Human3Diffusion）虽然更快更真实，但生成的avatar通常保持输入图像的原始姿态，导致自遮挡严重，难以绑骨动画；且输出分辨率受限于前馈3DGS生成器的固定分辨率（如LGM的256×256），丢失精细细节。

此外，基于SMPL模板的mesh重建方法（SiTH、SIFU）受限于固定拓扑结构，在松散衣物上表现不佳。

## 核心问题
如何从单张野外图片生成**同时具备高保真细节和可动画能力**的3D人体avatar？需要解决两个子问题：(1) 如何在姿态变化时保持多视角一致性并恢复遮挡区域的几何与外观？(2) 如何突破前馈3DGS生成器的分辨率瓶颈，获得精细局部细节？

## 方法详解
### 整体框架
AdaHuman是一个两阶段pipeline：
- **第一阶段**：姿态条件联合3D扩散（Pose-Conditioned 3D Joint Diffusion）——输入单张图片，通过在扩散过程中每一步交替进行多视角LDM去噪和3DGS重建，生成粗糙但3D一致的全身3DGS avatar $\mathcal{G}_{\text{coarse}}$，支持重建原姿态或生成标准A-pose。
- **第二阶段**：组合式3DGS细化（Compositional 3DGS Refinement）——将粗糙avatar的局部身体部位（头部、上半身、下半身）通过SDEdit精细化，再用visibility-aware组合策略融合成高细节的完整avatar $\mathcal{G}_{\text{refined}}$。

### 关键设计
1. **姿态条件联合3D扩散**：基于Stable Diffusion的U-Net架构，将2D自注意力替换为3D跨视角注意力。以SMPL语义pose map和camera ray map作为额外条件输入。关键创新在于每个去噪步$t$都插入3DGS生成器$\mathbf{G}$：先由LDM预测clean图像$\mathbf{x}^{t \to 0}$, 用$\mathbf{G}$生成3DGS $\mathcal{G}_t$，再渲染得到3D一致的clean图像$\hat{\mathbf{x}}^{t \to 0}$用于下一步去噪。通过简单切换pose条件即可生成任意目标姿态的avatar，无需标准姿态训练数据。

2. **Crop-aware Camera Ray Map**：解决局部视角与全局视角的3D坐标对应问题。对于局部裁切视角中的像素$(u,v)$，通过裁切框坐标映射回全局视角坐标$(i,j)$，然后用全局camera ray map方程计算Plücker ray embedding。这使得3DGS生成器能在统一的全局空间中同时处理全身和局部视角的输入。

3. **Visibility-aware 3DGS组合**：通过两个准则智能合并局部和全局3DGS：(1) **View Coverage**——统计每个Gaussian被多少输入视角覆盖，覆盖不足的视为不可靠而丢弃；(2) **Visibility Salience**——计算alpha通道在所有渲染视角上的梯度幅值，梯度低的视为噪声。同时：若某splat在更精细的部位（如头部优先于上身）有良好覆盖，则从较粗部位中删除冗余splat，避免重叠冲突。

### 损失函数 / 训练策略
- **LDM损失**：标准MSE噪声预测损失 $\mathcal{L}_{\text{LDM}} = \mathcal{L}_{\text{MSE}}(\epsilon, \epsilon_\theta)$
- **3DGS生成器损失**：$\mathcal{L}_\mathbf{G} = \lambda_{\text{MSE}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{LPIPS}} \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$（含表面正则化）
- 额外采样12个辅助视角对3DGS提供稠密监督
- **训练数据**：MVHumanNet（6209个subject的多相机视频）+ CustomHumans（589个mesh渲染）
- **训练流程**：先在全身视角训练20k步→加入局部视角训练30k步→重姿态微调10k步
- 16张A100 80GB，batch size=128，lr=5e-5

## 实验关键数据

### Avatar重建（CustomHumans数据集）
| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|------|-------|-------|--------|------|
| LGM | 18.99 | 0.8445 | 0.1664 | 122.3 |
| SiTH | 20.77 | 0.8727 | 0.1277 | 42.9 |
| SIFU | 20.59 | 0.8853 | 0.1359 | 92.6 |
| Human3Diffusion | 21.08 | 0.8728 | 0.1364 | 35.3 |
| **AdaHuman** | **21.46** | **0.8925** | **0.1087** | **27.3** |

### 新姿态合成（MVHumanNet数据集）
| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| SiTH | 21.21 | 0.8742 | 0.1261 |
| SIFU | 21.27 | 0.8722 | 0.1244 |
| AdaHuman + deform | 23.01 | 0.8825 | 0.1100 |
| **AdaHuman** | **24.64** | **0.9046** | **0.0863** |

### 用户偏好研究（SHHQ数据集，28人参与）
| 对比方法 | AdaHuman偏好率 |
|---------|--------------|
| vs SiTH | 88.3% |
| vs SIFU | 99.2% |
| vs Human3Diffusion | 79.7% |
| vs Coarse 3DGS（消融） | 93.8% |

### 消融实验要点
- **去掉组合细化（Coarse 3DGS only）**：FID从27.3退化到31.9，面部等精细区域明显模糊
- **直接合并（Direct Composition）**：FID上升至36.2，产生大量浮动伪影
- **可学习合并（Learnable Composition）**：FID=28.0，略有改善但仍有伪影且计算量增大
- **去掉联合扩散（No Joint Diffusion）**：PSNR从21.46降到20.79，视角不一致性增加
- **添加GT Pose条件**：PSNR可进一步提升至23.00，说明姿态估计精度仍有改善空间
- **更多身体部位（加middle部位）**：与3部位（upper/lower/head）效果相当（21.43 vs 21.46 PSNR），但效率更低

## 亮点
- **扩散过程中嵌入3DGS重建**是保证多视角一致性的优雅解法，比后处理式重建更有效
- **姿态条件生成**可以无需标准姿态训练数据就泛化到A-pose，是零样本泛化的有趣体现
- **Crop-aware ray map**用极简方式（坐标映射）解决了局部-全局视角在3D空间中的对应问题，无需复杂网络
- **Visibility-aware组合**基于view coverage和alpha梯度的两个简单准则，有效避免了naïve合并的伪影问题
- 在松散衣物的重姿态上表现出色，能生成逼真的衣物形变效果

## 局限性 / 可改进方向
- 手部和手臂等遮挡严重区域的局部细化效果较差，容易产生伪影
- 动画能力仍依赖SMPL的skinning weights对齐，面部表情、手势和衣物形变受限
- 推理时间约70秒（A100），相比前馈方法（LHM等秒级推理）较慢
- 姿态估计误差会传播影响重建质量（GT pose可提升约1.5 dB PSNR）
- 两种动画模式各有trade-off：直接重姿态更逼真但慢且时序不连贯，LBS动画快但衣物形变受限

## 与相关工作的对比
- **vs Human3Diffusion**：同为联合扩散+3DGS路线，但H3D无姿态条件（不能重姿态），无局部细化（细节模糊）。AdaHuman在PSNR上高约0.4dB，FID低约8分，且额外支持动画。
- **vs SiTH/SIFU**：基于SMPL模板的mesh方法，受限于固定拓扑，松散衣物重建质量差。用户偏好研究中AdaHuman以88-99%的压倒性优势胜出。SiTH/SIFU在重姿态时依赖LBS，衣物形变不真实。
- **vs IDOL/LHM（同期工作）**：IDOL和LHM走前馈路线追求效率，AdaHuman基于扩散模型利用更强的生成先验，牺牲速度换取更高的生成质量。

## 启发与关联
- **与[扩散模型幻想视角增强3DGS](../../../ideas/3d_vision/20260317_diffusion_view_augment_3dgs.md)的关联**：AdaHuman的"扩散过程中进行3DGS重建"和该idea的"扩散增强稀疏视角3DGS"思路类似，都是用扩散模型弥补视角稀疏的问题。AdaHuman进一步证明了在扩散步骤中嵌入3D重建（而非后处理式重建）的有效性。该idea中关于不确定性加权（对幻想视角施加方差估计权重）的思想，或可引入AdaHuman的组合策略中。
- **组合式细化的通用性**：crop-aware ray map + visibility-aware composition的方案不局限于人体，理论上可扩展到任何需要多尺度3DGS重建的场景（如大场景中的细节物体重建）。
- **与[过程感知对齐](../../../ideas/image_generation/20260316_process_aware_alignment.md)的关联**：AdaHuman在扩散过程的中间步骤引入3D约束，本质上也是一种过程感知的生成策略。若将过程感知的偏好对齐引入avatar质量评估，可能进一步提升细节质量。

## 评分
- 新颖性: ⭐⭐⭐⭐ 姿态条件联合扩散+组合式3DGS细化的双重创新有明确技术贡献，但联合扩散框架基于H3D的扩展
- 实验充分度: ⭐⭐⭐⭐⭐ 两个benchmark定量评测、用户研究、重姿态评测、详尽消融、in-the-wild展示，实验非常全面
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，方法描述详细，图表质量高。个别地方有typo
- 价值: ⭐⭐⭐⭐ 在单图3D人体生成领域推进了state-of-the-art，但推理速度是实际应用的瓶颈
