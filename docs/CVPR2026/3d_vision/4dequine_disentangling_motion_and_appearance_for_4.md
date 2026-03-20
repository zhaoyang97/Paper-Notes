# 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video

**会议**: CVPR 2026  
**arXiv**: [2603.10125](https://arxiv.org/abs/2603.10125)  
**代码**: [https://luoxue-star.github.io/4DEquine_Project_Page/](https://luoxue-star.github.io/4DEquine_Project_Page/)  
**领域**: 3D视觉 / 动态重建 / 动物建模  
**关键词**: 4D重建, 3DGS, 运动解耦, 外观解耦, 单目视频, VAREN  

## 一句话总结
将马科动物4D重建解耦为运动估计(AniMoFormer时空Transformer+后优化)和外观重建(EquineGS前馈3DGS)两个子任务，用VAREN参数化模型做桥梁，仅在合成数据(VarenPoser+VarenTex)上训练即在真实数据APT-36K和AiM上达到SOTA，并能零样本泛化到斑马和驴。

## 背景与动机
从单目视频重建马科动物的4D形态（几何+运动+外观）在畜牧、运动分析、动物福利等领域很有价值。现有方法存在两类核心瓶颈：(1) 通用4D重建方法（如Monst3R、Page-4D）无法从不完整观测中恢复物体的完整几何；(2) 基于SMAL模板的优化方法（GART、4D-Fauna等）需要对逐个视频做联合优化，计算昂贵且要求接近360°覆盖的视频捕获——这在真实世界场景中几乎不可能满足。需要一种既高效又能处理稀疏视角的方案。

## 核心问题
如何从普通的单目视频（非360°环绕拍摄）中高效地重建马科动物的4D表征（几何+运动+外观），同时避免逐视频的昂贵优化？核心挑战在于运动和外观的联合优化既复杂又对视角覆盖敏感。

## 方法详解
### 整体框架
将4D重建分解为两个独立子任务：**运动估计**（AniMoFormer：时空Transformer + 后优化 → 输出VAREN参数序列）和**外观重建**（EquineGS：前馈网络从单张图像 → 输出canonical 3DGS化身）。VAREN参数化马体模型作为桥梁连接两者——运动给出每帧的姿态/形状参数，外观在canonical空间生成Gaussian点云，通过LBS变形到每帧姿态。

### 关键设计
1. **AniMoFormer（运动估计）**: 两阶段设计。第一阶段是时空Transformer——先用Spatial Transformer（ViT-H backbone, 来自AniMer预训练）逐帧提取特征，再用Temporal Transformer在16帧窗口上做self-attention建模时序关系，回归VAREN的姿态/形状/相机参数。第二阶段是Post-Optimization——用可微渲染器将3D mesh投影到2D，与ViTPose++检测的2D关键点和Samurai提取的mask做对齐优化（分两步：先优化全部参数偏重关键点对齐，再冻结姿态偏重mask对齐）。
2. **EquineGS（外观重建）**: 前馈网络，从单张图像生成可动画的canonical 3DGS化身。先用VAREN模板mesh做细分（13873→55486顶点）作为Gaussian初始化位置；然后双流特征提取——DINOv3 ViT-L提取图像特征，Point Transformer编码3D点特征；最后用DSTG（Dual-Stream Transformer Gaussian）decoder融合两路特征，预测每个Gaussian的位移、旋转、缩放、颜色和不透明度。
3. **合成数据集创建**: VarenPoser（运动训练用）——将VAREN模型拟合到PFERD马动捕数据，用MV-Adapter生成纹理，模拟三种相机轨迹（固定/环绕/推拉），共1171个视频片段。VarenTex（外观训练用）——用UniTex多视角扩散模型生成一致多视图图像，共150K张，提供比VarenPoser更高质量的外观。

### 损失函数 / 训练策略
- AniMoFormer训练损失: $\mathcal{L} = \lambda_{varen}\mathcal{L}_{varen} + \lambda_{smooth}\mathcal{L}_{smooth} + \lambda_{2D}\mathcal{L}_{2D} + \lambda_{3D}\mathcal{L}_{3D}$，其中smooth项约束相邻帧参数变化
- 后优化损失: $\mathcal{L} = \lambda_{2D}\mathcal{L}_{2D} + \lambda_{smooth}\mathcal{L}_{smooth} + \lambda_{reg}\mathcal{L}_{reg} + \lambda_{mask}\mathcal{L}_{mask}$，分两阶段分别侧重关键点和mask
- EquineGS训练损失: $\mathcal{L} = \lambda_{image}(\|L1\| + LPIPS) + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{reg}\mathcal{L}_{reg}$
- AniMoFormer在单张4090上训练10小时(100K步)；EquineGS在8×4090上训练3天(100K步)
- 推理速度: 11秒/帧（A100），对比GART固定15分钟/视频

## 实验关键数据
| 数据集 | 指标 | 4DEquine | 之前SOTA(AniMer) | 提升 |
|--------|------|----------|-----------------|------|
| APT-36K | PCK@0.05 | **61.8** | 44.5 | +38.9% |
| APT-36K | PCK@0.1 | **83.9** | 76.6 | +9.5% |
| AiM | PCK@0.05 | **84.2** | 55.5 | +51.7% |
| AiM | Accel↓ | **21.8** | 26.2 | -16.8% |
| VarenPoser | CD↓ | **3.4** | 15.2 | -77.6% |

| 数据集 | 指标 | 4DEquine | GART(全优化) | Few-shot GART |
|--------|------|----------|-------------|--------------|
| AiM-Horse | SSIM↑ | **0.8364** | 0.7819 | 0.7550 |
| AiM-Horse | LPIPS↓ | **0.1720** | 0.2308 | 0.2452 |
| AiM-Zebra(零样本) | PSNR↑ | **15.54** | 15.21 | 14.31 |
| AiM-Zebra(零样本) | LPIPS↓ | **0.2000** | 0.2287 | 0.2973 |

### 消融实验要点
- 去掉后优化(PO): PCK@0.05降至37.7(APT-36K)，渲染质量LPIPS从0.172→0.217——PO是像素对齐的关键
- 去掉时空Transformer: 加速误差Accel明显升高——时序建模对运动平滑性贡献最大
- 去掉mesh细分(SubDiv): PSNR略高但可视化充满孔洞——低分辨率点云不足以形成连续表面
- DSTG vs 标准Cross-Attention: DSTG在SSIM/LPIPS上一致更优——双流融合优于单向cross-attention
- 窗口大小N: 4→8→16帧性能持续提升，N=32 OOM

## 亮点
- **解耦思路优雅**: 将4D重建拆成运动和外观，用参数化模型VAREN做桥梁，各自独立训练和推理
- **纯合成数据训练，零样本泛化到真实数据和未见物种（驴、斑马）**——说明合成数据+好的参数化先验可以弥合域差距
- 前馈外观重建只需单张图像，避免了GART等方法需要多帧优化的瓶颈
- 后优化作为可选后处理步骤的两阶段设计（先对齐关键点再对齐mask）值得借鉴

## 局限性 / 可改进方向
- VAREN模型本身不能很好地表示尾巴和鬃毛的物理形态，限制了这些区域的重建质量
- 不能处理动态光照变化
- EquineGS仅用单帧输入，对严重遮挡/截断的输入帧会失败——未来可融合多关键帧
- 仅限马科动物，泛化到猫狗等其他四足动物需要相应的参数化模型
- 训练成本不低（8×4090训练3天），不适合资源受限场景

## 与相关工作的对比
- **vs GART (CVPR 2024)**: GART是逐视频优化的方法，需要较好的视角覆盖，每个视频固定15分钟优化；4DEquine是前馈方法，单帧11秒推理。GART在PSNR上略优(16.19 vs 15.66)但感知指标(SSIM/LPIPS)不如4DEquine，在零样本斑马场景上全面落后
- **vs 4D-Fauna (ICCV 2025)**: 4D-Fauna是无模板的通用方法，可重建100+种四足动物但缺乏几何精度；4DEquine利用VAREN先验获得更高几何保真度，但仅限马科
- **vs AniMer**: 单帧估计方法，无时序建模导致运动抖动严重；4DEquine的时空Transformer在PCK@0.05上提升39-52%

## 启发与关联
- 解耦+参数化先验的思路可以推广到其他有参数化模型的领域（人体SMPL、手部MANO等）
- 合成数据训练→真实数据泛化的pipeline（MV-Adapter生成纹理 + UniTex生成多视图）是一个通用方案
- 与我的研究方向关联较弱，该工作聚焦于特定动物重建的垂直应用

## 评分
- 新颖性: ⭐⭐⭐⭐ 解耦运动和外观的思路清晰，但各个模块（时空Transformer、3DGS前馈）不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集+多baseline+详细消融+零样本泛化+失败案例分析，非常完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但论文较长，supplementary内容丰富
- 价值: ⭐⭐⭐ 在马科动物重建领域有实际价值，但应用面相对窄
