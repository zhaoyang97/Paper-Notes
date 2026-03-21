# 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video

**会议**: CVPR 2025  
**arXiv**: [2603.10125](https://arxiv.org/abs/2603.10125)  
**代码**: [项目页](https://luoxue-star.github.io/4DEquine_Project_Page/)  
**领域**: 3D视觉 / 动物重建 / 4D重建  
**关键词**: 4D重建, 马匹重建, 3D Gaussian Splatting, VAREN模型, 运动-外观解耦

## 一句话总结
将单目视频的4D马匹重建解耦为运动估计（AniMoFormer时空Transformer）和外观重建（EquineGS单图前馈3DGS），依托VAREN参数化模型和两个大规模合成数据集，在真实数据上达到SOTA几何+外观重建效果，且能零样本泛化到驴和斑马。

## 研究背景与动机

1. **领域现状**：单目4D动物重建主要有两条路线：(a) 无模板方法（BANMo、RAC等）学习可变形辐射场但缺乏几何先验，形状粗糙；(b) 基于SMAL/hSMAL参数化模型的方法（SMALR、Dessie等）有几何先验但纹理质量差，且需要逐视频优化。近期GART等方法用3D Gaussian Splatting做动物avatar，但仍需360°视频+逐实例优化。

2. **现有痛点**：(a) 逐视频优化方法（GART）计算昂贵且需要完整观测覆盖，真实视频往往视角有限导致优化失败；(b) 前馈方法（3D-Fauna、MagicPony等）虽不需逐实例优化，但必须牺牲形状真实感换取泛化性；(c) 现有方法不使用最新的高精度VAREN马匹模型，而是依赖较粗糙的SMAL。

3. **核心矛盾**：4D重建同时估计运动和外观，两者相互耦合——运动不准导致外观无法对齐，外观不好又反过来影响运动估计。联合优化既慢又对不完整观测敏感。

4. **本文要解决什么？**
   - 如何高效准确地从单目视频恢复马匹的逐帧运动（pose+shape序列）？
   - 如何从极少输入（甚至单张图像）前馈生成高保真可动画3D Gaussian avatar？
   - 如何在真实数据缺乏的情况下训练上述网络？

5. **切入角度**：将4D重建显式解耦为两个独立子问题——动态运动恢复 + 静态外观重建，用VAREN参数化模型作为桥梁连接两者。分别构建合成数据集训练各自网络。

6. **核心idea一句话**：运动和外观解耦后分别用专门的网络+合成数据解决，再通过VAREN模型的skinning将外观驱动到每帧姿态，实现高效高质量4D马匹重建。

## 方法详解

### 整体框架
输入单目视频，pipeline分两路：
1. **AniMoFormer**（运动分支）：时空Transformer + 后优化，对每帧预测VAREN模型的形状$\beta \in \mathbb{R}^{39}$、姿态$\theta \in \mathbb{R}^{38\times3}$和相机参数
2. **EquineGS**（外观分支）：从视频中选一张代表帧，前馈生成canonical空间的3D Gaussian avatar（55,486个Gaussian点）
3. **融合**：通过VAREN的LBS将canonical Gaussian变形到每帧的姿态空间，渲染得到4D结果

### 关键设计

1. **VarenPoser合成运动数据集**
   - 做什么：创建大规模合成马匹视频数据集，提供VAREN参数GT用于训练运动网络
   - 核心思路：将VAREN模型拟合到标记点运动捕捉数据集PFERD上获得姿态参数，随机分配形状参数增加多样性，用MV-Adapter生成纹理，关键创新是模拟三种真实相机运动轨迹（fix/dolly/orbit）
   - 设计动机：真实4D马匹数据标注极其困难，合成数据集包含1171个video clip、512×512分辨率、60FPS，提供精确的VAREN参数GT

2. **AniMoFormer时空Transformer**
   - 做什么：从16帧视频窗口回归时间一致的VAREN参数
   - 核心思路：两阶段设计——(1) Spatial Transformer逐帧提取空间特征，(2) Temporal Transformer在N帧窗口上做self-attention建模时序关系，(3) VAREN Transformer Decoder回归参数。训练损失：
     $$\mathcal{L} = \lambda_{varen}\mathcal{L}_{varen} + \lambda_{smooth}\mathcal{L}_{smooth} + \lambda_{2D}\mathcal{L}_{2D} + \lambda_{3D}\mathcal{L}_{3D}$$
     其中$\mathcal{L}_{smooth}$约束相邻帧参数变化平滑
   - 设计动机：单帧方法（AniMer）无法利用时序信息，导致帧间抖动。时空Transformer天然支持滑动窗口处理任意长视频

3. **Post-Optimization后优化**
   - 做什么：将Transformer预测的mesh与2D图像精确对齐
   - 核心思路：用可微分渲染器将3D mesh投影到2D，与ViTPose++提取的pseudo GT关键点和Samurai提取的mask对比，优化姿态参数使其pixel-aligned
   - 设计动机：Transformer预测的mesh可能与2D证据有偏移，后优化通过图像级监督弥补这一gap，两者缺一不可（消融证实去掉任何一个都会掉点）

4. **EquineGS前馈外观重建**
   - 做什么：从单张图像前馈生成可动画的3DGS avatar
   - 核心思路：
     - **点云初始化**：将VAREN模板mesh细分（每边中点+每面分四），从13,873顶点上采样到55,486个Gaussian初始位置
     - **双流特征提取**：DINOv3 ViT-L提取图像多尺度特征$F_I \in \mathbb{R}^{784\times1024}$，Point Transformer对3D点编码得$F_P \in \mathbb{R}^{N_G\times1024}$
     - **DSTG解码器**：改进的MMDiT块，三步融合——全局上下文向量提取→图像+点特征联合注意力→MLP预测每个Gaussian的属性（位置偏移、旋转、缩放、颜色、不透明度）
   - 设计动机：视频中马匹外观不变，且真实视频视角有限→单帧前馈比逐帧优化更合理。DSTG比标准cross-attention融合效果更好（消融验证）

5. **VarenTex合成外观数据集**
   - 做什么：生成150K张高质量多视角马匹图像用于训练EquineGS
   - 核心思路：从VarenPoser mesh渲染法线图和CCM→ControlNet生成参考图像→UniTex多视角扩散模型生成一致的多视角训练图像
   - 设计动机：VarenPoser的纹理质量不够高保真，且外观网络需要多视角数据而不是单目视频

### 损失函数 / 训练策略
- **AniMoFormer**：VAREN参数L2损失 + 平滑损失 + 2D/3D关键点L1损失
- **Post-Optimization**：2D关键点损失 + mask L1损失 + 平滑损失 + 正则化
- **EquineGS**：$\mathcal{L} = \lambda_{image}(\|I-\hat{I}\|_1 + \text{LPIPS}) + \lambda_{mask}\|M-\hat{M}\|_1 + \lambda_{reg}\mathcal{L}_{reg}$
- 训练全部在合成数据上，测试时直接泛化到真实数据

## 实验关键数据

### 主实验（运动估计）

| 方法 | APT36K PCK@0.05↑ | APT36K PCK@0.1↑ | APT36K Accel↓ | AiM PCK@0.05↑ | VarenPoser CD↓ |
|------|-----------------|-----------------|--------------|---------------|---------------|
| 3D-Fauna | 20.1 | 51.4 | 189.3 | 33.3 | 43.0 |
| 4D-Fauna | 25.5 | 53.5 | 177.7 | 46.5 | 38.5 |
| AniMer | 44.5 | 76.6 | 130.5 | 55.5 | 15.2 |
| **AniMoFormer** | **61.8** | **83.9** | **128.6** | **84.2** | **3.4** |

### 外观重建（Novel View/Pose on AiM）

| 方法 | Horse PSNR↑ | Horse SSIM↑ | Horse LPIPS↓ | Zebra PSNR↑ | Zebra LPIPS↓ |
|------|------------|------------|-------------|------------|-------------|
| 3D-Fauna | 12.20 | 0.7205 | 0.2782 | 12.33 | 0.3318 |
| GART (full opt.) | 16.19 | 0.7819 | 0.2308 | 15.21 | 0.2287 |
| GART* (few-shot) | 15.42 | 0.7550 | 0.2452 | 14.31 | 0.2973 |
| **4DEquine** | 15.66 | **0.8364** | **0.1720** | **15.54** | **0.2000** |

### 消融实验

| 配置 | APT36K PCK@0.05 | AiM PCK@0.05 | AiM Accel |
|------|----------------|--------------|-----------|
| w/o PO & Temporal | 37.1 | 45.1 | 30.6 |
| w/o PO | 37.7 | 47.8 | 25.7 |
| w/o Temporal | 57.9 | 82.9 | 24.7 |
| Full AniMoFormer | **61.8** | **84.2** | **21.8** |

### 关键发现
- **AniMoFormer大幅超越所有baseline**：PCK@0.05从AniMer的44.5提升到61.8（APT36K），CD从15.2降到3.4。核心来自VAREN模型+时空Transformer+后优化三者结合
- **Temporal模块显著降低Accel**（运动更平滑），Post-Optimization显著提升PCK（对齐更准）——两者互补
- **4DEquine在SSIM/LPIPS上超越GART**：虽然PSNR略低于完整优化的GART（15.66 vs 16.19），但感知指标更好，说明外观结构更准确，且只需单帧输入
- **零样本泛化到斑马**：在zebra子集上全面超越包括GART在内的所有方法，证明模型学到了通用特征而非记忆训练纹理
- **效率优势**：每帧11秒（A100），GART需固定15分钟优化

## 亮点与洞察
- **运动-外观解耦的设计哲学**：将4D重建拆成两个独立子问题各自解决，避免了联合优化的困难。这个思路可以迁移到其他4D重建任务——只要存在参数化模型作为桥梁，运动和外观就可以独立训练。例如人体4D重建也可以用SMPL作为桥梁做类似解耦
- **DSTG双流Transformer解码器**：改进MMDiT块来融合图像特征和3D点特征，比标准cross-attention更有效。核心是先提取全局上下文做调制，再做双流联合注意力
- **合成数据训练+真实数据泛化**：VarenPoser和VarenTex完全是合成数据，但模型在真实数据上表现优秀。关键因素包括：相机轨迹多样化（fix/dolly/orbit）、形状随机化、多视角扩散模型生成高质量纹理
- **单图生成完整avatar**：EquineGS仅用一张图像就能生成可动画的360°gaussian avatar，绕过了逐视频优化的瓶颈

## 局限性 / 可改进方向
- **尾巴和鬃毛重建不足**：VAREN模型本身对尾巴/鬃毛建模有限，影响这些部位的外观质量
- **不处理动态光照**：假设外观在视频中不变，无法处理光照变化场景
- **仅限马科动物**：虽然能零样本泛化到驴和斑马，但无法处理其他类型动物（如猫狗），需要对应的参数化模型
- **改进思路**：(1) 为尾巴/鬃毛引入物理仿真表示；(2) 加入relighting模块处理动态光照；(3) 融合多关键帧信息提升外观完整性

## 相关工作与启发
- **vs GART**：GART需要逐视频优化（15min），且需要完整观测。4DEquine是前馈方法（11s/帧），仅需单帧外观输入，在SSIM/LPIPS上更好
- **vs 4D-Fauna**：无模板方法缺乏几何先验，PCK@0.05仅46.5 vs 4DEquine的84.2。但4D-Fauna可处理100+物种
- **vs AniMer**：4DEquine的AniMoFormer直接在AniMer基础上扩展了时空Transformer，PCK@0.05从44.5→61.8

## 评分
- 新颖性: ⭐⭐⭐⭐ 运动-外观解耦+VAREN桥梁的设计很巧妙，两个合成数据集的构建有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多baseline、全面消融，还有零样本泛化实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，流程图直观
- 价值: ⭐⭐⭐⭐ 在马匹4D重建这个niche领域做到了SOTA，解耦思路有通用性
