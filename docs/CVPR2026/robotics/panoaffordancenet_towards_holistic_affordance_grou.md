# PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments

**会议**: CVPR 2026  
**arXiv**: [2603.09760](https://arxiv.org/abs/2603.09760)  
**代码**: [https://github.com/GL-ZHU925/PanoAffordanceNet](https://github.com/GL-ZHU925/PanoAffordanceNet) (有)  
**领域**: 具身智能 / 全景感知 / Affordance  
**关键词**: 全景affordance grounding, 等距矩形投影失真, 球面密化, 单样本学习, 360°室内感知  

## 一句话总结
提出PanoAffordanceNet，首次定义360°室内环境中的全局affordance grounding任务，通过失真感知光谱调制器(DASM)和全球面密化头(OSDH)解决ERP几何失真和稀疏激活问题，配合多级训练目标抑制语义漂移，在自建360-AGD数据集上大幅超越现有方法（KLD从2.853→1.270）。

## 背景与动机
具身智能体（如服务机器人）在360°物理空间中操作，需要全局感知能力。然而现有affordance grounding研究存在两个根本局限：（1）以对象为中心的范式——仅理解单个物体的功能，忽视了场景级的空间组织和多实例交互；（2）局限于透视视图——受限的视野与机器人360°的操作空间严重不匹配。当现有方法直接应用于全景图像时，性能急剧下降，因为全景图像不仅仅是视野的扩展，它从根本上改变了空间特征的分布模式。

## 核心问题
全景室内affordance grounding面临三大挑战：（1）ERP投影带来的纬度依赖几何失真，极区变形严重，模型难以同时保留局部交互细节和全局功能结构；（2）非均匀采样导致功能区域分布高度稀疏，初始激活难以聚合为语义连贯、边界一致的affordance区域；（3）缺少密集像素标注下，将抽象affordance语义精确对齐到多尺度区域极其困难，容易出现语义漂移。

## 方法详解

### 整体框架
PanoAffordanceNet是端到端的单样本学习框架，包含四个核心模块：（1）双编码器特征提取（DINOv2视觉+CLIP文本）带LoRA高效适配；（2）DASM失真感知光谱调制器；（3）球面感知层次解码器（含OSDH）；（4）多级训练目标。

### 关键设计
1. **Distortion-Aware Spectral Modulator (DASM)**: 针对ERP的纬度依赖失真，通过双频谱蒸馏分离任务相关几何信号。首先通过跨模态注意力将文本语义注入视觉特征，然后将特征分解为高频分量（Laplacian算子提取边界轮廓）和低频分量（高斯平滑保留结构）。高频增强模块(HFEM)锐化赤道区域的交互边界并抑制极区伪影；低频稳定模块(LFSM)维持极区全局结构一致性，防止拉伸导致的语义碎片化。最终通过混合门控机制（语言驱动通道门×自适应空间门）选择性融合。

2. **Omni-Spherical Densification Head (OSDH)**: 解决非均匀采样导致的稀疏激活问题。首先通过全局语义发现（轻量Transformer decoder用文本query交叉注意视觉特征）获取初始affordance map A_init。然后OSDH将视觉特征投影到单位超球面，构建余弦相似度亲和矩阵S，通过top-k选种、置信度引导噪声抑制、最大值传播等操作，将稀疏种子激活扩展为拓扑连续的完整功能区域。关键是利用球面自相似性作为结构归纳偏置，仅需可学习残差标量α控制传播强度。

3. **Multi-Level Training Objective（多级训练目标）**: 三层约束协同抑制语义漂移——像素级BCE确保定位准确；分布级KL散度保持预测热力图的全局形状和强度一致；区域-文本对比损失(InfoNCE)建立视觉区域与affordance概念的语义对应，关键在于消歧同一物体上的不同affordance（如沙发扶手的"抓握"vs坐垫的"坐"）。

### 损失函数 / 训练策略
- L_total = λ₁·L_BCE + λ₂·L_KL + λ₃·L_RTC
- 优化器：AdamW，初始lr=1e-5，cosine退火
- 训练：2×NVIDIA A6000，20k迭代，batch size 4
- 输入分辨率：560×1120
- 数据增强：随机翻转+颜色抖动+全景特有增强（±3°旋转、±5%缩放、水平环绕平移）
- LoRA适配：rank=16，应用于DINOv2注意力层
- 文本编码：CLIP ViT-B/16 + CoOp prompt learner

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 360-AGD Easy | KLD↓ | 1.270 | 2.853 (OS-AGDO) | -1.583 |
| 360-AGD Easy | SIM↑ | 0.506 | 0.124 (OS-AGDO) | +0.382 |
| 360-AGD Easy | NSS↑ | 4.490 | 1.299 (OS-AGDO) | +3.191 |
| 360-AGD Hard | KLD↓ | 1.306 | 2.965 (OS-AGDO) | -1.659 |
| 360-AGD Hard | SIM↑ | 0.474 | 0.115 (OS-AGDO) | +0.359 |
| AGD20K Seen | KLD↓ | 0.739 | 0.740 (OOAL) | ≈持平 |
| AGD20K Seen | SIM↑ | 0.616 | 0.577 (OOAL) | +0.039 |

### 消融实验要点
- 模块消融（Hard Split）：Baseline KLD=1.475 → +LoRA=1.421 → +DASM=1.359 → +OSDH=1.306，每个模块均有增益
- 损失消融：仅BCE KLD=1.596；+KL=1.430；+RTC=1.459；三者结合KLD=1.306（最优）
- LoRA rank：r=16最优，r<16表示不足，r≥24过拟合（KLD升至1.403@r=32）
- Top-k鲁棒性：k∈[5,20]范围内KLD仅波动0.006，说明OSDH对超参不敏感

## 亮点
- 首次提出360°室内环境的全局affordance grounding任务，从对象级理解转向场景级推理
- DASM的双频谱分解+纬度自适应校正设计精巧，很好地解决了ERP失真问题
- OSDH的球面自相似性传播机制让稀疏信号恢复拓扑连续性，且对top-k超参极不敏感
- 构建了首个全景affordance grounding数据集360-AGD，包含Easy和Hard两个难度级别
- 在自建数据集上的提升幅度巨大（KLD降低55%+），同时在透视视图AGD20K上保持竞争力
- 真实场景验证：通过Insta360 X4头戴采集的真实数据验证了模型的泛化能力

## 局限性 / 可改进方向
- 仅处理静态场景，缺乏时间维度推理（论文自述：未来将探索动态场景的时序reasoning）
- 360-AGD数据集规模相对有限，场景多样性可进一步扩展
- 基于关键点标注的弱监督策略可能遗漏被严重遮挡的交互区域
- 未探索与3D空间表示的跨模态协同（论文自述：未来方向）
- one-shot设定下的泛化能力已很好，但在需要细粒度区分的affordance（如扶手的握vs推）上仍有挑战

## 与相关工作的对比
- vs. OOAL：OOAL是透视视图的one-shot方法，在全景场景下严重语义漂移（360-AGD KLD 2.868 vs本文1.270）
- vs. OS-AGDO：同为one-shot，但无ERP失真校正和全景特有设计，同样在360°场景下失败
- vs. LOCATE/WSMA：弱监督方法，在透视视图AGD20K上表现良好，但均未涉及全景场景
- vs. WorldAfford：场景级affordance理解但依赖SAM分割+CoT推理，非端到端
- vs. AffordanceLLM：LLM驱动但仍是透视视图+对象中心

## 启发与关联
- 与idea [分层4K全景分割用于具身智能的空间推理增强](../../../ideas/segmentation/20260316_panoramic_spatial_reasoning.md)高度相关：全景affordance grounding为全景空间推理提供了功能层面的理解
- ERP失真的频域分解方法可推广到其他全景感知任务（语义分割、目标检测等）
- OSDH的球面自相似性传播思想值得在其他稀疏信号恢复任务中探索

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首创全景affordance grounding任务，DASM和OSDH设计精巧
- 实验充分度: ⭐⭐⭐⭐ 有自建数据集+跨域验证+详细消融+超参分析+真实场景测试
- 写作质量: ⭐⭐⭐⭐ 结构完整，动机清晰，但部分公式较密集
- 价值: ⭐⭐⭐⭐⭐ 为具身智能的全景场景级感知开辟了新方向，提供了完整的任务定义+数据集+方法+benchmark
