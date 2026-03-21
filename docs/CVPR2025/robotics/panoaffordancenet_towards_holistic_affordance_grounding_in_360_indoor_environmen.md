# PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments

**会议**: CVPR 2025  
**arXiv**: [2603.09760](https://arxiv.org/abs/2603.09760)  
**代码**: https://github.com/GL-ZHU925/PanoAffordanceNet  
**领域**: 机器人 / 3D理解  
**关键词**: 全景affordance, 360度场景, ERP畸变, 球面传播, one-shot学习

## 一句话总结
提出PanoAffordanceNet——首个360°全景affordance grounding框架，通过失真感知频谱调制器(DASM)处理ERP纬度依赖畸变、全球面致密化头(OSDH)恢复稀疏激活为拓扑连续区域、多层级训练目标抑制语义漂移，并构建首个全景affordance数据集360-AGD，全面超越现有方法。

## 研究背景与动机
1. **领域现状**：视觉affordance grounding（给定动作预测物体可交互区域）是具身AI的核心能力，已从全监督发展到弱监督和基础模型驱动的范式。但现有方法局限于以物体为中心的透视图设置。
2. **现有痛点**：机器人本质上在360°空间中操作，而现有affordance方法受限于有限视角。当直接扩展到全景场景时，性能急剧下降。根本原因在于三个全景特有挑战：(1) ERP(等距矩形投影)引入纬度依赖的几何畸变，极区严重拉伸——模型难以同时保留局部交互细节和全局功能结构；(2) 非均匀采样导致功能区域高度稀疏散乱，初始激活难以聚合为语义连贯的affordance区域；(3) 缺乏密集像素级标注，在复杂360°场景中精确对齐抽象affordance语义与多尺度区域极其困难。
3. **核心矛盾**：全景图提供了完整的环境信息但等距矩形投影从根本上改变了空间特征的分布模式，使得为透视图设计的方法失效。
4. **本文要解决什么？** 如何在360°全景室内环境中实现准确的场景级affordance grounding？
5. **切入角度**：在频域而非空间域处理ERP畸变（频率天然对应不同尺度的结构），同时利用球面自相似性传播种子激活恢复拓扑连续区域。
6. **核心idea一句话**：双频谱蒸馏做纬度自适应畸变校准 + 球面种子传播做稀疏激活致密化 = 全景affordance grounding。

## 方法详解

### 整体框架
端到端的one-shot学习框架。输入为全景ERP图像（560×1120）和one-shot示例。DINOv2-B/14（LoRA微调）提取视觉特征，CLIP文本编码器+CoOp提供affordance语义嵌入。特征经DASM校准后送入球面感知分层解码器（全局语义发现+OSDH致密化），多层级损失联合训练。

### 关键设计

1. **失真感知频谱调制器 (DASM)**:
   - 做什么：通过双频谱蒸馏隔离任务相关的几何信号，自适应校准ERP引起的纬度依赖畸变
   - 核心思路：先通过cross-attention注入文本语义到视觉特征，然后用拉普拉斯算子提取高频（边界/轮廓）和高斯平滑提取低频（全局结构）。赤道区高频增强模块(HFEM)锐化交互边界，极区低频稳定模块(LFSM)保持全局结构一致性。最后通过混合门控（语言驱动通道门 + 自适应空间门）选择性融合两个频段：$\mathbf{F}_{\text{freq}} = \mathbf{F}'_v + \sum_{k \in \{h,l\}} \lambda_k (\mathbf{g}_{\text{ch}} \odot \mathbf{g}_{\text{sp}} \odot \mathbf{F}_k)$
   - 设计动机：ERP造成赤道区边缘清晰但极区结构拉伸，在频域上恰好对应高频和低频的不平衡。分频段分别处理比空间域校正更自然优雅

2. **全球面致密化头 (OSDH)**:
   - 做什么：将稀疏碎片化的初始affordance激活恢复为拓扑连续的完整功能区域
   - 核心思路：先通过cross-attention得到初始affordance map $\mathbf{A}_{\text{init}}$。然后将视觉特征投射到单位超球面，构建余弦相似度亲和矩阵 $\mathcal{S} \in \mathbb{R}^{L \times L}$。通过top-k选取高置信种子，用置信度图抑制噪声，最后种子驱动传播：$\mathbf{A}_{\text{refined}} = \mathbf{A}_{\text{init}} + \alpha \cdot \max_{j \in \mathcal{K}} (\mathcal{S}_{ij} \cdot \mathcal{C}_j)$
   - 设计动机：全景图的非均匀采样导致功能区域稀疏分散。利用球面自相似性（相似外观的区域倾向于有相同affordance）作为结构归纳偏置，用种子传播恢复连续区域

3. **多层级训练目标**:
   - 做什么：从三个层级抑制语义漂移
   - 核心思路：像素级BCE确保定位精度；分布级KL散度保持全景拓扑连续性；Region-Text对比损失（InfoNCE）将视觉区域与affordance概念精确对齐。$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{BCE}} + \lambda_2 \mathcal{L}_{\text{KL}} + \lambda_3 \mathcal{L}_{\text{RTC}}$
   - 设计动机：单纯像素级损失无法处理同一物体上多个affordance（如沙发扶手"抓"vs坐垫"坐"）的语义歧义，需要区域-文本对比约束

### 损失函数 / 训练策略
- 总损失: $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{BCE}} + \lambda_2 \mathcal{L}_{\text{KL}} + \lambda_3 \mathcal{L}_{\text{RTC}}$，其中RTC基于InfoNCE将区域特征与affordance文本嵌入对齐
- AdamW优化器，学习率1e-5，余弦退火。2×A6000 GPU，20k迭代，batch size 4
- 全景专用数据增强：随机旋转±3°、随机缩放±5%、水平环绕平移（利用360°拓扑的旋转不变性）
- 标注为keypoint-based，经高斯核转为概率热力图作为软监督

## 实验关键数据

### 主实验

| 数据集 | 指标 | PanoAffordanceNet | OOAL | OS-AGDO |
|--------|------|------------------|------|---------|
| 360-AGD Easy | KLD↓ | **1.270** | 2.868 | 2.853 |
| 360-AGD Easy | SIM↑ | **0.506** | 0.117 | 0.124 |
| 360-AGD Easy | NSS↑ | **4.490** | 1.267 | 1.299 |
| 360-AGD Hard | KLD↓ | **1.306** | 3.067 | 2.965 |
| 360-AGD Hard | SIM↑ | **0.474** | 0.097 | 0.115 |
| 360-AGD Hard | NSS↑ | **4.398** | 1.484 | 1.484 |
| AGD20K Seen | KLD↓ | **0.739** | 0.740 | — |
| AGD20K Seen | SIM↑ | **0.616** | 0.577 | 0.390 |
| AGD20K Unseen | SIM↑ | **0.475** | 0.461 | 0.382 |

在360-AGD上SIM指标提升超过4倍，NSS提升超过3倍。在标准透视AGD20K上也保持竞争力。

### 消融实验（360-AGD Hard Split）

| 配置 | KLD↓ | SIM↑ | NSS↑ | 说明 |
|------|------|------|------|------|
| Baseline | 1.475 | 0.416 | 4.196 | 无LoRA/DASM/OSDH |
| + LoRA | 1.421 | 0.429 | 4.257 | 参数高效微调 |
| + LoRA + DASM | 1.380 | 0.450 | 4.317 | 畸变校准有效 |
| + LoRA + OSDH | 1.359 | 0.448 | 4.339 | 致密化有效 |
| + LoRA + DASM + OSDH | **1.306** | **0.474** | **4.398** | 完整模型最优 |

### 关键发现
- DASM和OSDH各自独立贡献了显著提升，组合后进一步提升
- 在AGD20K透视数据集上不降性能，说明全景处理能力是额外获得的，不以牺牲透视性能为代价
- 现有透视方法（OOAL、OS-AGDO）直接用于全景图时性能急剧下降（SIM从0.577降到0.117），验证了全景affordance的独特挑战

## 亮点与洞察
- **首个360° affordance grounding工作**：定义新任务+构建新数据集+提出专用框架，完整的开创性工作
- **频域处理ERP畸变**：高频/低频分别对应边界细节和全局结构，分频段处理比空间域滤波更自然
- **球面种子传播**：利用视觉自相似性作为归纳偏置，无需额外标注即可从稀疏激活恢复连续区域
- **One-shot设定**：仅需一个示例即可推广到新affordance类别，标注效率极高

## 局限性 / 可改进方向
- 360-AGD数据集规模和场景多样性有限，可能不足以覆盖真实机器人应用场景
- OSDH的亲和矩阵 $\mathcal{S} \in \mathbb{R}^{L \times L}$ 在高分辨率下计算量大
- 未与3D场景理解方法结合，可利用depth信息进一步提升
- 19个affordance类别可能不够覆盖机器人的实际交互需求

## 相关工作与启发
- **vs OOAL/OS-AGDO**: 透视图方法直接用于全景图效果极差（SIM 0.117 vs 0.506），验证了ERP畸变处理的必要性
- **vs 全景语义分割**: 全景分割处理语义类别，affordance grounding处理动作-区域关系——更接近机器人的交互需求
- **vs WorldAfford**: 使用LLM做场景级affordance但仍依赖SAM做物体分割，PanoAffordanceNet端到端更简洁

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个360° affordance grounding工作，任务定义+数据集+方法全面创新
- 实验充分度: ⭐⭐⭐⭐ 有消融和跨域验证，但新数据集规模有限
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 为具身AI在全景场景的应用奠定基础
