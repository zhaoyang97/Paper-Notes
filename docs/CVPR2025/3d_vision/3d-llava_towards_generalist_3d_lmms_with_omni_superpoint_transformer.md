# 3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer

**会议**: CVPR 2025  
**arXiv**: [2501.01163](https://arxiv.org/abs/2501.01163)  
**代码**: [https://github.com/djiajunustc/3D-LLaVA](https://github.com/djiajunustc/3D-LLaVA)  
**领域**: 多模态VLM / 3D场景理解 / 大语言模型  
**关键词**: 3D LMM, Superpoint Transformer, Visual Connector, Referring Segmentation, Scene Understanding  

## 一句话总结
提出3D-LLaVA，一个极简架构的通用3D大语言多模态模型，核心是**Omni Superpoint Transformer (OST)**作为多功能视觉连接器，同时充当视觉特征选择器、视觉提示编码器和分割掩码解码器，仅用点云输入就在ScanQA（92.6 CiDEr）、ScanRefer（43.3 mIoU）等5个基准上全面达到SOTA。

## 背景与动机
现有3D LMM存在三个问题：(1) 依赖复杂pipeline——离线多视角特征提取（如Chat-Scene需要预提取2D+3D实例特征）或额外的任务特定头；(2) 视觉连接器功能单一——MLP投影或Q-Former只做特征转换，无法处理视觉提示或做分割；(3) 大多数3D LMM只能生成文本，不能接地语言到3D掩码。2D LMM领域已有LLaVA这样的简洁范式，但3D领域缺乏同等简洁且强大的基线。

## 核心问题
如何设计一个**无需离线特征提取、无需额外任务特定模块**的统一3D LMM架构，使其既能理解对话/问答/描述，又能处理视觉提示交互和生成3D分割掩码？

## 方法详解

### 整体框架
点云 → Sparse 3D U-Net特征提取 → 超点池化聚合为~数百个超点 → OST处理（多功能：特征增强+选择+提示编码+掩码解码）→ 投影到语言空间 → LLM（Vicuna-7B + LoRA）→ 文本输出 + 可选[SEG]触发掩码解码。

### 关键设计
1. **Omni Superpoint Transformer (OST)**：架构基于自注意力（非交叉注意力），超点特征同时作为query和key/value。三合一功能：
   - **视觉特征选择器**：根据objectness分数选择top-K（默认100）超点token输入LLM，减少计算开销
   - **视觉提示编码器**：无参数Visual Sampler将点击/框/掩码提示通过三近邻插值或平均池化转为特征，拼接到超点query中送入OST，通过掩码注意力防止提示影响超点（单向信息流）
   - **掩码解码器**：LLM输出[SEG]时，对应hidden state投影为分割query，送入冻结的OST，mask head生成二值掩码

2. **距离自适应自注意力**：在自注意力中引入基于超点欧几里得距离的偏置项$\sigma \cdot D$，引导相近超点间更强交互

3. **混合预训练**：联合实例分割监督（ScanNet200标注）+ 2D-to-3D知识蒸馏（LLaVA-1.5的CLIP-ViT-L特征通过几何对应lift到3D），一箭双雕：获取感知先验 + 借2D特征做3D-语言对齐的桥梁

### 损失函数 / 训练策略
- **预训练（Stage 1）**：$\mathcal{L}_{Pre} = \mathcal{L}_{Cls} + \mathcal{L}_{Mask} + \mathcal{L}_{KD}$，512 epochs on ScanNet200
- **指令微调（Stage 2）**：$\mathcal{L}_{IFT} = \mathcal{L}_{text} + 0.1 \times \mathcal{L}_{mask}$，冻结3D编码器+OST+LLM主体，仅训练投影层+LoRA
- 8×RTX 3090，batch 2/GPU，梯度累积8步，AdamW，cosine调度，lr=2e-4
- 数据：295K条指令微调数据（ScanRefer+Nr3D+Multi3DRefer+ScanQA+SQA3D+Scan2Cap）

## 实验关键数据

### 主要结果（统一训练，无单独微调）
| 任务 | 数据集 | 指标 | 3D-LLaVA | 前最优 | 提升 |
|------|--------|------|----------|--------|------|
| 3D QA | ScanQA | CiDEr | **92.6** | 87.7 (Chat-Scene) | +4.9 |
| 3D QA | ScanQA | BLEU-4 | **17.1** | 14.3 (Chat-Scene) | +2.8 |
| Situated QA | SQA3D | EM | 54.5 | 54.6 (Chat-Scene) | -0.1 |
| 3D Dense Cap | Scan2Cap | C@0.5 | **78.8** | 77.2 (Chat-Scene) | +1.6 |
| Referring Seg | ScanRefer | mIoU | **43.3** | 41.7 (SegPoint*) | +1.6 |
| Referring Seg | Multi3DRefer | mIoU | **42.7** | 36.1 (SegPoint*) | +6.6 |

注意：3D-LLaVA仅用点云，不用多视角图像；Chat-Scene用PC+Image。

### 消融实验要点
- **视觉提示编码**：OST作为编码器（78.8 CiDEr）> 池化直接使用（76.8）> 坐标投影MLP（68.7）
- **视觉token数量**：100最优；50时CiDEr下降1.5%，200/400无明显提升
- **Box-level grounding**：虽然设计用于掩码级，直接转bbox后Acc@0.25=51.2%，优于大多数竞争方法

## 亮点
- **极简设计**：单一OST模块同时解决特征选择、提示编码、掩码解码三个任务，无需额外模块
- **仅用点云**：不依赖离线多视角图像特征，部署更简单，但性能超越使用2D+3D特征的方法
- **通用性强**：一次训练覆盖QA、描述、指称分割等所有3D理解任务，是首个同时覆盖文本生成和3D分割的统一3D LMM
- **混合预训练策略**：实例分割+2D知识蒸馏的组合为3D编码器提供了强perception prior

## 局限性 / 可改进方向
- 仅在ScanNet室内场景验证，室外/大规模场景效果未知
- Box-level grounding（Acc@0.5=40.6%）弱于Chat-Scene（50.2%），因为方法为掩码优化非bbox优化
- 3D数据仍是瓶颈，仅295K条指令数据。作者指出数据收集是下一步重点
- 预训练需要ScanNet200的实例标注，限制了对新场景的适应性

## 与相关工作的对比
- **Chat-Scene**：用离线Mask3D+2D+3D实例特征，pipeline复杂。3D-LLaVA在ScanQA上CiDEr高4.9%，且仅用点云
- **SegPoint**：首个结合LLM做referring segmentation的方法，但需微调。3D-LLaVA统一训练后ScanRefer mIoU高1.6%
- **Grounded 3D-LLM**：也用点云输入，但3D-LLaVA在ScanQA CiDEr上高19.9%
- **LEO**：ScanQA CiDEr 101.4似乎更高，但其设定不同（访问GT对象信息），不直接可比

## 启发与关联
- "一个视觉连接器解决所有任务"的设计思路适用于2D多模态模型——将project层升级为多功能Transformer
- 超点表示作为3D场景的token化方案比FPS采样更信息保留，可推广到其他3D-LLM

## 评分
- 新颖性: ⭐⭐⭐⭐ OST三合一设计优雅，但核心组件（超点、掩码Transformer）均为已有技术的巧妙组合
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集、多任务对比、丰富消融、定性可视化
- 写作质量: ⭐⭐⭐⭐⭐ Motivation清晰、对比系统、方法描述详尽
- 价值: ⭐⭐⭐⭐⭐ 简洁统一的架构+SOTA性能+开源代码，是3D LMM的强基线
