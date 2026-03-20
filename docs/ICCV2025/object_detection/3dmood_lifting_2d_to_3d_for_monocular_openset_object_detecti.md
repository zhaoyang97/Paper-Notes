# 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection

**会议**: ICCV 2025  
**arXiv**: [2507.23567](https://arxiv.org/abs/2507.23567)  
**代码**: [royyang0714.github.io/3D-MOOD](https://royyang0714.github.io/3D-MOOD) (有)  
**领域**: 目标检测  
**关键词**: 单目3D目标检测, 开放集检测, 2D-to-3D lifting, 几何感知query, canonical image space  

## 一句话总结
提出首个端到端的单目开放集3D目标检测器3D-MOOD，通过将开放集2D检测"提升"到3D空间，结合几何感知3D query生成与canonical image space设计，在Omni3D闭集和Argoverse 2/ScanNet开集基准上均达到SOTA。

## 背景与动机
单目3D目标检测（3DOD）仅依靠单张RGB图像来识别和定位3D物体，成本低但挑战大。现有方法几乎全部在**闭集**设定下工作——训练集和测试集共享相同场景和类别。但在机器人、AR/VR等实际应用中，模型经常遇到**新环境**和**新类别**的物体，闭集方法完全无法应对。

3D数据缺乏丰富的视觉-语言对（不像2D领域有海量image-text pairs），因此直接在3D上做开放集分类非常困难。另一方面，单目深度估计在跨数据集泛化上也面临固有挑战。这两个问题叠加使得开放集单目3DOD一直没有被真正解决。

## 核心问题
**如何让单目3D检测器同时具备：(1) 识别训练时未见过的新类别物体的能力（开放词汇分类），(2) 在未见过的场景中依然准确估计3D位置/尺寸/朝向的能力（跨域3D回归泛化）？**

关键挑战在于：3D数据标注稀缺且没有文本对——无法像2D那样通过视觉-语言对齐实现开放集。同时，跨不同数据集训练时，图像分辨率、相机内参差异巨大，传统的resize+padding策略会导致相机内参和图像尺寸不一致的歧义。

## 方法详解

### 整体框架
3D-MOOD建立在开放集2D检测器Grounding DINO (G-DINO)之上。输入为单目图像$\mathbf{I}$和语言提示$\mathbf{T}$（描述感兴趣的物体类别），输出为3D检测框$\mathbf{D}^{3D}$（包含3D中心坐标、长宽高、6D朝向）以及类别$\hat{\mathbf{C}}$。

核心思路是"**从2D lift到3D**"：先用G-DINO的视觉-语言融合能力得到开放集2D检测结果，再通过设计的3D bounding box head从2D object query中估计"提升参数"，将2D框差分地映射为3D框。整个pipeline端到端可训练，允许2D和3D任务联合优化。

Pipeline:
1. 图像编码器(Swin-T/B)提取图像特征$\mathbf{q}_{\text{image}}$，文本骨干(BERT)提取文本特征$\mathbf{q}_{\text{text}}$
2. 跨模态Transformer解码器逐层融合图像和文本特征，生成2D object query $\mathbf{q}_{2d}^i$
3. 2D box head预测2D检测框，类别通过query和文本嵌入的相似度确定（开放集分类）
4. Geometry-aware 3D Query Generation模块将2D query与相机嵌入和深度特征融合，生成3D query $\mathbf{q}_{3d}^i$
5. 3D bounding box head从3D query预测提升参数（投影3D中心偏移、对数深度、对数尺寸、6D旋转），结合2D框和相机内参$\mathbf{K}$得到最终3D检测

### 关键设计

1. **3D Bounding Box Head（2D→3D可微提升）**: 从2D object query预测12维3D属性。3D定位通过预测投影3D中心与2D框中心的偏移$[\hat{u}, \hat{v}]$加上缩放对数深度$\hat{d}$实现：$\hat{z} = \exp(\hat{d}/s_{\text{depth}})$。3D尺寸直接预测缩放对数值（不使用类别先验，因为开放集场景下类别先验不可用）。朝向采用6D旋转参数化（而非仅yaw角），支持室内外通用场景。整个提升过程可微分：$\hat{\mathbf{D}}^{3D}_i = \mathbf{Lift}(\text{MLP}^{3D}_i(\mathbf{q}_{3d}^i), \hat{\mathbf{D}}^{2D}_i, \mathbf{K})$。

2. **Canonical Image Space（规范图像空间）**: 解决跨数据集训练时图像分辨率和相机内参不一致的核心问题。之前方法（如Cube R-CNN按短边resize+右下padding、G-DINO按长边resize+右下padding）导致：(a) 大量无用的零填充浪费GPU资源；(b) 相机内参$\mathbf{K}$不随resize改变，训练和推理时内参不一致；(c) 打破了中心投影假设。3D-MOOD的方案是固定输入分辨率$[H_c \times W_c]$（800×1333），保持原始宽高比resize后做**中心padding**，同时相应调整内参。这样训练和推理时观测统一，GPU显存也减少（17GB vs 21-23GB per batch=2）。

3. **Auxiliary Metric Depth Estimation（辅助度量深度估计）**: 增加一个FPN提取深度特征$\mathbf{F}$，通过Transformer block生成深度特征$\mathbf{F}^d_{16}$，用相机嵌入$\mathbf{E}$（参考UniDepth的设计）条件化后上采样到1/8分辨率，预测全图对数深度$\hat{d}_{\text{full}}$，共享同一个深度缩放因子$s_{\text{depth}}$。用Scale-invariant log loss监督。提供全局场景几何理解，为后续query生成提供条件信号。

4. **Geometry-aware 3D Query Generation（几何感知3D查询生成）**: 为了让3D估计在跨域场景下泛化，将2D object query通过两步注入几何先验：(1) 先与相机嵌入$\mathbf{E}$做交叉注意力，使query感知场景特定的相机属性；(2) 再与深度特征$\mathbf{F}^d_8|\mathbf{E}$做交叉注意力，对齐深度估计和3D框估计。注意：深度特征交叉注意力部分**梯度截断（detach）**以稳定训练。消融显示此模块在开集设定下提升最大（+1.3 ODS），有效编码几何先验帮助泛化。

### 损失函数 / 训练策略
- **2D损失$L_{2D}$**: L1 + GIoU（框回归）+ 对比损失（预测物体和语言token的分类，继承GLIP设计）
- **3D损失$L_{3D}$**: L1 loss分别监督各3D属性（投影中心偏移、深度、尺寸、朝向）
- **辅助深度损失$L^{aux}_{depth}$**: Scale-invariant log loss，权重$\lambda_{\text{depth}}=10$
- 总损失：$L_{\text{final}} = \sum_{i=0}^{l}(L^i_{2D} + L^i_{3D}) + \lambda_{\text{depth}} L^{aux}_{\text{depth}}$
- 每层解码器都有独立的2D和3D head（多层监督）
- 深度GT来源：各Omni3D子数据集的深度GT、投影LiDAR点或SfM点
- 训练120 epochs，batch size 128，lr=0.0004；消融实验12 epochs，batch size 64
- 骨干网络：Swin-T(29M) / Swin-B(88M)

## 实验关键数据

### 开放集结果（Omni3D训练 → AV2/ScanNet测试）
| 数据集 | 指标 | 3D-MOOD (Swin-B) | Cube R-CNN | OVM3D-Det | 提升 |
|--------|------|------|----------|----------|------|
| Argoverse 2 | ODS | **23.8** | 8.9 | 8.8 | +14.9 vs Cube R-CNN |
| Argoverse 2 | ODS(N) 新类别 | **14.8** | 0.0 | 1.7 | 从零到可用 |
| ScanNet | ODS | **31.5** | 19.5 | 16.3 | +12.0 vs Cube R-CNN |
| ScanNet | ODS(N) 新类别 | **15.7** | 0.0 | 8.8 | +6.9 vs OVM3D-Det |

### 闭集结果（Omni3D test）
| 方法 | AP3D omni↑ |
|------|-----------|
| Cube R-CNN | 23.3 |
| Uni-MODE* | 28.2 |
| **3D-MOOD (Swin-T)** | **28.4** |
| **3D-MOOD (Swin-B)** | **30.0** |

### 消融实验要点
| 设定 | CI | Depth | GA | AP3D omni (闭集) | ODS open (开集) |
|------|-----|-------|-----|-----------------|----------------|
| Baseline | - | - | - | 24.1 | 23.6 |
| +Canonical Image | ✓ | - | - | 25.5 (+1.4) | 24.5 (+0.9) |
| +Depth | ✓ | ✓ | - | 26.2 (+0.7) | 24.7 (+0.2) |
| +Geometry-aware | ✓ | ✓ | ✓ | **26.8 (+0.6)** | **26.0 (+1.3)** |

- Canonical Image Space对闭集和开集都有明显帮助，且减少GPU占用（21→17 GB）
- 辅助深度头对闭集帮助较大（+0.7），对开集帮助有限（+0.2），作者推测是Omni3D深度数据不够多样
- 几何感知query生成在开集设定下提升最显著（+1.3 ODS），说明几何先验对跨域泛化至关重要
- Geometry-aware比Cube R-CNN的virtual depth收敛更快（12 epoch对比：26.8 vs 21.6）

## 亮点
- **首次定义并解决开放集单目3DOD问题**：建立了完整的benchmarks（Omni3D→AV2/ScanNet）和评估指标ODS
- **将开放集能力从2D"免费"继承到3D的思路很巧妙**：不需要3D领域的视觉-语言对，只需设计可微的2D→3D提升模块，分类能力完全来自2D开放集检测器
- **Canonical Image Space设计**：简洁但有效——固定分辨率+中心padding+同步调整内参，同时省显存
- **ODS评估指标**：用归一化距离而非IoU做匹配，对小/薄物体更公平；加入TP误差（ATE/ASE/AOE）综合评价，比单纯AP更全面
- **端到端联合训练的优势**：OVM3D-Det使用pipeline方式（G-DINO+SAM+UniDepth+LLM生成伪GT），无法端到端优化，效果明显不如3D-MOOD

## 局限性 / 可改进方向
- **推理速度较慢**：3D-MOOD (Swin-T) 17 FPS vs Cube R-CNN (DLA-34) 68 FPS，重型backbone+多个head的代价
- **辅助深度头对开集泛化帮助有限**：深度数据不够多样（仅Omni3D），若引入更多深度训练数据（如MiDAS/UniDepth预训练）可能进一步提升
- **深度估计精度仍有差距**：KITTI Eigen-split上绝对相对误差9.1%，远逊于UniDepth(4.21%)和Metric3Dv2(4.4%)
- **开集场景下IoU-based AP极低**：即使ODS很高，IoU-based AP3D仍然很低（如AV2上14.7%），说明3D定位精度仍有较大提升空间
- **可能可以引入更强的depth foundation model**作为frozen特征提供器，而非从头训练depth head
- **可探索将开放集能力推广到3D跟踪**（作者团队有CC-3DT等3D跟踪经验）

## 与相关工作的对比
| 对比方法 | 核心差异 | 优势 | 劣势 |
|---------|---------|------|------|
| **Cube R-CNN** | 闭集统一3DOD，用类别先验预测尺寸，virtual depth | 3D-MOOD在闭集上超越Cube R-CNN 6.7 AP，且可检测新类别 | Cube R-CNN速度更快(68 FPS) |
| **OVM3D-Det** | Pipeline方式：用foundation models生成伪3D GT | 3D-MOOD端到端训练、性能显著更好；OVM3D-Det的depth model还用了目标域数据训练 | OVM3D-Det不需要3D标注 |
| **Uni-MODE** | 域置信度+BEV检测器联合训练室内外 | 3D-MOOD在Omni3D闭集上略优(30.0 vs 28.2)且有开集能力 | Uni-MODE未开源无法直接对比开集 |

## 启发与关联
- **2D→3D lift的范式**非常通用：只要有好的2D开放集检测器，就能通过学习"提升参数"扩展到3D，这个思路可以迁移到开放集3D分割、3D跟踪等任务
- **Canonical Image Space**的中心padding+内参同步调整策略可以被任何跨数据集训练的检测/深度估计方法采用
- **ODS评估指标**可以作为这个领域未来工作的标准评估方案
- 论文证明了在3D缺乏视觉-语言数据时，从2D"借用"语义对齐能力是可行路径，启发后续在其他3D任务中也利用2D基础模型的开放集能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个端到端开放集单目3DOD，问题定义和benchmark贡献大，但单个技术模块（depth head、canonical space）不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 闭集+开集+跨域+消融+backbone对比+评估指标分析，非常完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机明确，但部分表格和公式排版可以更紧凑
- 价值: ⭐⭐⭐⭐⭐ 开创了开放集单目3DOD新方向，提供了benchmark和评估指标，后续工作必引
