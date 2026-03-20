# SAP: Segment Any 4K Panorama

**会议**: arXiv 2026 (HKUST(GZ) & LIGHTSPEED & HKU)  
**arXiv**: [2603.12759](https://arxiv.org/abs/2603.12759)  
**作者**: Lutao Jiang*, Zidong Cao*, Weikai Chen†, Xu Zheng, Yuanhuiyi Lyu, Zhenyang Li, Zeyu Hu, Yingda Yin, Keyang Luo, Runze Zhang, Kai Yan, Shengju Qian, Haidi Fan, Yifan Peng, Xin Wang, Hui Xiong, Ying-Cong Chen†
**代码**: Coming Soon（项目页：[https://lutao2021.github.io/SAP_Page/](https://lutao2021.github.io/SAP_Page/)）  
**领域**: 全景图像实例分割 / 分割基础模型  
**关键词**: 全景分割, SAM2, 4K高分辨率, 固定轨迹视频分割, 透视投影  

## 一句话总结
SAP将全景图实例分割重新建模为固定轨迹的透视视频分割任务，通过column-first zigzag扫描将ERP全景图分解为连续重叠的透视patch序列，并用InfiniGen合成183K张4K全景图微调SAM2，在真实4K全景图上零样本mIoU比原始SAM2平均提升+17.2。

## 背景与动机
360°全景相机在机器人、AR/VR、具身智能中日益普及，对高分辨率全景实例分割的需求迫切。但SAM/SAM2等基础模型在全景图上性能严重下降：(1) 4K全景图(4096×2048)被resize到1024²时丢失大量细节；(2) 等距圆柱投影(ERP)导致极区严重畸变、左右边缘不连续(seam artifact)；(3) 现有全景分割方法主要关注语义分割，高分辨率实例级分割几乎未被探索。更深层的问题是：SAM2的流式memory机制假设输入是时序连续的视频帧，而ERP滑窗patch破坏了这种视角连续性假设。

## 核心问题
全景图的球面拓扑结构与SAM2等streaming memory模型的时序连续性假设之间存在根本性的**拓扑-记忆对齐**问题（topology–memory alignment problem）。滑窗式直接裁切ERP图像会打破物理视角连续性，导致memory bank在错误假设下运行，造成身份传播不稳定和分割不一致。同时，大规模高分辨率全景实例标注数据的缺乏进一步制约了模型适配。

## 方法详解

### 整体框架
输入：一张ERP全景图 + 用户prompt点 → (1) 沿预定义扫描轨迹将ERP转换为N帧重叠透视patch伪视频 → (2) 将prompt点投影到可见的透视帧中 → (3) 用微调后的SAM2对伪视频进行分割 → (4) 将逐帧mask反投影回ERP平面，通过逐像素最大值融合得到最终全景分割mask。整个过程完全在透视投影空间中操作，避免了ERP畸变问题。

### 关键设计
1. **全景图→透视伪视频转换 (Pano-to-Perspective Video)**：通过三阶段几何投影（定义虚拟相机内外参 → 建立2D网格反投影到球面射线 → 球面坐标转换采样ERP像素）将全景图渲染为1024×1024的透视patch。FoV=90°，overlap ratio r=0.5，pitch步长45°，产生N_yaw×N_pitch个视点。用户prompt点也投影到对应透视帧中（检查z_c>0且在图像范围内）。
2. **Column-first Zigzag扫描轨迹 (Scanning Trajectory)**：核心要求是轨迹形成**无限循环**——从任意起始点出发都能回到该点。Column-first扫描通过上下往复运动实现无限循环（row-first做不到）。奇数列从上到下、偶数列从下到上，相邻帧只改变一个角度维度（yaw或pitch），保证视频般平滑过渡。训练时随机采样起始点，将轨迹复制2份拼接，取连续N帧窗口确保全覆盖。
3. **183K合成数据 + 混合训练**：用InfiniGen引擎耗费40,000 GPU小时合成183,440张4K ERP全景图，含6,409,732个实例mask（小物体37.84%、中物体25.70%、大物体36.47%）。训练时混合SAM2原始训练数据（SA-1B + SA-V），防止灾难性遗忘。

### 损失函数 / 训练策略
- 冻结Hiera-Large图像编码器，只更新memory attention、memory encoder、mask decoder和prompt encoder
- AdamW优化器，batch size 128，lr=2×10⁻⁴（cosine schedule），weight decay=0.1，gradient clipping=0.1
- 每张全景图转换为1024×1024透视视频，FoV=90°，50% overlap，column-first snake order扫描2个cycle

## 实验关键数据

### PAV-SOD基准（真实4K全景，零样本）
| 模型 | 1-click mIoU | Small | Medium | Large | 3-click mIoU | Small | Medium | Large |
|------|-------------|-------|--------|-------|-------------|-------|--------|-------|
| SAM2-tiny | 51.6 | 46.3 | 55.5 | 49.1 | 82.2 | 61.7 | 79.6 | 88.8 |
| SAP-tiny | **75.8** | 53.9 | 76.6 | 79.7 | **84.8** | 68.7 | 84.7 | 88.4 |
| Δ(tiny) | **+24.2** | +7.6 | +21.1 | +30.6 | +2.6 | +7.0 | +5.1 | -0.4 |
| SAM2-large | 66.3 | 50.7 | 71.9 | 64.4 | 84.3 | 65.5 | 81.7 | 90.5 |
| SAP-large | **77.3** | 61.1 | 76.2 | 81.7 | **86.1** | 71.8 | 84.3 | 90.6 |
| Δ(large) | **+11.0** | +10.4 | +4.3 | +17.3 | +1.8 | +6.3 | +2.6 | +0.1 |
| **跨4模型平均** | **+17.2** | | | | | | | |

### HunyuanWorld-1.0基准（8K合成卡通全景，零样本）
| 模型 | 1-click mIoU | 3-click mIoU |
|------|-------------|-------------|
| SAM2-large | 70.8 | 82.1 |
| SAP-large | **77.4** (+6.6) | **88.3** (+6.2) |

### InfiniGen基准（合成4K全景）
| 模型 | 1-click mIoU | 3-click mIoU |
|------|-------------|-------------|
| SAM2-large | 62.8 | 81.4 |
| SAP-large | **81.9** (+19.1) | **89.0** (+7.6) |

### Seam/Pole区域子集（76个实例）
| 方法 | SAM2 | SAM2+scan | SAP |
|------|------|-----------|-----|
| mIoU | 46.1 | 53.8 | **68.2** (+22.1) |

### 消融实验要点
- **合成数据至关重要**：只用SAM2原数据训练在PAV-SOD上1-click mIoU仅60.0；只用合成数据67.3；混合训练达77.3（+10.0），说明混合训练既提升全景能力又防止遗忘
- **Overlap是关键**：无overlap扫描的mIoU仅66.5，加入50% overlap提升到77.3（+10.8），因为SAM2 memory bank需要帧间重叠才能有效运行
- **透视投影 vs 直接裁切ERP**：perspective transform达77.3 mIoU，直接ERP patchify仅72.6（+4.7），说明消除ERP畸变域gap很重要
- **固定轨迹扫描 vs 直接resize全景图微调**：使用扫描策略77.3 vs 直接resize微调73.0（+4.3），证明轨迹对齐训练对稳定memory使用至关重要
- **Seam/Pole区域受益最大**：SAP在seam/pole区域mIoU达68.2，比vanilla SAM2的46.1提升+22.1，得益于扫描策略使seam区域连通、极区形状正常化

## 亮点 / 我学到了什么
- **视角转换的idea很巧妙**：不是设计新的球面卷积或畸变感知模块，而是将全景分割"退化"为普通透视视频分割，完美复用SAM2的时序memory机制——用问题变换代替模型改造
- **Column-first zigzag轨迹设计严谨**：数学上保证无限循环、全覆盖、任意起始点，且相邻帧只变一个维度——这种将几何约束转化为序列设计的思路很通用
- **大规模合成数据路线的成功案例**：40K GPU小时合成183K全景图+6.4M实例mask，加上混合训练策略，在真实场景零样本迁移效果显著——对数据稀缺领域有参考价值

## 局限性 / 可改进方向
- **仅支持单张全景图**，未扩展到全景视频（作者在Future Work中也提到），无法利用跨帧时序一致性
- **计算开销较大**：FoV=90°、50% overlap下每张全景图需要24帧1024×1024的推理（N_pitch=3, N_yaw=8），更密集的扫描（如Δ=30°→48帧）会进一步增加开销
- **代码和数据均未公开**（Coming Soon），可复现性暂时无法验证；合成数据的多样性和domain gap问题也值得关注

## 与相关工作的对比
1. **OmniSAM [ICCV 2025]**：同样利用SAM2处理全景图，但直接在ERP上做滑窗裁切——迫使模型处理极区畸变，未恢复视角连续性。SAP通过透视投影+轨迹对齐训练从根本上解决了拓扑-memory不匹配问题。实验中SAP在large模型上1-click mIoU比OmniSAM的ERP patchify策略高4.7。
2. **Trans4PASS [CVPR 2022]**：设计Deformable Patch Embedding处理球面畸变进行全景语义分割。属于"设计新模块适配畸变"路线，而SAP属于"消除畸变回归透视"路线，后者可复用强大的透视预训练基础。
3. **SAM2 [ICLR 2025]**：SAP的基座模型。SAP的核心贡献是揭示SAM2在全景图上失败的根本原因（拓扑-memory不匹配）并提出修复方案，而非替换SAM2。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐⭐ 将全景分割重新建模为拓扑-memory对齐问题，视角新颖且解法优雅
- 实验充分度: ⭐⭐⭐⭐ 3个benchmark、4种模型尺寸、5组消融，数据充分且逻辑清晰
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，数学推导完整，结构紧凑
- 对我的价值: ⭐⭐⭐⭐ "通过输入变换复用基础模型"的思路有通用借鉴意义
