# TRACE: Your Diffusion Model is Secretly an Instance Edge Detector

**会议**: ICLR 2026  
**arXiv**: [2503.07982](https://arxiv.org/abs/2503.07982)  
**代码**: [项目页](https://shjo-april.github.io/TRACE)  
**领域**: segmentation  
**关键词**: diffusion model, instance edge, unsupervised segmentation, panoptic segmentation, self-attention, boundary detection  

## 一句话总结
发现文本到图像扩散模型的自注意力图在去噪过程特定时间步隐式编码了实例边界信息，提出 TRACE 框架通过实例涌现点(IEP)和注意力边界散度(ABDiv)提取这些边界，并蒸馏为单步边缘检测器，在无监督实例分割和弱监督全景分割上大幅超越已有方法。

## 背景与动机
1. 高质量实例/全景分割依赖昂贵且不一致的密集标注(掩码/框/点)
2. 无监督方法(MaskCut等)基于 DINO 特征聚类，但常合并同类相邻物体或碎片化
3. 弱监督语义分割接近全监督精度，但扩展到全景仍需点/框标注来区分实例
4. 点标注存在人类偏差（标注者倾向标注物体中心，一致性差）
5. **核心发现**: 扩散模型自注意力在去噪过程中短暂但可靠地呈现实例级结构
6. 交叉注意力即使有提示也不能可靠分离相邻物体，但自注意力在特定步骤可以

## 方法详解
**Instance Emergence Point (IEP)**:
- 在去噪轨迹中测量连续步骤间自注意力的 KL 散度
- $t^\star = \arg\max_t D_{KL}(SA(X_{t_{prev}}) \| SA(X_t))$
- KL 峰值对应实例结构最显著的时间步

**Attention Boundary Divergence (ABDiv)**:
- 在 IEP 处的自注意力图上计算十字邻域散度
- $\text{ABDiv}(SA)_{i,j} = D_{KL}(SA_{i+1,j} \| SA_{i-1,j}) + D_{KL}(SA_{i,j+1} \| SA_{i,j-1})$
- 同实例内部像素注意力分布相似，跨实例边界处剧烈发散

**One-Step Self-Distillation**:
- 用 LoRA 微调扩散 backbone + 轻量边缘解码器
- 伪标签用 ABDiv 边缘图（三值化：边/内/不确定，不确定区域排除）
- 推理时单次前向(t=0)即产生连通边缘图，81× 加速

**Background-Guided Propagation (BGP)**: 用边缘作为分隔器，将碎片化掩码在实例边界内传播合并

## 实验关键数据
**无监督实例分割 (COCO 2017, fine-tuned)**:
| 方法 | AP_mk |
|------|-------|
| CutLER | 8.7 |
| + CutS3D (深度) | 10.7 |
| **+ TRACE** | **12.8 (+4.1)** |
| ProMerge+ | 8.9 |
| **+ TRACE** | **13.0 (+4.1)** |

**弱监督全景分割 (VOC 2012)**:
- DHR + TRACE (tag only): PQ **58.3**, 超越点监督 EPLD (56.6) +1.7
- COCO 上同样仅用 tag 就超越点监督基线

**效率**: 蒸馏后 81× 加速，仅 6% 运行时开销

## 亮点
- **深刻洞察**: 揭示扩散模型自注意力中隐藏的实例边界先验
- **无需标注**: 完全无监督地提取实例级边界
- **实用框架**: ABDiv 非参数化 + 单步蒸馏推理高效
- **Tag → 全景**: 仅用图像级标签就超越点监督的全景分割方法
- **模型无关**: 可插入已有分割流水线(MaskCut/ProMerge等)带来一致提升

## 局限性
- IEP 的最优时间步可能因图像内容和扩散模型版本而异
- 蒸馏需要训练边缘解码器，仍有一定成本
- 扩散模型本身计算量大，即使单步仍比纯 ViT 方案慢
- 对极小目标和高度遮挡场景的边界提取效果未充分讨论

## 相关工作
- **MaskCut/CutLER/UnSAM**: 基于 DINO 的无监督实例分割
- **CutS3D/CUPS**: 深度引导的实例分离，TRACE 表现更好
- **DiffCut/DiffSeg**: 利用扩散注意力做语义分割，但未做实例级
- **Point2Mask/EPLD**: 点监督全景分割，TRACE 仅用 tag 即超越

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (扩散模型作为实例边缘检测器，洞察深刻)
- 实验充分度: ⭐⭐⭐⭐⭐ (UIS + 全景 + 5种扩散backbone + 消融)
- 写作质量: ⭐⭐⭐⭐⭐ (图文并茂，讲故事能力强)
- 价值: ⭐⭐⭐⭐⭐ (开辟扩散模型用于实例分割的新范式)
