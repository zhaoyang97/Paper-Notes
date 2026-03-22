<!-- 由 src/gen_stubs.py 自动生成 -->
# FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis

**会议**: NEURIPS2025  
**arXiv**: [2509.20295](https://arxiv.org/abs/2509.20295)  
**代码**: [GitHub](https://github.com/Chhro123/fast-foreground-aware-anomaly-synthesis)  
**领域**: segmentation  
**关键词**: anomaly synthesis, industrial anomaly segmentation, diffusion model, foreground-aware, accelerated sampling  

## 一句话总结
提出 FAST，一个面向分割的工业异常合成框架，通过前景感知重建模块（FARM）和异常感知加速采样（AIAS）在仅 10 步去噪下生成高质量合成异常，在 MVTec-AD 上 mIoU 达 76.72%，超越所有先前方法。

## 背景与动机
- 工业异常分割需要像素级标注，但真实异常稀缺、多样、标注昂贵
- 合成异常数据是有前景的替代方案，但现有方法存在三大限制：
  1. GAN 方法缺乏可控性
  2. 手工方法（补丁替换）缺乏结构一致性和复杂度
  3. 扩散方法对所有空间区域一视同仁，忽略异常区域的统计特性，且需上千步采样

## 核心问题
如何在保持生成质量的同时大幅加速异常合成，并显式建模异常区域的前景-背景差异？

## 方法详解
1. **AIAS（异常感知加速采样）**：
   - 无训练采样策略，利用 Linear-Gaussian closure 引理
   - 将多步 DDPM 反向转换聚合为少量粗到细的解析更新
   - 从 1000 步降至 10 步（100× 加速），系数预计算
2. **FARM（前景感知重建模块）**：
   - 在每步从噪声潜变量中预测伪清洁异常潜变量
   - 对异常掩码区域重新注入异常感知噪声
   - 在前向和反向过程中都保持异常区域的显著性
   - 作为去噪器外部的重建路径运行
3. **整体流程**：基于 LDM，每一步先 AIAS 跳过多步，再 FARM 保持异常显著性

## 实验关键数据
- **MVTec-AD**（Segformer 下游）：
  - FAST: mIoU **76.72%**, Acc **83.97%**
  - vs DRAEM: 67.84% / 74.75% vs AnomalyDiffusion: 63.57% / 73.52%
  - 关键类别提升：capsule +11.83 mIoU, grid +4.70, transistor +7.58
- **BTAD**：同样一致超越所有基线
- **消融**：
  - w/o FARM: mIoU 从 76.72% 降至 65.33%（↓11.39），Acc 从 83.97% 降至 71.24%（↓12.73）
  - FARM 在细粒度类别上提升尤为显著：transistor +29.5 mIoU
- **多 backbone 验证**：在 Segformer、BiseNetV2、STDC 上均一致提升
- **效率**：仅 10 步即达 SOTA，100× 加速

## 亮点
- 理论有保证：基于 Linear-Gaussian closure 引理推导加速采样的解析解
- FARM 简洁有效：外部重建路径 + 掩码注入，概念清晰且提升巨大（+11.39 mIoU）
- 极致加速：10 步 SOTA，适合工业产线换型的实际需求
- 与下游分割器解耦：合成数据可搭配任意轻量分割网络

## 局限性 / 可改进方向
- 依赖预定义的异常掩码（几何增强 + LDM 合成），掩码质量影响上限
- 需要文本 prompt 描述异常语义，对 prompt 敏感
- 仅在工业异常数据集验证，通用分割场景未测试
- 合成数据与真实异常的 domain gap 仍存在

## 与相关工作的对比
| 方法 | 类型 | 可控性 | 速度 | MVTec mIoU |
|------|------|--------|------|-----------|
| CutPaste | 手工 | 低 | 快 | 55.00 |
| DRAEM | 手工 | 中 | 快 | 67.84 |
| DFMGAN | GAN | 中 | 中 | 64.96 |
| AnomalyDiffusion | 扩散 | 高 | 慢(1000步) | 63.57 |
| **FAST** | 扩散 | 高 | 快(10步) | **76.72** |

## 启发与关联
- 前景-背景解耦思路可推广到其他条件生成任务（如 inpainting、编辑）
- 解析加速采样（vs 学习加速如 distillation）是一条值得探索的路线
- 工业异常合成→分割的 pipeline 可作为工业质检的实用解决方案

## 评分
- 新颖性: ⭐⭐⭐⭐ (AIAS 的解析加速 + FARM 的前景感知设计有新意)
- 实验充分度: ⭐⭐⭐⭐ (2+数据集 × 3分割器 × 6+基线 + 消融)
- 写作质量: ⭐⭐⭐⭐ (理论推导清晰，实验组织好)
- 价值: ⭐⭐⭐⭐ (工业应用导向，实用性强)
