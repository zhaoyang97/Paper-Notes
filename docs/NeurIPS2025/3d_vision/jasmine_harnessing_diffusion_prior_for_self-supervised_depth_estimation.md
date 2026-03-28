<!-- 由 src/gen_stubs.py 自动生成 -->
# Jasmine: Harnessing Diffusion Prior for Self-supervised Depth Estimation

**会议**: NEURIPS2025  
**arXiv**: [2503.15905](https://arxiv.org/abs/2503.15905)  
**代码**: 待确认  
**领域**: 3d_vision / depth_estimation  
**关键词**: 自监督深度估计, Stable Diffusion, 扩散先验, 混合批次重建, Scale-Shift GRU  

## 一句话总结
首次将Stable Diffusion视觉先验引入自监督单目深度估计：提出Mix-Batch Image Reconstruction避免自监督噪声损坏SD先验，设计Scale-Shift GRU桥接SD的尺度偏移不变性(SSI)与自监督的尺度不变性(SI)深度，在KITTI上AbsRel达0.102且泛化性强。

## 背景与动机

1. **领域现状**：自监督深度估计避免昂贵GT标注但在遮挡/无纹理/光照变化区域退化严重。SD等基础模型有丰富视觉先验，但此前仅用于有监督深度估计。
2. **核心挑战**：自监督的重投影损失含噪且不精确，直接用其微调SD会损坏预训练先验（梯度扰动导致VAE潜空间退化）。
3. **另一挑战**：SD产出SSI深度（scale-shift不变），而自监督需要SI深度（尺度不变）——两者分布不匹配导致训练不稳定。

## 方法详解

### 整体框架
SD微调框架，通过代理任务保护先验 + GRU模块对齐SSI/SI深度。

### 关键设计1: Mix-Batch Image Reconstruction (MIR)
- 训练批次交替包含深度预测任务（KITTI真实图像）和图像重建任务（Hypersim合成图像）
- 图像重建用**photometric loss**而非latent loss——避免VAE的1/8分辨率导致块状伪影
- 合成数据上的重建任务保持SD先验不被自监督噪声损坏

### 关键设计2: Scale-Shift GRU (SSG)
- GRU迭代模块，用可学习的scale/shift查询与SD隐层状态做交叉注意力
- 输出scale因子 $s_c$ 和shift因子 $s_h$，将SSI深度转换为SI深度
- 迭代精炼对齐，而非一次性线性变换

### 关键设计3: Steady SD Finetuning
- 用预训练自监督教师（MonoViT）生成伪标签
- 伪标签损失权重逐渐衰减——初期稳定训练，后期让模型超越教师

## 实验关键数据

### KITTI（自监督方法对比）

| 方法 | AbsRel↓ | SqRel↓ | RMSE↓ | δ₁↑ |
|------|---------|--------|-------|-----|
| Monodepth2 | 0.115 | 0.903 | 4.863 | 0.877 |
| HR-Depth | 0.106 | 0.755 | 4.472 | 0.890 |
| MonoViT | 0.096 | - | - | - |
| **JASMINE*** | **0.102** | **0.540** | **3.728** | **0.907** |

- SqRel和RMSE显著领先
- 与零样本方法对比：优于Marigold(0.120)、E2E FT(0.112)、Lotus(0.110)

### 泛化性
- DrivingStereo和CityScapes零样本迁移优于监督方法

### 消融
- Photometric loss >> Latent loss（解决VAE块状伪影问题）
- 混合真实+合成图像关键——仅用一种类型失败

## 亮点
1. **首个SD自监督深度**：证明扩散先验可在无GT条件下有效利用
2. **MIR代理任务**：巧妙保护SD先验不被噪声自监督信号损坏
3. **SSG桥接模块**：优雅解决SSI/SI分布不匹配
4. **强泛化性**：跨数据集零样本迁移优于监督方法

## 局限性 / 可改进方向
1. 依赖预训练教师模型(MonoViT)生成伪标签——教师质量影响上限
2. AbsRel 0.102仍不如MonoViT(0.096)——但泛化性更好
3. SD微调的计算成本高于传统自监督方法

## 与相关工作的对比
- **vs Marigold/Lotus**：这些是零样本方法需要GT微调SD；JASMINE纯自监督
- **vs MonoViT**：强自监督基线，JASMINE在SqRel/RMSE/泛化性上更优
- **vs E2E FT**：端到端微调但需GT；JASMINE无需标注

## 启发与关联
- 代理任务保护预训练先验的思路可推广到其他SD微调场景
- SSI→SI的分布对齐问题在其他相对预测任务中也普遍存在
- 合成数据混合训练思路对自监督方法有普适参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个SD自监督深度+MIR+SSG创新组合
- 实验充分度: ⭐⭐⭐⭐ KITTI+跨域零样本+消融
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，设计动机清晰
- 价值: ⭐⭐⭐⭐ 开启SD自监督深度的新方向
