# AdaptGrad: Adaptive Sampling to Reduce Noise

**会议**: NeurIPS 2025  
**arXiv**: [2410.07711](https://arxiv.org/abs/2410.07711)  
**代码**: 无  
**领域**: 可解释性 / 梯度方法 / 模型解释  
**关键词**: gradient smoothing, SmoothGrad, saliency map, adaptive sampling, model interpretability  

## 一句话总结
通过卷积公式视角首次理论分析了SmoothGrad的噪声来源（越界采样），提出AdaptGrad方法通过概率界约束采样范围来抑制噪声，在不增加计算开销的前提下提升梯度显著性图的质量。

## 背景与动机
SmoothGrad是最流行的梯度平滑可解释性方法——通过在输入上添加高斯噪声并平均多次梯度来得到更清晰的显著性图。但SmoothGrad存在两个问题：(1) 为什么它有效？缺乏理论解释；(2) 噪声水平高，超参数（噪声标准差α）设置困难。越大的α理论上平滑效果越好，但实际中过大的α导致采样点超出输入值域，引入大量噪声。

## 核心问题
SmoothGrad中的噪声从何而来？如何在保持平滑效果的同时减少噪声？

## 方法详解

### 整体框架
将SmoothGrad的等效操作解释为卷积——在梯度场上做高斯卷积。从这一视角出发，识别出越界采样是噪声的主要来源，然后设计概率约束的采样策略来避免越界。

### 关键设计
1. **卷积公式分析**：证明SmoothGrad等价于梯度场与高斯核的卷积，从而将噪声分析转化为卷积边界问题
2. **越界采样识别**：当采样点超出有效值域(如图像像素[0,255])时，梯度信息无意义，成为纯噪声
3. **自适应采样(AdaptGrad)**：利用累积分布函数(CDF)将采样限制在可信区间[c-分位数, (1-c)-分位数]内，c∈{0.95, 0.99}。对于靠近值域边界的像素，自动缩小噪声范围。
4. **通用兼容性**：不仅适用于SmoothGrad，还可与GradientInput、Integrated Gradients、NoiseGrad组合

## 实验关键数据
| 指标 | AdaptGrad | SmoothGrad | 提升 |
|------|-----------|------------|------|
| 稀疏性(VGG16) | 更高 | 基线 | ~2-3% |
| 忠实性(deletion) | 更好 | 基线 | 显著 |
| SIC(IG-W) | 0.7221 | 0.7179 | +0.6% |
| 视觉质量 | 更清晰 | 较模糊 | 明显 |

- 模型：VGG16, ResNet50, InceptionV3, MLP
- 数据集：ILSVRC2012, MNIST
- 一致性和不变性指标与SmoothGrad持平
- 对抗生成：FGSM+AdaptGrad需要更少像素修改

### 消融实验要点
- c∈{0.95, 0.99, 0.995, 0.999}均有效，鲁棒
- AdaptGrad优于ClipGrad（硬裁剪）
- uniform-n帧采样优于first-n
- 目标定位精度：AdaptGrad > SmoothGrad

## 亮点
- **理论突破**：首次通过卷积公式解释了SmoothGrad为何有效以及噪声从何而来
- **简洁有效**：解决方案极其简单——约束采样范围即可，零额外计算开销
- **通用性**：不仅改善SmoothGrad，还能改善IG、NoiseGrad等其他梯度方法
- **跨任务验证**：可解释性指标 + 目标定位 + 对抗生成

## 局限性 / 可改进方向
- c是经验选择的，最优值可能因数据集/任务而异
- 忠实性指标本身有局限（长尾效应、insertion/deletion假设）
- 对高维稀疏数据的效果待探索

## 与相关工作的对比
- **vs SmoothGrad**：AdaptGrad是SmoothGrad的改进版，用概率约束减少噪声
- **vs NoiseGrad**：NoiseGrad对权重加噪声，AdaptGrad对输入加约束噪声，两者可组合
- **vs ClipGrad**：硬裁剪粗暴截断，AdaptGrad用概率分布软约束

## 评分
- 新颖性: ⭐⭐⭐⭐ 卷积视角的理论分析新颖，解决方案自然
- 实验充分度: ⭐⭐⭐⭐ 多模型、多指标、多与方法组合
- 写作质量: ⭐⭐⭐⭐ 问题→分析→解决的叙事结构清晰
- 价值: ⭐⭐⭐⭐ 对模型可解释性社区有实际贡献
