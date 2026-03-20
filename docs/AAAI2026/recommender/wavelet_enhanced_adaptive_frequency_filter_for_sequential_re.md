# Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation

**会议**: AAAI 2026  
**arXiv**: [2511.07028](https://arxiv.org/abs/2511.07028)  
**代码**: [https://github.com/xhy963319431/WEARec](https://github.com/xhy963319431/WEARec)  
**领域**: 序列推荐 / 频域信号处理  
**关键词**: sequential recommendation, frequency-domain filtering, wavelet transform, dynamic filter, personalized recommendation  

## 一句话总结
提出WEARec模型结合动态频域滤波(DFF)和小波特征增强(WFE)两个模块，分别捕获个性化全局频域信息和增强非平稳短期波动，在四个公开数据集上超越频域推荐SOTA基线，长序列场景提升可达11.4%。

## 背景与动机
- 频域方法(FMLPRec/BSARec等)通过傅里叶变换捕获周期模式取得进展
- **局限1**: 静态固定模式滤波器忽略行为模式的个性化——不同用户被不同频率成分驱动
- **局限2**: 全局DFT擅长长程依赖但模糊非平稳信号和短期波动——FMLPRec本质是低通滤波器

## 核心问题
1. 如何实现个性化频域滤波以适应不同用户行为模式？
2. 如何同时捕获全局频域分布和局部非平稳信号？

## 方法详解

### 整体框架
类Transformer编码器。Embedding层→L=2层WEARec块(DFF+WFE+FFN)→预测层。DFF+WFE替代self-attention。

### 关键设计
1. **DFF (动态频域滤波)**: 多头投影→1D FFT→用户上下文c_l(时域均值)→两个MLP生成缩放Δs和偏置Δb→个性化滤波器W_hat=W⊙(1+Δs), b_hat=b+Δb→频域线性变换→IFFT
2. **WFE (小波特征增强)**: Haar DWT沿item维度分解为低频A+高频D→自适应矩阵T缩放高频D_tilde=D⊙T→低频不变→IDWT重构
3. **Feature Integration**: H_hat = α⊙X_DFF + (1-α)⊙Y_WFE，α≈0.3最优

### 损失函数 / 训练策略
- 交叉熵损失, Adam, lr∈{0.0005,0.001}, 嵌入维度=64, N=50, batch 256
- 小波分解层1, α∈{0.1-0.9}, k∈{1,2,4,8}
- NVIDIA Tesla A100 GPU

## 实验关键数据
| 数据集 | 指标 | 本文 | BSARec | 提升 |
|--------|------|------|----------|------|
| Beauty | HR@10 | 0.1041 | 0.1008 | +3.27% |
| Sports | HR@10 | 0.0631 | 0.0612 | +3.10% |
| LastFM | HR@10 | 0.0899 | 0.0807 | +11.40% |
| ML-1M | HR@10/NG@10 | 0.2952/0.1696 | 0.2757/0.1568 | +2.10%/+1.37% |

**长序列(N=200)效率**: WEARec 66.46s/epoch vs BSARec 109.26s vs SLIME4Rec 120.43s

### 消融实验要点
- 去WFE/去DFF/去多头投影均导致所有数据集性能下降
- DFF是核心模块，去除后性能下降最显著
- 频谱可视化：WEARec覆盖所有频率，FMLPRec和SLIME4Rec偏向低频

## 亮点
- 动态滤波器根据用户上下文自适应调整，实现个性化频域处理
- 时间复杂度O(nd·log n)优于self-attention的O(n²d)
- 不用对比学习和self-attention，训练速度快39-45%
- 长序列场景优势更显著(LastFM +11.4%)

## 局限性 / 可改进方向
- 仅用Haar小波(最简单)，复杂小波基可能更优
- α为全局固定超参数，可学习为样本自适应
- 未考虑item content/side information

## 与相关工作的对比
时域方法(GRU4Rec/Caser/SASRec/DuoRec)和频域方法(FMLPRec/FamouSRec/FEARec/SLIME4Rec/BSARec)全面超越。

## 启发与关联
个性化动态滤波器设计可推广到其他序列建模。小波与傅里叶互补组合在信号处理中有普适性。

## 评分
- 新颖性: ⭐⭐⭐⭐ 动态频域滤波+小波增强组合设计新颖
- 实验充分度: ⭐⭐⭐⭐ 四数据集、长序列分析、频谱可视化齐全
- 写作质量: ⭐⭐⭐⭐ 信号处理基础充分，方法清晰
- 价值: ⭐⭐⭐⭐ 为频域推荐提供高效有效的新范式
