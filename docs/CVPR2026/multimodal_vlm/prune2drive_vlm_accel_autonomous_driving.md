# Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving

**会议**: CVPR 2026  
**arXiv**: [2508.13305](https://arxiv.org/abs/2508.13305)  
**代码**: [https://github.com/MinhaoXiong/Prune2Drive.git](https://github.com/MinhaoXiong/Prune2Drive.git)  
**领域**: 自动驾驶 / VLM加速 / Token剪枝  
**关键词**: 多视角VLM, 视觉token剪枝, 多样性感知采样, 视图自适应, 自动驾驶  

## 一句话总结
首个面向多视角自动驾驶VLM的即插即用token剪枝框架Prune2Drive，通过T-FPS（token级最远点采样）保持语义/空间多样性 + 视图自适应剪枝率优化自动分配不同视角的token预算,在DriveLM上仅保留10% token即实现6.40×prefill加速且性能仅降3%。

## 背景与动机
自动驾驶VLM(DriveMM/DriveLMM-o1)需处理6个环视摄像头的高分辨率图像(每图729 tokens),总计>4000 visual tokens,导致$O(n^2)$注意力计算极慢。现有token剪枝方法（FastV/SparseVLM）针对单图设计,有三个缺陷：(1) 依赖注意力权重→不兼容FlashAttention；(2) 存在位置偏差→后端token被系统性保留；(3) 忽视多视角的差异性贡献→前视和后视摄像头对驾驶决策的重要性不同,不应均匀剪枝。

## 核心问题
如何在多视角自动驾驶场景下,设计一个不依赖注意力权重、考虑视图贡献差异的training-free token剪枝方法？

## 方法详解

### 整体框架
两个核心组件：(1) **T-FPS (Token-wise Farthest Point Sampling)**：在token嵌入空间中用最远点采样迭代选择最具多样性的token子集；(2) **视图自适应剪枝率优化**：用TPE（Tree-structured Parzen Estimator）在小验证集上自动搜索每个摄像头视角的最优token保留率。

### 关键设计
1. **T-FPS多样性感知选择**：借鉴点云处理中的FPS算法,但用余弦距离替代欧氏距离。初始随机选一个token,每一步选择与已选集合余弦距离最大的token加入,直到达到目标数量。关键优势：(a) 不依赖注意力→兼容FlashAttention；(b) 最大化语义+空间覆盖→避免丢失低注意力但重要的物体（如远处车辆）；(c) 计算开销极低——N=729时仅0.02s,<0.1% FLOPs。

2. **视图自适应剪枝率优化**：将每个视角的保留率$\alpha_i$作为可优化变量,目标函数$\mathcal{M}(\alpha) = R(\alpha) - \lambda P(\alpha)$平衡性能奖励和token总量惩罚。用TPE在500个样本的小验证集上搜索最优解,仅需3 H100 GPU小时即可收敛。结果显示：前视摄像头自动获得更高保留率（对驾驶决策更重要），后视和侧视适度减少。

3. **理论保证**：证明了T-FPS（k-center贪心近似最小Hausdorff距离）+视图自适应率（按重要性加权分配预算）的组合方案,比均匀随机采样+等比例剪枝能提供更紧的误差界。

### 损失函数 / 训练策略
完全training-free。T-FPS在视觉编码器输出后直接应用,视图自适应率在小验证集上离线搜索一次即固定。兼容LLaVA-OneVision-7B(DriveMM)和InternVL2.5-8B(DriveLMM-o1)。

## 实验关键数据
**DriveLM (DriveMM, 10% token保留):**

| 方法 | Token/图 | Avg Score | Prefill加速 | FLOPs |
|--------|------|------|------|------|
| Vanilla | 729 | 59.1 | 1× | 100% |
| FastV | 72 | 54.1 | 5.78× | 14.2% |
| SparseVLM | 72 | 55.9 | 4.06× | 14.4% |
| **Prune2Drive** | **72** | **57.4** | **6.40×** | **13.4%** |

DriveLMM-o1: 10%保留→68.3 vs vanilla 74.2(下降6%),优于FastV(65.3)、DART(67.4)。

**通用VLM (LLaVA-1.5, 128 tokens)**: 97.3% avg performance,优于SparseVLM 96.2%。

**视频AD (OmniDrive)**: 49.0 vs vanilla 50.3,优于FastV 44.3和SparseVLM 46.8。

### 消融实验要点
- **距离度量**: 余弦 ≈ L1 ≈ L2 >> min距离（最近采样导致严重退化-15%,验证了多样性原则）
- **TPE > Evolutionary > GridSearch**: 差异较小(<1%),均优于手工设定
- **Match Score 25%保留甚至超原模型**(34.0 vs 33.9)：适度剪枝有正则化效果（去除冗余/干扰token）
- **T-FPS也在通用VLM上work**: LLaVA-1.5 64 tokens时94.3%（SparseVLM 89.5%）——差距更大
- **失败模式**：大面积均匀色块物体（橙色公交车）因特征相似可能被欠采样

## 亮点
- 首个专为多视角自动驾驶设计的token剪枝——不是简单迁移单图方法
- T-FPS的FPS思路极其优雅——在token嵌入空间而非3D空间做最远点采样,保证语义多样性
- 视图自适应率优化直接解决了"前视vs后视重要性不同"的实际问题
- 6.40×的prefill加速对实时驾驶系统有直接工业价值
- 兼容FlashAttention且开销极低（0.02s/图）

## 局限性 / 可改进方向
- 仅在离线benchmark上评估,缺少闭环仿真验证
- T-FPS对大面积均匀色块物体可能欠采样（因为特征相似度高→距离小→不被FPS选中）
- 视图自适应率是离线搜索的固定值,未做动态输入自适应
- 未与V2Drop/ApET等最新token压缩方法对比——仅对比FastV/SparseVLM/DART/PACT
- 仅测试了DriveMM和DriveLMM-o1两个AD模型

## 与相关工作的对比
- **vs FastV (注意力剪枝)**：FastV依赖第2层注意力,有位置偏差且不兼容FA。Prune2Drive不用注意力,10% token时57.4 vs 54.1(+3.3)
- **vs DART (相似度剪枝)**：DART用余弦相似度但不考虑视图差异。Prune2Drive加入视图自适应,10% token时57.4 vs DriveLM未报告
- **vs V2Drop (CVPR'26)**：V2Drop用层间变化量在LLM内部剪枝,Prune2Drive用FPS在编码器输出后剪枝——二者正交可组合
- **vs ApET (CVPR'26)**：ApET用近似误差做重要性评估,Prune2Drive用多样性最大化——思路不同但目标类似

## 启发与关联
- T-FPS的"嵌入空间FPS"策略可推广到所有多图VLM场景——如医学多视角成像、机器人多摄像头系统
- 视图自适应率优化的reward-penalty框架可用于其他需要跨模态/跨source预算分配的场景
- 与DUET-VLM的视觉侧聚类+语言侧剪枝思路互补——可以用T-FPS替代DUET的V2V聚类

## 评分
- 新颖性: ⭐⭐⭐⭐ T-FPS和视图自适应率均为清晰的新贡献，但各组件不算复杂
- 实验充分度: ⭐⭐⭐⭐⭐ 2个AD基准+1个视频AD+5个通用VLM、效率分析、消融、可视化、失败分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，理论分析加分，但部分表格格式不够精炼
- 价值: ⭐⭐⭐⭐⭐ 自动驾驶VLM的实时部署极需此类方法，6.40×加速有直接工业价值
