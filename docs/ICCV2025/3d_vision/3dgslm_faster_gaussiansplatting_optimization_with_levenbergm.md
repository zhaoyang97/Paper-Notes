# 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt

**会议**: ICCV 2025  
**arXiv**: [2409.12892](https://arxiv.org/abs/2409.12892)  
**代码**: [GitHub](https://github.com/lukasHoel/3DGS-LM) (有)  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, Levenberg-Marquardt, 优化加速, CUDA并行, 二阶优化  

## 一句话总结

提出3DGS-LM，用定制的Levenberg-Marquardt (LM) 优化器替代ADAM来加速3DGS重建，通过梯度缓存数据结构和自定义CUDA核实现高效GPU并行的PCG求解，在保持重建质量的同时将优化速度提升约20%，且与其他3DGS加速方法正交可叠加。

## 背景与动机

- 3D高斯泼溅 (3DGS) 通过可微光栅器将3D高斯投影为2D splat并α-blending得到像素颜色，实现了实时渲染和高质量新视角合成
- 现有加速3DGS的方法集中在两个方向：(1) 改进densification策略减少高斯数量 (GS-MCMC, Taming-3DGS, Mini-Splatting)；(2) 改进可微光栅器实现 (DISTWAR的warp reduction, gsplat的splat并行化)
- 但**所有方法仍然使用ADAM优化器**，需要数千次梯度下降迭代才能收敛，在高分辨率密集场景上训练可能耗时一小时
- 相比之下，RGB-D融合领域的3D重建任务中，Gauss-Newton/LM算法能以比SGD少一个数量级的迭代次数快速收敛——这为用二阶优化器加速3DGS提供了直接动机

## 核心问题

如何将Levenberg-Marquardt (二阶优化器) 高效适配到3DGS的优化流程中？关键挑战：(1) Jacobian矩阵规模巨大（百万高斯×数百高分辨率图像），无法显式存储；(2) 需要在GPU上高效并行计算Jacobian-向量积；(3) LM对初始化更敏感，直接用SfM点云初始化效果不佳。

## 方法详解

### 整体框架

采用**两阶段优化**策略：
1. **第一阶段 (ADAM)**: 运行标准3DGS优化20K迭代（含densification至15K迭代），得到未完全收敛但数量已确定的高斯初始化
2. **第二阶段 (LM)**: 仅需5次LM迭代（每次8次PCG内循环）即可收敛

两阶段设计的原因：梯度下降在初期进展快但后期收敛慢，而LM需要好的初始化才能发挥优势（直接从SfM初始化反而更慢）。这也避免了在LM的5-10次迭代中处理densification（3DGS需要140次densification操作）。

### 关键设计

**1. LM目标函数重构**

将3DGS的L1+SSIM损失改写为残差平方和形式以适配LM：

$$E(\mathbf{x}) = \sum_{i=1}^{N} \lambda_1 |c_i - C_i|^2 + \lambda_2 (1 - \text{SSIM}(c_i, C_i))^2$$

通过对原始损失取平方根得到两类残差 $r_i^{\text{abs}}$ 和 $r_i^{\text{SSIM}}$，使目标函数与原始3DGS完全等价但符合LM要求。

**2. 正规方程与PCG求解**

每次LM迭代求解正规方程获得更新方向Δ：

$$(J^T J + \lambda_{\text{reg}} \cdot \text{diag}(J^T J)) \Delta = -J^T F(\mathbf{x})$$

由于J矩阵无法显式存储，采用**预条件共轭梯度 (PCG)** 以matrix-free方式求解，最多8次PCG迭代。

**3. 梯度缓存与per-pixel-per-splat并行化**

这是本文最核心的技术贡献：

- **问题**: 原始3DGS的per-pixel并行化在PCG中效率太低，因为每次PCG迭代需要计算 $\mathbf{u}=J\mathbf{p}$ 和 $\mathbf{g}=J^T\mathbf{u}$，同一中间α-blending状态（$T_s$, $\partial c/\partial \alpha_s$, $\partial c/\partial c_s$）被重复计算多达18次
- **方案**: 在PCG开始前，用buildCache一次性缓存这些中间梯度，然后改为**per-pixel-per-splat并行化**——每个线程处理一条光线上一个splat的所有残差
- 利用链式法则将Jacobian分解为三个独立可并行计算的因子：$\frac{\partial r}{\partial x_i} = \frac{\partial r}{\partial c} \cdot \frac{\partial c}{\partial s} \cdot \frac{\partial s}{\partial x_i}$
- 缓存按高斯排序后可实现**合并内存访问 (coalesced access)**，applyJT核用**segmented warp reduction**高效聚合

**4. 图像子采样方案**

高分辨率密集场景中缓存可能过大，解决方案：
- 将图像分为 $n_b$ 个batch（每batch 25-70张图），独立求解正规方程
- 通过加权均值合并更新向量：$\Delta = \frac{\sum_i M_i \Delta_i}{\sum_k M_k}$，权重为PCG预条件器的逆 $M_i = \text{diag}(J_i^T J_i)$
- 权重的数学推导基于将完整正规方程解分解为子集解的加权组合，再用对角近似

**5. 线搜索与λ调整**

求得Δ后，在30%图像子集上做线搜索找最优步长γ。根据质量指标ρ调整正则化强度：ρ>1e-5时接受更新并减小λ，否则回退更新并加倍λ。

### 损失函数 / 训练策略

- **损失**: 与原始3DGS完全相同的L1+SSIM（λ₁=0.8, λ₂=0.2），只是通过平方根变换适配LM的残差平方和形式
- **SSIM梯度处理**: 将SSIM的局部邻域卷积梯度反传到中心像素（忽略对邻域其他像素的贡献），保持光线独立性以支持并行化
- **实现细节**: 第一阶段20K迭代用默认超参数；第二阶段仅5次LM迭代，每次8次PCG迭代；batch大小根据数据集分辨率调整（MipNeRF360: 25张×4batch, DeepBlending: 25张×3batch, Tanks&Temples: 70张×3batch）

## 实验关键数据

| 数据集 | 指标 | 3DGS | 3DGS+Ours | 加速(s) |
|--------|------|------|-----------|---------|
| MipNeRF360 | SSIM/PSNR/Time | 0.813/27.40/1271s | 0.813/27.39/972s | **23.5%** |
| DeepBlending | SSIM/PSNR/Time | 0.900/29.51/1222s | 0.903/29.72/951s | **22.2%** |
| Tanks&Temples | SSIM/PSNR/Time | 0.844/23.68/736s | 0.845/23.73/663s | **9.9%** |

**与其他加速方法的叠加效果** (DeepBlending):

| 方法 | 原始Time | +Ours Time | 加速 |
|------|----------|------------|------|
| DISTWAR | 841s | 672s | **20.1%** |
| gsplat | 919s | 716s | **22.1%** |
| Taming-3DGS | 447s | 347s | **22.4%** |

所有baseline加上LM优化器后平均加速约20%，质量指标基本不变。GPU内存消耗平均53GB（baseline仅6-11GB），体现了时间-内存的trade-off。

### 消融实验要点

1. **L1/SSIM vs L2目标**: L2损失下3DGS和3DGS+Ours质量都明显下降（SSIM 0.862→0.854），证明L1/SSIM损失的重要性；两种损失下LM都实现了加速
2. **batch大小**: 从100张减至40张仅轻微影响质量（PSNR 33.77→33.51），但运行时间从242s降至212s，GPU内存从32.5GB降至15.4GB——图像子采样不影响LM收敛
3. **LM vs 多视角ADAM**: 在相同多视角约束下（75张图/step），LM仅需5次迭代即达到ADAM 10K迭代的质量。ADAM增大batch到50/130张仍需更多迭代才能收敛到相同质量，证明LM的二阶更新方向质量远高于一阶梯度
4. **初始化迭代数**: K=6000或K=8000次ADAM初始化后LM收敛更快；K过小时LM反而更慢，验证了两阶段策略的必要性

## 亮点

- **正交性设计精妙**: 从优化器角度切入3DGS加速，与改rasterizer、改densification的方法完全正交，可直接叠加获得更大加速
- **缓存驱动的并行化**: 通过一次缓存中间梯度 + 重新排序，将per-pixel并行改为per-pixel-per-splat并行，applyJT比buildCache快4.8倍
- **工程上极度高效**: 完整第二阶段仅需5次LM × 8次PCG = 40次核调用就收敛，而ADAM需要额外10K次迭代
- **SSIM梯度的巧妙处理**: 只对中心像素反传梯度保持光线独立性，使SSIM可以无缝集成到并行PCG中
- **残差权重预计算**: 将 $(\partial r/\partial c)^2$ 提取到核外计算，避免核内额外全局内存读取，使L1/SSIM残差的计算开销几乎与L2相同

## 局限性 / 可改进方向

1. **GPU内存消耗大**: 梯度缓存使内存从6-11GB增至53GB，高分辨率/大规模场景可能需要CPU offloading
2. **仍依赖ADAM做densification**: 两阶段设计意味着第一阶段仍需较长时间；若能将densification集成到LM中（仅5-10次迭代难以支持140次densification），可进一步加速
3. **加速比在不同数据集上波动**: Tanks&Temples上加速仅约10%，远低于MipNeRF360的23.5%——可能因为低分辨率图像更多、LM的开销相对更大
4. **缓存排序开销**: sortCacheByGaussians虽然比PCG本身快，但在每次LM迭代中是固定开销

## 与相关工作的对比

| 方法 | 加速方式 | 与3DGS-LM的关系 |
|------|---------|----------------|
| DISTWAR | warp reduction加速反向传播 | 正交，可叠加（DISTWAR+Ours: 764s vs DISTWAR: 966s） |
| gsplat | splat并行化改进光栅器 | 正交，可叠加 |
| Taming-3DGS | 改进densification减少高斯数量 | 正交，可叠加（最快baseline+Ours: 347s） |
| GS-MCMC | 用MCMC替代启发式densification | 改densification策略，与改优化器互补 |
| Rasmuson et al. | Gauss-Newton优化低分辨率NeRF体素 | 受其启发，但3DGS-LM利用显式高斯原语实现更高效的并行JVP |
| 3DGS² (Lan et al. 2025) | 近二阶收敛的3DGS优化 | 同方向（二阶优化加速3DGS），值得关注 |

## 启发与关联

- **通用优化器替换思路**: 不仅限于3DGS，任何基于可微渲染+ADAM的重建管线（如NeRF变体、可微网格优化）都可能受益于类似的LM替换
- **缓存+重排序的GPU优化模式**: 先缓存中间状态再重排序以获得合并内存访问，这是一个通用的CUDA优化模式，可应用于其他需要重复计算相同中间量的场景
- **图像子采样+加权均值**: 这种将大规模最小二乘分解为小batch独立求解再加权合并的策略，具有一定的分布式优化/联邦学习的味道
- **两阶段训练哲学**: 一阶方法快速初始化 + 二阶方法精细收敛的策略在多个优化问题中有效

## 评分

- 新颖性: ⭐⭐⭐⭐ [从优化器角度加速3DGS是一个简洁有效的新视角，但LM本身不是新算法，核心贡献在工程实现]
- 实验充分度: ⭐⭐⭐⭐⭐ [3个数据集13个场景×4个baseline的完整实验，消融覆盖目标函数/batch大小/初始化/多视角对比，per-scene结果和profiling分析详尽]
- 写作质量: ⭐⭐⭐⭐⭐ [逻辑清晰，从动机到方法到实验层层递进，两阶段设计的motivation through Fig.4很好，算法伪代码和缓存示意图直观易懂]
- 价值: ⭐⭐⭐⭐ [20%加速+质量不变+正交可叠加，实用价值高；但53GB内存消耗限制了实际部署场景]

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐ — 待深读后评价
- 实验充分度: ⭐⭐⭐ — 待深读后评价
- 写作质量: ⭐⭐⭐ — 待深读后评价
- 对我的价值: ⭐⭐⭐⭐ — 待深读后评价
