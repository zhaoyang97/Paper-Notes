# 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt

**会议**: ICCV 2025  
**arXiv**: [2409.12892](https://arxiv.org/abs/2409.12892)  
**代码**: [GitHub](https://github.com/lukasHoel/3DGS-LM) (有)  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, Levenberg-Marquardt, 优化加速, CUDA并行, 二阶优化  

## 一句话总结
将3D Gaussian Splatting的ADAM优化器替换为定制化的Levenberg-Marquardt（LM）二阶优化器，通过高效CUDA并行化的PCG算法和梯度缓存结构实现Jacobian-向量积加速，在保持相同重建质量的前提下将优化时间缩短约20%。

## 研究背景与动机

1. **领域现状**：3D Gaussian Splatting（3DGS）已成为新视角合成的主流方法，通过3D高斯原语的可微光栅化实现实时渲染和高质量图像合成。当前优化通常使用ADAM优化器，需要30K次迭代、耗时长达1小时。
2. **现有痛点**：已有加速方法主要从两条路线入手——加速光栅化器实现（如DISTWAR的warp reduction、gsplat的并行模式）或减少高斯数量（如GS-MCMC、Taming-3DGS的新densification方案），但它们都没有触及底层优化器本身，仍依赖ADAM这个一阶优化器逐步收敛。
3. **核心矛盾**：ADAM作为一阶方法，每步只利用梯度方向信息，收敛需要数千次迭代才能到达局部最优；而二阶方法（如LM）通过求解法方程近似二阶更新，理论上可以用少得多的迭代次数收敛，但在3DGS场景下面临数百万高斯参数×高分辨率图像的Jacobian矩阵过大无法显式存储的挑战。
4. **本文要解决什么**：如何将LM优化器高效应用于3DGS，在GPU上实现可扩展的Jacobian-向量积计算？
5. **切入角度**：利用3DGS高斯原语的稀疏性——每个像素只受少量高斯贡献，Jacobian矩阵极度稀疏——设计缓存友好的per-pixel-per-splat并行策略，将中间梯度缓存一次后在PCG迭代中复用。
6. **核心idea一句话**：通过梯度缓存+per-pixel-per-splat CUDA并行化实现矩阵无关PCG求解，将LM嫁接到3DGS优化的第二阶段，仅需5次LM迭代替代10K次ADAM迭代。

## 方法详解

### 整体框架
方法分两阶段：**第一阶段**使用原始3DGS的ADAM优化器运行前20K次迭代完成densification，得到未收敛的高斯初始化；**第二阶段**切换到定制LM优化器，仅需5-10次LM迭代即可收敛到与30K次ADAM相当的质量。输入是位姿图像+SfM点云，输出是优化后的3D高斯场景。

### 关键设计

1. **LM优化器的3DGS适配**:
   - 做什么：将3DGS的渲染损失重构为平方和能量函数以适配LM框架
   - 核心思路：对L1和SSIM损失项分别取平方根得到残差 $r_i^{\text{abs}} = \sqrt{\lambda_1|c_i - C_i|}$ 和 $r_i^{\text{SSIM}} = \sqrt{\lambda_2(1-\text{SSIM}(c_i, C_i))}$，使得目标函数 $E(\mathbf{x}) = \sum r_i^2$ 成为标准最小二乘形式。每步通过求解法方程 $(\mathbf{J}^T\mathbf{J} + \lambda_{\text{reg}}\text{diag}(\mathbf{J}^T\mathbf{J}))\Delta = -\mathbf{J}^T\mathbf{F}$ 获取更新方向，再通过line search找最优步长 $\gamma$。
   - 设计动机：保留了原始L1+SSIM目标（比纯L2质量更好，实验验证），同时使LM能利用曲率信息做更高质量的更新步。

2. **梯度缓存与per-pixel-per-splat并行化**:
   - 做什么：将PCG中反复需要的Jacobian-向量积 $\mathbf{J}\mathbf{p}$ 和 $\mathbf{J}^T\mathbf{u}$ 通过缓存中间梯度加速
   - 核心思路：原3DGS的per-pixel并行每个线程处理一条光线上所有splat，导致 $\alpha$-blending中间状态（$T_s$, $\partial c/\partial \alpha_s$, $\partial c/\partial c_s$）在PCG中被重复计算多达18次。本文改为：buildCache阶段一次性缓存所有中间梯度 $\partial c/\partial s$，之后PCG每步通过per-pixel-per-splat并行（一个线程只处理一条光线的一个splat）直接读缓存完成计算。缓存先按像素排序存储，再通过sortCacheByGaussians重排以保证合并访存。
   - 设计动机：消除PCG迭代中的冗余计算，将计算粒度从per-pixel拆解到per-pixel-per-splat，大幅提升GPU占用率和并行度。

3. **图像子采样方案**:
   - 做什么：控制缓存内存使用，使方法可扩展到高分辨率密集采集场景
   - 核心思路：将图像分为 $n_b$ 个批次，每批独立求解法方程得到更新向量 $\Delta_i$，最终通过加权平均合并：$\Delta = \sum_i \frac{\mathbf{M}_i \Delta_i}{\sum_k \mathbf{M}_k}$，权重为各批次Jacobi预条件器的对角项 $\mathbf{M}_i = \text{diag}(\mathbf{J}_i^T\mathbf{J}_i)$。实际使用25-70张图片/批，最多4个批次。
   - 设计动机：高分辨率场景下全部图像的缓存会超出GPU显存（完整方案需~53GB），批次化处理将内存使用降低到可控范围，同时加权合并保证了跨批次更新方向的一致性。

### 损失函数 / 训练策略
- 目标函数与原始3DGS完全一致（L1 + SSIM），仅改变优化器
- LM正则化强度 $\lambda_{\text{reg}}$ 根据更新质量指标 $\rho$ 自适应调节：$\rho > 10^{-5}$ 时降低正则化（$\lambda_{\text{reg}} *= 1-(2\rho-1)^3$），否则回退更新并加倍 $\lambda_{\text{reg}}$
- 第一阶段20K迭代（含densification），第二阶段仅5次LM迭代（每次8轮PCG）
- Line search在30%图像子集上运行以节省渲染开销

## 实验关键数据

### 主实验

| 方法 | 数据集 | SSIM↑ | PSNR↑ | 时间(s) | 加速比 |
|------|--------|-------|-------|---------|--------|
| 3DGS | MipNeRF360 | 0.813 | 27.40 | 1271 | - |
| 3DGS + Ours | MipNeRF360 | 0.813 | 27.39 | 972 | 23.5% |
| gsplat | MipNeRF360 | 0.814 | 27.42 | 1064 | - |
| gsplat + Ours | MipNeRF360 | 0.814 | 27.42 | 818 | 23.1% |
| DISTWAR | Deep Blending | 0.899 | 29.47 | 841 | - |
| DISTWAR + Ours | Deep Blending | 0.902 | 29.60 | 672 | 20.1% |
| Taming-3DGS | Tanks&Temples | 0.833 | 23.76 | 366 | - |
| Taming-3DGS + Ours | Tanks&Temples | 0.832 | 23.72 | 310 | 15.3% |

### 消融实验

| 配置 | PSNR↑ | 时间(s) | 说明 |
|------|-------|---------|------|
| ADAM L1/SSIM (30K) | 27.23 | 1573 | 原始3DGS基线 |
| LM L1/SSIM | 27.29 | 1175 | 本文方法，快25%质量略好 |
| ADAM L2 only | 27.31 | 1528 | 纯L2质量差但PSNR高 |
| LM L2 only | 27.48 | 1131 | LM+L2也能加速 |
| Batch=100 images | 33.77 | 242 | 全图像效果最好 |
| Batch=60 images | 33.69 | 223 | 质量轻微下降，速度快 |
| Batch=40 images | 33.51 | 212 | 内存15GB，质量可接受 |
| Multi-view ADAM (50 iters) | 29.54 | 962 | 相同迭代数下LM更优 |
| LM (5 iters) | 29.72 | 951 | 仅5次迭代达到更好质量 |

### 关键发现
- LM替代ADAM在所有基线上均获得约20%加速，证明了优化器层面改进与光栅化器/densification改进正交互补
- 仅5-10次LM迭代就能替代10K次ADAM迭代，单步更新质量远高于一阶方法
- 图像子采样对质量影响很小（100→40张仅掉0.26 PSNR），但能大幅减少显存
- 初始化质量对LM至关重要：从SfM直接跑LM反而更慢，需要ADAM先做好初始化才能发挥二阶方法优势

## 亮点与洞察
- **缓存驱动的并行化方案**是核心技巧：通过一次缓存中间α-blending梯度，将PCG中高达18次的冗余计算降为0，并开启了更细粒度的per-pixel-per-splat并行。这个思路可以迁移到任何需要反复求解涉及可微渲染Jacobian的优化问题中。
- **两阶段策略**精准利用了一阶和二阶优化器各自的优势：ADAM擅长从差初始化快速进展+完成densification，LM擅长在好初始化附近快速收敛。这种"粗调+精调"范式具有通用价值。
- **加权批次合并**（公式8）巧妙地用 $\text{diag}(\mathbf{J}^T\mathbf{J})$ 作为权重，本质上让对当前批次图像贡献大的高斯参数获得更大的更新权重，保证了跨批次更新方向的物理一致性。

## 局限性 / 可改进方向
- **显存开销大**：完整方案需约53GB GPU显存（vs 基线6-11GB），限制了在消费级GPU上的应用
- **初始化依赖**：必须用ADAM先跑20K步，LM从头训练反而更慢，两阶段切换点的选择是启发式的
- **densification阶段未优化**：LM目前只优化固定数量高斯的参数，如何在densification过程中也使用二阶信息是开放问题
- 改进思路：可以探索更低内存的隐式Jacobian计算方案（如随机化Hutchinson trace估计），或者设计自适应的阶段切换策略

## 相关工作与启发
- **vs ADAM-based 3DGS**：本文证明在3DGS优化中一阶方法不是唯一选择，二阶方法在好初始化下收敛更快
- **vs Taming-3DGS**：Taming通过减少高斯数量加速，本文通过换优化器加速，两者正交可叠加（联合使用在T&T上达到310s vs 原始736s）
- **vs RGB-D fusion中的GN/LM**：RGB-D重建领域长期使用Gauss-Newton，本文首次成功将其引入3DGS并解决了可扩展性问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 将LM优化器引入3DGS并非全新想法，但缓存驱动的GPU并行化方案是显著的工程创新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个标准数据集、四个基线、详细消融（目标函数、批大小、初始化、多视图ADAM对比）
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码和图示辅助理解好
- 价值: ⭐⭐⭐⭐ 20%加速实用且与现有方法正交可叠加，但高显存限制了实际部署

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
