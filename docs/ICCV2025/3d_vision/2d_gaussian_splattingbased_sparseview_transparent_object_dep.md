# TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update

**会议**: ICCV 2025  
**arXiv**: [2507.11069](https://arxiv.org/abs/2507.11069)  
**代码**: [项目主页](https://jeongyun0609.github.io/TRAN-D/) (有)  
**领域**: 3D视觉  
**关键词**: 2D Gaussian Splatting, 透明物体深度重建, 稀疏视角, 物理仿真, 场景更新  

## 一句话总结

提出TRAN-D，一种基于2D Gaussian Splatting的稀疏视角透明物体深度重建方法，通过分割引导的object-aware损失优化遮挡区域Gaussian分布，并利用物理仿真（MPM）实现物体移除后的场景动态更新，仅需单张图像即可完成场景刷新。

## 背景与动机

- **透明物体深度重建是计算机视觉中的难题**：透明物体由于反射、折射等物理特性，传统ToF传感器和神经渲染方法均难以获取准确深度
- **已有方法的不足**：
  - NeRF-based方法（Dex-NeRF, NFL, Residual-NeRF）需要大量训练图像且训练时间长；Residual-NeRF还依赖背景图像
  - GS-based方法（TranSplat, TransparentGS）虽然质量好，但依然需要稠密多视角输入
  - 稀疏视角NVS方法（InstantSplat, FSGS）依赖3D基础模型，对透明物体存在泛化偏差，无法正确区分透明物体和背景
- **动态场景问题未解决**：当物体被移动或移除时，现有方法需要重新扫描整个场景，非常耗时
- **关键洞察**：将透明物体从背景中分离出来,对物体对应的Gaussian进行专注优化是关键

## 核心问题

1. 如何在稀疏视角（仅6张图像）下准确重建透明物体的深度？
2. 如何处理遮挡区域（从任何视角都看不到的表面），避免Gaussian过拟合？
3. 物体被移除后，如何高效更新场景表示（无需重新扫描），处理因物体移除引起的连锁运动？

## 方法详解

### 整体框架

TRAN-D包含三个模块：
1. **透明物体分割模块**：基于fine-tuned Grounded SAM进行透明物体实例分割
2. **Object-aware 2DGS模块**：用分割mask和object index one-hot向量联合优化2D Gaussian，配合object-aware 3D损失处理遮挡区域
3. **场景更新模块**：利用MPM（Material Point Method）物理仿真预测物体移除后的连锁运动，仅需单张鸟瞰图进行场景刷新

### 关键设计

1. **透明物体分割（Fine-tuned Grounded SAM）**：
   - 使用不在词典中的特定类别文本提示"786dvpteg"来表示"透明物体"，避免"glass"/"transparent"等通用词导致的误分割
   - 仅fine-tune图像骨干（GroundedDINO），冻结文本骨干（BERT），在合成TRansPose数据上训练1个epoch
   - 所有透明物体统一为一个类别，分配唯一标识符作为类别特定prompt
   - 确保多视角间一致的实例分割mask

2. **分割Mask渲染与Object Index One-Hot渲染**：
   - 每个Gaussian $\mathcal{G}_i$ 被赋予颜色向量 $\mathbf{m}_i \in \mathbb{R}^3$ 表示其所属物体
   - 渲染方程与颜色渲染类似：$m(x) = \sum_i m_i \alpha_i \hat{\mathcal{G}}_i(u(x)) \prod_{j=1}^{i-1}(1-\alpha_j \hat{\mathcal{G}}_j(u(x)))$
   - 同时维护object index one-hot向量 $\mathbf{o}_i \in \mathbb{R}^{N+1}$（N个物体+1个背景），通过softmax归一化后用dice loss优化
   - 分割mask的联合优化防止透明物体的Gaussian不透明度在训练中塌缩为零

3. **Object-aware 3D Loss（核心创新）**：
   - **问题**：稀疏视角+遮挡导致部分区域梯度极弱，仅靠视角空间位置梯度无法优化这些Gaussian
   - **解决方案**：基于3D距离的层级化损失
   - 选取 $n_g$ 个最远的Gaussian作为组中心，每组包含 $n_n$ 个最近邻Gaussian
   - **距离方差损失** $\mathcal{L}_d = \text{Var}(d_1, ..., d_{n_g})$：促使组中心间距均匀，将遮挡区域波动较大的距离拉向可见区域稳定的距离
   - **局部密度损失** $\mathcal{L}_S = \text{Var}(S_1, ..., S_{n_g})$：促使各组内部密度一致，吸引Gaussian到稀疏区域
   - **三级层级分组策略**：$(n_g, n_n) = (16,16), (32,16), (64,32)$，适应不同优化阶段Gaussian数量的变化

4. **基于物理仿真的场景更新（MPM）**：
   - 物体移除后通过object index one-hot向量识别并删除对应Gaussian
   - 从2D Gaussian渲染深度图生成mesh，用于物理仿真
   - 使用Taichi实现的MPM模拟物体移除后的连锁运动（100个时间步）
   - 材料参数：杨氏模量 $5 \times 10^4$ Pa，泊松比0.4
   - 仿真后进行100次迭代的Gaussian重优化（省略object-aware损失），仅需单张鸟瞰图

### 损失函数 / 训练策略

总损失函数：
$$\mathcal{L} = a_\text{color}\mathcal{L}_c + a_\text{mask}\mathcal{L}_m + a_\text{one-hot}\mathcal{L}_\text{one-hot} + \mathcal{L}_\text{obj}$$

- $\mathcal{L}_c$：RGB重建损失（L1 + D-SSIM），$a_\text{color} = 0.5$
- $\mathcal{L}_m$：分割mask损失（L1 + D-SSIM），$a_\text{mask} = 0.5$
- $\mathcal{L}_\text{one-hot}$：Dice loss，$a_\text{one-hot} = 1.0$
- $\mathcal{L}_\text{obj}$：object-aware 3D损失，汇总各层级各物体的 $\mathcal{L}_S$（$a_S=10000/3$）和 $\mathcal{L}_d$（$a_d=1/3$）

训练细节：
- 从随机点初始化（不依赖SfM或3D基础模型）
- one-hot学习率从0.1衰减到0.0025，1000次迭代
- GPU：NVIDIA RTX 2080 Ti
- 场景更新时仅需100次迭代优化

## 实验关键数据

| 数据集 | 指标 | TRAN-D | 之前SOTA (TranSplat) | 提升 |
|--------|------|--------|---------------------|------|
| TRansPose (t=0) | MAE | **0.0380** | 0.0632 | 39.9%↓ |
| TRansPose (t=0) | δ<2.5cm | **69.11%** | 43.01% | +26.1% |
| TRansPose (t=1) | δ<2.5cm | **48.46%** | 31.62% | 1.53× |
| ClearPose (t=0) | MAE | **0.0461** | 0.0905 | 49.1%↓ |
| ClearPose (t=0) | δ<2.5cm | **54.38%** | 31.95% | +22.4% |

效率对比（19个场景平均）：

| 方法 | t=0总训练时间 | t=1总训练时间 | Gaussian数量(t=0) |
|------|-------------|-------------|-------------------|
| TRAN-D | **54.1s** | **13.8s** | **33.5k** |
| InstantSplat | 78.8s | 95.5s | 850.1k |
| TranSplat | 596.0s | 612.7s | 297.8k |
| 2DGS | 440.9s | 447.6s | 227.8k |

### 消融实验要点

- **Object-aware Loss**：去掉该损失后MAE从0.0419升至0.0447（t=0），RMSE从0.1059升至0.1136，且Gaussian数量略增（35983 vs 33482），表明该损失在减少过拟合的同时还能压缩Gaussian数量
- **物理仿真**：去掉物理仿真后t=1的MAE从0.0886升至0.0891，不使用仿真会导致物体在Z轴方向位置不变，过拟合训练图像并丢失物体形状
- **视角数量**：3/6/12视角下MAE分别为0.0405/0.0419/0.0448（差异很小），证明方法对稀疏视角非常鲁棒；6视角时性能趋于饱和
- TRAN-D在3视角下MAE(0.0405)优于InstantSplat在12视角下(0.2062)数倍

## 亮点

- **将物理仿真引入透明物体场景重建**：优雅地解决了物体移除后连锁运动的问题，避免重新扫描
- **Object-aware 3D Loss设计巧妙**：不依赖任何额外网络，通过3D距离的方差约束让Gaussian自发覆盖遮挡区域，既简洁又有效
- **从随机点初始化**：完全避免了对SfM或3D基础模型的依赖，反而性能更好（因为这些模型对透明物体有泛化偏差）
- **极致高效**：t=0仅需54秒，t=1场景更新仅需13.8秒（含物理仿真），Gaussian数量仅33k（对比InstantSplat的850k）
- **场景更新仅需单张图像**：结合物理仿真，用单张鸟瞰图即可更新场景，精度达到使用6张图的baseline的1.5倍
- 用非字典词"786dvpteg"作为透明物体的文本prompt是一个聪明的工程trick

## 局限性 / 可改进方向

- **严重依赖分割质量**：分割失败（tracking failure、强光照、边界模糊）会直接导致重建和物理仿真失败
- **只能处理部分物体移除或轻微运动**：无法处理更复杂的动态场景（如任意物体的添加、大幅移动）
- **只渲染物体不渲染背景**：与其他方法的比较不完全公平（虽然任务定义如此）
- **未来方向**：开发不依赖分割的方法；处理更复杂的动态和光照环境；扩展至更多样的真实场景

## 与相关工作的对比

| 方法 | 类型 | 视角需求 | 透明物体特化 | 动态场景 | 训练时间 |
|------|------|---------|------------|---------|---------|
| Dex-NeRF | NeRF | 稠密 | ✓ | ✗ | 慢 |
| NFL | NeRF | 稠密 | ✓ | ✗ | 慢 |
| TranSplat | 3DGS+Diffusion | 稠密 | ✓ | ✗ | 中等 |
| TransparentGS | 3DGS+BSDF | 稠密 | ✓ | ✗ | 慢 |
| InstantSplat | 3DGS+3D基础模型 | 稀疏 | ✗ | ✗ | 快 |
| FSGS | 3DGS+基础模型 | 稀疏 | ✗ | ✗ | 慢 |
| **TRAN-D** | **2DGS+物理仿真** | **稀疏** | **✓** | **✓** | **极快** |

## 启发与关联

- **物理仿真+神经表示的结合范式**值得关注：先用神经方法重建，再用物理引擎推理动态变化，最后用少量数据微调，这种pipeline可以拓展到更多动态场景理解任务
- **分割引导的Gaussian优化**：将分割信息作为额外通道联合splatting的思路非常通用，可用于任何需要物体级别控制的GS任务
- **3D空间正则化**替代依赖额外网络的方案：不用预训练深度网络或3D基础模型，而是设计3D几何约束来引导优化，在数据稀缺场景更鲁棒
- 对机器人操作场景（抓取透明物体）有直接应用价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 物理仿真用于场景更新的思路新颖，object-aware 3D loss设计巧妙；但整体框架是已有组件（2DGS+Grounded SAM+MPM）的组合
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据集全面评估，消融实验完整，效率对比清晰；但真实场景无法定量评估是遗憾
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详细，图表质量高；整体结构合理
- 价值: ⭐⭐⭐⭐ 在机器人操作透明物体场景有实际价值，极快的速度和对稀疏视角的鲁棒性是实用的；但对分割的强依赖限制了通用性

## 评分
- 新颖性: ⭐⭐⭐ — 待深读后评价
- 实验充分度: ⭐⭐⭐ — 待深读后评价
- 写作质量: ⭐⭐⭐ — 待深读后评价
- 对我的价值: ⭐⭐⭐⭐ — 待深读后评价
