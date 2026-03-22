<!-- 由 src/gen_stubs.py 自动生成 -->
# EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale

**会议**: NeurIPS 2025  
**arXiv**: [2510.24173](https://arxiv.org/abs/2510.24173)  
**代码**: https://github.com/ASK-Berkeley/EddyFormer (有)  
**领域**: 科学计算 / 流体力学  
**关键词**: Turbulence Simulation, Spectral Element Method, Transformer, LES, Neural PDE Solver

## 一句话总结
提出 EddyFormer，一种基于谱元法 (SEM) 的 Transformer 架构，将流场分解为 LES（大尺度）和 SGS（小尺度）两路并行流，在 256³ 分辨率 3D 湍流上达到 DNS 级精度且加速 30 倍，并在未见的 4× 更大域上泛化良好。

## 研究背景与动机

1. **领域现状**：湍流模拟需要 $Re^{9/4}$ 分辨率做 DNS，成本极高。LES 只解析大尺度结构，用理论模型近似小尺度，但在壁面紧围、各向异性湍流中有困难。
2. **现有痛点**：(a) FNO 等神经算子在大规模湍流上扩展困难；(b) Transformer 的二次复杂度随网格分辨率增长；(c) 大多数 ML 方法仅在 2D 小规模流上验证。
3. **核心矛盾**：如何在保持谱方法的高精度的同时，利用注意力机制高效捕获多尺度相互作用？
4. **切入角度**：将 SEM（谱元法）作为 tokenization：粗元素作为 token，元素内用谱展开。序列长度仅为元素数 $N^3$，远小于总模态数 $N^3 M^3$。
5. **核心 idea 一句话**：用谱元法做 tokenization + LES/SGS 双流架构，将湍流的多尺度特性显式建模到模型设计中。

## 方法详解

### 整体框架
EddyFormer 将 PDE 初始条件用 SEM 插值，分成 LES stream（全局大尺度）和 SGS stream（局部小尺度）并行处理，输出 $\mathbf{u} = \mathbf{u}_{LES} + \mathbf{u}_{SGS}$。

### 关键设计

1. **SEM Tokenization**:
   - 将域分为 $H = N^3$ 个粗元素，每个元素内用 $M^3$ 阶谱基函数展开
   - 元素 = token，元素内谱展开 = token 特征。序列长度仅为 $N^3$

2. **SGS Stream (小尺度局部动力学)**:
   - 用 SEM-based 卷积 (SEMConv) 建模局部涡旋交互
   - 基于 Kolmogorov 假设，小尺度动力学具有普遍性

3. **LES Stream (大尺度全局动力学)**:
   - 用 SEM-based 自注意力 (SEMAttn) 捕获全局远程依赖
   - Rotary Position Encoding 保持平移不变性

### 损失函数
- 时间平均单步相对误差

## 实验关键数据

### 主实验 — 3D 均匀各向同性湍流
| 模型 | 参数量 | 单步误差 |
|------|--------|----------|
| EddyFormer | **2.3M** | **最低** |
| FNO | 17.6M | 中 |

### 加速比较
| 方法 | 256³ 模拟时间 | L2 误差 |
|------|-------------|--------|
| DNS (256³) | 152 sec | 16.3% |
| **EddyFormer** | **4.86 sec** | 18.2% |

### 关键发现
- 30× 加速 vs DNS，精度相当
- 域泛化：在未见的 4× 更大域上保持物理不变量指标精度
- The Well benchmark 上解决了其他 ML 模型无法收敛的湍流案例

## 亮点与洞察
- **物理启发的架构设计**：LES/SGS 双流直接对应湍流的多尺度结构
- **SEM tokenization 非常巧妙**：把谱元法的粗元素作为 token，截断了 attention 输入规模
- **域泛化**通过 attention masking 实现

## 局限性 / 可改进方向
- 仅测试各向同性湍流，各向异性、壁面紧围湍流未验证
- 固定时间步长预测

## 相关工作与启发
- **vs FNO**: FNO 用全局 Fourier 核但参数效率低，EddyFormer 通过 SEM 将全局与局部分离

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SEM tokenization + LES/SGS 双流架构很新颖且物理启发强
- 实验充分度: ⭐⭐⭐⭐⭐ 3D 湍流 + 2D 域泛化 + The Well benchmark
- 写作质量: ⭐⭐⭐⭐⭐ 物理背景和方法描述都很详细
- 价值: ⭐⭐⭐⭐⭐ 30× 加速3D湍流达到DNS级精度，实用价值很高
