# Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation

**会议**: ICLR 2026 / **arXiv**: [2602.10489](https://arxiv.org/abs/2602.10489)  
**代码**: [GitHub](https://github.com/gxingyu/ADAlign)  
**领域**: llm_alignment / 图域适应  
**关键词**: graph domain adaptation, characteristic function, spectral alignment, minimax optimization, neural spectral discrepancy  

## 一句话总结
提出**ADAlign**框架，利用**神经谱差异(NSD)**在频域自适应对齐源/目标图嵌入分布，通过可学习频率采样器自动发现并优先对齐每个迁移场景中最关键的分布差异维度。

## 背景与动机
1. 图域适应(GDA)需要将标注源图的知识迁移到无标注目标图，但面临复杂的多维分布偏移
2. 现有方法依赖人工选择对齐元素（节点属性、度分布等），需手工设计图滤波器提取特征
3. 不同迁移场景的主导差异维度不同（如B→E场景feature 2,3最大，U→E则feature 1,2,4），固定策略无法适应
4. MMD只对齐低阶矩，KL散度在高维不稳定——需要完备且可优化的距离度量
5. 特征函数(CF)唯一刻画概率分布，提供Fourier域的完备表示，但在图领域尚未被利用

## 方法
- **GNN编码**: 共享GNN编码器将源/目标图映射为节点嵌入 $Z^S$, $Z^T$
- **特征函数变换**: 计算嵌入的经验CF $\Psi^S(\mathbf{t}), \Psi^T(\mathbf{t})$，在频域比较分布
- **NSD度量**: $|\Psi_{Z^S}(\mathbf{t}) - \Psi_{Z^T}(\mathbf{t})|^2$ 分解为**振幅差**（全局结构变化）和**相位差**（关系模式偏移），参数 $\kappa$ 控制两者权重
- **自适应频率采样器**: 参数化的Normal Scale Mixture分布 $p_{\mathcal{T}}(\mathbf{t};\phi)$，通过对抗训练自动聚焦到差异最大的频率区域
- **Minimax优化**: $\min_\delta \max_\phi [\mathcal{L}_{source} + \lambda \mathcal{L}_{align}]$，GNN最小化分类+对齐损失，频率采样器最大化对齐损失以暴露最大差异
- **PAC-Bayes理论**: 证明目标域误差上界由源域margin loss + 加权频域差异控制

## 实验
| 设置 | 关键发现 |
|------|---------|
| Citation (A→C, C→A等6个任务) | ADAlign在所有任务上超越SOTA，如A→C: 82.21% vs 次优80.93% |
| Airport (U→B, B→E等6个任务) | 大幅领先，U→B: 63.18% vs 次优60.16%（不同迁移场景优势各异） |
| Blog (B1→B2, B2→B1) | 55.08%/52.42% vs 次优49.53%/46.78%，提升5-6个点 |
| Twitch (DE→EN, EN→DE) | 在跨平台社交网络迁移上也保持优势 |
| 效率 | 相比UDAGCN等方法，ADAlign内存消耗更低、训练更快 |
| 消融 | 自适应频率采样器和振幅-相位分解均对性能有显著贡献 |

## 亮点
- 首次将**神经特征函数**引入图域适应，提供理论上完备的分布度量
- 振幅-相位分解赋予NSD清晰的物理含义：振幅→分散程度差异，相位→关系结构偏移
- 自适应频率采样避免固定网格的信息损失，实现"按需对齐"
- PAC-Bayes界为框架提供理论保障，且自适应权重 $\omega_m$ 直接对应实际频率采样器
- 10个数据集16个迁移任务全面验证，鲁棒性强

## 局限性
- 仅验证节点分类任务，图分类/链接预测等其他图任务未涉及
- 假设源/目标图共享GNN编码器，跨架构场景未考虑
- Normal Scale Mixture的表达能力有限，更复杂的采样分布可能进一步提升
- 需要调节 $\kappa$（振幅-相位平衡）和 $\lambda$（对齐权重），敏感性分析有限

## 相关工作
- UDAGCN(WWW'20)/StruRW(ICML'23): 基于对抗/随机游走的GDA → ADAlign用频域统一替代
- SA-GDA/GRADE/PairAlign: 手工分解图偏移 → ADAlign自适应发现主导偏移
- Li et al.(2017,2020): CF在生成模型中的应用 → 本文首次拓展到图域适应

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐
