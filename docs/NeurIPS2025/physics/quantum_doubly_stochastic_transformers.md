<!-- 由 src/gen_stubs.py 自动生成 -->
# Quantum Doubly Stochastic Transformers

**会议**: NEURIPS2025  
**arXiv**: [2504.16275](https://arxiv.org/abs/2504.16275)  
**代码**: 待确认  
**领域**: others  
**关键词**: quantum computing, doubly stochastic matrices, Vision Transformer, attention mechanism, variational quantum circuit  

## 一句话总结
提出 QDSFormer，用变分量子电路（QontOT）替换 ViT 中的 softmax 生成双随机注意力矩阵，在多个小规模图像识别任务上超越标准 ViT 和 Sinkformer，并显著稳定训练。

## 背景与动机
- Transformer 中的 softmax 让注意力矩阵成为右随机矩阵，但已知会导致 entropy collapse、rank collapse、token uniformity 等问题
- Sinkformer 通过 Sinkhorn 算法将注意力矩阵强制为双随机矩阵（DSM），在多种任务上提升性能
- 但 Sinkhorn 算法是迭代近似、非参数化、梯度不稳定的
- 近期证明：变分量子电路（QontOT）可以参数化地生成 DSM，且没有已知的经典对应方法

## 核心问题
如何利用量子电路的参数化 DSM 生成能力来替代 softmax，构建一种更灵活、更稳定的双随机 Transformer？

## 方法详解
1. **DSM 生成方式对比**：对比了 Sinkhorn、QR 分解、Birkhoff 投影、QontOT 量子电路等多种 DSM 生成算子
2. **QontOT 集成**：将 QontOT 量子电路扩展为矩阵级 DSM 输出，替换 ViT 自注意力中的 softmax
3. **电路训练策略**：
   - Differentiable：电路参数与 Transformer 联合优化（最慢）
   - Mixed：Transformer 训练交替进行梯度无关的电路优化
   - Static：使用预训练电路参数，纯推理模式（效果最好）
4. **量子灵感经典替代**：提出基于 QR 分解的双随机 Transformer（QRFormer）作为经典替代

## 实验关键数据
- **FashionMNIST/MNIST**：2-4 层 ViT 上 QDSFormer 在大多数配置下显著超越 softmax 和 Sinkhorn
  - FashionMNIST 4层：QontOT 90.3% vs Softmax 89.7% vs Sinkhorn 89.1%
  - MNIST 4层：QontOT 98.8% vs Softmax 98.8% vs Sinkhorn 97.9%
- **MedMNIST**（7 个数据集）：QDSFormer 在 5/7 数据集上最优，平均准确率 74.3% vs ViT 73.0%
- **Eureka 组合任务**：QDSFormer 在 100 epoch 内学会（vs 标准 ViT 需数百 epoch），准确率提升 ~30%
  - lr=5e-4 时 QontOT 达 89.4% 且 5/5 run 出现 Eureka moment；Softmax 仅 61.1%（1/5）
- **训练稳定性**：QDSFormer 在所有实验中性能方差一致低于其他方法

## 亮点
- 首个参数化双随机 Transformer，利用量子电路生成 DSM 且无已知经典等价方法
- Static 模式不需额外训练电路参数，却性能最强——说明量子 DSM 的归纳偏置本身就有价值
- 在组合推理任务上大幅提前 Eureka Moment，显示双随机注意力自然地稳定了 ViT 训练
- 量子硬件噪声保留了注意力矩阵的排序（Spearman ρ>0.9），可能反而有正则化效果

## 局限性 / 可改进方向
- 所有实验在小规模数据集上，受限于量子模拟器的扩展性
- 量子硬件上运行尚不可行（需 ~640K shots/sample，当前硬件 kHz 级频率）
- 端到端训练反而劣于 Static，可能受 Barren plateaus 影响
- 未与 ESPFormer、LOTFormer 等并发工作比较

## 与相关工作的对比
| 方法 | 注意力类型 | 参数化 | DSM 保证 |
|------|-----------|--------|---------|
| ViT（Softmax） | 右随机 | ✓ | ✗ |
| Sinkformer | 双随机 | ✗ | 近似（迭代） |
| QRFormer（本文提出） | 双随机 | ✗ | 精确 |
| **QDSFormer** | 双随机 | ✓ | 精确 |
| ESPFormer | 双随机 | ✗ | Sliced OT |
| LOTFormer | 双随机+线性 | ✓ | Conditional OT |

## 启发与关联
- 量子归纳偏置的一般性启示：不一定需要量子优势，量子电路的结构约束本身可作为有用的归纳偏置
- Static 模式成功暗示：随机（但结构良好的）DSM 可能已足够好，关键在于双随机性而非特定的 DSM
- 与注意力温度调节的关系：双随机注意力自动增加 entropy 且不使其均匀，避免了手动调温的需求

## 评分
- 新颖性: ⭐⭐⭐⭐ (量子电路替换 softmax 思路新颖)
- 实验充分度: ⭐⭐⭐ (仅小规模数据集，但对比全面)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，理论与实验平衡)
- 价值: ⭐⭐⭐ (概念验证阶段，实际应用受限于量子硬件)
