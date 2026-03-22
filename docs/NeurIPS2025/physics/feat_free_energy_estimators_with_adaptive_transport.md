<!-- 由 src/gen_stubs.py 自动生成 -->
# FEAT: Free energy Estimators with Adaptive Transport

**会议**: NEURIPS2025  
**arXiv**: [2504.11516](https://arxiv.org/abs/2504.11516)  
**代码**: [GitHub](https://github.com/jiajunhe98/FEAT)  
**领域**: others  
**关键词**: free energy estimation, stochastic interpolants, Jarzynski equality, Crooks theorem, molecular simulation  

## 一句话总结
提出 FEAT 框架，基于随机插值学习自适应传输，通过 escorted Jarzynski 等式和 Crooks 定理提供一致、最小方差的自由能差估计器，统一了平衡与非平衡方法。

## 背景与动机
- 自由能估计是统计力学、药物发现、量子场论中的核心问题
- 传统方法局限：
  - FEP (Free Energy Perturbation)：分布重叠不足时方差大
  - BAR (Bennett Acceptance Ratio)：仍需分布重叠
  - TI (Thermodynamic Integration)：需要中间分布的精确采样
  - Jarzynski 等式：非平衡方法但方差可能大
- 近期神经网络方法（Targeted FEP with flow matching、Neural TI）取得进展，但非平衡方法在深度学习中仍未充分探索

## 核心问题
如何设计一个统一框架，利用学习到的传输（transport）来实现高效、鲁棒的自由能差估计，同时兼顾平衡和非平衡两种范式？

## 方法详解
1. **随机插值框架**：学习连接两个态 S_a 和 S_b 的随机传输过程
   - 参数化速度场 v_t^ψ 和能量插值 U_t^θ
   - 通过随机插值模型训练传输
2. **估计器设计**：
   - **Escorted Jarzynski 估计器**：利用带护送项的 Jarzynski 等式，支持正向/反向单侧估计
   - **Controlled Crooks 估计器**：基于 Crooks 涨落定理，实现最小方差估计（类似 BAR 的推广）
   - **变分界**：同时提供自由能差的上界和下界
3. **One-sided FEAT**：用扩散模型学习单态的绝对自由能，再取差值（适用于大系统）
4. **关键优势**：
   - 无需中间分布的精确采样（非平衡优势）
   - 避免了耗时的散度计算（相比 Targeted FEP）
   - 对离散化和网络学习误差有天然鲁棒性

## 实验关键数据
- **Targeted FEP 对比**（Table 1）：
  - GMM-40: FEAT 0.04±0.04 vs TargetFEP 0.09±0.26（接近真值 0）
  - LJ-128: FEAT 595.04±6.52 vs TargetFEP 无法计算（散度太贵）
  - ALDP-S: FEAT 29.38±0.04 vs TargetFEP 29.47±0.22（参考值 29.43）
  - ALDP-T: FEAT -4.56±0.08 vs TargetFEP -4.78±0.32（参考值 -4.25）
- **Neural TI 对比**（Table 2）：
  - GMM-40: 有预处理 0.1±0.2 vs 无预处理 -181.6±6.7（高度依赖预处理）
  - FEAT 无需问题特定预处理
- **大规模系统**（Table 3, One-sided FEAT）：
  - ALA-4 (66维): 109.91±2.55 vs 参考 107.5
  - Chignolin (蛋白质): 320.02±0.70 vs 参考 320.19（极高精度）
- **量子场论**：在 φ⁴ 理论上通过 umbrella sampling 成功恢复磁化分布的对称性

## 亮点
- 理论统一性：将 FEP、BAR、TI、Jarzynski 等式全部纳入一个框架的特殊情况
- 避免散度计算：相比 Targeted FEP (flow matching) 在大系统上计算效率大幅提升
- 对离散化误差的鲁棒性：非平衡方法的核心优势
- 在 Chignolin 蛋白质系统上的精度令人印象深刻

## 局限性 / 可改进方向
- 基于路径空间的重要性采样，方差可能大于状态空间方法（bias-variance trade-off）
- 需要两个端态的样本（未来可探索 Vargas et al. 的方法放宽此限制）
- LJ-128 上标准差较大（6.52），大系统方差控制仍需改进
- One-sided FEAT 需要额外训练两个模型

## 与相关工作的对比
| 方法 | 类型 | 需精确中间采样 | 散度计算 | 参数化 |
|------|------|--------------|---------|--------|
| FEP | 平衡 | ✗ | ✗ | ✗ |
| BAR | 平衡 | ✗ | ✗ | ✗ |
| TI | 平衡 | ✓ | ✗ | ✗ |
| Neural TI | 平衡 | ✓ | ✗ | ✓（需预处理） |
| Targeted FEP (FM) | 非平衡 | ✗ | ✓（昂贵） | ✓ |
| **FEAT** | 非平衡 | ✗ | ✗ | ✓ |

## 启发与关联
- FEAT 的统一视角说明：平衡方法是非平衡方法的特殊情况，非平衡范式提供更大设计空间
- 随机插值作为通用传输学习框架的潜力，可能扩展到其他需要分布间桥接的任务
- 药物发现中的绑定自由能计算是最直接的应用方向

## 评分
- 新颖性: ⭐⭐⭐⭐ (统一框架视角新颖，escorted Jarzynski + 学习传输的结合有创意)
- 实验充分度: ⭐⭐⭐⭐ (从玩具到蛋白质到量子场论，覆盖面广)
- 写作质量: ⭐⭐⭐⭐ (理论推导严谨，实验设置清晰)
- 价值: ⭐⭐⭐⭐ (对计算物理/化学有实际应用价值)
