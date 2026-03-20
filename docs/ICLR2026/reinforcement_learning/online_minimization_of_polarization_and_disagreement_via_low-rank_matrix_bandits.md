# Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits

**会议**: ICLR2026  
**arXiv**: [2510.00803](https://arxiv.org/abs/2510.00803)  
**代码**: [GitHub](https://github.com/FedericoCinus/online-min-pol)  
**领域**: llm_agent, online learning  
**关键词**: 观点极化, Friedkin-Johnsen模型, 低秩矩阵bandit, 遗憾最小化, 社交网络干预  

## 一句话总结
将社交网络中极化与分歧最小化问题建模为在线低秩矩阵bandit问题，提出两阶段算法OPD-Min-ESTR（先估计子空间再低维线性bandit），将维度从 $|V|^2$ 降至 $O(|V|)$，实现 $\tilde{O}(\max\{1/\kappa, \sqrt{|V|}\}\sqrt{|V|T})$ 累积遗憾。

## 背景与动机
1. 社交平台加剧观点极化和社会分裂，需要算法干预来缓解
2. Friedkin-Johnsen (FJ)模型是研究观点动力学的经典基础：agent的表达观点是内在观点与邻居表达观点的加权平均
3. 现有工作(Musco et al. 2018)假设完全知道所有agent的内在观点——现实中获取内在观点代价高昂甚至不可能
4. 已有部分放松假设的工作（二值观点/SDP上界/部分查询），但均未解决**在线设置**下观点完全未知、仅能通过序列干预学习的问题
5. 直接转化为线性bandit特征维度为 $|V|^2$，遗憾随 $|V|^2$ 增长，不可接受
6. 现有低秩矩阵bandit依赖连续空间（如高斯）采样，不适用于离散的图Laplacian约束

## 方法详解
**问题建模：OPD-Min (Online Polarization & Disagreement Minimization)**

- FJ均衡：$\bm{z}^* = (\mathbf{I} + \mathbf{L})^{-1}\bm{s}$，极化+分歧 = $f(\bm{s}, \mathbf{L}) = \bm{s}^\top (\mathbf{I}+\mathbf{L})^{-1}\bm{s}$
- 令forest矩阵 $\mathbf{X} = (\mathbf{I}+\mathbf{L})^{-1}$，$\Theta^* = \bm{s}\bm{s}^\top$（秩1），则 $f(\mathbf{X}) = \langle \Theta^*, \mathbf{X} \rangle$ → 低秩矩阵bandit
- 每轮选择干预 $\mathbf{L}_t$，观察含噪标量反馈 $Y_t = f(\mathbf{X}_t) + \eta_t$

**两阶段算法 OPD-Min-ESTR：**

1. **子空间探索（Stage 1）**：$T_1$轮随机探索 + 核范数正则化最小二乘估计 $\hat{\Theta}$，提取顶特征向量 $\hat{\bm{s}}$
   - 关键挑战：action集是结构化的forest矩阵（非高斯），需要新的RSC分析
   - 估计误差界：$\|\hat{\Theta} - \Theta^*\|_F^2 \leq 36\log(2|V|/\delta) / (\kappa^2 T_1)$
2. **降维+线性Bandit（Stage 2）**：用 $\hat{\bm{s}}$ 构造旋转矩阵，将 $|V|^2$ 维降至 $2|V|-1$ 维，运行标准OFUL

## 实验关键数据
- 在Erdős-Rényi和SBM两类图上测试，$|V| \in \{8, 16\}$，T=10000
- OPD-Min在所有设置下累积遗憾和运行时间均显著优于全维OFUL基线
- $|V|=16$时差异尤为明显：OFUL遗憾大幅升高且运行极慢
- OPD-Min能快速追上oracle子空间基线，有效弥合子空间估计的初始差距
- 在真实网络(Karate club, Les Misérables等)和大规模图($|V|$到1024)上验证了可扩展性
- 噪声鲁棒性：$\sigma_\eta \in \{0.01, 0.1, 1.0\}$ 下表现稳定

## 亮点
- 首次将FJ模型下极化最小化问题形式化为在线遗憾最小化，建立了社交干预与bandit理论的桥梁
- 利用问题的秩1结构实现 $|V|^2 \to O(|V|)$ 的维度降低，理论和实际效率均大幅提升
- 遗憾界 $\tilde{O}(\sqrt{T})$ 对时间horizon是最优的
- 针对forest矩阵约束提出新的RSC分析，解决了现有低秩bandit方法依赖连续采样的局限
- 有完整的理论证明和代码开源

## 局限性 / 可改进方向
- RSC参数 $\kappa$ 取全局最坏情况，可能过于保守（实验显示实际有效曲率更高）
- 当前仅处理单个话题的标量观点，多话题/多维观点场景需扩展
- 标量反馈假设较强，实际平台可能提供更丰富但更嘈杂的信号（如社区级极化）
- 实验图规模较小（主要 $|V| \leq 16$），大规模实验在附录（最大1024）
- 干预框架可能被恶意滥用（反转目标即可最大化极化），需治理保障

## 与相关工作的对比
- vs. Musco et al. 2018（离线，已知观点）：本文是首个在线+未知观点设置
- vs. Chaitanya et al. 2024（SDP上界）：不需要任何观点先验，且计算更轻量
- vs. Cinus et al. 2025（部分查询）：不需要查询任何agent观点
- vs. 全维OFUL：维度从 $|V|^2$ 降至 $O(|V|)$，遗憾和运行时间大幅改善
- vs. Lu et al. 2021/Kang et al. 2022（低秩bandit）：解决了离散结构化action空间的适配问题

## 评分
- 新颖性: ⭐⭐⭐⭐ (FJ模型+bandit的首次结合，问题建模elegant)
- 实验充分度: ⭐⭐⭐⭐ (合成+真实网络+可扩展性+鲁棒性，较全面)
- 写作质量: ⭐⭐⭐⭐⭐ (数学推导严谨，结构清晰，reproducibility excellent)
- 价值: ⭐⭐⭐⭐ (理论贡献扎实，落地有一定距离但方向重要)
