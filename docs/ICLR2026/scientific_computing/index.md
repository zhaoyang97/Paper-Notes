<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧮 科学计算

**🔬 ICLR2026** · 共 **8** 篇

**[Astral: Training Physics-Informed Neural Networks with Error Majorants](astral_training_physics-informed_neural_networks_with_error_majorants.md)**

:   提出 Astral 损失函数（基于函数型后验误差上界/error majorant），替代传统 PiNN 中的残差损失来训练物理信息神经网络，实现训练过程中可靠的误差估计，并在扩散方程、Maxwell 方程等多种 PDE 上取得了更好或相当的精度。

**[Deep Learning for Subspace Regression](deep_learning_for_subspace_regression.md)**

:   将缩减阶建模中的子空间预测问题形式化为 Grassmann 流形上的回归，设计适用于子空间数据的损失函数和神经网络参数化，并提出子空间嵌入（embedding）技术——预测比目标更大的子空间——理论和实验证明可显著降低学习复杂度并提升精度。

**[DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)**

:   提出 DRIFT-Net 双分支神经算子，通过受控低频混合（谱分支）和局部细节保真（图像分支）的带宽融合（radial gating），解决窗口注意力中全局谱耦合不足导致的自回归漂移问题，在 Navier-Stokes 基准上误差降低 7%-54%。

**[Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)**

:   系统实证分析将 KAN（Kolmogorov-Arnold Networks）集成到硬约束递归物理信息架构（HRPINN）中的表现——发现小型 KAN 在单变量多项式残差（Duffing）上具有竞争力，但在乘法项（Van der Pol）上严重失败且超参数极度脆弱，标准 MLP 稳定性远优。

**[HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning](hyperkkl_enabling_non-autonomous_state_estimation_through_dynamic_weight_conditi.md)**

:   提出 HyperKKL，用超网络编码外源输入信号并即时生成 KKL 观测器参数，使非自治非线性系统的状态估计无需重新训练或在线梯度更新，在 Duffing、Van der Pol、Lorenz、Rössler 四个系统上验证有效。

**[Learning-guided Kansa Collocation for Forward and Inverse PDE Problems](learning-guided_kansa_collocation_for_forward_and_inverse_pde_problems.md)**

:   将基于径向基函数(RBF)的无网格Kansa方法从单变量线性PDE扩展到耦合多变量和非线性PDE场景，结合自调参技术和多种时间步进方案，并系统对比了与PINN、FNO等神经PDE求解器在正问题和反问题上的表现。

**[One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)**

:   论证标准神经 PDE 求解器在边界条件变化时实际学习的是"边界条件索引的算子族"而非单一边界无关算子，形式化为条件风险最小化导出不可识别性结果，实验验证边界分布偏移下的急剧退化。

**[Policy myopia as a mechanism of gradual disempowerment in Post-AGI governance, Circa 2049](policy_myopia_as_a_mechanism_of_gradual_disempowerment_in_post-agi_governance_ci.md)**

:   论证政策短视（policy myopia）不是注意力分配问题，而是后 AGI 治理中产生不可逆人类失权的**机制**——通过显著性捕获、能力级联和价值锁定三个耦合机制，跨经济/政治/文化系统产生自我强化的人类边缘化均衡。
