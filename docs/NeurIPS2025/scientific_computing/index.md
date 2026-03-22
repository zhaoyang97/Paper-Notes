<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧮 科学计算

**🧠 NeurIPS2025** · 共 **7** 篇

**[DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)**

:   提出 DeltaPhi 框架：不直接学习 PDE 的输入→输出映射，而是学习**相似物理状态之间的残差**，利用物理系统稳定性实现隐式数据增强，在数据稀缺场景下显著提升各类神经算子的性能。

**[Eddyformer Accelerated Neural Simulations Of Three-Dimensional Turbulence At Sca](eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)**

:   提出 EddyFormer，一种基于谱元法 (SEM) 的 Transformer 架构，将流场分解为 LES（大尺度）和 SGS（小尺度）两路并行流，在 256³ 分辨率 3D 湍流上达到 DNS 级精度且加速 30 倍，并在未见的 4× 更大域上泛化良好。

**[Enforcing Governing Equation Constraints In Neural Pde Solvers Via Training-Free](enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)**

:   评估两种无训练后处理投影方法（非线性 LBFGS 优化和线性化 Jacobian 投影）来强制神经 PDE 求解器满足控制方程约束，在 Lorenz/KS/NS 三个动力学系统上大幅减少约束违反并提升精度，尤其 LBFGS 投影远优于 physics-informed 训练。

**[GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)**

:   首次提出可扩展的5D神经网络代理模型 GyroSwin，将 Swin Transformer 扩展至5维回旋动力学相空间，通过交叉注意力实现3D↔5D交互、通道式模态分离捕获带状流，在等离子体湍流模拟中实现比传统准线性方法更高的精度，且比数值求解器（GKW）快3个数量级。

**[Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)**

:   揭示了标准 BSDE 方法性能不如 PINNs 的根本原因是 Euler-Maruyama 积分引入的不可消除离散化偏差，提出基于 Stratonovich 形式的 Heun-BSDE 方法彻底消除该偏差，在高维 PDE 上与 PINNs 竞争。

**[Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)**

:   挑战了"神经 PDE 模拟器精度受限于训练数据（数值求解器）精度"的传统认知，发现并严格定义了 **emulator superiority** 现象——仅在低精度求解器数据上训练的神经网络，在以高精度参考解评估时竟能超越其训练求解器本身。

**[Towards Universal Neural Operators Through Multiphysics Pretraining](towards_universal_neural_operators_through_multiphysics_pretraining.md)**

:   研究基于 Transformer/SSM 的神经算子在多物理场 PDE 上的迁移学习能力，通过 adapter-based 架构（lifting/projection 作为问题特定适配器，算子层共享）实现跨 PDE 类型的预训练和微调，在多个场景中显著提升模型质量并减少微调成本。
