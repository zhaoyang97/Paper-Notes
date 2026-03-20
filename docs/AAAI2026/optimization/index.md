<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**🤖 AAAI2026** · 共 **4** 篇

**[A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds](a_distributed_asynchronous_generalized_momentum_algorithm_wi.md)**

:   提出一种完全异步（totally asynchronous）的广义动量（Generalized Momentum）分布式优化算法，无需假设通信/计算延迟的上界即可保证线性收敛，在 Fashion-MNIST 分类任务上比梯度下降快 71%、比 Heavy Ball 快 41%、比 Nesterov 加速梯度法快 19%。

**[A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication](a_unified_convergence_analysis_for_semi-decentralized_learni.md)**

:   本文在统一的收敛分析框架下，首次系统比较了半去中心化联邦学习中两种服务器-设备通信原语（S2S仅返回被采样设备 vs. S2A广播给所有设备），揭示了S2S在高组间异质性下更优、S2A在低异质性下更优的不同regime，并给出了实用的系统配置指南。

**[Explore How to Inject Beneficial Noise in MLLMs](explore_how_to_inject_beneficial_noise_in_mllms.md)**

:   提出 Multimodal Noise Generator (MuNG)，通过变分推断框架从图文对中动态生成"有益噪声"注入冻结的MLLM视觉特征中，以抑制无关语义、增强跨模态表征对齐，仅需约1%额外参数即可超越全参数微调和LoRA等PEFT方法。

**[On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)**

:   在二层过参数化线性网络上理论分析 Label Noise SGD 的学习动力学，揭示了两阶段行为——Phase I 中权重范数逐渐缩小使模型从 lazy regime 逃逸到 rich regime，Phase II 中权重与真实插值器对齐并收敛——并将该理论扩展到 SAM 优化器。
