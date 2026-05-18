
### 1. Executive Summary
The paper introduces **GotenNet** (Geometric Tensor Network), a novel architecture for 3D equivariant graph neural networks designed to model complex molecular geometries. The authors address a persistent methodological dichotomy in geometric deep learning: the trade-off between computational efficiency and geometric expressiveness. GotenNet resolves this by employing a spherical-scalarization paradigm that utilizes inner-product-based geometric tensor operations, thereby eliminating the need for irreducible representations (irreps) and Clebsch-Gordan (CG) tensor products. The framework demonstrates strict E(3) equivariance, superior predictive accuracy across scalar and vector/tensor properties, and favorable computational scaling on large-scale molecular datasets.

### 2. Problem Formulation & Motivation
Contemporary equivariant GNNs for molecular modeling are broadly categorized into two paradigms:
- **Scalarization-based models:** Project 3D geometric information into invariant scalar features prior to reconstruction. While computationally efficient, this projection attenuates higher-order spatial relationships, limiting expressiveness.
- **High-degree steerable models:** Employ irreps and CG transforms to manipulate geometric features in higher-resolution representation spaces. Although theoretically rigorous and highly expressive, these models incur substantial computational overhead due to dense tensor product operations.

The authors posit that explicit CG coefficients are not strictly necessary for capturing high-degree geometric relationships. Recent theoretical insights indicate that inner-product operations can equivalently encode these relationships with significantly reduced computational complexity. GotenNet operationalizes this insight into a practical, scalable architecture.

### 3. Methodological Framework
GotenNet comprises four principal components, each engineered to preserve geometric consistency while optimizing representational capacity:

| Component | Function | Key Mechanism |
|-----------|----------|---------------|
| **Unified Structural Embedding** | Initializes node and edge representations | Aggregates neighbor information via radial basis functions and learnable embeddings. Simultaneously encodes semantic (atomic number) and geometric (interatomic distances) information without irrep/CG dependencies. |
| **Geometry-Aware Tensor Attention (GATA)** | Enhances message-passing with spatial context | Infuses geometric encoding into self-attention keys. Splits attention outputs into degree-specific coefficients, enabling precise, degree-aware aggregation of steerable features. |
| **Hierarchical Tensor Refinement (HTR)** | Iteratively updates edge representations | Projects high-degree steerable features via SO(3)-equivariant linear transformations, computes aggregated similarities through inner products, and applies residual updates to edge scalars. |
| **Equivariant Feed-Forward (EQFF)** | Non-linear feature transformation | Separately processes scalar and steerable channels. Applies degree-preserving linear mappings and element-wise operations, ensuring stable non-linear expansion while maintaining equivariance. |

### 4. Theoretical Guarantees
The manuscript provides rigorous mathematical proofs (Appendices A–C) demonstrating that GotenNet maintains **strict E(3) equivariance** across all layers:
- **Initialization:** Node and edge scalar features are proven invariant; steerable features transform covariantly under Euclidean operations.
- **GATA & HTR:** Attention coefficients and inner-product aggregations are invariant under rotation and translation. The Hadamard product between invariant scalars and equivariant tensors preserves transformation properties.
- **EQFF:** Channel-wise operations are constructed to commute with E(3) group actions.
By induction, the composition of these equivariant modules guarantees end-to-end geometric consistency, satisfying the requirements of physical molecular systems.

### 5. Empirical Evaluation & Results
GotenNet was evaluated across four benchmark datasets, encompassing both equilibrium property prediction and molecular dynamics force field modeling:

- **QM9:** GotenNet (S, B, L variants) achieves state-of-the-art performance across 12 molecular targets. Even the smallest variant surpasses prior baselines on 9/12 targets, with the large variant reducing standard MAE by >16% compared to the best baseline.
- **Molecule3D:** Demonstrates robust scalability to >3 million graphs. Achieves lowest errors across µ, ε_HOMO, ε_LUMO, and ∆ε, outperforming SaVeNet-L by 24–32%.
- **MD22 & rMD17:** Exhibits superior accuracy in energy and force predictions for flexible biomolecules and supramolecular systems (up to 370 atoms). Reduces force MAE by up to 31.3% and energy MAE by >35% on complex systems like the Buckyball catcher and double-walled nanotube.

Ablation studies confirm that structural embedding, geometric encoding, and HTR are individually critical to performance. The model scales predictably with layer depth and embedding dimension, with diminishing returns observed only when dataset size becomes a bottleneck.

### 6. Computational Efficiency & Scalability
GotenNet’s architectural design yields substantial efficiency gains relative to attention-based equivariant baselines:
- **Latency & Throughput:** Training and inference latencies scale favorably with node count. While dense-attention models (e.g., Geoformer, EquiformerV2) exhibit O(n²) computational bottlenecks, GotenNet maintains near-linear scaling.
- **Resource Utilization:** The base variant requires ≤2 GPU days for full training on QM9, with inference latencies 25–42% lower than competing architectures. Parameter counts remain competitive (6.1M–18.3M across scales).
- **Coefficient Design:** An ablation comparing shared vs. degree-specific spherical harmonic coefficients reveals that individual coefficients yield marginally higher expressiveness at the cost of increased parameterization, providing practitioners with a tunable efficiency-accuracy trade-off.

### 7. Limitations & Future Directions
The authors acknowledge several avenues for extension:
- **Scale Equivariance:** Current formulations assume fixed spatial scales; incorporating scale-equivariant operations could enhance generalization across molecular conformations.
- **Higher-Order Representations:** Extending beyond second-degree tensors (L_max=2) may capture more complex anisotropic interactions.
- **Sparse Implementations:** Optimizing for sparse neighborhood graphs would further reduce memory overhead for ultra-large biomolecular systems.
- **Theoretical Analysis:** Formal characterization of the expressiveness-efficiency boundary and generalization bounds for geometric tensor networks remain open research questions.

### 8. Conclusion
GotenNet constitutes a significant methodological advancement in geometric deep learning. By substituting computationally intensive CG transforms with geometry-aware inner-product operations, hierarchical refinement, and degree-specific attention mechanisms, the architecture successfully decouples high geometric expressiveness from prohibitive computational costs. Its rigorous equivariance guarantees, consistent state-of-the-art performance across diverse molecular benchmarks, and favorable scaling properties establish it as a versatile and scalable framework for next-generation molecular property prediction, force field modeling, and broader applications in computational chemistry and materials science.

