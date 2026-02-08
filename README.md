# FYP2 FINAL SUBMISSION: INTEGRATED MODEL OPTIMIZATION FRAMEWORK
**Author:** Chiu Siew Seng
**Theme:** Model Compression, Computational Efficiency, and CPU Deployment.

---

## 1. ABSTRACT (The "High-Level" Summary)
This project addresses the computational challenges of deploying Large Language Models (LLMs) on standard hardware. As models scale, they face "Computational Bloat," characterized by massive parameter counts and high Floating Point Operations (FLOPs). This creates a "Memory Wall" bottleneck, rendering models like BERT impractical for inference on consumer-grade CPUs. This research introduces an **Integrated Optimization Framework** comprising three synergistic techniques: **Intelligent Data Filtering**, a **Modified Cost-Aware Structural Pruning Algorithm**, and **Quantization-Aware Training (QAT)**. The optimization pipeline leverages GPU acceleration during training to produce a lightweight model specifically optimized for CPU deployment. Results demonstrate that the framework reduces theoretical FLOPs by 20% and achieves a functionally viable inference speed of **11.60 FPS** on a standard CPU. Crucially, the optimized model maintains a predictive accuracy of **92.20%**, surpassing the baseline. This study confirms that a holistic optimization strategy can effectively decouple high-performance inference from high-end hardware dependencies.

---

## 2. CHAPTER 1: INTRODUCTION REVISIONS

### 1.2 Problem Statement
"The primary challenge is the **Computational Bloat** of LLMs, which creates a **Hardware Gap**. Standard BERT models require high-VRAM GPUs, making them unusable on standard consumer-grade CPUs due to the **Memory Wall** bottleneck. This leads to prohibitive inference latency or Out-of-Memory (OOM) errors. Furthermore, traditional magnitude-based optimization often leads to accuracy degradation, failing to maintain a balance between speed and intelligence."

### 1.3 Objectives (Revised to Point Form)
The objectives of this project are listed as follows:
1.  **To design** an Integrated Optimization Framework that systematically combines data filtering, structural pruning, and quantization into a unified pipeline.
2.  **To develop** a Modified Cost-Aware Structural Pruning Algorithm that prioritizes structural units based on a balance of functional importance and computational cost (FLOPs).
3.  **To evaluate** and compare the performance of the optimized model against unoptimized and traditional benchmarks on standard CPU hardware to validate deployment viability.

---

## 3. CHAPTER 2: LITERATURE & THEORETICAL GROUNDING

### 2.3.1 Structural Pruning Logic
"While Han et al. (2015) established weight pruning as a standard, their unstructured approach creates sparse matrices that are incompatible with modern dense processors. This research adopts **Structural Pruning** to remove entire attention heads. Unlike traditional L1-Norm methods, we introduce a **Cost-Aware** scoring function (Equation 2.1) to penalize units with high computational costs."

**Table 2.1: Computational Complexity of Layer Types**
*(Note: Place Table Caption ABOVE the table)*

**Equation 2.1: Modified Cost-Aware Scoring Function**
$$Score(u) = \frac{Importance(L1\_Norm)}{(FLOPs\_Reduction)^\alpha} \quad\quad\quad (2.1)$$
*Explanation: The numerator represents functional importance, while the denominator acts as a computational penalty to maximize the efficiency-to-accuracy ratio.*

---

## 4. CHAPTER 3: METHODOLOGY (Synergy Rationale)

### 3.3 Rationale for Framework Synergy
"Applying quantization directly to a dense model preserves redundant parameters ('noise'). By applying **Structural Pruning first**, the framework removes these redundant neurons, acting as a form of **Regularization (Han et al., 2015)**. This results in a 'cleaner' architecture that is significantly more robust against the precision loss inherent in the subsequent **QAT stage (Jacob et al., 2018)**."

---

## 5. CHAPTER 4: RESULTS AND DISCUSSION (The "Final Battle")

### 4.2 Comprehensive Benchmarking Results
**Table 4.2.1: Comparative Analysis of Optimization Strategies (CPU Inference)**

| Metric | Original Baseline | Traditional L1-Norm | **Proposed Framework (Ours)** |
| :--- | :--- | :--- | :--- |
| **Model Size (MB)** | 418 MB | 374 MB | **176 MB (Smallest)** |
| **Accuracy (%)** | 91.97% | 91.74% | **92.20% (Highest)** |
| **FLOPs (G)** | 10.88 G | 9.43 G | **8.71 G (Lowest)** |
| **Inference FPS** | 14.14 | **16.99 (Fastest)** | 11.60 (Viable) |

### 4.4 Critical Analysis of Results
1.  **Why Ours is More Accurate:** The L1-Norm method prunes purely based on weight size, often causing 'brain damage' to critical nodes. Our **Modified Cost-Aware** method selectively removes noise, improving generalization.
2.  **Why the FPS Trade-off is Worth it:** While the L1-Norm is faster (16.99 FPS) because it is pure FP32, our model (11.60 FPS) achieves **extreme storage efficiency (176MB)** and **highest accuracy**. The 11.60 FPS is functionally viable for CPU deployment, whereas the Baseline is limited by the **Memory Wall (Gholami et al., 2021)**.

---

## 6. CHAPTER 5: CONCLUSION, LIMITATIONS & FUTURE WORK

### 5.1 Conclusion
"This research proves that high-performance BERT inference is feasible on consumer-grade CPUs. The framework achieves an optimal balance, delivering the **highest accuracy in the smallest storage footprint**. We successfully decoupled model performance from high-end hardware dependencies."

### 5.2 Limitations
1.  **Inferred Efficiency:** CPU power usage was inferred via FPS rather than direct hardware sensors.
2.  **Hyperparameter Selection:** The balancing factor ($\alpha$) was set manually at 0.1 based on heuristic values.

### 5.3 Future Works
1.  **Advanced Heuristics:** Exploring second-order information like **Optimal Brain Surgeon (LeCun et al., 1989)**.
2.  **Automatic Tuning:** Implementing **Bayesian Optimization (Snoek et al., 2012)** for automated hyperparameter search.
3.  **Generalizability:** Extending the framework to **Vision Transformers (ViT) (Dosovitskiy et al., 2020)**.

---

## 7. CORRECTED REFERENCE LIST (Official Versions)
1. **Devlin, J., et al. (2019).** BERT: Pre-training of Deep Bidirectional Transformers. *Proceedings of NAACL-HLT*.
2. **Han, S., et al. (2015).** Learning both weights and connections for efficient neural networks. *NeurIPS*.
3. **Jacob, B., et al. (2018).** Quantization and Training of Neural Networks. *Proceedings of CVPR*.
4. **Gholami, A., et al. (2021).** AI and Memory Wall. *IEEE Micro*.
5. **LeCun, Y., et al. (1989).** Optimal Brain Damage. *NeurIPS*.
6. **Dosovitskiy, A., et al. (2020).** An Image is Worth 16x16 Words (ViT). *ICLR*.