# Equivariant Spherical CNNs for Neonatal Diffusion MRI

This repository provides the official implementation of our paper:

> **Equivariant Spherical CNNs for Accurate Fiber Orientation Distribution Estimation in Neonatal Diffusion MRI with Reduced Acquisition Time**  
> *Haykel Snoussi and Davood Karimi*  
> Department of Radiology, Boston Children’s Hospital & Harvard Medical School


## Summary

We introduce a geometric deep learning framework based on **rotationally equivariant Spherical CNNs (sCNNs)** to estimate **Fiber Orientation Distributions (FODs)** from neonatal diffusion MRI (dMRI) with **only 30% of the full acquisition protocol**. The model was trained and evaluated on 43 neonatal dMRI datasets from the Developing Human Connectome Project (dHCP).

### Highlights
- **SO(3)-equivariant** spherical convolutions that respect rotational symmetries of diffusion signals.
- **Shell-attention mechanism** to dynamically weight contributions from different b-value shells.
- **Spatial-domain loss function** for perceptually meaningful FOD reconstructions.
- Achieves **superior FOD estimation** and tractography compared to standard MLP and MSMT-CSD.

### 🧪 Method Overview

<img src="figures/Fig3.pdf" alt="sCNN Architecture and Pipeline" width="100%"/>

*Figure 3: Overview of the full data processing pipeline and sCNN architecture.*



## 📄 Citation

If you use this work, please cite:

```bibtex
@article{snoussi2025scnn,
  title={Equivariant Spherical CNNs for Accurate Fiber Orientation Distribution Estimation in Neonatal Diffusion MRI with Reduced Acquisition Time},
  author={Snoussi, Haykel and Karimi, Davood},
  journal={arXiv preprint arXiv:},
  year={2025}
}
