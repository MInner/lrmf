# Log-Likelihood Ratio Minimizing Flows: Towards Robust and Quantifiable Neural Distribution Alignment

We introduce a new adversarial log-likelihood ratio domain alignment objective, show how to upper-bound it with a simple stable minimization objective, if the domain transformation is a normalizing flow, and show its relation to Jensen–Shannon divergence and GANs.

**[Ben Usman](https://cs-people.bu.edu/usmn/), [Nick Dufour](#), [Avneesh Sud](#), [Kate Seanko](http://ai.bu.edu/ksaenko.html)** </br>
NeurIPS 2020 </br>
<a href="https://arxiv.org/abs/2003.12170">arxiv</a> / <a href="https://crossminds.ai/video/log-likelihood-ratio-minimizing-flows-towards-robust-and-quantifiable-neural-distribution-alignment-606fe2f7f43a7f2f827c0167/">video [3min]</a> / <a href="pdf/lrmf_poster.pdf">poster</a> / <a href="https://papers.nips.cc/paper/2020/hash/f169b1a771215329737c91f70b5bf05c-Abstract.html">proceedings </a> / <a href="bib/lrmf.bib">bib</a>

> Distribution alignment has many applications in deep learning, including domain adaptation and unsupervised image-to-image translation. Most prior work on unsupervised distribution alignment relies either on minimizing simple non-parametric statistical distances such as maximum mean discrepancy or on adversarial alignment. However, the former fails to capture the structure of complex real-world distributions, while the latter is difficult to train and does not provide any universal convergence guarantees or automatic quantitative validation procedures. In this paper, we propose a new distribution alignment method based on a log-likelihood ratio statistic and normalizing flows. We show that, under certain assumptions, this combination yields a deep neural likelihood-based minimization objective that attains a known lower bound upon convergence. We experimentally verify that minimizing the resulting objective results in domain alignment that preserves the local structure of input domains.

<p align="center">
  <img src="https://cs-people.bu.edu/usmn/img/lrmf_large.png" />
</p>

## Experiments

This repo constains four (mostly) self-contained experiments implemented in [JAX](https://github.com/google/jax) and [Tensorflow Probability](https://github.com/tensorflow/probability). Some of them are heavy enough to make the github viewer fail, so we suggest using colab links to view experiments in the following order:
- `jax_gaussian_and_real_nvp.ipynb` [[colab]](https://colab.research.google.com/github/MInner/lrmf/blob/main/jax_gaussian_and_real_nvp.ipynb): we define 1D and 2D Gaussian likelihood-ratio minimizing flows (LRMF) and see that they sucessfully align normally-distributed distributions, and first two moments of moons distributions. After that we define a RealNVP LRMF and apply it to normally distributed datasets confirming that LRMF works correctly in the **over**parameterized regime.
- `tfp_moons_real_nvp_ffjord.ipynb` [[colab]](https://colab.research.google.com/github/MInner/lrmf/blob/main/tfp_moons_real_nvp_ffjord.ipynb): we define RealNVP and FFJORD LRMFs and show that they sucesfully align moon datasets. We also show that our approach produces more semantically meaningful alignment on this dataset than alternatives.
- `tfp_digit_embeddings_real_nvp.ipynb` [[colab]](https://colab.research.google.com/github/MInner/lrmf/blob/main/tfp_digit_embeddings_real_nvp.ipynb): we embed MNIST and USPS into a lower-dimentional space using a shared VAEGAN and train RealNVP LRMF to align resulting embedding clouds.
- `jax_gmm_vanishing_gradient.ipynb` [[colab]](https://colab.research.google.com/github/MInner/lrmf/blob/main/jax_gmm_vanishing_gradient.ipynb): we show that even 1D Equal Gaussian Mixture LRMF suffers from poly-exponentially vanishing generator gradients as distributions are drawn further apart (see Example 2.2 in the paper).

## Visualizations
An animated version of Figure 4 from the paper. 

Columns (left to right): MMD, EMD, back-to-back flow, LRMF. 

Rows: distributions, graduent, class label, classification accuracy, loss.

<p align="center">
  <img src="https://cs-people.bu.edu/usmn/img/gifs/lrmf_compressed.gif" />
</p>
