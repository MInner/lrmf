# Log-Likelihood Ratio Minimizing Flows: Towards Robust and Quantifiable Neural Distribution Alignment

We introduce a new adversarial log-likelihood ratio domain alignment objective, show how to upper-bound it with a simple stable minimization objective, if the domain transformation is a normalizing flow, and show its relation to Jensen–Shannon divergence and GANs.

**[Ben Usman](https://cs-people.bu.edu/usmn/), [Nick Dufour](#), [Avneesh Sud](#), [Kate Seanko](http://ai.bu.edu/ksaenko.html)** </br>
NeurIPS 2020 </br>
<a href="https://arxiv.org/abs/2003.12170">arxiv</a> / <a href="https://crossminds.ai/video/log-likelihood-ratio-minimizing-flows-towards-robust-and-quantifiable-neural-distribution-alignment-606fe2f7f43a7f2f827c0167/">video [3min]</a> / <a href="pdf/lrmf_poster.pdf">poster</a> / <a href="https://papers.nips.cc/paper/2020/hash/f169b1a771215329737c91f70b5bf05c-Abstract.html">proceedings </a> / <a href="bib/lrmf.bib">bib</a>

> Distribution alignment has many applications in deep learning, including domain adaptation and unsupervised image-to-image translation. Most prior work on unsupervised distribution alignment relies either on minimizing simple non-parametric statistical distances such as maximum mean discrepancy or on adversarial alignment. However, the former fails to capture the structure of complex real-world distributions, while the latter is difficult to train and does not provide any universal convergence guarantees or automatic quantitative validation procedures. In this paper, we propose a new distribution alignment method based on a log-likelihood ratio statistic and normalizing flows. We show that, under certain assumptions, this combination yields a deep neural likelihood-based minimization objective that attains a known lower bound upon convergence. We experimentally verify that minimizing the resulting objective results in domain alignment that preserves the local structure of input domains.

<p align="center">
  <img src="https://cs-people.bu.edu/usmn/img/lrmf_large.png" />
</p>
