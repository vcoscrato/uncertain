Uncertain
=======

Uncertainty-driven recommendation systems in Python.

:warning: This project is currently under development and should be used very carefully.

This project provides pytorch-lightning [:link:](https://github.com/PyTorchLightning/pytorch-lightning) based implementations of several recommendation algorithms:

* Explicit feedback:

    * FunkSVD Factorization Model [:link:](https://sifter.org/~simon/journal/20061211.html)
    
    * Confidence-aware Probabilistic Matrix Factorization (CPMF) [:newspaper:](https://ojs.aaai.org/index.php/AAAI/article/view/11251)

* Implicit feedback:

    * Logistic Matrix Factorization (logMF) [:newspaper:](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)
    
    * Bayesian personalized ranking Matrix Factorization (bprMF) [:newspaper:](https://arxiv.org/pdf/1205.2618.pdf)
    
* Ordinal feedback:
    
    * OrdRec [:newspaper:](https://dl.acm.org/doi/abs/10.1145/2043932.2043956)
