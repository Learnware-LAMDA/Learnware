.. _spec:
================================
Specification
================================

Learnware Specification
-----------------------------

The learnware specification should ideally provide essential information about every model in the learnware market, enabling efficient and accurate identification for future users. As mentioned in Section 3.1, our current specification design has two components. The first part consists of a string of descriptions or tags assigned by the learnware market based on developer-submitted information. These descriptions or tags help identify the model's specification island. Different learnware market enterprises may use different descriptions or tags.

The second part of the specification is crucial for determining the model's position in the functional space $F: \mathcal{X} \mapsto \mathcal{Y}$ with respect to obj. A recent development in this area is the RKME (Reduced Kernel Mean Embedding) specification, which builds on the reduced set of KME (Kernel Mean Embedding) techniques. KME is a powerful method for mapping a probability distribution to a point in RKHS (Reproducing Kernel Hilbert Space), while the reduced set retains this ability with a concise representation that doesn't reveal the original data.

The RKME specification assumes that each learnware is a well-performed model on its training data. The RKME specification is based on RKME $\widetilde{\Phi}$, which aims to provide a good representation by constructing a reduced set to approximate the empirical KME $\Phi=\int_{\mathcal{X}} k(\boldsymbol{x}, \cdot) \mathrm{d} P(\boldsymbol{x})$ of the underlying distribution. Theoretically, when the kernel function satisfies $k(\boldsymbol{x}, \boldsymbol{x}) \leq 1$ for all $x \in \mathcal{X}$, we have the guarantee that

$$
\|\widetilde{\Phi}-\Phi\|_{\mathcal{H}} \leq 2 \sqrt{\frac{2}{n}}+\sqrt{\frac{1}{m}}+\sqrt{\frac{2 \log (1 / \delta)}{m}},
$$

with a probability of at least $1-\delta$, where $n, m$ are the sizes of the RKME reduced set and the original data, respectively. It is known that when using characteristic kernels such as the Gaussian kernel, KME can capture all information about the distribution. Additionally, when the RKHS of the kernel function is finite-dimensional, RKME has a linear convergence rate $O\left(e^{-n}\right)$ to empirical KME; for infinite-dimensional RKHS, it has been proved constructively that RKME can enjoy $O(\sqrt{d} / n)$ convergence rate under $L_{\infty}$ measure, where $d$ is the dimension of the original data.

Therefore, RKME is guaranteed to be a good estimation of KME and a valid representation for data distribution that encodes the ability of a trained model.

Under certain assumptions, the risk for the user task can be bounded, such as assuming that the distribution corresponding to the user's task matches that of a learnware, or that it can be approximated by a mixture of distributions corresponding to a set of learnwares' tasks, i.e.,

$$
\mathcal{D}_u=\sum_{i=1}^N w_i \mathcal{D}_i
$$

where $\mathcal{D}_u$ is the distribution corresponding to the user's task, $N$ is the number of learnwares, and $\mathcal{D}_i$ are their corresponding distributions, 
