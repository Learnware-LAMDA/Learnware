.. _spec:
================================
Specification
================================

Concepts & Types
======================================

The search of helpful learnwares can be divided into two stages: statistical specification and semantic specification.

Statistical Specification
---------------------------

The learnware specification should ideally provide essential information about every model in the learnware market, enabling efficient and accurate identification for future users. Our current specification design has two components. The first part consists of a string of descriptions or tags assigned by the learnware market based on developer-submitted information. These descriptions or tags help identify the model's specification island. Different learnware market enterprises may use different descriptions or tags.

The second part of the specification is crucial for determining the model's position in the functional space :math:`F: \mathcal{X} \mapsto \mathcal{Y}` with respect to obj. A recent development in this area is the RKME (Reduced Kernel Mean Embedding) specification, which builds on the reduced set of KME (Kernel Mean Embedding) techniques. KME is a powerful method for mapping a probability distribution to a point in RKHS (Reproducing Kernel Hilbert Space), while the reduced set retains this ability with a concise representation that doesn't reveal the original data.

The RKME specification assumes that each learnware is a well-performed model on its training data. The RKME specification is based on RKME :math:`\widetilde{\Phi}`, which aims to provide a good representation by constructing a reduced set to approximate the empirical KME :math:`\Phi=\int_{\mathcal{X}} k(\boldsymbol{x}, \cdot) \mathrm{d} P(\boldsymbol{x})` of the underlying distribution. Theoretically, when the kernel function satisfies :math:`k(\boldsymbol{x}, \boldsymbol{x}) \leq 1` for all :math:`x \in \mathcal{X}`, we have the guarantee that

.. math::

   \|\widetilde{\Phi}-\Phi\|_{\mathcal{H}} \leq 2 \sqrt{\frac{2}{n}}+\sqrt{\frac{1}{m}}+\sqrt{\frac{2 \log (1 / \delta)}{m}},

with a probability of at least :math:`1-\delta`, where :math:`n, m` are the sizes of the RKME reduced set and the original data, respectively. It is known that when using characteristic kernels such as the Gaussian kernel, KME can capture all information about the distribution. Additionally, when the RKHS of the kernel function is finite-dimensional, RKME has a linear convergence rate :math:`O\left(e^{-n}\right)` to empirical KME; for infinite-dimensional RKHS, it has been proved constructively that RKME can enjoy :math:`O(\sqrt{d} / n)` convergence rate under :math:`L_{\infty}` measure, where :math:`d` is the dimension of the original data. Therefore, RKME is guaranteed to be a good estimation of KME and a valid representation for data distribution that encodes the ability of a trained model.

Under certain assumptions, the risk for the user task can be bounded, such as assuming that the distribution corresponding to the user's task matches that of a learnware, or that it can be approximated by a mixture of distributions corresponding to a set of learnwares' tasks, i.e.,

.. math::

   \mathcal{D}_u=\sum_{i=1}^N w_i \mathcal{D}_i

where :math:`\mathcal{D}_u` is the distribution corresponding to the user's task, :math:`N` is the number of learnwares, and :math:`\mathcal{D}_i` are their corresponding distributions. We have :math:`\sum_{i=1}^N w_i=1` and :math:`w_i \geq 0`. These two assumptions are known as task-recurrent and instance-recurrent assumptions. Additionally, assume that all learnwares are well-performed ones:

.. math::

   \mathbb{E}_{\mathcal{D}_i}\left[\ell\left(\widehat{f}_i(\boldsymbol{x}), \boldsymbol{y}\right)\right] \leq \epsilon, \forall i \in[N],

where :math:`\widehat{f}_i` is the function corresponding to the :math:`i`-th learnware, :math:`\ell` is the loss function, and :math:`\boldsymbol{y}` is assumed to be determined by a ground-truth global function :math:`h`. Under these assumptions, recent studies have attempted to bound the risk on the user's task. With the task-recurrent assumption and selecting the learnware :math:`\left(\widehat{f}_i, \tilde{\Phi}_i\right)` with the smallest RKHS distance :math:`\eta` according to RKME, given the loss function

.. math::

   \left|\ell\left(\widehat{f}_i(\boldsymbol{x}), h(\boldsymbol{x})\right)\right| \leq U, \forall \boldsymbol{x} \in \mathcal{X}, \forall i \in[N],

we have

.. math::

   \mathbb{E}_{\mathcal{D}_u}\left[\ell\left(\widehat{f}_i(\boldsymbol{x}), \boldsymbol{y}\right)\right] \leq \epsilon+U \eta+O\left(\frac{1}{\sqrt{m}}+\frac{1}{\sqrt{n}}\right).

As for the instance-recurrent assumption and the 0/1-loss

.. math::

   \ell_{01}(f(\boldsymbol{x}), \boldsymbol{y})=\mathbb{I}(f(\boldsymbol{x}) \neq \boldsymbol{y}),

a more general result has been achieved:

.. math::

   \mathbb{E}_{\mathcal{D}_u}\left[\ell_{01}(f(\boldsymbol{x}), \boldsymbol{y})\right] \leq \epsilon+R(g),

where :math:`R(g)=\sum_{i=1}^N w_i \mathbb{E}_{\mathcal{D}_1}\left[\ell_{01}(g(\boldsymbol{x}), i)\right]` represents the weighted risk of any learnware selector :math:`g(x)`, which takes unlabeled data as input and assigns it to the appropriate model, :math:`f(\boldsymbol{x})=\widehat{f}_{g(\boldsymbol{x})}(\boldsymbol{x})` is the final model for the user's task.

Efforts have been made to enable the learnware market to handle unseen tasks, where the user's task involves some unseen aspects that have never been addressed by the current learnwares in the market. A more general theoretical analysis has been presented based on mixture proportion estimation.


Semantic Specification
---------------------------

The semantic specification describes the characteristics of user's task and the market will identify potentially helpful leaarnwares whose models solve tasks similar to your requirements. The detail semantic specification is in `Indentification Learnwares <../workflow/identify.html>`_.


Regular Specification
======================================


Table Specification
--------------------------

Image Specification
--------------------------

Text Specification
--------------------------
Different from tabular data, each text input is a string of different length, so we should first transform them to equal-length arrays. Sentence embedding is used here to complete this transformation. We choose the model ``paraphrase-multilingual-MiniLM-L12-v2``, a lightweight multilingual embedding model. Then, we calculate the RKME specification on the embedding, 
just like we do with tabular data. Besides, we use the package ``langdetect`` to detect and store the language of the text inputs for further search. We hope to search for the learnware which supports the language of the user task.

System Specification
======================================