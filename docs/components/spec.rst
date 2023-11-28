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

Image data lives in a higher dimensional space than other data types. Unlike lower dimensional spaces, metrics defined based on Euclidean distances (or similar distances) will fail in higher dimensional spaces. This means that measuring the similarity between image samples becomes difficult. 

To address these issues, we use the Neural Tangent Kernel (NTK) based on Convolutional Neural Networks (CNN) to measure the similarity of image samples.  As we all know, CNN has greatly advanced the field of computer vision and is still a mainstream deep learning technique. 

Usage & Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

In this part, we show that how to generate Image Specification for the training set of the CIFAR-10 dataset. 
Note that the Image Specification is generated on a subset of the CIFAR-10 dataset with ``generate_rkme_image_spec``. 
Then, it is saved to file "cifar10.json" using ``spec.save``. 

In many cases, it is difficult to construct Image Specification on the full dataset. 
By randomly sampling a subset of the dataset, we can construct Image Specification based on it efficiently, with a strong enough statistical description of the full dataset.

.. tip::
   Typically, sampling 3,000 to 10,000 images is sufficient to generate the Image Specification.

.. code-block:: python

   import torchvision
   from torch.utils.data import DataLoader
   from learnware.specification import generate_rkme_image_spec

   SAMPLED_SIZE = 5000

   full_set = torchvision.datasets.CIFAR10(
      root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
   loader =  DataLoader(full_set, batch_size=SAMPLED_SIZE, shuffle=True)
   sampled_X, _ = next(iter(loader))

   spec = generate_rkme_image_spec(sampled_X)
   spec.save("cifar10.json")

Privacy Protection
^^^^^^^^^^^^^^^^^^^^^^^^^^

In the third row of the figure, we show the eight pseudo-data with the largest weights :math:`\beta` in the Image Specification generated on the CIFAR-10 dataset.
Notice that the Image Specification generated based on Neural Tangent Kernel (NTK) protects the user's privacy very well.

In contrast, we show the performance of the RBF kernel on image dat in the first row of the figure below. 
The RBF not only exposes the real data (plotted in the corresponding position in the second row), but also fails to fully utilise the weights :math:`\beta`.

.. image:: ../_static/img/image_spec.png
   :align: center

Text Specification
--------------------------


System Specification
======================================