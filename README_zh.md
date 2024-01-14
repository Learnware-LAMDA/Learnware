<div align=center>
  <img src="./docs/_static/img/logo/logo.svg"  width="420" height="auto" style="max-width: 100%;"/>
  <br/>
  <br/>
</div>

<p align="center">
    <a href="https://pypi.org/project/learnware/#files">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/learnware.svg?logo=python&logoColor=white">
    </a>
    <a href="https://pypi.org/project/learnware/#files">
        <img alt="Platform" src="https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey">
    </a>
    <a href="https://pypi.org/project/learnware/#history">
        <img alt="PypI Versions" src="https://img.shields.io/pypi/v/learnware">
    </a>
    <a href="https://img.shields.io/pypi/dm/example-package">
        <img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/example-package">
    </a>
    <a href="https://learnware.readthedocs.io/en/latest/?badge=latest">
        <img alt="Documentation Status" src="https://readthedocs.org/projects/learnware/badge/?version=latest">
    </a>
    <a href="LICENSE">
        <img alt="License" src="https://img.shields.io/pypi/l/learnware">
    </a>
</p>

<p>
    <h3 align="center">
        <b>中文</b> |
        <a href="README.md">English</a>
    </h3>
</p>

# 简介

学件范式由周志华教授在2016年提出 [1, 2]，旨在构建一个巨大的模型平台系统：即学件基座系统，系统地组织管理世界各地的机器学习开发者分享的模型，并通过统一的方式识别、利用已有模型的能力快速解决新的机器学习任务。

本项目开发的 `learnware` 包对学件范式中的核心组件和算法进行了实现，全流程地支持学件上传、检测、组织、查搜、部署和复用等功能。基于良好的结构设计，`learnware` 包具有高度可扩展性，为后续相关算法和功能的开发打下坚实基础。

此外，`learnware` 包被用于「[北冥坞系统](https://bmwu.cloud)」中，作为系统的核心引擎支撑整个系统的运转。科研人员也可以使用 `learnware` 包高效地探索学件相关研究。

[1] Zhi-Hua Zhou. Learnware: on the future of machine learning. Frontiers of Computer Science, 2016, 10(4): 589–590 <br/>
[2] 周志华. 机器学习: 发展与未来. 中国计算机学会通讯, 2017, vol.13, no.1 (2016 中国计算机大会 keynote)

## 学件范式

学件由性能优良的机器学习模型和描述模型的**规约**构成，即「学件 = 模型 + 规约」。
学件的规约由「语义规约」和「统计规约」两部分组成：

- 语义规约通过文本对模型的类型及功能进行描述；
- 统计规约则通过各类机器学习技术，刻画模型所蕴含的统计信息。

学件的规约刻画了模型的能力，使得模型能够在未来用户事先对学件一无所知的情况下被充分识别并复用，以满足用户需求。

<div align="center">
  <img src="./docs/_static/img/learnware_market-zh.svg" width="700" height="auto" style="max-width: 100%;" />
</div>

如上图所示，在学件范式中，系统的工作流程主要分为以下两个阶段：

- **提交阶段**：开发者自发地提交各式各样的学件到学件基座系统，而系统会对这些学件进行质量检查和进一步的组织。
- **部署阶段**：当用户提交任务需求后，学件基座系统会根据学件规约推荐对用户任务有帮助的学件并指导用户进行部署和复用。

## 架构设计

<div align="center">
  <img src="./docs/_static/img/learnware_framework.svg" width="700" height="auto" style="max-width: 100%;"/>
</div>

架构设计的原则包括：解耦 (Decoupling)、自治 (Autonomy)、可重用性 (Reusability) 以及可扩展性 (Scalability)。
上图从模块和工作流程的角度对整个架构进行了阐述。

- 针对工作流程 (Workflow)，`learnware` 包括「提交阶段」和「部署阶段」。

<div align=center>

|  阶段 | 描述  |
|  ----  | ----  |
| 提交阶段  | 开发者自发地将学件提交到学件市场中，随后市场会进行学件检测并对这些学件进行相应地组织。 |
| 部署阶段 | 学件市场根据用户的任务需求推荐学件，并提供高效的学件部署和复用的方法。 |

</div>

- 针对学件范式下各类模块 (Module)，`learnware` 是一个包含 `Learnware`, `Market`, `Specification`, `Model`, `Reuse` 和 `Interface` 等模块的平台。

<div align=center>

|  模块 | 描述  |
|  ----  | ----  |
| `Learnware`  | 学件由规约模块和模型模块组成。 |
| `Market` | 设计用于学件的组织、查搜和检测。 |
| `Specification` | 生成并存储学件的统计和语义规约，可用于学件的查搜和复用。 |
| `Model` | 包括模型和模型容器，可以提供统一的模型调用接口并自动创建隔离的模型运行环境。 |
| `Reuse` | 包括数据无关和数据相关的复用方式与异构学件对齐方式，可用于学件的部署和复用。 |
| `Interface` | 与北冥坞系统进行网络通讯的接口。|

</div>


# 快速上手

## 环境安装

`learnware` 包目前托管在 [PyPI](https://pypi.org/project/learnware/) 平台，其具体安装方式如下：

```bash
pip install learnware
```

在 `learnware` 包中，除了基础类之外，许多核心功能（如学件规约生成、学件部署等）都需要依赖 `torch` 库。用户可选择手动安装 `torch`，或直接采用以下命令安装 `learnware` 包：

```bash
pip install learnware[full]
```

但需要特别注意的是，由于用户本地环境可能较为复杂，安装 `learnware[full]` 并不能确保 `torch` 能够在用户的本地环境成功调用 `CUDA`。

## 学件准备

在 `learnware` 包中，每个学件都是一个 `zip` 包，其中至少需要包含以下四个文件：

- `learnware.yaml`：学件配置文件；
- `__init__.py`：提供使用模型的方法；
- `stat.json`：学件的统计规约，其文件名可自定义并记录在 learnware.yaml 中；
- `environment.yaml` 或 `requirements.txt`：指明模型的运行环境。

为方便大家构建学件，我们提供了「[学件模板](https://www.bmwu.cloud/static/learnware-template.zip)」，大家可在其基础上构建自己的学件。
关于学件 `zip` 包中各文件的详细描述可参考文档：[学件准备](https://learnware.readthedocs.io/en/latest/workflows/upload.html#prepare-learnware)。

## 工作流程

用户可根据以下步骤实现 `learnware` 中的工作流程。

### 初始化学件市场

`EasyMarket` 类提供了学件市场的核心功能。根据如下代码，可以实例化一个名为 "demo" 的基础学件市场：

```python
from learnware.market import instantiate_learnware_market

# 实例化学件市场
demo_market = instantiate_learnware_market(market_id="demo", name="easy", rebuild=True)
```

### 上传学件

在将学件上传到「学件市场」之前，需要创建相应的语义规约，即 `semantic_spec`。这涉及选择或输入预定义的语义标签的值，以描述你的任务和模型的特性。

例如，以下代码示例生成了适用于教育场景的 `Scikit-Learn` 类型模型的语义规约。该模型用于对表格数据执行分类任务：

```python
from learnware.specification import generate_semantic_spec

semantic_spec = generate_semantic_spec(
    name="demo_learnware",
    data_type="Table",
    task_type="Classification",
    library_type="Scikit-learn",
    scenarios="Education",
    license="MIT",
)
```

得到语义规约后，可以使用如下代码上传学件：

```python
demo_market.add_learnware(zip_path, semantic_spec)
```

其中 `zip_path` 为待上传学件 `zip` 包的路径。

### 语义规约查搜

为了找到与你的任务目标相符的学件，你需要提供一个名为 `user_semantic` 的语义规约，来概述你的任务特点。随后，学件市场将通过 `user_semantic` 进行语义查搜，识别与你的任务需求相近的学件。

```python
# 构造包含语义规约的 user_info
user_info = BaseUserInfo(id="user", semantic_spec=semantic_spec)

# search_learnware: 当 user_info 不包含统计规约时，仅执行语义规约查搜
search_result = easy_market.search_learnware(user_info)
single_result = search_results.get_single_results()

# single_result: 语义规约查搜返回的 Tuple[Score, Learnware] 列表
print(single_result)
```

### 统计规约查搜

如果提供统计规约文件 `stat.json`，学件市场可以基于上述查搜结果进一步进行更准确的查搜。
此阶段的查搜将利用统计信息来识别一个或多个对你的任务有帮助的学件。

以下代码展示了使用 Reduced Kernel Mean Embedding (RKME) 作为统计规约进行查搜的例子：

```python
import learnware.specification as specification

user_spec = specification.RKMETableSpecification()

# unzip_path: 解压缩的学件文件夹路径
user_spec.load(os.path.join(unzip_path, "rkme.json"))
user_info = BaseUserInfo(
    semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_spec}
)
search_result = easy_market.search_learnware(user_info)

single_result = search_results.get_single_results()
multiple_result = search_results.get_multiple_results()

# search_item.score: 根据 MMD 距离，按降序排列
# search_item.learnware.id: 学件 id, 根据查搜匹配度按降序排列
for search_item in single_result:
    print(f"score: {search_item.score}, learnware_id: {search_item.learnware.id}")

# mixture_item.learnwares: 可结合使用的学件集合
# mixture_item.score: `mixture_item.learnwares` 中各学件集合的查搜匹配度
for mixture_item in multiple_result:
    print(f"mixture_score: {mixture_item.score}\n")
    mixture_id = " ".join([learnware.id for learnware in mixture_item.learnwares])
    print(f"mixture_learnware: {mixture_id}\n")
```

### 多学件复用

使用上一步中返回的学件列表 `mixture_learnware_list`，你可以轻松地复用它们对自己的数据进行预测，而无需从头开始训练模型。我们提供了两种方法来重用学件集合：`JobSelectorReuser` 和 `AveragingReuser`。将以下代码片段中的 `test_x` 替换为你的测试数据，即可实现学件复用：

```python
from learnware.reuse import JobSelectorReuser, AveragingReuser

# 使用 jobselector reuser 复用查搜到的学件, 并对 text_x 进行预测
reuse_job_selector = JobSelectorReuser(learnware_list=mixture_item.learnwares)
job_selector_predict_y = reuse_job_selector.predict(user_data=test_x)

# 使用 averaging ensemble reuser 复用查搜到的学件, 并对 text_x 进行预测
reuse_ensemble = AveragingReuser(learnware_list=mixture_item.learnwares)
ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)
```

我们还提供了两种方法，可基于用户的有标记数据来复用给定的学件集合：`EnsemblePruningReuser` 和 `FeatureAugmentReuser`。
参考下述代码，其中 `test_x` 为测试数据，`train_x, train_y` 为有标记的训练数据：

```python
from learnware.reuse import EnsemblePruningReuser, FeatureAugmentReuser

# 使用 ensemble pruning reuser 复用查搜到的学件, 并对 text_x 进行预测
reuse_ensemble = EnsemblePruningReuser(learnware_list=mixture_item.learnwares, mode="classification")
reuse_ensemble.fit(train_x, train_y)
ensemble_pruning_predict_y = reuse_ensemble.predict(user_data=test_x)

# 使用 feature augment reuser 复用查搜到的学件, 并对 text_x 进行预测
reuse_feature_augment = FeatureAugmentReuser(learnware_list=mixture_item.learnwares, mode="classification")
reuse_feature_augment.fit(train_x, train_y)
feature_augment_predict_y = reuse_feature_augment.predict(user_data=test_x)
```

### 自动工作流程示例

`learnware` 包提供了自动化的工作流程示例，包括准备学件、在学件市场中上传和删除学件，以及使用语义和统计规约查搜学件。
工作流程示例可参考 `test/test_workflow/test_workflow.py` 文件。

# 实验示例

我们构建了各种类型的实验场景，并进行了充分的测试，以评估规约生成、学件查搜以及在表格、图像和文本数据上学件复用的基线算法。

## 实验配置

所有实验均在一台 Linux 服务器上完成，其具体规格如下表所示。服务器的所有处理器均用于训练和评估。

<div align=center>

| 系统               | GPU                | CPU                      |
|----------------------|--------------------|--------------------------|
| Ubuntu 20.04.4 LTS   | Nvidia Tesla V100S | Intel(R) Xeon(R) Gold 6240R |

</div>

## 表格场景实验

在各种表格数据集上，我们首先评估了从学件市场中识别和复用与用户任务具有相同特征空间的学件的性能。另外，由于表格任务通常来自异构的特征空间，我们还评估了从不同特征空间中识别和复用学件的性能。

### 实验设置

我们的实验利用了销量预测领域的三个公共数据集：[Predict Future Sales (PFS)](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)，[M5 Forecasting (M5)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) 和 [Corporacion](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)。为了扩大实验规模，我们对这些数据集应用了多种特征工程方法。然后，我们将每个数据集按店铺划分，并进一步将每个店铺的数据划分为训练集和测试集。我们在每个 Corporacion 和 PFS 训练集上训练了一个 LightGBM 模型，而测试集和 M5 数据集被用于构建用户任务。基于上述方式，我们构建了一个包含 265 个学件的学件市场，涵盖了五种特征空间和两种标签空间。所有这些学件都已上传至[北冥坞学件基座系统](https://bmwu.cloud/)。

### 基线算法

复用学件的最基本方式是 Top-1 复用 (Top-1 reuser)，即直接使用由 RKME 规约选择的单个学件。此外，我们实现了两种数据无关复用器和两种数据相关复用器，它们可用于复用从市场中识别出的单个或多个有用的学件。当用户无标记的数据时，JobSelector 复用器通过训练一个任务选择器为不同的样本选择不同的学件；AverageEnsemble 复用器使用集成方法进行预测。在用户有测试数据和少量有标记训练数据的情况下，EnsemblePruning 复用器有选择地集成一组学件，选择最适合用户任务的学件；FeatureAugment 复用器将每个接收到的学件视为特征增强器，将其输出视为新特征，然后在增强的特征集上构建一个简单的模型。JobSelector 和 FeatureAugment 只对表格数据有效，而其他方法也适用于文本和图像数据。

### 同构场景

在同构场景中，PFS 数据集中的 53 家商店被视为 53 个独立的用户。每个商店使用自己的测试数据作为用户数据，并应用与学件市场相同的特征工程方法。这些用户随后可以在市场内搜索与其任务具有相同特征空间的同构学件。

当用户没有标记的数据或只有少量有标记数据时，我们对不同的基线算法进行了比较。下表显示了所有用户的平均损失。结果表明，我们提供的方法远远优于从市场中随机选择一个学件的结果。

<div align=center>

| Setting                           | MSE    |
|-----------------------------------|--------|
| Mean in Market (Single)           | 0.897  |
| Best in Market (Single)           | 0.756  |
| Top-1 Reuse (Single)              | 0.830  |
| Job Selector Reuse (Multiple)     | 0.848  |
| Average Ensemble Reuse (Multiple) | 0.816  |

</div>

下图展示了当用户提供不同数量有标记数据的结果；对于每个用户，我们进行了多次实验，并计算了损失的均值和标准差；图中展示了所有用户的平均损失。其表明，当用户只有有限的训练数据时，识别和复用单个或多个学件相对于用户自己训练的模型表现出更好的性能。

<div align=center>
  <img src="./docs/_static/img/Homo_labeled_curves.svg"  width="500" height="auto" style="max-width: 100%;"/>
</div>

### 异构场景

基于学件市场中学件与用户任务之间的相似性，异构情况可以进一步分为不同的特征工程和不同的任务场景。

#### 不同特征工程的场景

我们将 PFS 数据集中的 41 家商店视为用户，采用与市场中学件不同的特征工程方法生成他们的用户数据。因此，尽管市场上的某些学件也是为 PFS 数据集设计的，但特征空间并不完全一致。

在这个实验设置中，我们研究了各种数据无关复用器。下表中的结果表明，即使用户缺乏标记数据，市场也能表现出较强的性能，特别是使用多学件复用方法 AverageEnsemble 时。

<div align=center>

| Setting                           | MSE    |
|-----------------------------------|--------|
| Mean in Market (Single)           | 1.149  |
| Best in Market (Single)           | 1.038  |
| Top-1 Reuse (Single)              | 1.105  |
| Average Ensemble Reuse (Multiple) | 1.081  |

</div>


#### 不同的任务场景

我们在 M5 数据集的所有十家商店上采用了三种不同的特征工程方法，总共生成了 30 个用户。尽管销量预测的总体任务与市场上的学件所处理的任务相符，但没有一个学件是为 M5 销量预测任务专门设计的。

在下图中，我们展示了用户自行训练的模型和几种学件复用方法的损失曲线。显然，异构学件在用户标记数据有限的情况下表现出了对用户任务的有效性。

<div align=center>
  <img src="./docs/_static/img/Hetero_labeled_curves.svg"  width="500" height="auto" style="max-width: 100%;"/>
</div>


## 图像场景实验

其次，我们在图像数据集上评估了我们的算法。值得注意的是，不同尺寸的图像可以通过调整大小进行标准化，无需考虑异构特征情况。

### 实验设置

我们选择了经典的图像分类数据集 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)，其中包含 10 个类别的 60000 张 32x32 的彩色图像。总共上传了 50 个学件：每个学件包含一个卷积神经网络，该网络在一个不平衡的子集上进行训练，包括来自四个类别的 12000 个样本，采样比例为 `0.4:0.4:0.1:0.1`。
总共测试了 100 个用户任务，每个用户任务包含 3000 个 CIFAR-10 样本，分为六个类别，采样比例为 `0.3:0.3:0.1:0.1:0.1:0.1`。

### 实验结果

我们使用 `1 - Accuracy` 作为损失度量来评估各种方法的平均性能。下述实验结果显示，当用户面临标记数据的稀缺或仅拥有有限数量的标记数据（少于 2000 个实例）时，利用学件市场可以获得更好的性能。

<div align=center>

| Setting                           | Accuracy |
|-----------------------------------|----------|
| Mean in Market (Single)           | 0.655    |
| Best in Market (Single)           | 0.304    |
| Top-1 Reuse (Single)              | 0.406    |
| Job Selector Reuse (Multiple)     | 0.406    |
| Average Ensemble Reuse (Multiple) | 0.310    |

</div>

<div align=center>
  <img src="./docs/_static/img/image_labeled_curves.svg"  width="500" height="auto" style="max-width: 100%;"/>
</div>

## 文本场景实验

最后，我们在文本数据集上对我们的算法进行了评估。文本数据的特征天然异构，但这个问题可以通过使用句子嵌入提取器 (Sentence Embedding Extractor) 来解决。

### 实验设置

我们在经典的文本分类数据集上进行了实验：[20-newsgroup](http://qwone.com/~jason/20Newsgroups/)，该数据集包含大约 20000 份新闻文档，包含 20 个不同的新闻组。
与图像实验类似，我们一共上传了 50 个学件。每个学件都是在一个子集上进行训练，该子集仅包括三个超类中一半样本的数据，其中的模型为 `tf-idf` 特征提取器与朴素贝叶斯分类器的结合。我们定义了 10 个用户任务，每个任务包括两个超类。

### 实验结果

结果如下表和图所示。同样地，即使没有提供标记数据，通过学件的识别和复用所达到的性能可以与市场上最佳学件相匹敌。此外，利用学件市场相对于从头训练模型可以减少约 2000 个样本。

<div align=center>

| Setting                           | Accuracy |
|-----------------------------------|----------|
| Mean in Market (Single)           | 0.507    |
| Best in Market (Single)           | 0.859    |
| Top-1 Reuse (Single)              | 0.846    |
| Job Selector Reuse (Multiple)     | 0.845    |
| Average Ensemble Reuse (Multiple) | 0.862    |

</div>


<div align=center>
  <img src="./docs/_static/img/text_labeled_curves.svg"  width="500" height="auto" style="max-width: 100%;"/>
</div>


# 引用

如果你在研究或工作中使用了我们的项目，请引用下述论文，感谢你的支持！

```bibtex
@article{zhou2022learnware,
  author = {Zhou, Zhi-Hua and Tan, Zhi-Hao},
  title = {Learnware: Small Models Do Big},
  journal = {SCIENCE CHINA Information Sciences},
  year = {2024},
  volume = {67},
  number = {1},
  pages = {1--12},
}
```

# 关于

## 如何贡献

`learnware` 还很年轻，可能存在错误和问题。我们非常欢迎大家为 `learnware` 做出贡献。
我们为所有的开发者提供了详细的[项目开发指南](https://learnware.readthedocs.io/en/latest/about/dev.html)，并设置了相应的 commit 格式和 pre-commit 配置，请大家遵守。
非常感谢大家的贡献！

## 关于我们

`learnware` 由 LAMDA 北冥坞研发团队开发和维护，更多信息可参考：[团队简介](https://docs.bmwu.cloud/zh-CN/about-us.html)。