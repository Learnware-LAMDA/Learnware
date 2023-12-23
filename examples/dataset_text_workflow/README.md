# Text Dataset Workflow Example

## Introduction

We conducted experiments on the widely used text benchmark dataset: `20-newsgroup <http://qwone.com/~jason/20Newsgroups/>`_.
20-newsgroup is a renowned text classification benchmark with a hierarchical structure, featuring 5 superclasses {comp, rec, sci, talk, misc}.

In the submitting stage, we enumerated all combinations of three superclasses from the five available, randomly sampling 50% of each combination from the training set to create datasets for 50 uploaders.

In the deploying stage, we considered all combinations of two superclasses out of the five, selecting all data for each combination from the testing set as a test dataset for one user. This resulted in 10 users.
The user's own training data was generated using the same sampling procedure as the user test data, despite originating from the training dataset.

Model training comprised two parts: the first part involved training a tfidf feature extractor, and the second part used the extracted text feature vectors to train a naive Bayes classifier.

Our experiments comprises two components:

* ``unlabeled_text_example`` is designed to evaluate performance when users possess only testing data, searching and reusing learnware available in the market.

* ``labeled_text_example`` aims to assess performance when users have both testing and limited training data, searching and reusing learnware directly from the market instead of training a model from scratch. This helps determine the amount of training data saved for the user.


## Run the code

Run the following command to start the ``unlabeled_text_example`.

```bash
python workflow.py unlabeled_text_example
```

Run the following command to start the ``labeled_text_example`.

```bash
python workflow.py labeled_text_example
```

## Results

### ``unlabeled_text_example``:

The accuracy of search and reuse is presented in the table below:

| Top-1 Performance   | Job Selector Reuse  | Average Ensemble Reuse |
|---------------------|----------------------|-------------------------|
| 0.859 +/- 0.051     | 0.844 +/- 0.053      | 0.858 +/- 0.051         |


### ``labeled_text_example``:

We present the change curves in classification error rates for both the user's self-trained model and the multiple learnware reuse(EnsemblePrune), showcasing their performance on the user's test data as the user's training data increases. The average results across 10 users are depicted below:

<div style="text-align:center;">
  <img src="../../docs/_static/img/text_example_labeled_curves.png" alt="Text Limited Labeled Data" style="width:50%;" />
</div>

From the figure above, it is evident that when the user's own training data is limited, the performance of multiple learnware reuse surpasses that of the user's own model. As the user's training data grows, it is expected that the user's model will eventually outperform the learnware reuse. This underscores the value of reusing learnware to significantly conserve training data and achieve superior performance when user training data is limited.
