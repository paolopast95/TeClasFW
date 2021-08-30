# TeClasFW

TeClasFW (which stands for Text Classification Framework) is a framework that allows training different machine learning models at once, using a configuration file in which all the details about the algorithms should be defined. 

This tool can be seen as a wrapper for the most popular libraries for NLP and allows the programmer not to worry about the implementation details.

TeClasFW allows also the definition of different combinations of parameters for each algorithm, storing the results of each configuration on a file.

In other words, the user can run many experiments using a simple configuration file.

There have been implemented different pre-processing operations and the most popular algorithms for text classification (Naive Bayes, SVM, LSTM, GRU, CNN, etc.)

## Installation

For the installation, all the necessary packages are listed in the requirements.txt.
Important notes: Python 3.8, CUDA 11.0 and cuDNN v8.0.4 (for CUDA 11.0) are needed for the correct execution of the tool.

```bash
pip install -r requirements.txt
```

## Usage
If you are interested in executing the classic algorithms like SVM, DecisionTrees, or Naive Bayes, the script *classical_algorithms_pipeline.py* has to be executed.

If you are interested in Deep Learning algorithms, you need to run the script *nn_algorithms_pipeline.py* 


## Configuration Files
To customize the training process and define the algorithms that should be executed (and all the parameters to test), the user should modify the two configuration files in the *config_files* directory.

For instance, the configuration file for the standard algorithms (SVM or DecisionTree) is shown below
```yaml
experiment:
  dataset_filename: "news_category.csv"
  output_folder_name: "news_category"
  preprocessing:
    tokenization: "wordpunct"
    stemming: "wordnet"
    stopword_removal: true
    embedding_strategy: tfidf
    embedding_file: "word2vec-google-news-300"
  models:
    - model_name: svm
      params:
        kernels: [rbf, linear, poly]
        gammas: [auto]
        Cs: [1, 10, 100, 500]
        degrees: [2]
    - model_name: naive_bayes
      params: null
    - model_name: decision_tree
      params:
        criteria: [gini, entropy]
        splitters: [random, best]
        max_depths: [10,100,1000, 10000]
        min_samples_splits: [2,3,4]
        min_samples_leaves: [10, 100, 200]
  evaluation:
    validation_set:
      test_size: 0.2
    metrics: [accuracy, precision, recall, f1score]
```

In the preprocessing section you can define all the operation that should be executed. The following list shows the possible value for that section

- tokenizazion: whitespace, word, wordpunct
- stemming: wordnet, porter, lancaster, snowball
- stopword_removal: true, false
- embedding_strategy: count, tfidf, word2vec (this field is needed in order to define how to convert the raw sentences into their numerical representation)
- embedding_file: name_of_pretrained_embedding_file (you can use your own embedding file or put the name of one of the pretrained file inside the gensim library)

For the models that need to be tested, you can define all the hyper parameters defined in the sklearn library:

- kernels, gamma values, C values for the SVM
- criterion, splitters, max_depths etc. for the DecisionTree

The same approach is used for the Deep Learning models, where you can define the number of recurrent (or convolutional) layers, the number of dense layers and other parameters in order to test all the models and store the results for each combination.
