[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Data augmentation for text classification

This is an open source implementation of my Master's Thesis: "Data augmentation for text classification" in collaboration with Kustomer, Inc. 

The main goal of this research is to demonstrate whether the use of the data augmentation techniques help to build text classifiers with better performance.

This README is a superficial overview of the work. All the aspects are deeper explained in the published [Master's Thesis report](https://riunet.upv.es/handle/10251/151669).

## Data augmentation techniques (DAT)
The three data augmentation techniques (DAT) used in this project are the nexts:

- Easy Data Augmentation (EDA):

  EDA applies very simple modifications on the original example, like word synonym substitution, removing words, inserting words and swapping words positions.

- Back-translation (BT):

  BT translates the original example in language A with the automatic machine translation system A→B to the pivot language B. The translated example obtained is translated another time, but using the automatic machine translation system B→A, in this way you obtain a reconstruction of the original example Sometimes the reconstructed example is the same as the original one, but in the majority of the cases is not equal, but it keeps practically the same semantic meaning. 

- Word substitution using pre-trained models (SUS):

  SUS masks some tokens of the original example, then uses a pre-trained language model like BERT to unmask them. BERT to unmask the tokens takes into account the right and left context of the visible tokens, therefore this approach applies a contextual augmentation to the original example. This technique generates synthetic examples with better quality than the two previous techniques, as a consequence, this approach is more expensive.

## Machine learning models
The two models used for the experimentation are not comparable (see item list below), because they belong to different nature. Nevertheless, it can be interesting to analyze how each model behaves.

- XGBoost with word embeddings features

- Pre-trained RoBERTa for sequence classification

## Dependencies

- Python 3.7
- [PyTorch (recommend GPU version)](https://github.com/pytorch/pytorch#from-source)
- [fastai](https://github.com/fastai/fastai)
- [langid.py](https://github.com/saffsd/langid.py)
- [fastText](https://github.com/facebookresearch/fastText#building-fasttext-for-python)
- Additionally, we need the following packages (using pip):

```
pip install pandas
pip install scikit-learn
pip install transformers
pip install nltk
pip install wandb
pip install hyperopt
pip install xgboost
```

## Acquire data
The comands below download all the corpora. 
```
cd corpus
./dataset-sst-2.sh
./dataset-snips.sh
./dataset-trec.sh
./dataset-replydotai.sh
./dataset-question-topic.sh
```
The comand below prepares the differents corpus for the experimentation.
```
cd data
./generate-datasets-for-experimenting.sh
```


## Experiments

In the experimentation we explore the effectiveness of data augmentation in different training scenarios:
- Using all the training examples available for each dataset.
- Simulating low-resource training environments, limiting the number of training examples per class (NEPC).


### Baselines

The first experiments are the baselines that do not apply any data augmentation in the training set, therefore the performance obtained in these experiments, we will try to improve it in the nexts experiments where we will apply data augmentation techniques.
 
 
The commands below  launch the baselines of XGBoost and RoBERTa respectively. The argument `$VALID-SIZE` is the dimension of the validation: `full` or `red`.

```
cd scripts
./experiments-baselines-xgboost.sh $VALID-SIZE
./experiments-baselines-roberta.sh $VALID-SIZE
```

### Full scenario 

Use the commands below for generating the augmented training datasets for each corpus using the three different methods proposed: EDA, BT and SUS; on the full training section. After that, it will be launched the experiments with XGBoost and RoBERTa models.

```
cd scripts
./da-full.sh
./experiments-da-xgboost-full.sh 
./experiments-da-roberta-full.sh 
```

The table below shows the results obtained, where we can see that these techniques do not work in the classifiers based in RoBERTA, and in the classifiers based on XGBoost the improvement obtained is very small.

![](https://github.com/franborjavalero/data-augmentation-for-text-classification/blob/master/results/jpg/da_full.jpg?raw=true)

### Low-resource scenarios

We proposed three low-resources scenarios, where the number of training examples per class (NEPC) are 20, 50 and a maximum of 80.
 
Use the commands below for generating the augmented training datasets for each corpus using the three different methods proposed on the low-resource training scenario. After that, it will be launched the experiments with XGBoost and RoBERTa models.

```
./da-eda-reduced.sh
./da-bt-reduced.sh
./da-sus-reduced.sh
./experiments-da-xgboost-reduced.sh 
./experiments-da-roberta-reduced.sh 
```

The table below shows the results obtained in the low-resource scenario where NEPC is 80. In these experiments BT seems to work in RoBERTa, but the improvement is insignificant. For XGBoost all the data augmentation techniques work, but the improvement is very small.

![](https://github.com/franborjavalero/data-augmentation-for-text-classification/blob/master/results/jpg/da_nepc-80.jpg?raw=true)

The table below shows the results obtained in the low-resource scenario where NEPC is 50. The results obtained are very similar to previous ones, where NEPC is 80.

![](https://github.com/franborjavalero/data-augmentation-for-text-classification/blob/master/results/jpg/da_nepc-50.jpg?raw=true)


The table below shows the results obtained in the low-resource scenario where NEPC is 20. The results obtained are the same as in the two previous scenarios, but in this case the improvement obtained applying the data augmentation technique are bit higher, but it continues being small.

![](https://github.com/franborjavalero/data-augmentation-for-text-classification/blob/master/results/jpg/da_nepc-20.jpg?raw=true)

### Statistical significance
The tests of statistical significance are used for ensuring that the performance of the two models is statically different. In this way we can know if the data augmentation techniques are useful or not.
 
The table below shows the *p value* obtained comparing the results obtained by the model that uses data augmentation techniques with respect to its correspondiente that does not use. In the case of the models based on XGBoost all the models in bold improve the performance, where SUS is the best technique. However, in the case of RoBERTa there are not any model that improves the baselines, only there one that worsen.



![](https://github.com/franborjavalero/data-augmentation-for-text-classification/blob/master/results/jpg/statistical_significance.jpg?raw=true)

## Conclusion

The main lesson learned after analyzing the results obtained on the different experiments is that the pre-trained version of RoBERTa for sequence classification is robust enough for text classification, it does not need any data augmentation technique for improving its performance, because the benefit obtained is not statically significant.

## Reference 

[Master's Thesis report](https://riunet.upv.es/handle/10251/151669)

## License
 
The MIT License (MIT)

Copyright (c) 2020 Francisco de Borja Valero
