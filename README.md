# ProtoInfoMax
Code and Data sets for the EMNLP-2021-Findings Paper "ProtoInfoMax: Prototypical Networks with Mutual Information Maximization for Out-of-Domain Detection"

A Pytorch implementation

| Table of Contents |
|-|
| [How To](#howto)|
| [Setup](#setup)|
| [Data](#data)|
| [Training](#training)|
| [Evaluation](#evaluation)|
| [Result](#result)|
| [Citation](#citation)|

## HowTo
1. Clone or download the current git repo
2. Create directory ~/data/ ```mkdir -p data```
3. Create directory ~/embeddings/ ```mkdir -p embeddings```
4. Install dependencies: [>>Setup](#setup)
5. Download preprocessed data sets: [>>Data](#data)
6. Before running bash scripts (\*.sh) from our repo, do not forget to: <br> ```Change the second line in the bash file into the path location where you installed anaconda or miniconda```
7. Run training script: [>>Training](#training)
8. Run evaluation script: [>>Evaluation](#evaluation)
9. Read log summary: [>>Result](#result)
10. Additional: See notebooks for investigating model reliability

## Setup
### Dependencies

Install libraries and dependecies:
```bash
conda create -n protoinfomax_env python=3.6
conda activate protoinfomax_env
conda install cudatoolkit=10.1 -c pytorch -n protoinfomax_env 

pip install -r requirement.txt
```

## Data 

Our preprocessed data can be downloaded at

Amazon data (sentiment classification): [>>to-download](https://drive.google.com/file/d/1ETckW4TZQdNMqhFnuazgoamHqP_jM4FC/view?usp=sharing)<br />
AI conversational data (intent classification): [>>to-download](https://drive.google.com/file/d/1TLjN4xuU3D18ZGGWDo0R8rZwQlXMz_pi/view?usp=sharing)<br />

Unzip the above compressed files into ~/data/

#### Word embeddings
FastText Pretrained Binary (English): [>>to-download](#)<br />
Finetuned FastText on Sentiment: [>>to-download](#)<br />
Finetuned FastText on Intent: [>>to-download](#)<br />

Unzip the above compressed word embeddings into ~/embeddings/

#### NLTK library
Before running the script, make sure that you have downloaded nltk_data in your home directory. Our code is based on manual installation of nltk_data.
See https://www.nltk.org/data.html ```Manual installation```.
We use the following set up to call the required data or library. Change ```"~/nltk_data/"``` to the location where you downloaded nltk_data.
```
import nltk
nltk.data.path.append("~/nltk_data/")
```


### Amazon data (Sentiment Classification)

```
AmazonDat
│
└───train
│   │   workspace_list                  #list of training domains or categories, read by batch iterator
│   │   workspace_list_kw               #list for data with keyword auxiliaries: Kws_xxx.train
│   │   Apps_for_Android.train
│   │   Books.train
│   │   ...
│   │   Kws_Apps_for_Android.train
│   │   Kws_Books.train
│   │   ...
│   
└───dev
│   │   workspace_list
│   │   Automotive.train
│   │   Automotive.test
│   │   ...
└───test
    │   workspace_list
    │   Digital_Music.train
    │   Digital_Music.test
    │   ...
```

### AI Conversation data (Intent Classification)

```
IntentDat
│
└───train
│   │   workspace_list
│   │   workspace_list_kw
│   │   Assistant.train
│   │   Atis.train
│   │   ...
│   │   Kws_Assistant.train
│   │   Kws_Atis.train
│   │   ...
│   
└───dev
│   │   workspace_list
│   │   Alarm.train
│   │   Alarm.test
│   │   ...
└───test
    │   workspace_list
    │   Balance.train
    │   Balance.test
    │   ...
```

### Custom Data

If you want to use your own data, please follow the data structure exemplified in the above data.<br />
For preparing your own data with keyword auxiliaries, run the following script on your data.<br />
```prep/extract_keywords.sh```<br />
Note that in the above script, we utilize TfIdf keyword extractor. If you want to use your own keyword extraction method (e.g. topic model, deep keyword generator), please follow structure exemplified by Kws_xxx.train. <br />

## Training

By default, the following training and evaluation scripts are configured with:
```
K=100 # number of K-shot per training episode
hidden_size=200 # dimension for Bidirectional GRU
sampling_classes=2 # number of distinct supervision class in training data set (e.g. ``positive'' and ``negative'' sentiment labels)
```

For running the scripts under different configuration set-ups, change parameters in ```config/config_sentiment``` and ```src\all_parameters_sentiment.py``` accordingly.


| Model                                                |  Benchmark               |   Bash script                                                |
| ---------------------------------------------------- | ------------------------ | ------------------------------------------------------------ |
| ProtoNet                                             | Sentiment Classification |  train/train_protonet_sentiment.sh                           |
| ProtoNet                                             | Intent Classification    |  train/train_protonet_intent.sh                              |
| OProto                                               | Sentiment Classification |  train/train_oproto_sentiment.sh                             |
| OProto                                               | Intent Classification    |  train/train_oproto_intent.sh                                |
| ProtoInfoMax                                         | Sentiment Classification |  train/train_protoinfomax_sentiment.sh                       |
| ProtoInfoMax                                         | Intent Classification    |  train/train_protoinfomax_intent.sh                          |
| ProtoInfoMax++                                       | Sentiment Classification |  train/train_protoinfomax_kws_sentiment.sh                   |
| ProtoInfoMax++                                       | Intent Classification    |  train/train_protoinfomax_kws_intent.sh                      |


## Evaluation

| Model                                                |  Benchmark               |   Bash script                                                |
| ---------------------------------------------------- | ------------------------ | ------------------------------------------------------------ |
| ProtoNet                                             | Sentiment Classification |  eval/eval_protonet_sentiment.sh                           |
| ProtoNet                                             | Intent Classification    |  eval/eval_protonet_intent.sh                              |
| OProto                                               | Sentiment Classification |  eval/eval_oproto_sentiment.sh                             |
| OProto                                               | Intent Classification    |  eval/eval_oproto_intent.sh                                |
| ProtoInfoMax                                         | Sentiment Classification |  eval/eval_protoinfomax_sentiment.sh                       |
| ProtoInfoMax                                         | Intent Classification    |  eval/eval_protoinfomax_intent.sh                          |
| ProtoInfoMax++                                       | Sentiment Classification |  eval/eval_protoinfomax_kws_sentiment.sh                   |
| ProtoInfoMax++                                       | Intent Classification    |  eval/eval_protoinfomax_kws_intent.sh                      |

## Result

## Citation

Please cite our paper if you find this repo useful :)

```BibTeX
to-be-added
```

----

Issues and pull requests are welcomed.
