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
| [Results](#results)|
| [GPU Specs](#gpus)|
| [Citation](#citation)|

## HowTo
1. Clone or download the current git repo
2. Get into the main directory ```cd protoinfomax```
3. Create directory ~/data/ ```mkdir -p data```
4. Create directory ~/embeddings/ ```mkdir -p embeddings```
5. Install dependencies: [>>Setup](#setup)
6. Download preprocessed data sets: [>>Data](#data)
7. Before running bash scripts (\*.sh) from our repo, do not forget to: <br> ```Change the second line in the bash file into the path location where you installed anaconda or miniconda```
8. Run training script: [>>Training](#training)
9. Run evaluation script: [>>Evaluation](#evaluation)
10. Read log summary: [>>Result](#result)
11. Additional: See notebooks for investigating model reliability

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
Due to large size of the pretrained and finetuned word embeddings, we could not release our finetuned version.<br>
For smoothly training with our code implementation:
1. Download English vector 100-dim from fasttext.cc. See [https://fasttext.cc/docs/en/crawl-vectors.html](https://fasttext.cc/docs/en/crawl-vectors.html). Located it in ```~/embeddings/```.
2. Run ```prep_scripts/finetuning_fastext.sh```.
3. The finetuned model and vocabulary will be stored in your ```~/embeddings/```.

Please be aware that storing these files will require ~6GB disk space in total.<br>

You could also use any pretrained word embeddings (e.g. GloVe), but please be advised to change the code accordingly.

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
```prep_scripts/extract_keywords.sh```<br />
Note that in the above script, we utilize TfIdf keyword extractor. If you want to use your own keyword extraction method (e.g. topic model, deep keyword generator), please follow structure exemplified by Kws_xxx.train. <br />

## Training

By default, the following training and evaluation scripts are configured with:
```
K=100 # number of K-shot per training episode
hidden_size=200 # dimension for Bidirectional GRU
sampling_classes=2 # number of distinct supervision class in training data set (e.g. ``positive'' and ``negative'' sentiment labels)
```

For running the scripts under different configuration set-ups, change parameters in ```config/config_sentiment``` and ```src/all_parameters_sentiment.py``` accordingly.


| Model                                                |  Benchmark               |   Bash script                                                |
| ---------------------------------------------------- | ------------------------ | ------------------------------------------------------------ |
| ProtoNet                                             | Sentiment Classification |  train_scripts/train_protonet_sentiment.sh                           |
| ProtoNet                                             | Intent Classification    |  train_scripts/train_protonet_intent.sh                              |
| OProto                                               | Sentiment Classification |  train_scripts/train_oproto_sentiment.sh                             |
| OProto                                               | Intent Classification    |  train_scripts/train_oproto_intent.sh                                |
| ProtoInfoMax                                         | Sentiment Classification |  train_scripts/train_protoinfomax_sentiment.sh                       |
| ProtoInfoMax                                         | Intent Classification    |  train_scripts/train_protoinfomax_intent.sh                          |
| ProtoInfoMax++                                       | Sentiment Classification |  train_scripts/train_protoinfomax_kws_sentiment.sh                   |
| ProtoInfoMax++                                       | Intent Classification    |  train_scripts/train_protoinfomax_kws_intent.sh                      |


## Evaluation

| Model                                                |  Benchmark               |   Bash script                                                |
| ---------------------------------------------------- | ------------------------ | ------------------------------------------------------------ |
| ProtoNet                                             | Sentiment Classification |  eval_scripts/eval_protonet_sentiment.sh                           |
| ProtoNet                                             | Intent Classification    |  eval_scripts/eval_protonet_intent.sh                              |
| OProto                                               | Sentiment Classification |  eval_scripts/eval_oproto_sentiment.sh                             |
| OProto                                               | Intent Classification    |  eval_scripts/eval_oproto_intent.sh                                |
| ProtoInfoMax                                         | Sentiment Classification |  eval_scripts/eval_protoinfomax_sentiment.sh                       |
| ProtoInfoMax                                         | Intent Classification    |  eval_scripts/eval_protoinfomax_intent.sh                          |
| ProtoInfoMax++                                       | Sentiment Classification |  eval_scripts/eval_protoinfomax_kws_sentiment.sh                   |
| ProtoInfoMax++                                       | Intent Classification    |  eval_scripts/eval_protoinfomax_kws_intent.sh                      |

## Results

## GPUs

Our GPU specifications for running the experiments:
- 4 GPUs ASUS Turbo GeForce GTX 1080 Ti (11GB RAM, 3584 CUDA cores, compute capability 6.1). 2 CPUs Intel Xeon 4110 @ 2.1Ghz (32 hyperthreads, RAM: 384GB).
- 4 GPUs Nvidia Tesla V100 (16GB RAM, 2560 tensor cores, 10480 CUDA cores, compute capability 7.0). 1 CPU Intel Xeon E5-2698v4 @ 2.2GHz (40 hyperthreads, RAM: 256GB).

## Citation

Please cite our paper if you find this repo useful :)

```BibTeX
to-be-added
```

----

Issues and pull requests are welcomed.
