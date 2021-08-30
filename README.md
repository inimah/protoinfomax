# ProtoInfoMax
Code and Data sets for the EMNLP-2021-Findings Paper "ProtoInfoMax: Prototypical Networks with Mutual Information Maximization for Out-of-Domain Detection"

pytorch implementation

| Table of Contents |
|-|
| [Setup](#setup)|
| [Data Preparation](#prepare)|
| [Training](#training)|
| [Evaluation](#evaluation)|
| [Result](#result)|

## Setup
### Dependencies

Install other dependecies:
```bash
conda create -n protoinfomax_env python=3.6
conda activate protoinfomax_env
conda install cudatoolkit=10.1 -c pytorch -n protoinfomax_env 

pip install -r requirement.txt
mkdir data
```

## Data Preparation

Our preprocessed data can be downloaded at

Amazon data (sentiment classification):

AI conversational data (intent classification):

Unzip the above compressed files into ~/data/

### Amazon data (Sentiment Classification)

```
Amazon
│
└───train
│   │   workspace_list
│   │   workspace_list_kw
│   │   ...
│   
└───dev
│   │   workspace_list
│   │   
│   │   ...
└───test
    │   workspace_list
    │   
```

### AI Conversation data (Intent Classification)

## Training

## Evaluation

## Result
