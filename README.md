# ProtoInfoMax
Code and Data sets for the EMNLP-2021-Findings Paper "ProtoInfoMax: Prototypical Networks with Mutual Information Maximization for Out-of-Domain Detection"

pytorch implementation

| Table of Contents |
|-|
| [Setup](#setup)|
| [Data Preparation](#data)|
| [Training](#training)|
| [Evaluation](#evaluation)|
| [Result](#result)|

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

Amazon data (sentiment classification):
AI conversational data (intent classification):

Unzip the above compressed files into ~/data/

### Amazon data (Sentiment Classification)

```
AmazonDat
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

If you want to use your own data, please follow the data structure exemplified in the above data.\n
For preparing your own data with keyword auxiliaries, please run the following script on your data.\n
```script/extract_keywords.sh```\n
Note that in the above script we utilize, TfIdf keyword extractor. \n
If you want to use your own keyword extraction method (e.g. topic model, deep keyword generator), follow structure exemplified by Kws_xxx.train.\n

## Training

## Evaluation

## Result
