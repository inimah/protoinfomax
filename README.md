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
| [Citation](#citation)|

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

Amazon data (sentiment classification):[to-download](https://drive.google.com/file/d/1ETckW4TZQdNMqhFnuazgoamHqP_jM4FC/view?usp=sharing)<br />
AI conversational data (intent classification):[to-download](https://drive.google.com/file/d/1TLjN4xuU3D18ZGGWDo0R8rZwQlXMz_pi/view?usp=sharing)

Unzip the above compressed files into ~/data/

### Amazon data (Sentiment Classification)

```
AmazonDat
│
└───train
│   │   workspace_list
│   │   workspace_list_kw
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

If you want to use your own data, please follow the data structure exemplified in the above data.<br />
For preparing your own data with keyword auxiliaries, please run the following script on your data.<br />
```script/extract_keywords.sh```<br />
Note that in the above script, we utilize TfIdf keyword extractor. If you want to use your own keyword extraction method (e.g. topic model, deep keyword generator), please follow structure exemplified by Kws_xxx.train. <br />

## Training

## Evaluation

## Result

## Citation

Please cite our paper if you find this repo useful :)

```BibTeX
to-be-added
```

----

Issues and pull requests are welcomed.
