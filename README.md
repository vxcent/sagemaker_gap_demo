# sagemaker_gap_demo

## 0. Set up Sagemaker Studio Lab instance 

## 1. Clone the github repo
`git clone https://github.com/vxcent/sagemaker_gap_demo.git`

## 2. Setup Python Environment
```bash 
conda activate default

conda install python=3.7

conda install -c conda-forge jsonnet openjdk

conda install pytorch=1.5 cudatoolkit=10.2 -c pytorch
```
## 3. Install dependencies

`cd sagemaker_gap_demo/rat-sql-gap`

```bash
pip install -r requirements.txt

python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```


## 4. Install the Stanford NLP Core Library
```bash
mkdir third_party
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip -d third_party/
```

### 5. Download the checkpoint
```bash
mkdir -p logdir/bart_run_1/bs\=12\,lr\=1.0e-04\,bert_lr\=1.0e-05\,end_lr\=0e0\,att\=1/
mkdir ie_dirs
aws s3 cp s3://gap-text2sql-public/checkpoint-artifacts/gap-finetuned-checkpoint logdir/bart_run_1/bs\=12\,lr\=1.0e-04\,bert_lr\=1.0e-05\,end_lr\=0e0\,att\=1/model_checkpoint-00041000

mkdir -p pretrained_checkpoint
aws s3 cp s3://gap-text2sql-public/checkpoint-artifacts/pretrained-checkpoint pretrained_checkpoint/pytorch_model.bin
```