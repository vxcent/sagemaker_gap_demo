# sagemaker_gap_demo

## 0. Set up Sagemaker Studio Lab instance 

## 1. Clone the github repo
```bash
git clone https://github.com/vxcent/sagemaker_gap_demo.git
```

## 2. Setup Python Environment
```bash 
conda activate default
conda install -y python=3.7
conda install -y -c conda-forge jsonnet openjdk
conda install -y pytorch=1.5 cudatoolkit=10.2 -c pytorch
```
## 3. Install dependencies
```bash
cd sagemaker_gap_demo
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```


## 4. Install the Stanford NLP Core Library
```bash
cd rat-sql-gap
mkdir third_party
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip -d third_party/
```

### 4.1 Start the Stanford Core NLP LIbrary
```bash
pushd third_party/stanford-corenlp-full-2018-10-05
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8999 -timeout 15000 > server.log &
popd
```

## 5. Download Large File Assets

### 5.1 Download finetune checkpoint
```bash
mkdir -p logdir/bart_run_1/bs\=12\,lr\=1.0e-04\,bert_lr\=1.0e-05\,end_lr\=0e0\,att\=1/
aws s3 cp s3://gap-text2sql-public/checkpoint-artifacts/gap-finetuned-checkpoint logdir/bart_run_1/bs\=12\,lr\=1.0e-04\,bert_lr\=1.0e-05\,end_lr\=0e0\,att\=1/model_checkpoint-00041000
```
### 5.2 Download pretrained model 
```bash
mkdir -p pretrained_checkpoint
aws s3 cp s3://gap-text2sql-public/checkpoint-artifacts/pretrained-checkpoint pretrained_checkpoint/pytorch_model.bin
```

## 6. Enjoy the notebook! 
### (rat-sql-gap/notebook.ipynb)