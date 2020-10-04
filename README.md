# arcface-pytorch
pytorch implement of arcface 


## データセットの準備
```
├── data
│   ├── Datasets
│   │   └── invoices
│   │       ├── test
│   │       │   ├── IMI
│   │       │   ├── adachi
│   │       │   └── yuyama
│   │       ├── train
│   │       │   ├── IMI
│   │       │   ├── adachi
│   │       │   └── yuyama

```
上図のようなディレクトリ構造で画像を用意する。各クラスに画像を格納する
その後、
```bash
cd data
python make_file_names.py
```
を実行し、データセットリストを作成する

# 実行環境（学習）


gpu環境で
```bash
docker run --gpus all -t -v /home/r_takenaka/arcface-pytorch/:/workspace/arcface-pytorch --name arcface -d -p 8097:8097 pytorch bash
docker exec -it arcface bash
```
docker 内で
```bash
# 学習状況のプロット用（必要なければconfigでdisply = False でoff)
python -m visdom.server
python train.py
```

##  evaluate
分類精度の確認
```bash
python estimate.py
```
t-SNE,Umapでのmetrics learningの可視化
```bash
python show_t_sne.py
```

## using openmax to predict

1 secondly train to make full connected layer
```bash
python train.py --train_second
```
2 prepare features of evaluation images
```bash
python imageNet_Features.py
python MAV_Compute.py
python compute_distances.py
```
3 predict criteria by openmax
```bash
python estimate_openmax.py
```


### evaluate用には必要なライブラリは別途用意必要



# References
https://github.com/deepinsight/insightface

https://github.com/auroua/InsightFace_TF

https://github.com/MuggleWang/CosFace_pytorch

# pretrained model and lfw test dataset
the pretrained model and the lfw test dataset can be download here. link: https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA pwd: b2ec
the pretrained model use resnet-18 without se. Please modify the path of the lfw dataset in config.py before you run test.py.
