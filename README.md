# IR-Recommend_System-ZY2506320
这是BUAA信息检索课程2025-2026学年第一学期本人期末大作业。

所使用数据集为[Movielens-1M](https://grouplens.org/datasets/movielens/1m/)。

数据集需经预处理，请下载数据集后，运行[```data/ml-1m_data_process.py```](data/ml-1m_data_precess.py)提取预处理文件。预处理策略详见[```data/README```](data/README.md)。

本仓库已将预处理文件上传[```data\ml-1m2.txt```](data/ml-1m2.txt)。

-----------
训练命令
```
python main.py --dataset=ml-1m2 --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```
其中dataset为经预处理后的数据集文件，train_dir表示训练日志、模型参数等存放路径，maxlen表示输入序列最大长度。

-----------
测试命令
```
python main.py --device=cuda --dataset=ml-1m2 --train_dir=default --state_dict_path="ml-1m2_default\RecModel.epoch=1000.lr=0.001.layer=2.head=1.hidden=56.maxlen=200.pth" --inference_only=true --maxlen=200
```
