请先下载[movielens-1m](https://grouplens.org/datasets/movielens/1m/)数据集，在当前目录（data/）下存为```ml-1m/```文件夹。
```
data
├─── ml-1m
|    ├─ movies.dat
|    ├─ ratings.dat
|    └─ users.dat
└─── ml-1m_data_process.py
```
实际上本项目只需使用到```ml-1m/ratings.dat```数据。

然后进入当前路径```cd data```，运行```python ml-1m_data_process.py```，生成输入数据文件```ml-1m.txt```。

本项目预处理逻辑为：由于检索阶段只需快速地筛选出相关物品，即只需要用户的历史交互序列，故首先对每个用户的交互记录按时间戳升序排列，确保序列的时间顺序性，然后将原始用户ID和电影ID重新映射为从1开始的连续整数，最后将交互序列按时间顺序保存为每行"用户ID 电影ID"的格式。
