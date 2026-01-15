import pandas as pd
import numpy as np
from collections import defaultdict

def preprocess_movielens_1m(input_file, output_file):
    """
    预处理MovieLens-1M数据集
    https://grouplens.org/datasets/movielens/1m/
    参数:
        input_file: 原始数据文件路径 (ratings.dat)
        output_file: 输出文件路径
    """
    
    # 读取原始数据
    print("正在读取原始数据...")
    
    # MovieLens-1M的ratings.dat格式: UserID::MovieID::Rating::Timestamp
    # 使用::作为分隔符
    df = pd.read_csv(input_file, sep='::', engine='python', 
                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    print(f"原始数据形状: {df.shape}")
    print(f"原始用户数: {df['user_id'].nunique()}")
    print(f"原始电影数: {df['movie_id'].nunique()}")
    
    # 按时间戳排序，确保时间顺序
    print("正在按时间戳排序...")
    df = df.sort_values(by=['user_id', 'timestamp'])
    
    # 创建映射：将原始ID映射为从1开始的连续ID
    print("正在创建ID映射...")
    
    # 用户ID映射
    unique_users = df['user_id'].unique()
    user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_users, start=1)}
    
    # 电影ID映射
    unique_movies = df['movie_id'].unique()
    movie_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_movies, start=1)}
    
    # 应用映射
    df['mapped_user_id'] = df['user_id'].map(user_mapping)
    df['mapped_movie_id'] = df['movie_id'].map(movie_mapping)
    
    print(f"映射后用户数: {len(user_mapping)}")
    print(f"映射后电影数: {len(movie_mapping)}")
    
    # 按用户分组，获取每个用户的交互序列
    print("正在构建用户交互序列...")
    
    # 按映射后的用户ID和时间戳排序
    df_sorted = df.sort_values(by=['mapped_user_id', 'timestamp'])
    
    # 写入文件：每行格式为 "用户ID 电影ID"
    print("正在写入输出文件...")
    
    with open(output_file, 'w') as f:
        for _, row in df_sorted.iterrows():
            line = f"{int(row['mapped_user_id'])} {int(row['mapped_movie_id'])}\n"
            f.write(line)
    
    # 可选：保存映射关系，便于后续分析
    print("正在保存映射关系文件...")
    
    # 保存用户ID映射
    user_mapping_df = pd.DataFrame(list(user_mapping.items()), columns=['original_id', 'mapped_id'])
    user_mapping_df.to_csv('user_mapping.csv', index=False)
    
    # 保存电影ID映射
    movie_mapping_df = pd.DataFrame(list(movie_mapping.items()), columns=['original_id', 'mapped_id'])
    movie_mapping_df.to_csv('movie_mapping.csv', index=False)
    
    # 统计信息
    print("\n=== 数据预处理完成 ===")
    print(f"输出文件: {output_file}")
    print(f"总交互数: {len(df_sorted)}")
    print(f"映射后用户数: {df_sorted['mapped_user_id'].nunique()}")
    print(f"映射后电影数: {df_sorted['mapped_movie_id'].nunique()}")
    
    # 检查是否符合要求
    print("\n=== 数据格式验证 ===")
    
    # 读取前几行验证格式
    with open(output_file, 'r') as f:
        lines = [next(f) for _ in range(5)]
    
    print("前5行数据示例:")
    for i, line in enumerate(lines, 1):
        print(f"  第{i}行: {line.strip()}")
    
    # 验证ID是否从1开始
    df_check = pd.read_csv(output_file, sep=' ', names=['user_id', 'movie_id'], nrows=100)
    print(f"\n用户ID范围: {df_check['user_id'].min()} 到 {df_check['user_id'].max()}")
    print(f"电影ID范围: {df_check['movie_id'].min()} 到 {df_check['movie_id'].max()}")
    
    return df_sorted, user_mapping, movie_mapping


# 使用示例
if __name__ == "__main__":
    # 原始数据文件路径
    input_file = "ml-1m/ratings.dat"  # MovieLens-1M的ratings.dat文件
    
    # 输出文件路径
    output_file = "ml-1m.txt"  # 符合模型要求的格式
    
    try:
        # 运行预处理
        df_processed, user_mapping, movie_mapping = preprocess_movielens_1m(input_file, output_file)
        
        # 额外的统计信息
        print("\n=== 详细统计信息 ===")
        
        # 每个用户的平均交互数
        user_interactions = df_processed.groupby('mapped_user_id').size()
        print(f"每个用户平均交互数: {user_interactions.mean():.2f}")
        print(f"每个用户最少交互数: {user_interactions.min()}")
        print(f"每个用户最多交互数: {user_interactions.max()}")
        
        # 交互的密度
        num_users = len(user_mapping)
        num_movies = len(movie_mapping)
        total_interactions = len(df_processed)
        density = total_interactions / (num_users * num_movies)
        print(f"交互矩阵密度: {density:.6f}")
        
        # 检查时间顺序
        print("\n=== 时间顺序验证 ===")
        sample_user = df_processed['mapped_user_id'].iloc[0]
        user_data = df_processed[df_processed['mapped_user_id'] == sample_user]
        print(f"用户 {sample_user} 的前3次交互:")
        for i, (_, row) in enumerate(user_data.head(3).iterrows(), 1):
            print(f"  第{i}次: 电影 {row['mapped_movie_id']}, 原始时间戳: {row['timestamp']}")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        print("请确保ratings.dat文件在当前目录中")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")