import pandas as pd
import numpy as np
import os

def preprocess_data1(input_file, output_file):
    print(f"正在处理 {input_file} ...")
    if not os.path.exists(input_file):
        print(f"未找到文件 {input_file}，请检查路径。")
        return
        
    # data1_raw 为单层表头，直接读取
    df = pd.read_excel(input_file)
    
    # 1. 去除完全重复的行
    df = df.drop_duplicates()
    
    # 2. 缺失值处理：物理规律数据，使用线性插值
    df = df.interpolate(method='linear', limit_direction='both')
    
    # 3. 简单的异常值修正：电流、间隙不可能为负数，若有则修正为 0 或上一时刻的正常值
    for col in df.columns:
        if '电流' in col or '间隙' in col:
            df.loc[df[col] < 0, col] = np.nan
    df = df.ffill().bfill() # 填补刚才被设为 nan 的负数异常值

    # 导出文件
    df.to_excel(output_file, index=False)
    print(f"已成功导出至 {output_file}\n")

def preprocess_timeseries_data(input_file, output_file):
    print(f"正在处理 {input_file} ...")
    if not os.path.exists(input_file):
        print(f"未找到文件 {input_file}，请检查路径。")
        return
        
    # data2, data3, data4 的第一行(索引0)是分类说明，第二行(索引1)才是具体的列名
    # 使用 header=1 读取，直接以具体的变量名作为 DataFrame 的列
    df = pd.read_excel(input_file, header=1)
    
    # 识别时间列（包含"时间"字样的列）
    time_cols = [col for col in df.columns if '时间' in str(col)]
    if len(time_cols) > 0:
        time_col = time_cols[0]
        # 1. 去除时间戳重复的行，保留第一次出现的数据
        df = df.drop_duplicates(subset=[time_col], keep='first')
        # 2. 确保数据严格按时间递增排序
        df = df.sort_values(by=time_col).reset_index(drop=True)
    else:
        df = df.drop_duplicates()

    # 3. 缺失值处理：时序传感器数据最佳方案是时间线性的插值
    df = df.interpolate(method='linear', limit_direction='both')
    
    # 4. 异常处理：因为涉及“故障检测”题目，这里不做强烈的去极值操作（如 3-sigma），
    # 以免把故障突变当成异常值删掉。仅对明显非法的负数(如电磁力、电流)做前向填充保护。
    for col in df.columns:
        if '电磁力' in col or '电流' in col or '间隙' in col:
            # 如果出现负数（假设物理上电流和电磁力为正向），将其设为 nan 然后用前一个正常值填充
            df.loc[df[col] < 0, col] = np.nan
            
    df = df.ffill().bfill()

    # 导出文件
    df.to_excel(output_file, index=False)
    print(f"已成功导出至 {output_file}\n")

if __name__ == "__main__":
    # 1. 处理静态测算数据 data1
    preprocess_data1("data1_raw.xlsx", "data1.xlsx")
    
    # 2. 处理时序多电磁铁传感器数据 data2, data3, data4
    preprocess_timeseries_data("data2_raw.xlsx", "data2.xlsx")
    preprocess_timeseries_data("data3_raw.xlsx", "data3.xlsx")
    preprocess_timeseries_data("data4_raw.xlsx", "data4.xlsx")
    
    print("所有数据预处理完成！")