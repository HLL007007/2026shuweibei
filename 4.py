import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNet, ARDRegression
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. 绘图与中文字体设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 物理参数与常数
# ==========================================
mc = 33000.0                       
mf = 3000.0 + 16 * 500.0           
g = 9.8                            
K = 0.079987                       

# ==========================================
# 2. 读取监测数据 (data4.xlsx)
# ==========================================
print("正在读取 data4.xlsx 监测数据...")
df = pd.read_excel('data4.xlsx')

t_data = df.iloc[:, 0].values          
h_data = df.iloc[:, 1].values          
ddz_c_data = df.iloc[:, 2].values      
I_matrix = df.iloc[:, 3:19].values     

dt = t_data[1] - t_data[0]
N = len(t_data)

# ==========================================
# 3. 动力学数据预处理
# ==========================================
print("正在提取系统动力学特征...")
ddh_data = savgol_filter(h_data, window_length=201, polyorder=3, deriv=2, delta=dt)

# 客观需求总力 Y
F_req = mc * ddz_c_data - mf * ddh_data + (mc + mf) * g

# 16个电磁铁理想受力矩阵 X
I_squared_sgn = np.sign(I_matrix) * (I_matrix ** 2)
F_ideal_matrix = K * I_squared_sgn / (h_data[:, None] ** 2)

# 总理想受力 (正常状态下的理论合力)
F_ideal_total = np.sum(F_ideal_matrix, axis=1)

# ==========================================
# 4. 【新增】系统受力线性相关性检验
# ==========================================
print("\n" + "="*45)
print("★ 全局受力相关性检验 (理想合力 vs 需求合力) ★")
pearson_corr, p_p = pearsonr(F_ideal_total, F_req)
spearman_corr, p_s = spearmanr(F_ideal_total, F_req)
print(f"-> Pearson (皮尔逊) 线性相关系数 : {pearson_corr:.4f}")
print(f"-> Spearman(斯皮尔曼) 秩相关系数 : {spearman_corr:.4f}")
if pearson_corr > 0.8:
    print("结论: 理想受力与实际需求力呈现强正相关，具备极佳的线性回归基础！")
print("="*45 + "\n")

# ==========================================
# 5. 【升级】二层稀疏学习器参数辨识 (滑动窗口)
# ==========================================
print("正在执行 二层稀疏学习算法 (ElasticNet + SBL) 辨识故障点...")
print("物理假设: 极短时间内至多存在 1-2 个电磁铁发生故障 (即故障具备极强稀疏性)")

window_size = 500 
step_size = 50

time_centers = []
eta_results = []  

# 【关键修改 1】降低 alpha 值 (从0.1降到0.002)，允许真实的轻微故障浮出水面
# 增加 max_iter 保证收敛
layer1_model = ElasticNet(alpha=0.002, l1_ratio=0.9, fit_intercept=False, max_iter=5000)

# 第二层学习器：SBL (稀疏贝叶斯学习)
layer2_model = ARDRegression(fit_intercept=False, threshold_lambda=1e4)

for start_idx in range(0, N - window_size, step_size):
    end_idx = start_idx + window_size
    
    X_window = F_ideal_matrix[start_idx:end_idx, :]
    Y_req_window = F_req[start_idx:end_idx]
    Y_ideal_total_window = F_ideal_total[start_idx:end_idx]
    
    scale_factor = 1e5
    X_scaled = X_window / scale_factor
    Y_res_scaled = (Y_req_window - Y_ideal_total_window) / scale_factor
    
    # ----------------------------------------------------
    # 第一层：ElasticNet 粗筛 
    # ----------------------------------------------------
    layer1_model.fit(X_scaled, Y_res_scaled)
    delta_eta_l1 = layer1_model.coef_
    
    # 【关键修改 2】降低阈值 (从0.05降到0.01)，只要偏差超过1%就送入第二层审查
    active_indices = np.where(np.abs(delta_eta_l1) > 0.01)[0]
    
    delta_eta_final = np.zeros(16)
    
    # ----------------------------------------------------
    # 第二层：SBL 稀疏贝叶斯精细估计
    # ----------------------------------------------------
    if len(active_indices) > 0:
        # 如果选出的嫌疑特征太多（比如超过5个），说明可能是整体性震荡而不是稀疏故障
        # 强制只取 L1 绝对值最大的前 3 个（强化 1-2 点故障假设）
        if len(active_indices) > 3:
            active_indices = active_indices[np.argsort(np.abs(delta_eta_l1[active_indices]))[-3:]]
            
        X_active = X_scaled[:, active_indices]
        layer2_model.fit(X_active, Y_res_scaled)
        
        delta_eta_final[active_indices] = layer2_model.coef_
    
    eta_window = 1.0 + delta_eta_final
    
    eta_results.append(eta_window)
    time_centers.append(t_data[start_idx + window_size // 2])

time_centers = np.array(time_centers)
eta_results = np.array(eta_results)

# 稍微减弱平滑窗口（从10降为5），防止故障尖峰被磨平
eta_smoothed = pd.DataFrame(eta_results).rolling(window=5, min_periods=1, center=True).mean().values

# ==========================================
# 6. 生成 4x4 矩阵时序图表
# ==========================================
print("分析完成！生成二层稀疏学习诊断图谱...")

fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True, sharey=True)
fig.suptitle(r'基于弹性网络与SBL级联的功率放大系数 $\eta_i(t)$ 稀疏诊断 (1-2点故障假设)', fontsize=18, fontweight='bold')

for i in range(16):
    row, col = divmod(i, 4)
    ax = axes[row, col]
    
    eta_line = eta_smoothed[:, i] 
    
    ax.axhspan(0.8, 1.2, color='#2ca02c', alpha=0.15)
    ax.axhline(0.8, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.axhline(1.2, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.8)
    
    # 使用较深的颜色绘制信号
    ax.plot(time_centers, eta_line, color='#1f77b4', linewidth=1.8)
    ax.set_title(f'电磁铁 {i+1} 号', fontsize=12)
    ax.set_ylim(0.4, 1.6)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 判断该电磁铁在整个时段内是否有显著偏差
    if np.max(eta_line) > 1.2:
        ax.set_facecolor('#fff2cc')
        ax.text(0.05, 0.85, "检测到: 激增故障", transform=ax.transAxes, color='darkgoldenrod', fontweight='bold')
    elif np.min(eta_line) < 0.8:
        ax.set_facecolor('#ffe6e6')
        ax.text(0.05, 0.85, "检测到: 衰减故障", transform=ax.transAxes, color='darkred', fontweight='bold')
    else:
        ax.text(0.05, 0.85, "状态: 稳健正常", transform=ax.transAxes, color='green')

for ax in axes[-1, :]: ax.set_xlabel('时间 t (s)', fontsize=12)
for ax in axes[:, 0]: ax.set_ylabel(r'放大系数 $\eta$', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()



# ==========================================
# 7. 【进阶可视化】典型故障电磁铁叠加对比图
# ==========================================
print("正在提取并绘制典型故障叠加对比图...")

decay_faults = []  # 衰减故障集合
surge_faults = []  # 激增故障集合

# 自动科学分类：计算每个电磁铁偏离 1.0 的最大向下和向上幅度
for i in range(16):
    eta_line = eta_smoothed[:, i]
    max_drop = 1.0 - np.min(eta_line)  # 最大衰减幅度
    max_spike = np.max(eta_line) - 1.0 # 最大激增幅度
    
    # 只有偏离超过 20% (即达到0.8或1.2界限) 才被认定为真正故障
    if max_drop > 0.2 or max_spike > 0.2:
        if max_drop > max_spike:
            decay_faults.append(i)
        else:
            surge_faults.append(i)

# 开始绘图：上下双子图结构
fig_comp, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig_comp.suptitle('典型故障电磁铁 $\eta_i(t)$ 动态演化特征对比分析', fontsize=18, fontweight='bold')

# -----------------
# 子图 1：衰减故障组
# -----------------
ax1.axhspan(0.8, 1.2, color='#2ca02c', alpha=0.15, label='[0.8, 1.2] 正常容限安全区')
ax1.axhline(0.8, color='#d62728', linestyle='--', linewidth=1.5)
ax1.axhline(1.0, color='gray', linestyle=':', linewidth=1)

# 使用高级色板，避免颜色重复
colors_decay = plt.cm.tab10(np.linspace(0, 1, len(decay_faults)))
for idx, color in zip(decay_faults, colors_decay):
    ax1.plot(time_centers, eta_smoothed[:, idx], linewidth=2.5, alpha=0.85, 
             color=color, label=f'电磁铁 {idx+1} 号')

ax1.set_title('类型一：功率衰减故障群 (主要特征: $\eta < 0.8$)', fontsize=15)
ax1.set_ylabel(r'功率放大系数 $\eta$', fontsize=13)
ax1.set_ylim(0.3, 1.3)
ax1.grid(True, linestyle='--', alpha=0.5)
if decay_faults:
    ax1.legend(loc='lower left', ncol=len(decay_faults)+1, fontsize=11)

# -----------------
# 子图 2：激增故障组
# -----------------
ax2.axhspan(0.8, 1.2, color='#2ca02c', alpha=0.15, label='[0.8, 1.2] 正常容限安全区')
ax2.axhline(1.2, color='#d62728', linestyle='--', linewidth=1.5)
ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1)

colors_surge = plt.cm.Set1(np.linspace(0, 1, len(surge_faults)))
for idx, color in zip(surge_faults, colors_surge):
    ax2.plot(time_centers, eta_smoothed[:, idx], linewidth=2.5, alpha=0.85, 
             color=color, label=f'电磁铁 {idx+1} 号')

ax2.set_title('类型二：功率激增故障群 (主要特征: $\eta > 1.2$)', fontsize=15)
ax2.set_xlabel('时间 t (s)', fontsize=14)
ax2.set_ylabel(r'功率放大系数 $\eta$', fontsize=13)
ax2.set_ylim(0.5, 2.4)
ax2.grid(True, linestyle='--', alpha=0.5)
if surge_faults:
    ax2.legend(loc='upper left', ncol=len(surge_faults)+1, fontsize=11)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()



# 生成高可用性的故障诊断定量汇总表

print("\n" + "="*50)
print("正在自动聚合时间序列，生成故障诊断定量汇总表...")

fault_records = []

# 定义聚合阈值：如果两次故障间隔小于 0.5秒，则视作同一个故障区间
merge_gap = 0.5 

def get_fault_intervals(bool_array, time_array, gap):
    """提取连续的故障时间段，智能聚合相近的故障点"""
    indices = np.where(bool_array)[0]
    if len(indices) == 0:
        return []
    
    intervals = []
    start_idx = indices[0]
    prev_idx = indices[0]
    
    for idx in indices[1:]:
        if time_array[idx] - time_array[prev_idx] <= gap:
            prev_idx = idx
        else:
            intervals.append((time_array[start_idx], time_array[prev_idx]))
            start_idx = idx
            prev_idx = idx
    intervals.append((time_array[start_idx], time_array[prev_idx]))
    return intervals

# 遍历 16 个电磁铁
for i in range(16):
    eta = eta_smoothed[:, i]
    
    # 1. 检测衰减故障 (eta < 0.8)
    decay_intervals = get_fault_intervals(eta < 0.8, time_centers, merge_gap)
    for start, end in decay_intervals:
        mask = (time_centers >= start) & (time_centers <= end)
        min_val = np.min(eta[mask])
        duration = end - start
        
        fault_records.append({
            "电磁铁编号": f"{i+1} 号",
            "诊断状态": "衰减故障 (η < 0.8)",
            "发生时间区间": f"{start:.2f}s - {end:.2f}s" if duration > 0.05 else f"瞬态 ({start:.2f}s)",
            "持续时长 (s)": f"{duration:.2f}",
            "极值系数 η": f"{min_val:.3f}"
        })
        
    # 2. 检测激增故障 (eta > 1.2)
    surge_intervals = get_fault_intervals(eta > 1.2, time_centers, merge_gap)
    for start, end in surge_intervals:
        mask = (time_centers >= start) & (time_centers <= end)
        max_val = np.max(eta[mask])
        duration = end - start
        
        fault_records.append({
            "电磁铁编号": f"{i+1} 号",
            "诊断状态": "激增故障 (η > 1.2)",
            "发生时间区间": f"{start:.2f}s - {end:.2f}s" if duration > 0.05 else f"瞬态 ({start:.2f}s)",
            "持续时长 (s)": f"{duration:.2f}",
            "极值系数 η": f"{max_val:.3f}"
        })

# 如果没有检测到任何故障（备用）
if not fault_records:
    print("全车系统正常，未检测到显著故障！")
else:
    # 转化为 DataFrame
    df_faults = pd.DataFrame(fault_records)
    
    # 按电磁铁编号排序，保证表格整洁
    df_faults['排序列'] = df_faults['电磁铁编号'].apply(lambda x: int(x.split(' ')[0]))
    df_faults = df_faults.sort_values(by=['排序列', '发生时间区间']).drop(columns=['排序列']).reset_index(drop=True)
    
    # 1. 在控制台打印美观的表格
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.unicode.east_asian_width', True) # 对齐中文
    
    print("\n★ 全车 16 台悬浮电磁铁故障在线诊断定量汇总表 ★")
    print("-" * 75)
    print(df_faults.to_string(index=True))
    print("-" * 75)
    
    # 2. 导出到 Excel (重点！方便你写论文)
    excel_name = "Problem4_Fault_Report.xlsx"
    df_faults.to_excel(excel_name, index=False)
    print(f"\n[提示] 诊断表格已成功导出为 Excel 文件: '{excel_name}'，请直接在当前文件夹下查收并复制进论文。")
print("="*50 + "\n")



# 蒙特卡洛噪声扰动鲁棒性检验

from tqdm import tqdm # 进度条库
print("\n" + "="*45)
print("启动蒙特卡洛(Monte Carlo)噪声扰动验证...")
print("="*45)

M_simulations = 50    # 蒙特卡洛实验次数 
noise_level = 0.03    # 注入 3% 的高斯白噪声

# 预先计算标准差作为噪声基准
std_h = np.std(h_data)
std_ddz = np.std(ddz_c_data)
std_I = np.std(I_matrix, axis=0)

# 用于存储每次实验的 eta 结果, 形状为 (M_simulations, 窗口数, 16)
mc_eta_results = []

# 为了提高蒙特卡洛运算效率，稍微放大滑动步长
mc_window_size = 500
mc_step_size = 100 
mc_time_centers = [t_data[start_idx + mc_window_size // 2] for start_idx in range(0, N - mc_window_size, mc_step_size)]

# 蒙特卡洛循环
for m in tqdm(range(M_simulations), desc="MC Simulations Progress"):
    
    # 1. 注入独立高斯白噪声
    noisy_h = h_data + np.random.normal(0, noise_level * std_h, N)
    noisy_ddz = ddz_c_data + np.random.normal(0, noise_level * std_ddz, N)
    noisy_I = I_matrix + np.random.normal(0, noise_level * std_I, (N, 16))
    
    # 2. 重新 SG 滤波 (模拟真实的带噪滤波过程)
    noisy_ddh = savgol_filter(noisy_h, window_length=201, polyorder=3, deriv=2, delta=dt)
    
    # 3. 动力学重建
    F_req_noisy = mc * noisy_ddz - mf * noisy_ddh + (mc + mf) * g
    I_squared_sgn_noisy = np.sign(noisy_I) * (noisy_I ** 2)
    F_ideal_matrix_noisy = K * I_squared_sgn_noisy / (noisy_h[:, None] ** 2)
    F_ideal_total_noisy = np.sum(F_ideal_matrix_noisy, axis=1)
    
    current_eta_result = []
    
    # 4. 二层模型参数辨识
    for start_idx in range(0, N - mc_window_size, mc_step_size):
        end_idx = start_idx + mc_window_size
        
        X_w = F_ideal_matrix_noisy[start_idx:end_idx, :] / 1e5
        Y_res_w = (F_req_noisy[start_idx:end_idx] - F_ideal_total_noisy[start_idx:end_idx]) / 1e5
        
        layer1_model.fit(X_w, Y_res_w)
        delta_l1 = layer1_model.coef_
        
        active_idx = np.where(np.abs(delta_l1) > 0.01)[0]
        delta_final = np.zeros(16)
        
        if len(active_idx) > 0:
            if len(active_idx) > 3:
                active_idx = active_idx[np.argsort(np.abs(delta_l1[active_idx]))[-3:]]
            layer2_model.fit(X_w[:, active_idx], Y_res_w)
            delta_final[active_idx] = layer2_model.coef_
            
        current_eta_result.append(1.0 + delta_final)
        
    mc_eta_results.append(current_eta_result)

# 转换为 Numpy 数组方便统计 (M, 窗口数, 16)
mc_eta_results = np.array(mc_eta_results)

# 计算均值和标准差
eta_mean = np.mean(mc_eta_results, axis=0)
eta_std = np.std(mc_eta_results, axis=0)

# ==========================================
# 9. 蒙特卡洛置信区间误差带可视化 (选3个典型代表)
# ==========================================
print("生成蒙特卡洛鲁棒性置信区间图...")

# 挑选三个最具代表性的电磁铁：4号(正常), 3号(衰减), 15号(激增)
target_magnets = [3, 2, 14] # 对应编号 4(索引3), 3(索引2), 15(索引14)
titles = ["【稳健正常】电磁铁 4 号", "【衰减故障】电磁铁 3 号", "【激增故障】电磁铁 15 号"]
colors = ['#2ca02c', '#d62728', '#ff7f0e']

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(r'蒙特卡洛扰动测试：传感器注入 3% 高斯白噪声的系统诊断鲁棒性 ($M=50$ 次)', fontsize=16, fontweight='bold')

for ax_idx, mag_idx in enumerate(target_magnets):
    ax = axes[ax_idx]
    
    mean_curve = eta_mean[:, mag_idx]
    std_curve = eta_std[:, mag_idx]
    
    # 画 95% 置信带 (mean ± 1.96 * std)
    lower_bound = mean_curve - 1.96 * std_curve
    upper_bound = mean_curve + 1.96 * std_curve
    
    # 正常容限区背景
    ax.axhspan(0.8, 1.2, color='green', alpha=0.1, label='正常运行容限 [0.8, 1.2]')
    ax.axhline(0.8, color='red', linestyle='--', linewidth=1, alpha=0.6)
    ax.axhline(1.2, color='red', linestyle='--', linewidth=1, alpha=0.6)
    
    # 绘制均值线和误差带
    ax.fill_between(mc_time_centers, lower_bound, upper_bound, color=colors[ax_idx], alpha=0.3, label='95% 置信区间 (CI)')
    ax.plot(mc_time_centers, mean_curve, color=colors[ax_idx], linewidth=2, label='蒙特卡洛辨识均值 $\overline{\eta}$')
    
    ax.set_title(titles[ax_idx], fontsize=13)
    ax.set_ylabel(r'参数 $\eta$', fontsize=12)
    ax.set_ylim(0.4, 1.6)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    if ax_idx == 0:
        ax.legend(loc='upper right', ncol=3)

axes[-1].set_xlabel('时间 t (s)', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()