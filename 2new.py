import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 物理参数定义
# ==========================================
mc = 33000.0          # 车体质量 (kg)
mf = 3000.0 + 16*500  # 悬浮架总质量 (kg)
k = 2e7               # 空气弹簧刚度 (N/m)
c = 8e4               # 空气弹簧阻尼 (N.s/m)
g = 9.8               # 重力加速度 (m/s^2)
total_weight = (mc + mf) * g  # 满载系统总重力 ≈ 431200 N

# ==========================================
# 2. 读取数据与接触模型参数
# ==========================================
# 轨道接触模型参数 (模拟极其坚硬的铁轨)
k_track = 5e8         # 轨道等效刚度 (非常大)
c_track = 5e6         # 轨道等效阻尼 (吸收冲击能量)

print("正在读取附件2数据...")
try:
    df = pd.read_excel('data2.xlsx', skiprows=[1]) 
    time_data = df.iloc[:, 0].values  
    F_mag_data = df.iloc[:, 1:17].sum(axis=1).values  
except Exception as e:
    print("未找到 data2.xlsx，将使用【动态测试数据】进行演示！")
    # 构造一个测试场景：前2秒吸力不足(趴在轨道上)，第2秒开始吸力增大(起飞)，第5秒达到平稳悬浮
    time_data = np.linspace(0, 10, 1000)
    F_mag_data = np.ones_like(time_data) * 400000  # 初始 40万N < 总重力 43.1万N
    F_mag_data[time_data > 2.0] = 450000           # 大于重力，开始起浮
    F_mag_data[time_data > 5.0] = total_weight     # 等于重力，平稳悬浮

F_mag_interp = interp1d(time_data, F_mag_data, kind='linear', fill_value='extrapolate')

# ==========================================
# 3. 建立包含接触力学状态的微分方程
# ==========================================
def train_dynamics(t, y):
    zc, zc_dot, zf, zf_dot = y
    
    # 获取电磁力
    F_mag = F_mag_interp(t)
    
    # 空气弹簧力 (作用于车体)
    F_air_spring = -k * (zc - zf) - c * (zc_dot - zf_dot)
    
    # 核心修正：轨道的支撑力 (罚函数法接触模型)
    if zf < 0:
        # 当发生向下穿透时，轨道产生巨大的向上抵抗力
        F_track = -k_track * zf - c_track * zf_dot
        # 轨道只能往上托，不能往下拽
        F_track = max(0, F_track)
    else:
        # 悬浮在空中时，无轨道支撑力
        F_track = 0.0
    
    # 运动方程
    zc_ddot = (F_air_spring - mc * g) / mc
    # 悬浮架受到：弹簧反作用力、电磁力、重力、轨道支撑力(触底时生效)
    zf_ddot = (-F_air_spring + F_mag - mf * g + F_track) / mf
        
    return [zc_dot, zc_ddot, zf_dot, zf_ddot]

# ==========================================
# 4. 初始条件与仿真求解
# ==========================================
zc_0 = - (mc * g) / k
y0 = [zc_0, 0.0, 0.0, 0.0]

t_span = (0, 10)
t_eval = np.linspace(0, 10, 2000) 

print("开始求解刚性动力学微分方程...")
# 必须使用 LSODA 或 BDF 求解器，因为巨大的 k_track 会导致系统呈现刚性
sol = solve_ivp(train_dynamics, t_span, y0, t_eval=t_eval, method='LSODA', max_step=0.01)
print("求解完成！\n")

# ==========================================
# 5. 结果提取与可视化
# ==========================================
t_sol = sol.t
zc_sol = sol.y[0]
zf_sol = sol.y[2]
z_gap = 0.06 - zf_sol  # 悬浮间隙 z = 0.06 - z_f

idx_9s = np.argmin(np.abs(t_sol - 9.0))
z_gap_9s = z_gap[idx_9s]

print("====== 核心结果输出 ======")
print(f"9秒时刻的悬浮间隙 z(9): {z_gap_9s:.6f} 米")
print("=========================\n")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 图1：位移曲线
ax1.plot(t_sol, zc_sol, label='车体垂向位移 $z_c$', color='blue', linewidth=2)
ax1.plot(t_sol, zf_sol, label='悬浮架垂向位移 $z_f$', color='red', linewidth=2)
ax1.set_title('磁浮列车车体与悬浮架垂向位移响应 (含物理轨道约束)', fontsize=14)
ax1.set_xlabel('时间 (s)', fontsize=12)
ax1.set_ylabel('位移 (m)', fontsize=12)
# 标注铁轨地面
ax1.axhline(y=0, color='black', linewidth=3, label='铁轨支撑面 ($z=0$)')
ax1.legend()
ax1.grid(True, alpha=0.5)

# 图2：悬浮间隙曲线
ax2.plot(t_sol, z_gap, label='悬浮间隙 $z$', color='green', linewidth=2)
ax2.axvline(x=9.0, color='gray', linestyle='--', label='9秒时刻')
ax2.scatter(9.0, z_gap_9s, color='red', zorder=5)
ax2.annotate(f'z(9s) = {z_gap_9s:.4f} m', 
             xy=(9.0, z_gap_9s), xytext=(8.5, z_gap_9s + 0.005),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)
ax2.set_title('列车悬浮间隙动态变化曲线', fontsize=14)
ax2.set_xlabel('时间 (s)', fontsize=12)
ax2.set_ylabel('间隙大小 (m)', fontsize=12)
ax2.set_ylim(0, 0.065)
ax2.legend()
ax2.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()