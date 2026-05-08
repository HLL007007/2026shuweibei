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

# ==========================================
# 2. 读取附件2数据并预处理
# ==========================================
print("正在读取附件2数据，数据量较大请稍候...")
# 假设前两行是表头和单位，跳过第一行，保留列名进行读取
df = pd.read_excel('data2.xlsx', skiprows=[1]) 

time_data = df.iloc[:, 0].values  # 第1列：时间 t
# 第2到17列：16个电磁铁的力，按行求和得到总电磁力 F_mag(t)
F_mag_data = df.iloc[:, 1:17].sum(axis=1).values  

# 构建连续时间的电磁力插值函数，方便在ODE求解器中调用
# fill_value='extrapolate' 允许在略微超出范围时外推，防止求解器报错
F_mag_interp = interp1d(time_data, F_mag_data, kind='linear', fill_value='extrapolate')

# ==========================================
# 3. 建立状态空间微分方程
# ==========================================
# 状态变量向量 y = [z_c, z_c_dot, z_f, z_f_dot]
def train_dynamics(t, y):
    zc, zc_dot, zf, zf_dot = y
    
    # 获取当前时刻的总电磁力
    F_mag = F_mag_interp(t)
    
    # 计算空气弹簧力 (对车体)
    F_air_spring = -k * (zc - zf) - c * (zc_dot - zf_dot)
    
    # 车体加速度: mc * zc_ddot = F_air_spring - mc * g
    zc_ddot = (F_air_spring - mc * g) / mc
    
    # 悬浮架受力 (不含轨道支持力): -F_air_spring + F_mag - mf * g
    frame_force = -F_air_spring + F_mag - mf * g
    
    # 引入轨道支撑约束 (非平滑切换)
    if zf <= 0 and frame_force <= 0:
        # 如果还在轨道上且合力向下，被轨道托住，加速度和速度均为0
        zf_ddot = 0.0
        zf_dot = 0.0  # 强制速度归零，防止数值穿透
    else:
        # 起浮状态
        zf_ddot = frame_force / mf
        
    return [zc_dot, zc_ddot, zf_dot, zf_ddot]

# ==========================================
# 4. 初始条件与仿真求解
# ==========================================
# z_c(0) 由弹簧初始静力平衡决定: -k*z_c(0) - mc*g = 0
zc_0 = - (mc * g) / k
y0 = [zc_0, 0.0, 0.0, 0.0]

t_span = (0, 10)  # 仿真时间 0 到 10 秒
# 我们希望获取高精度的输出点
t_eval = np.linspace(0, 10, 10000) 

print("开始求解动力学微分方程...")
# 使用 RK45 求解器 (如果出现刚性问题可换用 'Radau' 或 'BDF')
sol = solve_ivp(train_dynamics, t_span, y0, t_eval=t_eval, method='RK45', max_step=0.01)
print("求解完成！\n")

# ==========================================
# 5. 结果提取与分析
# ==========================================
t_sol = sol.t
zc_sol = sol.y[0]
zf_sol = sol.y[2]
z_gap = 0.06 - zf_sol  # 悬浮间隙 z = 0.06 - z_f

# 找出 t=9 秒时的索引和数值
idx_9s = np.argmin(np.abs(t_sol - 9.0))
z_gap_9s = z_gap[idx_9s]

print("====== 核心结果输出 ======")
print(f"9秒时刻的车体位移 zc(9): {zc_sol[idx_9s]:.6f} 米")
print(f"9秒时刻的悬架位移 zf(9): {zf_sol[idx_9s]:.6f} 米")
print(f"★ 9秒时刻的悬浮间隙 z(9): {z_gap_9s:.6f} 米 ★")
print("=========================\n")

# ==========================================
# 6. 可视化绘图 (论文用图)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 图1：位移曲线
ax1.plot(t_sol, zc_sol, label='车体垂向位移 $z_c$', color='blue', linewidth=2)
ax1.plot(t_sol, zf_sol, label='悬浮架垂向位移 $z_f$', color='red', linewidth=2)
ax1.set_title('磁浮列车车体与悬浮架垂向位移响应 (0-10s)', fontsize=14)
ax1.set_xlabel('时间 (s)', fontsize=12)
ax1.set_ylabel('位移 (m)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.5)

# 图2：悬浮间隙曲线
ax2.plot(t_sol, z_gap, label='悬浮间隙 $z$', color='green', linewidth=2)
ax2.axvline(x=9.0, color='gray', linestyle='--', label='9秒时刻')
ax2.scatter(9.0, z_gap_9s, color='red', zorder=5)
ax2.annotate(f'z(9s) = {z_gap_9s:.4f} m', 
             xy=(9.0, z_gap_9s), xytext=(9.2, z_gap_9s + 0.002),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)
ax2.set_title('列车悬浮间隙动态变化曲线', fontsize=14)
ax2.set_xlabel('时间 (s)', fontsize=12)
ax2.set_ylabel('间隙大小 (m)', fontsize=12)
# 正常悬浮间隙一般在 0.01 左右，标出基准线
ax2.axhline(y=0.01, color='orange', linestyle=':', label='额定悬浮间隙参考(约10mm)')
ax2.legend()
ax2.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()