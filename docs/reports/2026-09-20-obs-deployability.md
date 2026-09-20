# kick_amp 观测空间盘点与实物部署可行性评估（2026-09-20）

> 结论先行：观测本身是按"可实机部署"设计的（IMU + 编码器 + 相机球检测 + 航向估计），
> checkpoint 导出格式 / 控制频率 / 执行器建模均已与 `booster_deploy` 对齐；
> 真正的工程量在 deploy 侧补感知输入通道与任务包。
> **⚠️ 第 0 前提：当前最好成绩（76.2% 进球）是 benchmark profile（`perfect_perception=True`）下测的，
> 实机不存在完美感知——部署前必须用默认 profile（虚拟感知噪声全开）复测或重训。**

调研方法：7 个并行 agent 分工读取 + 2 轮对抗校验（共 210 次工具调用），
所有结论均有 文件:行号 依据；校验修正了 5 处细节（检测概率 clamp 语义、特权 14 维 COM 占位、
hip_roll scale 数字等），维度 79/93 逐项核实无误。

## 一、策略观测（actor）：79 维

配置 `source/booster_train/booster_train/tasks/manager_based/kick_amp/robots/k1/kick_amp/env_cfg.py:134-146`，
实现 `mdp/observations.py` + `mdp/commands.py`。K1 = 22 自由度（2 头 + 8 臂 + 12 腿）。

| # | 项 | 维 | 内容 / 坐标系 | 噪声 | 真机来源 |
|---|---|---|---|---|---|
| 1 | `project_gravity` | 3 | 重力基座系投影 | σ0.01 | ✅ IMU 加速度计 |
| 2 | `base_ang_vel` | 3 | 基座角速度 | σ0.1 | ✅ IMU 陀螺 |
| 3 | `ball_obs` | 3 | yaw 系球相对位置 (x,y) + 检测 flag | 内建虚拟感知 | ❌ 需视觉 |
| 4 | `goal_pos` | 2 | yaw 系球门中心（世界常量 (7,0) − 机器人世界真值） | σ0.5 | ❌ 需全局定位 |
| 5 | `base_yaw` | 2 | 世界航向 (cosθ, sinθ) | σ0.1 | ⚠️ 需状态估计（IMU 航向/磁力计/动捕） |
| 6 | `joint_pos` | 22 | 关节角 − 默认角（默认角被 startup 随机偏 ±0.05 rad 且策略不可见） | σ0.01 | ✅ 编码器 |
| 7 | `joint_vel` | 22 | 关节速度 ×0.1 | σ0.01 | ✅ 编码器差分 |
| 8 | `actions` | 22 | 上一步原始动作（未乘 scale；`action_manager.py:262-264`） | 无 | ✅ 自身输出 |

合计 3+3+3+2+2+22+22+22 = **79**（与 `env_cfg.py:130` 注释、`rsl_rl/amp/modules.py:4` 一致）。
所有 term 无 clip；关节序为 PhysX BFS 序非 URDF 序（`commands.py:207-209` 注释）。

### 1.1 虚拟球感知（`commands.py:426-457, 587-605`）

`ball_obs` 底层是仿真球真值，但强制过一层对真机视觉的显式仿真（参数源自论文 ~1h 真机标定）：

- 25 Hz 刷新（隔一个 50 Hz policy 步）；
- 延迟：每 episode 采样 `clamp(randn·1+6, 0, 19)` 步，从 20 深环形缓冲取值（典型 ≈120 ms）；
- 距离噪声：每维独立高斯 σ = 0.149 + 0.124·d 米；
- FOV：头部相机 87°×58°（安装偏置 `(0.054, 0, 0.102)`，`commands.py:184-187`）；
- 检测概率 ≡ `min(0.9, 2.475 − 0.225·d)`（代码 `dist.clamp(min=7.0)`：d≤7 m 恒 0.9，11 m 归零；
  **校验结论：clamp 是概率封顶 0.9，不是方向反了的 bug**）；
- 未检出时 xy 置零；球在身后但 x>−2 m 时 1% 假阳性；
- `perfect_perception=True` 可旁路（cfg 默认 False；**benchmark 训练 profile 为 True**，`Project.md:81`）。

### 1.2 50 帧历史 + encoder-decoder POMDP

- `num_stack=50`（`ppo_cfg.py:17`）：runner 维护 (N, 50, 79) **归一化**观测历史（1 s @50 Hz），
  每步 roll 后写底、done 清零（`runner.py:226-229 _push_obs`、`runner.py:393`）；
- encoder `Linear(79×50→1024→128→64)` 压成 64 维 latent 与当前 79 维拼接进 actor；
  decoder 从 latent 重建 14 维特权量做辅助监督（`modules.py:36-49, 71-79`；
  论文实测球位置估计 RMSE 0.344→0.186 m——**策略自带滤波器，实机检测流无需外部卡尔曼**）；
- EmpiricalNormalization 只作用于 policy obs（critic 不归一化，`runner.py:123-125`）。

策略故意看不到：基座线速度（POMDP 子集）、球真值、动力学参数。

## 二、critic 观测：93 维（仅训练用，部署丢弃）

同样 8 项（无噪声，`ball_obs`→真值 `ball_obs_true`：yaw 系真值球位 2 + in_view 1）＋
`privileged` 14 维（`commands.py:573-585`）：

COM 占位 **0×3**（`events.py:136` 硬编码零）+ trunk 质量比 1 + 基座线速度 3 + 高度 1 +
球滚阻 xy 2 + 球真值位置 2 + 球真值速度 2。（校验修正：COM 虽被物理随机化但从不进特权观测。）

AMP 判别器另有 39 维风格观测（`commands.py:549-571`），同样仅训练用。

## 三、动作：22 维关节位置目标

`ActionsCfg`（`env_cfg.py:176-186`）：目标 = `default_joint_pos + scale·action`，
逐关节 `scale = 0.25·effort_limit/stiffness`（`booster.py:105-116`）；
50 Hz（decimation 4 × sim.dt 0.005）；
无动作级延迟——`BoosterDelayedPDActuator` 已含 0-4 substep（0-20 ms）指令延迟
（`env_cfg.py:178-180` 注释中"2-8 substeps"是陈旧文本，实际 `booster.py:48-49`）。

## 四、部署可行性

### 4.1 已具备（逐行核实）

1. **policy.pt 实测可加载**：AmpRunner 训练中自动导出 TorchScript（`runner.py:231-251, 434-441`），
   `forward(obs_raw[1,79], stacked_obs[1,50,79]) → action[1,22]`，encoder + EmpiricalNormalization
   冻进图内（`m.obs_norm` 可直接调用）；判别器/双 critic 不需要。deploy 只吃 `torch.jit`——格式兼容。
2. **频率一致**：训练 50 Hz = deploy 真机 `policy_dt=0.02`（`controller_cfg.py:95`，
   `booster_robot_controller.py:541-553`）。唯一差异是物理子步（PhysX 5 ms vs MuJoCo 2 ms）。
3. **执行器保真**：真实电机型号（腿 E6408/E4315/E4310/E6416、踝并联 wrapper、臂 R14、头 HT4438）
   + 分段线性 T-N 曲线限幅 + armature + 0-20 ms 延迟随机 + PD ±5%（`actuator.py:85-163, 322-396`）。
4. **deploy 框架有实机先例**：locomotion / beyond_mimic 跑通 500 Hz `/low_state`→`/joint_ctrl`，
   含 sim↔real 关节重排（`base_controller.py:29-30`）；v3 增益配置修复过执行器换代脱节
   （memory: booster-deploy-actuator-mismatch）。
5. **论文先例**：2511.03996 在 T1 上零样本部署（RealSense + YOLOv8s → BEV 25 Hz 直喂 +
   地标/里程计给球门方位），RoboCup 2025 成人组 + 2025 世界人形机器人运动会夺冠。

### 4.2 缺口（按优先级）

0. **训练侧前提**：复测默认 profile（虚拟感知）下的成绩——见文首警告。
1. **感知输入通道（最大缺口）**：deploy 全仓无 ball/goal 通道；真机 `root_pos_w`/`root_lin_vel_w`
   恒填零（`booster_robot_controller.py:244-249`），唯一外部输入是 3 维摇杆。需补：
   球检测（相机+YOLO+BEV，25 Hz，(x,y)+检测位）与球门方位/绝对航向（地标+里程计，
   或起步版"已知初始朝向 + IMU yaw 积分"，注意漂移直接进 `goal_pos`/`base_yaw` 观测）。
2. **deploy 任务包**：新建 `tasks/kick_amp/`——79 维按序拼接（sim 关节序 + real2sim 重排）、
   双输入推理 + 50×79 归一化历史栈（episode 边界清零）、动作解码 `action[sim2real]·scale + default`
   （v3 数字）、`default_joint_pos` 改训练 crouch（hip_pitch −0.19 / knee 0.50,0.53 /
   ankle_pitch −0.18,−0.22，`env_cfg.py:71-75`；deploy 现值腿部全零 `robots/booster.py:69-75`）、
   真机 kp/kd 写训练派生值、跌倒停机回退（per-policy，`locomotion.py:63-70` 先例）。
3. **sim2sim 先行**：`booster_assets/scene/soccer_field_14x9.xml` 已有球（r=0.11、m=0.43，与训练一致）
   + 双门 + 围板，但 `MujocoController` 只载 `K1_22dof.xml`——需合并场景并把球 freejoint 纳入控制循环；
   球摩擦 0.7 → 对齐训练 `ball_friction_range=(0.2,0.2)`。
4. **次要 sim2real 缺口**（论文协议自身取舍，非阻塞；真机草坪/垫子需注意）：
   无重力随机化、纯平地无 height-scan、球接触材料用 PhysX 默认值、本体感知零延迟、
   电机力矩上限不随机、肢体惯量完全可信、速度推扰仅 ±0.1 m/s。

### 4.3 论文 vs 本项目的结构差异（移植注意）

- 论文 actor 有"球门法向 (cos,sin)"，本项目用 `base_yaw_cos_sin` 替代；
- 论文用 encoder→64 维 latent（`num_stack` 是本项目等价实现，维度 50×79 堆叠）；
- 论文 critic 特权量含球速度/摩擦等，本项目 14 维构成见 §二（COM 三维是占位零）。

## 五、下一步（本报告的执行项）

1. ✅ 本报告落盘；
2. ✅ sim2sim 落地（见 §六：`booster_deploy/tasks/kick_amp/` 增量新增，现有任务零改动）；
3. 实机感知（YOLOv8s + RealSense + BEV）留到实机阶段——sim2sim 的球观测来自 MuJoCo 真值
   （本 ckpt 为 perfect_perception 训练），虚拟感知管线已移植待噪声版 ckpt。

## 六、sim2sim 实施记录（2026-09-20）

### 6.1 交付物（均在 `/data/rl_robot/BoosterRobotics/booster_deploy`，纯增量未提交）

- `tasks/kick_amp/scene/k1_soccer_14x9.xml`：K1_22dof.xml 机器人树 + 14x9 球场合并
  （逐字搬运 + 修正：hip_roll forcerange 43→76（E4315，Project.md:213 已知问题）、
  头关节 armature 0.002→0.001（HT4438）、**球门与围墙改纯视觉**（训练判进球纯几何，
  实体小门会堵 ~24% 进球带且门网会困住判出界的球））
- `tasks/kick_amp/kick_amp.py`：79 维观测 1:1 复刻 + 虚拟感知移植 + 50 帧归一化历史栈
- `tasks/kick_amp/kick_amp_mujoco.py`：按名寻址控制器（球 freejoint 场景 nq=36）+
  T-N 转速曲线限矩 + 子步粒度延迟（含首步无延迟语义）+ 球滚阻外力 + 回合协议
  （验收摆位 scenario_layout 逐行移植，首事件终止语义）
- `tasks/kick_amp/__init__.py`：任务注册 + 全精度增益（取自 run env_cfg.txt）
- `scripts/sim2sim_kick_amp.py`：入口（headless 批跑 / --view 交互 / --perception virtual）
- `tasks/kick_amp/models/kick_amp_it6400_policy.pt`：= run 2026-09-19_13-45-20 的
  model_6400 TorchScript 导出（195/256、0 摔倒；**该 run perfect_perception=True、
  min_delay=max_delay=2（固定 10ms）、全部 DR 事件关闭**）；另备份至
  `booster_train/models_backup/`（logs 曾被清理过一次，防再丢）

### 6.2 调试记录（两个真 bug，均已修，值得留档）

1. **历史栈错位一拍**：训练 rollout（runner.py:341-356+393）在动作 n 读到的栈**包含当前
   观测本身**（上一 step 的 push 写入的是 post-action obs），回合首步栈为零。初版"推理后压栈"
   慢一拍 → 修正为"除首步外先压当前 obs 再推理"。
2. **延迟单位错一级**：训练 DelayBuffer 的单位是 5ms 物理子步；初版按 policy 步（20ms）索引
   队列 → `--delay 5` 实为 100ms（训练上限 20ms）→ 关节跟踪误差 0.6 rad、全部 3s 内摔倒。
   修正为子步粒度切换（20ms policy 步内前 delay 子步用上一步目标），并补上 IsaacLab
   "buffer 未满返回当前值"的首步无延迟语义。

诊断方法可复用：观测分块 dump 对照（spawn 态 8 块全对）→ PD 定靶测试（原版舞蹈任务组合
也塌 → 排除环境回归；舞蹈**策略**跑 10s 稳定 → 锁定接口侧）→ 离屏渲染看行为（步态自然、
触球后摔倒）→ 单变量矩阵（延迟 0/10/20ms × solref × 站姿）。

### 6.3 对抗校验（3 透镜并行，核心全部 CONFIRMED-CORRECT）

79 维观测构成/顺序、yaw 系旋转、四元数约定、real2sim/sim2real 方向、历史栈语义、
动作解码（scale=0.25·effort/stiffness 逐关节对上）、终止几何（ball_in_goal/出场/摔倒/高速）、
摆位协议（scenario_layout 逐行）均与训练源码对证一致。校验发现并已修复：
`--view` 永不运行（start() 时序）、`deploy.py --list` 在 base python 被模块级 import 弄崩
（回归，已惰性+路径兜底）、虚拟感知 FOV 漏相机偏移、T-N 曲线缺失、场景 inertial 两处抄写误差。

### 6.4 结果（32 回合、验收摆位、seed 123、perfect 感知、10ms 延迟）

| 版本 | 配置 | 超时 | 摔倒 | 踢出界 | 触球 | 平均存活 | 进球 |
|---|---|---|---|---|---|---|---|
| v1 | 修栈+延迟后，20ms | 11/32 | 19/32 | 2/32 | 44% | 16.2s | 0 |
| **v2** | +T-N/球门视觉/armature/首步语义，**10ms（训练值）** | **18/32** | **13/32 (41%)** | 1/32 | 44% | **21.6s** | 0 |
| v3 | it7200（I 臂达标 ckpt） | 21/32 | **9/32 (28%)** | 2/32 | 37.5% | 21.3s | 0 |
| IsaacSim 参照 | it6400 验收 256 场景 | — | **0 (0%)** | 出界 10.5% | 100%* | 30s | **76.2%** |

it7200（2026-09-20 训练的达标检查点，IsaacSim 侧摔倒 2.0%/出界 7.0%）在 MuJoCo 里摔倒率
41%→28%——训练侧的平衡改善部分迁移，但 sim2real/sim2sim 摔倒 gap 仍是部署前的头号问题。

\* 该 ckpt 的验收触球口径与 sim2sim 的 0.30m 距离代理不同。延迟扫描（8 回合/档）：
0ms 摔 6/8、10ms 摔 5/8、20ms 摔 3/8（n=8 噪声大，最终按训练值取 10ms）。

**解读**：策略在异构物理引擎里能走、能追球、能踢（离屏视频确认自然摆臂步态、2s 内逼至
球边触球、能将球踢出边界），v2 修正后 56% 回合撑满 30s。**剩余 gap = 41% 摔倒率 vs
IsaacSim 0%**——这是当前 sim2sim 的核心结论：控制策略部分转移成功，但平衡裕度不足，
直接上实机会有四成概率摔倒。候选原因（按可验证性排序）：脚部碰撞几何（MuJoCo 单个
box 角点接触 vs 训练 URDF convex hull 网格脚）、有效摩擦（MuJoCo 1.0 vs 训练 URDF
默认材质，组合方式不同）、踝关节并联机构用串联铰链近似、求解器接触柔顺性差异。

### 6.5 下一步建议

1. （训练侧，优先）用默认 profile（虚拟感知 + DR）重训/续训——现有 ckpt 从未见过感知噪声，
   这是实机前置条件，也可能顺带提升鲁棒性降低摔倒；
2. （sim2sim 侧）脚部碰撞几何对齐实验：给 MuJoCo 脚换 convex hull 网格（URDF 同源），
   或把脚 box 改为两侧圆角胶囊，量化摔倒率变化；
3. 摔倒时刻回放定位（--view 复现单回合），区分"步态中失稳"vs"踢球后失稳"；
4. （实机侧，未动）感知输入通道 + 任务包在 BoosterRobotPortal 路径的接线，等 1-2 收敛后做。

## 附：关键文件索引

- 训练观测：`tasks/manager_based/kick_amp/robots/k1/kick_amp/env_cfg.py:130-186`、
  `mdp/observations.py`、`mdp/commands.py:280-360, 426-470, 549-605`
- 导出接口：`rsl_rl/amp/runner.py:226-251, 353, 393, 433-441`、`rsl_rl/amp/modules.py`
- 执行器：`assets/robots/booster.py:46-119`、`assets/robots/actuator.py:85-396`
- deploy 侧：`booster_deploy/booster_deploy/controllers/*`、`tasks/beyond_mimic/__init__.py:65-113`
  （v3 增益先例）、`tasks/locomotion/__init__.py`（obs/安全先例）
- 论文：`/data/rl_robot/BoosterRobotics/2511.03996v2.pdf`（§2.1/§2.4/§4.3、附录 A/E/F）
