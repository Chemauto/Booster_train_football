# K1 自主踢球 · 项目配置与设计（Project.md）

论文 2511.03996 的 **AMP 端到端足球/踢球任务**在 Booster K1 上的复现。
**目标**：机器人自主走向足球、把球送进 7 m 外的球门。
**验收**（256 场景固定队列）：进球 ≥30%、触球 ≥60%、摔倒 ≤2%、出界 ≤10%。

- 怎么跑 → **[QUICKSTART.md](QUICKSTART.md)**
- 完整实验过程（每次失败的原因与效果）→ **[docs/reports/2026-09-19-experiment-log.md](docs/reports/2026-09-19-experiment-log.md)**
- 当前成绩：进球 **76.2%** ✅ / 触球 **100%** ✅ / 摔倒 **0%** ✅ / 出界 **10.5–14.8%** ❌（唯一未达标项）

---

## 1. 任务定义与场地几何

| 概念 | 值 | 说明 |
|---|---|---|
| 场地 | x ∈ [−7, 7]、y ∈ [−4.5, 4.5] | RoboCup 成人尺寸的一半（14 × 9 m） |
| 球门 | `GOAL_X = 7.0`、`GOAL_HALF_WIDTH = 1.3`、`GOAL_HEIGHT = 1.8` | **门线 = 场地边界线**（本项目最关键的几何事实） |
| 进球判据（验收） | 球心单步越过 `x = 7.11` 且 \|y\| + 0.11 < 1.3、球顶低于横梁 | `evaluate_kick_amp.py`，与训练侧判据同源 |
| 训练侧进球终止 | 同一几何（`mdp.ball_in_goal`） | 进球即结束回合 |
| 球 | 半径 0.11 m、质量 0.43 kg | |
| 机器人出界 | \|x\| > 7.9 或 \|y\| > 5.4（0.9 m 缓冲，t1.py） | |
| 球出界 | x < −7 或（x > 7 且 \|y\| > 1.3）或 \|y\| > 4.5 | 只重置球（训练时） |

> ⚠️ `GOAL_X == FIELD_HALF_LENGTH` 意味着"把球送进门"与"把自己带出场外"是同一动作的两面。
> 这是距离型边界惩罚在本任务上三次失败的根源（见实验记录 §4.3/4.5）。

## 2. 仓库结构

```
source/booster_train/booster_train/
  tasks/manager_based/kick_amp/          # 本任务
    mdp/            commands.py（soccer 命令：状态机/奖励组/球与门的几何常量）、
                    rewards.py、observations.py、events.py、geometry.py、mirror.py
    robots/k1/kick_amp/  env_cfg.py（场景/观测/动作/奖励/终止/事件）、ppo_cfg.py
    curriculum.py   # approach↔soccer 两个训练阶段的配置覆盖
  rsl_rl/amp/       runner.py（双 critic PPO + AMP 判别器）、policy.py
  assets/robots/booster.py               # K1/T1 的 articulation 与动作尺度
scripts/
  run_kick_amp_stages.py   # 分段训练→评估的编排器（推荐入口）
  rsl_rl/train_kick_amp.py # 训练 / --play / --completion_file
  evaluate_kick_amp.py     # 256 场景验收协议 + 诊断字段
  render_kick_amp.py       # 追踪相机录 mp4
  convert_paper_data.py    # 论文 CSV → AMP npz
  validate_motion_dataset.py, amp_data_build.py, ...  # 数据管线
tests/                     # 151 项 / 84 子测试（CPU，不需 GPU）
docs/reports/              # 按阶段的专题报告（含完整实验记录）
```

## 3. 任务与场景

| 项 | 配置 |
|---|---|
| 任务 id | `Booster-K1-KickAMP-v0`（训练）、`Booster-K1-KickAMP-v0-Play`（播放：2 env、120 s、无干预） |
| 机器人 | K1，22 个驱动关节 + 浮动基座；URDF 由 `booster_assets` 提供 |
| 默认站姿 | 由 walk 数据导出的微屈姿（hip −0.19、knee 0.50/0.53、ankle −0.18/−0.22）；**直腿零位在该执行器模型下不是稳定平衡**（膝会塌、躯干掉到摔倒阈值以下） |
| 球门 | 门柱与横梁**仅视觉**（进球由几何判定，不需碰撞体） |
| 地面/光照 | 平面 + 穹顶光 |
| 场景规模 | cfg 默认 4096 env / `env_spacing=6.0`；**本机 8 GB 卡实际用 1024** |

## 4. 仿真与动作

| 项 | 值 |
|---|---|
| `sim.dt` × `decimation` | 0.005 s × 4 → **策略 50 Hz**（`step_dt = 0.02 s`） |
| `episode_length_s` | 60 s（Play 120 s）；评估侧抬到 ≥ 步数，故评估中不出现超时 |
| 动作 | `JointPositionAction`，22 关节，`use_default_offset=True` |
| 动作尺度 | `K1_ACTION_SCALE[n] = 0.25 × effort_limit_sim / stiffness`（T-N 曲线缩放，本机 0.268–0.887） |
| 执行器延迟 | `BoosterDelayedPDActuator` 自带 2–8 子步（10–40 ms）延迟；**不再叠加** `DelayedJointPositionAction`（否则双重计延迟）。benchmark profile 固定为 2 步 = 10 ms |
| 接触传感 | 开启（软限位/碰撞惩罚依赖它；曾因索引错位把左脚落地判成非法接触） |

## 5. 观测与特权信息

**策略观测**（`PolicyObsCfg`）：`projected_gravity`、`base_ang_vel`、`ball_obs`（虚拟球感知，带延迟）、
`goal_pos_b`（球门在机身系）、`base_yaw_cos_sin`、`joint_pos_rel`、`joint_vel_scaled (×0.1)`、`last_action`，
外加 **50 帧历史堆叠**（`num_stack=50`，即 1 s @50 Hz）与观测噪声（`enable_corruption`）。

**critic 观测**（`CriticObsCfg`）：同上但无噪声，并把 `ball_obs` 换成 `ball_obs_true`（真实球位），
再加 `privileged_obs`（质量/质心等物理随机化参数）。

球感知有**通信延迟**（`ball_delay_steps`，默认 6 步 = 120 ms），benchmark profile 下用完美感知。

## 6. 奖励

### 6.1 任务奖励（命令组，`SoccerStateCommand` 的 `rew_groups[0]`）

| 项 | 尺度 | 说明 |
|---|---:|---|
| 进球（在门内每步） | `goal_scale = 15.0` | 球在门内期间持续给分 |
| 球远离机器人 | `ball_distance_scale = 50.0` | 奖励"把球踢出去"的位移（论文项，方向无关） |
| 球靠近球门中心 | `goal_distance_scale = 500.0` | **势能项**（只能挣净进展，不能来回刷）；方向性的主要来源 |

### 6.2 辅助项（`RewardCfg`）

| 项 | 权重 | 现状与备注 |
|---|---:|---|
| `survival` | **+3.0** | 每步恒定；是"站着"的基准收益 |
| `termination` | **−1000** | 摔倒/出界；**已排除 goal 原因**（否则进球被罚 1000，策略会远离门线） |
| `pos_still` | **−100** | 站桩罚；参数 `penalize_pos=0.45`（**必须低于实测步速 0.623**）、`penalize_pos_distance=0.3`（球豁免半径 = 接触距离） |
| `kick_ball` | −20 | 罚"脚尖前捅"（前两个鞋底角点贴球的 +x 速度） |
| `side_kick_ball` | +20 | 奖励内侧扫踢（左脚 −y、右脚 +y），与球门方向正交 |
| `face_ball_pitch/yaw` | −0.5 / −0.5 | 头部相机对准球 |
| `root_acc` | −1e-3 | 基座加速度 |
| `action_rate` / `head_action_rate` | −1 / −15 | 动作变化率 |
| `dof_pos_limits` / `collision` | −100 / −100 | 关节软限位外 5% 区间 / 非法碰撞 |
| `feet_min_distance` / `feet_slide` | −5 / −1 | 双脚间距 / 滑步 |
| `track_lin_vel_ball` / `feet_air_time_biped` | +3.0 / +2.0 | 行走课程项，**随课程 (1−c) 衰减**，靠近球时 gate 掉 |
| `feet_clearance` | +1.0 | 抬脚高度（曾因"静止满分"诱导贴地不动，已加摆动 gate） |
| `boundary` / `boundary_outward` / `ball_lateral` | **0** | 三者均**默认关闭**且经实测判定无效或有害（见实验记录 §4.3/4.5/4.10） |

## 7. 终止与回合结构

| 终止项 | 语义 | 惩罚 |
|---|---|---|
| `time_out` | 达到 episode 长度 | 无（`time_out=True`） |
| `goal` | **球在门内（验收几何）→ 进球即结束回合** | **无**（`termination` 奖励显式排除该原因） |
| `out_of_field` | 机器人越界（0.9 m 缓冲） | −1000 |
| `base_contact` | 躯干低于 0.35 m（摔倒；站立 ≈0.57，深蹲 ≈0.43） | −1000 |
| `high_velocity` | 基座速度平方和 > 50 | −1000 |

进球终止的意义：**否则"进球 → 继续玩 → 跟着球冲出场外"是回合结构的必然**。
实测（臂 G）：137 个进球里 **125 个（95%）结束时机器人已出界**，出界 225/256。
加上该终止后出界降到 27/256（臂 H2）。

进球标志在**球重置之前**缓存（`ball_in_goal_now`）：进球会在同一次 `_update_command` 里触发
"只重置球"的传送，任何在传送之后读实时球位置的代码都会看到"不在门内"。

## 8. 课程与权重调度

| 机制 | 值 | 说明 |
|---|---|---|
| `task_curriculum(env)` | `min(common_step_counter / (16000×24), 1)` | `common_step_counter = 总迭代数 × horizon`，即 **c = 迭代数 / 16000** |
| 行走课程项 | 权重 × (1 − c) | `track_lin_vel_ball`、`feet_air_time_biped` 等 |
| `task_weight_floor`（CLI） | `max(c, floor)` | **满权 1.0 可绕开课程斜坡**；步态与站桩项仍按原始 c 衰减 |
| 任务优势权重 | `weights[0] ×= task_policy_weight(env)` | 缩放的是**标准化后的优势**，不是原始奖励 |

诊断依据：停滞点的 goal critic 优势权重只有 `2.0 × 0.1875 = 0.375`，任务信号被压住 ——
这是"站着不动"退化解的直接原因之一。

## 9. AMP 先验

| 项 | 值 |
|---|---|
| 数据 | `amp_paper_waistfix`：**24 段 / 5314 帧 / 50 Hz**（12 走 + 12 踢；踢球是 6 左 6 右的**精确镜像**） |
| 来源 | 论文仓库自带的重定向 CSV（`code/data/*.csv`）→ `scripts/convert_paper_data.py` |
| 该数据集的修复 | T1 腰 yaw 在腿的上游而 K1 没有 → 把 `q_waist`（实测中位 5.4°、最大 24.6°）加到两个 hip yaw |
| 判别器 | `amp_coef = 1.0`、判别器 lr `1e-4`、梯度惩罚配置 5（runner 内部另 ×10 ⇒ 有效 50，与论文一致）、`symmetric_coef = 10`、`reconstruction_coef = 1`、`bound_coef = 100` |
| 风格奖励 | `amp_reward_coef = 0.3`；用 `+1+tanh` 形式（论文 Eq.4 的负号与其判别器目标矛盾，**不要**为字面对齐而反转） |
| 对称性 | `mirror.py` 做左右镜像 + 对称损失；已知**默认膝角左右不对称**（0.50/0.53）使镜像漏掉 0.03/0.04 rad 的仿射原点项（未修，见 §13） |

参考数据里 `walk_back` / `walk_turn_around` / `side_step_left|right` 齐备 ——
"球在正后方"所需的转身与后退素材本身是有的，该分支的失败是训练分布问题（臂 H1）。

## 10. PPO / 双 critic runner

| 参数 | 值 |
|---|---|
| 结构 | `AmpRunner`（`rsl_rl/amp/runner.py`）：两个 critic，优势按 `advantage_coef = (2.0, 1.0)` 合并（进球组 : 辅助组） |
| `num_envs` / `max_iterations` | cfg 4096 / 20000（本机用 1024） |
| `horizon_length` | 24（与 `TASK_RAMP_STEPS` 的分母一致） |
| `num_learning_epochs` / `num_mini_batches` | 5 / 4 |
| `learning_rate` / 自适应 | 1e-3，按**迭代平均 KL** 调整（`desired_kl = 0.01`） |
| `gamma` / `lam` | 0.995 / 0.95 |
| `entropy_coef` | −0.01（在 `loss += coef·entropy` 下是**鼓励探索**） |
| 观测历史 | `num_stack = 50`（1 s @50 Hz） |
| 导出 | 深拷贝导出（曾因引用 live normalizer，`eval()` 后使训练归一化永久失效） |

`--reset_optimization`：保留 actor、重置 critic/PPO 状态与探索（sigma 回 0.15），
用于"奖励或回合结构发生阶段级变化"时。

## 11. 训练入口与运维

**推荐入口** `scripts/run_kick_amp_stages.py`：分段 `训练 → 独立评估`，串行占用 GPU，具备：

- **源码哈希门**：`source/**/*.py` 或 `scripts/**/*.py` 变化 → 中止后续阶段（避免混合实验）
- **数据哈希**：`--motion_dir` 会被固定并记录 sha256
- **完成文件校验**：训练子进程必须写出 `training_result.json`（含实际迭代计数），否则阶段失败
- **优雅停止**：SIGINT/SIGTERM 转发给子进程 → 保存 checkpoint 后退出
- 状态落盘 `status.json`：全部 command、checkpoint、物理指标；阶段结束只标 `requires_review`，
  **不自动把"存活"或"进程正常结束"当作足球成功**

**原始训练** `scripts/rsl_rl/train_kick_amp.py`：`--checkpoint` 续训、不填则从零；
`--play` 交互播放；`--completion_file` 写完成报告。

**8 GB 显存的硬约束**：同一时刻只能有一个 IsaacSim 实例（两个会 Mutex 崩）；
训练 1024 env ≈ 6.1 GB；**渲染 ≤13 env**（13 env + 相机 ≈ 7.2 / 8.2 GB）。

## 12. 评估协议（验收仪器）

```bash
python scripts/evaluate_kick_amp.py --headless --checkpoint <CKPT> \
  --scenario soccer --num_envs 256 --steps 1500 --seed 123 --perfect_perception --output logs/eval.json
```

- **256 场景**、**首轮 episode**、**均值（确定性）动作**、固定 `seed 123`、`--perfect_perception`
- 场景布局是**确定性**的：`yaw = [0,±90,180][id%4]`、`bearing = [0,±90,180][(id//4)%4]`、距离 U(0.8,2.0)
  → 可用 `(env_id // 4) % 4` 按**球相对机器人的方位**拆解
- **重置前快照**（否则会把复位位移当成移动）；接触是保守几何代理（脚在球 0.25 m 内且球随后位移 >0.2 m）
- 输出指标：`validated_goal_count/fraction`、`contact_proxy_count`、`fall_count/fraction`、
  `termination_causes`（含 `goal`）、`first_episode_survival_fraction`、
  `first_episode_success_or_survival_fraction`（进球即结束回合后，前者会把"成功"算成"没活到结束"）
- 诊断字段（本项目加的）：`ball_end_zone_counts`（球终点分区）、`final_ball_pos_m`、`final_robot_pos_m`、
  `max_ball_x_m`、`crossing_y_m`（**插值到门线**的横向位置 = 偏了多少）、
  `max_contact_lateral_speed_mps`、`crossing_lateral_speed_mps`

⚠️ **不要用 64 场景做决策**：同一配置在 64 场景下进球 1/64，256 场景 11/256（低估 2.7 倍）。

## 13. 已知限制与未完成项

| 项 | 状态 |
|---|---|
| **出界 10.5–14.8%（目标 ≤10%）** | 唯一未达标指标。机制已测清：球到线率 95%+，其中 ~85% 进门；**瞄准精度随训练上升**（70.7% → 85.5% → 83.7%），是继续训练的依据 |
| 摔倒波动 | it6000 曾出现 28 次（5σ），集中在"球在正后方、2 秒内、球没碰到"的单一分支；it6400 回 0。选检查点应用独立队列（`--seed 456`），报告数用 `seed 123` |
| 已判负的改动 | 距离型边界项（三次复现"用进球换出界"）、速度型/横向速度型罚项（H4 实测进球 −52、出界 +56）、ACCAD 数据集（奖励相容性差 60 倍）——详见实验记录 |
| `GOAL_SUCCESS_STEPS = 50` | 论文常量，注释写"=> episode success"，但实际几乎永不成立（球进门后机器人继续踢）。现以**验收几何**作为成功判据，两者不一致这件事本身记录在案 |
| 执行器限矩不一致 | K1 hip roll：URDF `effort_limit` = 43 Nm，motor cfg `effort_limit_sim` = 76 Nm；两套值都参与动作尺度计算（未核实硬件应取哪个） |
| 镜像损失原点 | 默认膝角左右不对称（0.50/0.53）使镜像漏掉 0.03/0.04 rad 的仿射项 |
| 数据集硬限位 | 论文数据 17/24 段含 K1 硬限位外动作（T1 踝 roll ±0.44 vs K1 ±0.345）—— 需要受 K1 约束的重定向，不能简单裁剪 |

## 14. 测试

`python -m pytest tests/ -q` → **151 passed / 84 subtests**（纯 CPU，不需要 GPU/IsaacSim）。
覆盖：奖励接线（每个 `mdp.<name>` 引用必须被导出，含 DoneTerm）、终止与进球几何的跨文件一致性、
放球分布（把真实方法提升出来跑，验证全圆采样）、评估字段语义（重置前快照）、
状态时序（进球/传送/感知延迟）、动作单位（弧度边界）、镜像归一化、训练编排（哈希门/完成文件）。

每个新增测试都做过**变异检验**：破坏实现或绕过接线时必须变红 —— 这条纪律在本项目中
抓到过真实缺陷（`GOAL_X` 用了没导入、奖励项接线漏了一半）。
