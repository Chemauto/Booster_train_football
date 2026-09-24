# ⚽ K1 自主踢球 · QUICKSTART

K1 人形机器人**自主走向足球、把球踢进 7 m 外球门**（论文 2511.03996 的 AMP 端到端复现）。

本页只讲怎么跑 🏃 —— 原理与全部配置见 **[Project.md](Project.md)**，
完整实验过程（每次因为什么错、改了什么、效果如何）见
**[docs/reports/2026-09-19-experiment-log.md](docs/reports/2026-09-19-experiment-log.md)**。

> 下文 `python` 请替换为你装了 Isaac Lab 的解释器（本机是
> `/home/xcj/miniconda3/envs/env_isaaclab/bin/python`）。
> `$ASSETS` = **本仓库自带的 `assets/`**（K1 模型 + AMP 动作 + 球场，见 [assets/README.md](assets/README.md)）。
> 不需要单独克隆 `booster_assets`。前置安装（Isaac Lab / `pip install -e source/booster_train`）见 [README](README.md)。

---

## 🎬 先看效果

四段演示视频（验收协议的四个方位：正前 / 左 / 正后 / 右）在 `logs/videos/`：

| 文件 | 球相对机器人 | 30 秒内进球 |
|---|---|---:|
| `01_front.mp4` | 正前方 | 3 |
| `02_left.mp4` | 左侧 | 2 |
| `03_behind.mp4` | **正后方**（先转身 180°） | 1 |
| `04_right.mp4` | 右侧 | 1 |

## ▶️ 播放策略

```bash
# 交互播放（带窗口，看它自己走位踢球）
python scripts/rsl_rl/train_kick_amp.py --task Booster-K1-KickAMP-v0-Play \
    --checkpoint <CKPT> --play --play_steps 1500 --num_envs 1

# 录成 mp4（无窗口，追踪相机跟拍机器人与球）
python scripts/render_kick_amp.py --headless --enable_cameras \
    --checkpoint <CKPT> --num_envs 13 --env_index 0 --steps 1500 \
    --out logs/videos/clip.mp4
```

仓库随附保留的最佳检查点（未提交，被 `.gitignore` 忽略）：

```
logs/rsl_rl/k1_kick_amp_soccer/2026-09-19_13-45-20/model_6400.pt   # 进球 76.2% / 摔倒 0%
logs/rsl_rl/k1_kick_amp_soccer/2026-09-19_14-15-10/model_6800.pt   # 进球 77.3%
```

⚠️ 渲染与训练**不能同时跑**，渲染 `--num_envs` **≤ 13**（详见文末）。

---

## 🏋️ 训练

### 🎯 要训多少轮

| 项 | 值 | 出处 |
|---|---|---|
| **总预算** | **20000 次迭代** | `ppo_cfg.py: max_iterations = 20000`（论文日程） |
| 步态课程交班 | **16000 次迭代** | `TASK_RAMP_STEPS = 16000 × horizon`，之后行走课程项衰减到 0 |
| 1 迭代 = | 1024 env × 24 步（`horizon_length = 24`） | |
| 实测耗时 | **约 3 s/迭代**（1024 env，8 GB 卡） | G/H 臂：400 迭代 ≈ 19 min |
| 全程 | 20000 迭代 ≈ **17 小时** | |

这是个**长训任务**：论文日程要到 16000 迭代才把行走课程完全交给任务奖励，总预算 20000。
仓库随附的检查点停在 **iteration 6800（约 34%）**，而进球率在 3600→6800 之间从 28.9% 升到 76%
且仍在上升 —— 想完全复现就继续往下训。

分段的总量就是 **N × M** 次迭代（例：`--stages 3 --updates 400` = 1200 迭代 ≈ 1.2 h）。

### 🧭 训练阶段：训练过程中策略在学什么

课程进度 `c = 迭代数 / 16000`（`TASK_RAMP_STEPS`）。行走奖励项按 **(1−c)** 衰减，
任务奖励组的**优势权重**按 **c** 上升（任务奖励本身不缩放，缩放发生在 PPO 合并优势时）：

| 迭代区间 | 奖励里谁在主导 | 策略在学什么 |
|---|---|---|
| 0 → 数千 | 行走课程项（`track_lin_vel_ball` 3.0、`feet_air_time_biped` 2.0、`feet_clearance` 1.0），任务优势权重 ×c ≈ 0 | 站稳、会走、不摔 |
| 中段（c 爬到 1，到 iteration 16000 交班完） | 行走项 ×(1−c) 渐弱，任务项渐强 | 朝球移动、触球 |
| 16000 → 20000 | 纯任务奖励 + AMP 风格 | 瞄准、把球送进门、边界控制 |

**`--task_weight_floor 1` 是绕开第一、二阶段的一步**：直接把任务优势权重提到满权。
本项目实测过不这么做的代价 —— 课程斜坡把 goal critic 的优势权重压到 **0.375**，
策略于是学会"停在球边不动"（撑到 episode 结束拿 survival）。当然也可以老老实实按 c 爬，
那就是"训满 16000 迭代才真正开始踢球"。

另有 `--training_phase`：`soccer`（正式任务，默认）或 `approach`（只用 walk 数据学走路，
任务奖励置零、任务优势权重为 0，用来先把步态隔离练好）。

### ⚙️ 参数含义

| 参数 | 含义 | 建议 |
|---|---|---|
| `--checkpoint <pt>` | 从该检查点续训；**不填 = 从零** | 续训时填 |
| `--num_envs` | 并行环境数（每迭代每 env 采样 24 步） | **1024**（8 GB 卡上限） |
| `--max_iterations` | 一次性训练的**总迭代数** | **20000** |
| `--stages` / `--updates` | 分段训练的段数 / 每段迭代数（总量 = 二者相乘） | 3 / 400 |
| `--eval_envs` | 每段评估场景数 | **256**（别用 64） |
| `--task_weight_floor` | 任务优势权重下限，1 = 满权（见上一节） | **1** |
| `--ball_spawn_bearing_deg` | 放球方位半角：默认只在机器人前方 ±46°；**180 = 全圆**，与验收协议的 0/±90/180° 对齐 | **180** |
| `--training_profile` | `benchmark` = 无物理随机化 + 完美感知 + 前方球（先隔离控制能力）；`default` = 完整随机化 + 感知噪声 | **benchmark** |
| `--motion_dir` | AMP 参考动作目录（需含 `walk/` 与 `kick/` 两个子目录） | `…/amp_paper_waistfix` |
| `--training_phase` | `soccer` 正式任务 / `approach` 只学走 | soccer |
| `--reset_optimization` | 保留 actor，重置 critic + PPO 状态 + 探索 sigma=0.15（奖励或回合结构发生阶段级变化时用） | 按需 |
| `--play` / `--play_steps` | 只播放不训练 | — |
| `--completion_file` | 训练结束写出**实际迭代计数**（分段编排器靠它判"这段真的跑完了"） | 分段时自动加 |
| `--boundary_weight` / `--boundary_outward_weight` / `--ball_lateral_weight` | 三个边界/横向速度罚项 | **保持 0**（已实测无效或有害） |
| 其余（`--task`、`--seed`、`--headless`、`--device`、`--experiment_name` …） | 任务 id / 随机种子 / 无窗口 / 设备 / 日志目录名 | 用默认 |

### 方式 A：一次性训练（最直观）

```bash
python scripts/rsl_rl/train_kick_amp.py --headless --num_envs 1024 \
    --max_iterations 20000 \
    --task_weight_floor 1 --ball_spawn_bearing_deg 180 \
    --training_profile benchmark \
    --motion_dir $ASSETS/motions/K1/amp_paper_waistfix
```

不填 `--checkpoint` 就是从零；每 200 迭代存一次，Ctrl-C 会保存后退出。

### 方式 B：分段训练（本仓库推荐，原因在下面三条）

```bash
python scripts/run_kick_amp_stages.py \
    --checkpoint <CKPT> --output logs/my_run \
    --stages 3 --updates 400 --num_envs 1024 --eval_envs 256 \
    --task_weight_floor 1 --ball_spawn_bearing_deg 180 \
    --training_profile benchmark \
    --motion_dir $ASSETS/motions/K1/amp_paper_waistfix
```

它只是把方式 A **切成几段串行跑**，每段结束跑一次 256 场景评估再继续。
推荐它不是因为分段本身更好，而是这台机器上的三个现实问题：

1. **17 小时的单进程很容易白跑**：中途 OOM / 崩溃会丢掉进行中的进度。
   本项目吃过静默失败的亏：一次 CUDA OOM 让某阶段只跑到 2000 次迭代（目标 2445），
   而当时只凭子进程返回码就继续了后续阶段 —— 于是那一段的"已完成"是假的。
   现在的编排器会核对训练子进程写出的**实际迭代计数**，不符即判该段失败。
2. **每段给你一个独立评估点**：得到"进球 vs 迭代"曲线而不是只有终点。
   "进球靠训练量能到 53.5% 且仍在升"、"摔倒集中在球在正后方"这两个结论都是靠曲线判断的，单点做不到。
3. **护栏**：`--motion_dir` 与 `source/`+`scripts/` 会被哈希固定，运行中的改动会让后续阶段自动停止，
   避免两套配置混进同一条曲线。

**运行期间不要改 `source/**/*.py` 或 `scripts/**/*.py`**（源码哈希校验会中止该次实验）。

**两个不在命令行里的关键参数**（都在 `env_cfg.py`）：

- `rewards.pos_still.params.penalize_pos` = 站桩判定阈值，**必须低于实测步速**
  （本机 0.623 m/s，当前 0.45）。给一个交不起的阈值 = 永久赶路压力，会把 reset 瞬态摔成 24%。
- `commands.soccer.goal_scale / ball_distance_scale / goal_distance_scale` = 15 / 50 / 500，
  任务奖励的三个尺度（在门内每步 / 球远离机器人 / 球靠近球门中心的势能）。

### 参考动作数据

`motions/K1/amp_paper_waistfix`（24 段：12 走 + 12 踢，50 Hz）由论文自带 CSV 转换而来：

```bash
python scripts/convert_paper_data.py --out $ASSETS/motions/K1/amp_paper_waistfix
```

数据不在本仓库内，随 `booster_assets` 分发。

---

## 📊 评估（256 场景验收协议）

```bash
python scripts/evaluate_kick_amp.py --headless --checkpoint <CKPT> \
    --scenario soccer --num_envs 256 --steps 1500 --seed 123 --perfect_perception \
    --output logs/eval.json
```

判读四个指标：**进球 ≥30% / 触球 ≥60% / 摔倒 ≤2% / 出界 ≤10%**。
JSON 里还有 `ball_end_zone_counts`（球终点分区）、`crossing_y_m`（越过门线时的横向位置）
和 `per_env`——按初始方位拆解用 `(env_id // 4) % 4`。

⚠️ **不要用 64 场景做决策**：它会把进球率低估约 2.7 倍（同一配置 1/64 vs 11/256）。

## 🧪 测试

```bash
python -m pytest tests/ -q      # 151 passed
```

---

## ⚠️ 8 GB 显卡的三条硬规矩

1. **同一时刻只能有一个 IsaacSim 实例** —— 两个会以 Mutex 崩溃。
2. **渲染时 `--num_envs` ≤ 13**（13 env + 相机 ≈ 7.2 / 8.2 GB）；训练 1024 env ≈ 6.1 GB。
3. **分段训练运行期间不要改 `source/**/*.py` 或 `scripts/**/*.py`**（哈希门会中止实验）。
