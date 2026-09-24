# 服务器重训手册：带虚拟感知的 K1 kick_amp（2026-09-24）

> 目标：训出**能处理丢检**（`ball_obs.flag=0`）的 checkpoint。
> 现有 it6400 / it7200 是 `perfect_perception=True` 训的，虚拟感知下 **32/32 全摔、0 进球**
> （见 `2026-09-24-xfrc-and-foot-ab.md` §五），不能直接上真机。
> 本文给：训练来历 → 服务器配方 → 验收判据 → 与 it7200 的对照基线。

---

## 一、之前是怎么训的（结论先行）

| 问题 | 答案 |
|---|---|
| 能力是从 0 训出来的吗？ | **是**（9/16–18，两阶段 approach→soccer，无预训练步态；BC 预训练是负结果已弃） |
| it6400 / it7200 是从 0 吗？ | **不是**。是续训链尾巴（见下表） |
| 见过感知噪声吗？ | **没有**。全程 `--perfect_perception` |
| 见过物理 DR 吗？ | **没有**。全程 `--nominal_physics` |

### 1.1 续训链（`logs/rsl_rl/k1_kick_amp_soccer/*/launch_args.json`）

```
…从0（9/16–18）… → model_6000
   └─ 2026-09-19_13-45-20 (H2)   6000 → 6400   ★ it6400  76.2% 进球 / 0 摔
        ├─ 2026-09-19_14-15-10 (H3/H4)  6400 → 6800
        └─ 2026-09-20_13-17-47 (I s1)   6400 → 6800
             └─ 2026-09-20_13-49-01 (I s2)  6800 → 7200   ★ it7200  86.3% / 2.0% 摔 / 7.0% 出界
                  └─ 2026-09-20_14-20-26 (I s3)  7200 → 7600
```

### 1.2 交付 ckpt 的启动参数（全部相同，benchmark profile）

| 参数 | 值 | 含义 |
|---|---|---|
| `--perfect_perception` | **True** | 球真值 + `flag≡1` |
| `--nominal_physics` | **True** | 关掉全部物理 DR |
| `--near_ball` | True | 球出生 0.8–2.0 m |
| `--ball_spawn_bearing_deg` | 180 | 方位全圆（H1 关键改动） |
| `--task_weight_floor` | 1.0 | 任务奖励满权 |
| `--num_envs` | 1024 | 本机 8 GB 卡上限 |
| `--max_iterations` | 400 | 每段 |
| `--seed` | 42 | |
| `--motion_dir` | `amp_paper_waistfix` | 24 段 AMP |
| `--training_phase` | soccer | |

**这就是感知缺口的根源**：策略从未见过 `flag=0`，也从未见过物理随机化。

### 1.3 已固化进代码、从 0 也会自动拿到的东西

今天代码里已经包含 9/16–9/20 全部消融结论，**从 0 起点比当时的基线好得多**：

- 进球即终止回合（否则进球后追球出界）
- `pos_still` 豁免半径 0.3、`penalize_pos=0.45`（掐掉"停车位"）
- `task_weight_floor` 满权机制
- `ball_spawn_bearing` 可配全圆
- AMP 数据集 `amp_paper_waistfix`（腰 yaw 修正）
- 接触传感器索引 / yaw / 归一化与镜像可交换 等 10 类修复
- `boundary` / `ball_lateral` 保持关闭（实测无效或有害）

---

## 二、服务器环境清单

```text
Isaac Lab 2.2 + Isaac Sim 5.0        # README 推荐 conda 装法
Booster_train_football/              # 本仓库 = 全部所需（K1 模型 + AMP + 球场已内置）
```

**`$ASSETS` = 本仓库的 `assets/`**（自 2026-09-24 起自带，见 `assets/README.md`），
**不需要**再克隆 `booster_assets`。

安装：

```bash
# 1) IsaacLab（略，见官方安装指南）
# 2) 本仓库（一个仓库搞定）
git clone https://github.com/Chemauto/Booster_train_football.git
cd Booster_train_football
git log --oneline -1                 # 应看到 3c21feb 或更新

# 3) 本任务包（用装了 IsaacLab 的解释器）
python -m pip install -e source/booster_train

# 4) 定义 ASSETS
export ASSETS=$PWD/assets
```

冒烟：

```bash
python -c "import sys; sys.path.insert(0,'assets'); import booster_assets; print(booster_assets.BOOSTER_ASSETS_DIR)"
ls $ASSETS/motions/K1/amp_paper_waistfix/walk | wc -l   # 12
python scripts/list_envs.py | grep KickAMP
python -m pytest tests/ -q          # 期望全绿
```

### 显存参考

| num_envs | 显存（本机实测/估算） |
|---|---|
| 1024 | ≈ 6.1 GB |
| 4096 | cfg 默认；≥24 GB 稳 |
| 8192+ | 论文用 16384；卡大就往上加 |

每迭代每 env 采样 24 步（`horizon_length`）；`20000 iters × 4096 env` 是论文量级预算。

---

## 三、主配方：从 0 + default profile（推荐）

```bash
ASSETS=/path/to/booster_assets
python scripts/rsl_rl/train_kick_amp.py --task Booster-K1-KickAMP-v0 --headless \
    --num_envs 4096 --max_iterations 20000 --seed 42 \
    --experiment_name k1_kick_amp_perception \
    --training_phase soccer \
    --task_weight_floor 1.0 \
    --ball_spawn_bearing_deg 180 \
    --near_ball \
    --motion_dir $ASSETS/motions/K1/amp_paper_waistfix
```

**关键：不要加 `--perfect_perception`，也不要加 `--nominal_physics`。**

| 选项 | 作用 | 本次 |
|---|---|---|
| `--perfect_perception` | 真球 + flag≡1 | **禁止**（加了等于白训） |
| `--nominal_physics` | 关物理 DR | **禁止**（顺带补 sim2real） |
| `--near_ball` | 球 0.8–2.0 m | 保留（对齐验收几何） |
| `--ball_spawn_bearing_deg 180` | 方位全圆 | 保留（H1：摔倒 11.7%→1.2%） |
| `--task_weight_floor 1.0` | 任务满权 | 保留 |
| `--checkpoint` | 续训 | **不填 = 从 0** |
| `--reset_optimization` | 重置 critic/优化器 | 仅续训时考虑 |

产物（`logs/rsl_rl/k1_kick_amp_perception/<stamp>/`）：

- `model_XXXX.pt` — AmpRunner 检查点（续训用）
- **`policy.pt`** — TorchScript 导出（sim2sim / 真机用，每 100 iter 自动刷新）
- `launch_args.json` / `agent_cfg.yaml` / `env_cfg.txt` / `git_info.txt` / `source_snapshot.tgz`

### 3.1 可选：分段跑 + 自动评估（推荐长跑用）

```bash
python scripts/run_kick_amp_stages.py \
    --output logs/perception_run \
    --stages 5 --updates 400 --num_envs 4096 --eval_envs 256 \
    --seed 42 --task_weight_floor 1.0 --ball_spawn_bearing_deg 180 \
    --training_profile default \
    --motion_dir $ASSETS/motions/K1/amp_paper_waistfix
```

- `--training_profile default` = **完整随机化 + 感知噪声**（`benchmark` 才是老的完美感知）
- `--checkpoint` 不填 = 从 0；填 = 续训
- 每段 `训练 → 独立评估` 串行占 GPU，`status.json` 记全程；源码/motion 被哈希锁定，跑动中改代码会自动停

### 3.2 变体（不推荐做主线，仅供对照）

| 方案 | 差异 | 风险 |
|---|---|---|
| **A. 从 0 + default（主推）** | 上表 | 要跑满预算，但干净 |
| B. 从 it7200 续训 + 开感知 | `--checkpoint .../model_7200.pt`，去掉 `--perfect_perception`，建议加 `--reset_optimization` | 完美感知先验可能赖着不走，丢检行为学不扎实 |
| C. 先 benchmark 冒烟服务器管线 | 先加 `--perfect_perception --nominal_physics` 跑 500 iter 确认能涨分 | 多花 1–2 h，换确定性 |

**要走 B 时务必记得**：it7200 在 `logs/rsl_rl/k1_kick_amp_soccer/2026-09-20_13-49-01/model_7200.pt`
（备份 `models_backup/k1_kick_amp_it7200_model.pt`）。

---

## 四、验收判据（两条协议都要跑）

### 4.1 老协议 —— 与 it7200 的 86.3% 可比

```bash
python scripts/evaluate_kick_amp.py --headless --checkpoint <CKPT> \
    --scenario soccer --num_envs 256 --steps 1500 --seed 123 \
    --perfect_perception --output logs/eval_perfect.json
```

| 指标 | 达标线 | it7200 基线 |
|---|---|---|
| 进球 | ≥77 (30%) | **221 (86.3%)** |
| 触球 | ≥154 (60%) | 251 |
| 摔倒 | ≤5 (2%) | **5 (2.0%)** |
| 出界 | ≤25 (10%) | 18 (7.0%) |

**判据 ①**：不能把控制能力练丢 —— 进球 ≥30%、摔倒 ≤2%。

### 4.2 新协议 —— 本次真正目标

```bash
python scripts/evaluate_kick_amp.py --headless --checkpoint <CKPT> \
    --scenario soccer --num_envs 256 --steps 1500 --seed 123 \
    --output logs/eval_virtual.json
# 不加 --perfect_perception → 虚拟感知（FOV 门控 + 距离丢检 + 噪声 + 6 步延迟）
```

| 指标 | it7200 基线（灾难） | **成功线** |
|---|---|---|
| 进球 | **0 / 256** | ≥30% |
| 摔倒 | **256 / 256 (100%)** | **≤20%**（越低越好） |
| 平均存活 | 2.0 s（中位 1.4 s） | 接近完美感知档 |
| fall before touch | 32/32 | 应有 after_touch 出现（说明摸到球了） |

**判据 ②**：虚拟感知下摔倒 **绝不能再 100%** —— 这是本次重训的成败线。

### 4.3 sim2sim 复测（MuJoCo，含视频）

```bash
python3 deploy/sim2sim_kick_amp.py --episodes 32 --seed 123 \
    --checkpoint <NEW_policy.pt 改名后放入 deploy/tasks/kick_amp/models/> \
    --perception virtual --out /tmp/new_virtual_e32.json

python3 deploy/render_sim2sim.py --episodes 3 --seed 128 \
    --perception virtual --checkpoint <NEW> --out logs/videos/sim2sim_new
```

对照表（同 seed 123 / 32 回合 / box 脚）：

| | perfect | virtual（it7200） | virtual（**新 ckpt 目标**） |
|---|---|---|---|
| 进球 | 56.3% | **0%** | ≥30% |
| 摔倒 | 25.0% | **100%** | ≤25% |
| 平均存活 | 9.2 s | 2.0 s | >5 s |

---

## 五、导出与交接

训练过程会自动写 `policy.pt`（JIT：`forward(raw_obs[1,79], stacked[1,50,79]) → 22 关节目标`）。
训练结束后：

```bash
cp logs/rsl_rl/k1_kick_amp_perception/<stamp>/policy.pt \
   deploy/tasks/kick_amp/models/kick_amp_perception_policy.pt
cp logs/rsl_rl/k1_kick_amp_perception/<stamp>/model_XXXX.pt \
   models_backup/k1_kick_amp_perception_model.pt
```

sim2sim 用 `--checkpoint kick_amp_perception_policy.pt`。

---

## 六、与 it7200 对照的完整基线数字

| 维度 | it6400 | it7200 | 本次目标 |
|---|---|---|---|
| 训练方式 | 从 0 累积后续训 | 从 6800 续训 | **从 0** |
| perfect_perception | True | True | **False** |
| nominal_physics | True | True | **False** |
| Isaac 进球（256/seed123） | 195 (76.2%) | 221 (86.3%) | ≥30%（完美感知档） |
| Isaac 摔倒 | 0 (0%) | 5 (2.0%) | ≤2% |
| **虚拟感知 摔倒** | 未测 | **32/32 (100%)** | **≤20%** |
| **虚拟感知 进球** | 未测 | **0%** | **≥30%** |
| MuJoCo perfect 进球 | 0%* | 56.3% | — |
| MuJoCo perfect 摔倒 | 41%* | 25.0% | — |
| MuJoCo virtual 进球 | 未测 | **0%** | ≥20% |
| MuJoCo virtual 摔倒 | 未测 | **100%** | ≤30% |

\* it6400 的 MuJoCo 数字是 **xfrc 修复前** 的旧值，仅作历史记录，不再可比。

---

## 七、注意（踩过的坑）

1. **不要**用 `--perfect_perception` 训"能上真机"的模型 —— 那是控制基准专用。
2. **不要**在部署端给丢检填"上次位置"或真值 —— 训练契约是 `(0,0,flag=0)`，
   记忆靠 50 帧 history stack；部署端造假 = 又一次 obs 分布偏移。
3. `boundary` / `boundary_outward` / `ball_lateral` **保持 0**（实测无效或有害，见实验记录 §4.3/4.5/4.10）。
4. 选检查点用**独立队列**（如 `--seed 456`），报告数再用协议 `seed 123` —— it6000 摔倒尖峰是这么来的教训。
5. 渲染与训练不能同时跑；渲染 `--num_envs ≤ 13`。
6. 本仓库 `deploy/` 侧 2026-09-24 修了两个 sim2sim bug（`xfrc_applied` 力/矩写反、
   foot A/B 的 `body_contype` 编译期问题），服务器训练**不依赖**这些，但 sim2sim 验收必须用修复后的代码。

---

## 八、一页速查

```bash
# 训（从 0，default profile = 虚拟感知 + 物理 DR）
python scripts/rsl_rl/train_kick_amp.py --task Booster-K1-KickAMP-v0 --headless \
  --num_envs 4096 --max_iterations 20000 --seed 42 \
  --experiment_name k1_kick_amp_perception --training_phase soccer \
  --task_weight_floor 1.0 --ball_spawn_bearing_deg 180 --near_ball \
  --motion_dir $ASSETS/motions/K1/amp_paper_waistfix

# 验 ①：完美感知（和 it7200 可比）
python scripts/evaluate_kick_amp.py --headless --checkpoint <CKPT> \
  --scenario soccer --num_envs 256 --steps 1500 --seed 123 --perfect_perception

# 验 ②：虚拟感知（成败线）
python scripts/evaluate_kick_amp.py --headless --checkpoint <CKPT> \
  --scenario soccer --num_envs 256 --steps 1500 --seed 123

# 验 ③：sim2sim + 录像
python3 deploy/sim2sim_kick_amp.py --episodes 32 --seed 123 \
  --checkpoint <NEW_policy.pt> --perception virtual
python3 deploy/render_sim2sim.py --episodes 3 --seed 128 \
  --perception virtual --checkpoint <NEW_policy.pt> --out logs/videos/sim2sim_new
```

相关文档：
- 训练来历与全部消融：`docs/reports/2026-09-19-experiment-log.md`、`2026-09-19-soccer-3arm.md`
- 观测契约 / 特权信息：`docs/reports/2026-09-20-obs-deployability.md`
- sim2sim 崩溃数字：`docs/reports/2026-09-24-xfrc-and-foot-ab.md`
- 配置参考：`Project.md`；怎么跑：`QUICKSTART.md`
