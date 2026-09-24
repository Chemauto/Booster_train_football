# sim2sim 接口复核、xfrc_applied 修复与 Foot Collision A/B（2026-09-24）

> 结论先行：
> 1. **`xfrc_applied` 力/矩写反是确定性 bug，且是"0 进球"的主因**。修复后 32 回合进球
>    0 → **56.3%**（seed 123 / it7200 / box 脚）。
> 2. **脚碰撞几何是极高敏感项**（A/B 摔倒率 22.7% ↔ 96.9%），但**方向与预期相反**：
>    MuJoCo 里 flat box 远好于 `Left_Foot.STL` convex hull。mesh 模式下脚**根本不产生
>    地面接触**，机器人 1.2–2.2 s 自由落体摔倒（123/124 为 before_touch）。
> 3. 因此剩余 ~23% 摔倒 gap **不能**靠"把 box 换回 STL hull"来修；box 是当前更可用的
>    近似。下一步应修 mesh 接触（或换 sole 薄片/圆角），并在 box 上扫 friction/solref。
> 4. 79 维 obs / 关节映射 / 50 帧栈 / action scale / PD / T-N / 延迟 / 控制频率等接口
>    本轮复核**未发现新问题**，摔倒 gap 不是低级接口错误。

代码提交：`73134cd deploy: fix xfrc force/torque swap, foot-collision A/B, fall-phase stats`

## 一、本轮已排除的接口项（复核通过）

以下与训练侧逐项对证，暂不怀疑：

| 项 | 状态 |
|---|---|
| 79 维 observation 顺序 | ✅ |
| real ↔ sim 关节映射 | ✅ |
| 50 帧 history stack | ✅ |
| action scale | ✅ |
| PD kp/kd | ✅ |
| T-N 转矩-转速限制 | ✅ |
| policy 50 Hz / MuJoCo 2 ms × 10 / Isaac 5 ms × 4 | ✅ |
| 10 ms actuator delay | ✅ |
| `root_ang_vel_b` 坐标系（MuJoCo free joint 角速度本体） | ✅ |

## 二、确定性 bug：`xfrc_applied` 力/矩写反

位置：`deploy/tasks/kick_amp/kick_amp_mujoco.py` `ctrl_step()`。

MuJoCo 布局是 `xfrc_applied[body] = [Force(3), Torque(3)]`，修复前代码：

```python
# ball rolling resistance  —— 写到了 Torque[0:2]，球根本没吃到 0.2 N 滚阻
self.mj_data.xfrc_applied[self.ball_bid, 3:5] = -self._friction_force * vxy / speed
# trunk push —— 力/矩完全对调
self.mj_data.xfrc_applied[self.trunk_bid, 3:5] = self._push_xy
self.mj_data.xfrc_applied[self.trunk_bid, 0:3] = self._push_torque
```

修复后（与训练 `mdp/commands.py:459-489` 的 `set_external_force_and_torque(..., is_global=True)` 对齐）：

```python
self.mj_data.xfrc_applied[self.ball_bid, :] = 0.0
self.mj_data.xfrc_applied[self.ball_bid, 0:2] = -self._friction_force * vxy / speed
self.mj_data.xfrc_applied[self.trunk_bid, :] = 0.0
if push_active:
    self.mj_data.xfrc_applied[self.trunk_bid, 0:2] = self._push_xy
    self.mj_data.xfrc_applied[self.trunk_bid, 3:6] = self._push_torque
```

影响：球在被施加旋转力矩而不是平移滚阻，踢球轨迹全错 → 与"能踢到球但 0 进球"一致。
默认 `push=False` 时机器人本体不吃这个错误扰动，**解释进球，解释不了摔倒**。

### 2.1 修复后 32 回合（seed 123 / it7200 / box / perfect 感知 / 10 ms 延迟）

| 指标 | 修复前（v3, it7200） | **修复后** |
|---|---|---|
| 进球 | **0/32** | **18/32 (56.3%)** |
| 摔倒 | 9/32 (28%) | 8/32 (25.0%) |
| 出界 | 2/32 | 6/32 (18.8%) |
| 平均存活 | 21.3 s | 9.2 s* |
| 触球代理 (0.30 m) | 37.5% | 65.6% |
| 物理球-机接触 | — | 71.9% |

\* 存活变短是进球/出界提前终止，不是变差。
对照 IsaacSim 验收（it6400, 256 场景）：进球 76.2%、摔倒 0–2%。
**进球已接近；摔倒 gap 仍在。**

结果 JSON：`/tmp/kick_amp_xfrcfix_e32_seed123_it7200.json`

## 三、摔倒时刻：fall before touch / after touch

口径：`ball_robot_in_contact()`（MuJoCo `mj_data.contact`，球 geom ↔ 任一非 world 机器人 geom）。
`fall` 含 `high_velocity` 终止。

### 3.1 32 回合（seed 123 / box）

| | 次数 |
|---|---|
| **before_touch**（步态失稳） | 3 |
| **after_touch**（触球后失稳） | **5** |

### 3.2 128 回合（seed 128 / box）

| | 次数 |
|---|---|
| **before_touch** | 12 |
| **after_touch** | **17** |

解读：**触球后失稳略占多数**（踢球接触瞬间稳定性 / ball-foot dynamics 是大头），
但 before_touch 接近一半，其中有一簇 1.7–2.5 s 的早摔（180°/−90° 航向居多）——
纯步态/落脚问题也真实存在，不能全推给踢球。

## 四、Foot Collision A/B（单变量，seed 128，各 128 回合，it7200，xfrc 已修）

A = 当前 flat box；B = `Left_Foot.STL` / `Right_Foot.STL` convex hull（URDF collision 同源）。
其余参数全部不动（摩擦、solref、延迟、感知、checkpoint 均一致）。

| | **A: box** | **B: STL convex hull** |
|---|---|---|
| 进球 | **70/128 (54.7%)** | **0/128 (0%)** |
| 摔倒 | 29/128 (**22.7%**) | 124/128 (**96.9%**) |
| 出界 | 29/128 (22.7%) | 0 |
| timeout | 0 | 4 |
| fall before touch | 12 | **123** |
| fall after touch | 17 | 1 |
| 触球代理 | 67.2% | 2.3% |
| 物理球-机接触 | 75.0% | 0.8% |
| 平均存活 | 10.2 s | **2.7 s** |
| 摔倒时长中位 | 3.2 s | **1.6 s**（1.12–4.64 s） |

结果 JSON：`/tmp/kick_amp_footAB_e128_seed128_{box,mesh}.json`

### 4.1 具体问题：mesh 模式下脚没有地面接触（已修复，根因是 `body_contype`）

> **根因（已定位并修复）**：MuJoCo 在**编译期**把 geom 的 `contype/conaffinity` 折叠进
> `body_contype/body_conaffinity`，broadphase 先看 **body** 级开关。XML 里
> `left_foot_mesh_col` 写了 `contype="0"`，于是 ankle body 的 `body_contype=0`，
> **整个 body 被排除出碰撞**；运行时把 `geom_contype` 改回 1 **不会**更新
> `body_contype`，mesh 对地面永远不产生接触。
>
> 最小对照（单 mesh + plane，40 步）：
>
> | 模式 | body_contype | 接触 |
> |---|---|---|
> | 编译 `contype=1` | 1 | `floor\|mcol` ✅ |
> | 编译 `contype=0`，运行时改 1 | **0** | **无** ❌ |
> | 编译两个都开，运行时关掉不用的那个 | 1 | 只剩选用的那个 ✅ |
>
> **修复**：两种脚 geom 都以 `contype=1 conaffinity=1` 编译，A/B 切换改为
> 运行时把**不用的那个** `geom_contype=0`（关闭是每步生效的，打开不是）。
> 修复后站姿冒烟：`field|left_foot_mesh_col` / `field|right_foot_mesh_col` 均出现，
> `trunk_z 0.550 → 0.530`，不再自由落体。
>
> 提交：`deploy: compile both foot collision geoms; runtime-select via geom_contype`


场景级诊断（spawn 后 `mj_forward`，ball 摆在原点仅作对照）：

**box 模式** —— 有地面接触（站立）：

```
ncon=6
  field <-> left_foot_box   dist=-0.00171   ← 脚-地接触（轻微侵入 1.7 mm，可站立）
  field <-> right_foot_box  (同上)
```

**mesh 模式** —— **完全没有脚-地接触**：

```
ncon=2
  ball_collision <-> g52
  ball_collision <-> g64
  （无任何 field <-> *_foot_mesh_col）
```

再用默认 crouch 站姿跑 50 个 policy step（mesh 模式）：

```
step   0 trunk_z=0.548  ncon=0  foot_pairs=[]
step  10 trunk_z=0.444  ncon=2  foot_pairs=[]
step  20 trunk_z=0.444  ncon=3  foot_pairs=[]
step  49 trunk_z=0.429  ncon=4  foot_pairs=[]
```

- `foot_pairs` 始终为空：**`left_foot_mesh_col` / `right_foot_mesh_col` 从未进入 contact 列表**；
- `trunk_z` 0.548 → 0.429 持续下沉 = 自由落体，与 A/B 里 1.2–2.2 s 早摔完全吻合。

world AABB（spawn）：

| geom | world z |
|---|---|
| `left_foot_box` | **[-0.0017, +0.0573]**（sole 略入地，正常） |
| `left_foot_mesh_col` | **[+0.0025, +0.0802]**（sole 浮在地面上方，且包到 +8 cm） |

上述 mesh 浮空/无接触是 `body_contype=0` 的**症状**，不是几何本身的问题
（编译期打开后，同一 `Left_Foot.STL` hull 能正常 `floor|mcol` 接触）。
几何上仍有差异，但那是修复后的 A/B 要量的东西，不是"站不住"的原因：

1. mesh hull 的 sole 比 box 低/高约 4 mm 量级（`mesh_pos/mesh_quat` 被 MuJoCo 主轴对齐过）；
2. hull 把整只脚包到 z≈+8 cm（box 只有 3.6 cm 厚 sole plate），convex hull 会填平
   脚底凹陷（`Left_Foot.STL` 底面并非平面），着地截面更接近弧面/楔形。

box 之所以"碰巧"有接触：`left_foot_box` 在 XML 里**默认 `contype=1`**，
ankle body 的 `body_contype` 由它撑起来；mesh 模式把 box 关掉、mesh 打开，
body 级仍在，但 mesh geom 在编译期是关的 → 依旧无接触。这是 A/B 开关路径的 bug，
不是 MuJoCo mesh-plane 接触本身不可用。

### 4.2 结论修正（A/B 数字在开关修复后作废，需重跑）

- ⚠️ **4 节的 22.7% ↔ 96.9% 对比是开关 bug 下的伪结果**：mesh 组不是"接触更差"，
  而是**根本没接触**。该组数字只能证明开关有 bug，**不能**用来给脚几何排序。
- ✅ 脚几何仍是高敏感候选（box sole 平面 vs hull 填平凹陷），但**必须在开关修复后
  重跑 A/B** 才能定量。
- ❌ 同样不能据此说"STL hull 不如 box"。

## 五、当前原因排序（修订）

| # | 原因 | 证据 | 权重 |
|---|---|---|---|
| ① | **脚-地接触模型**（box vs STL hull，待重测） | 开关 bug 修复前无法定量；几何差真实存在 | ⭐⭐⭐⭐⭐ |
| ② | friction / contact solver（MuJoCo 1.0 vs 训练材质；solref 0.001） | 未扫 | ⭐⭐⭐⭐ |
| ③ | checkpoint 缺 physics DR（0% 摔是 in-distribution） | it6400 0 fall 仅在 nominal PhysX | ⭐⭐⭐⭐ |
| ④ | 踢球接触瞬间稳定性（after_touch 17/29） | fall 统计 | ⭐⭐⭐ |
| ⑤ | 踝并联用串联铰链近似 | 结构性 | ⭐⭐ |
| ⑥ | observation / 控制接口 | 本轮复核通过 | ⭐ |

已修复（不再是原因）：`xfrc_applied` 力/矩写反；foot A/B 开关的 `body_contype` 编译期问题。

## 六、下一步（按性价比）

1. **重跑 Foot A/B**（开关已修，128 回合 / seed 128）—— 之前的 mesh 组作废：
   ```bash
   python3 deploy/sim2sim_kick_amp.py --episodes 128 --seed 128 \
       --checkpoint kick_amp_it7200_policy.pt --foot-collision box
   python3 deploy/sim2sim_kick_amp.py --episodes 128 --seed 128 \
       --checkpoint kick_amp_it7200_policy.pt --foot-collision mesh
   ```
2. 可加第三档 **`Left_Foot_Collision.STL`**（60 三角简化壳，脚底更平）或
   **sole 薄片 + 圆角胶囊**，与 box / full STL 并列。
3. **在 box 上扫 friction 0.5 / 0.7 / 1.0 + solref** —— 打剩余摔倒的主路径。
3. **after_touch 专题**：17/29 触球后失稳，需要看踢球瞬间的支撑脚 CoP / 角动量
   （`--view` 回放 seed 128 中 `fall_phase=after_touch` 的回合）。
4. **训练侧**：用默认 profile（虚拟感知 + DR）续训，提升鲁棒性；
   **不要**再用 nominal PhysX 的 0% 摔倒当验收。
5. **真机**：仍缺 ball/goal/odom 输入通道，先不动真机足球策略。

## 七、附：本轮改动与产物

提交 `73134cd`：

| 文件 | 改动 |
|---|---|
| `deploy/tasks/kick_amp/kick_amp_mujoco.py` | xfrc 力/矩修复；`_set_foot_collision` A/B 开关（后修：两 geom 编译期全开、运行时关掉不用的那个）；`ball_robot_in_contact`；fall before/after 统计 |
| `deploy/tasks/kick_amp/scene/k1_soccer_14x9.xml` | 脚 box 命名 + 增加 `*_foot_mesh_col`（默认关闭） |
| `deploy/tasks/kick_amp/__init__.py` | `foot_collision: str = "box"` |
| `deploy/sim2sim_kick_amp.py` | `--foot-collision box\|mesh`；输出文件名带 A/B 标签 |
| `deploy/tasks/__init__.py` | **新增**：让 `tasks.kick_amp` 解析到本副本（原先因缺 `__init__.py` 被 `booster_deploy/tasks` 同名包抢走，改动实际没生效——这是本轮第一个 smoke 假失败的根因） |

复现：

```bash
# 32 回合验收（xfrc 修复后基线）
python3 deploy/sim2sim_kick_amp.py --episodes 32 --seed 123 \
    --checkpoint kick_amp_it7200_policy.pt

# Foot A/B
python3 deploy/sim2sim_kick_amp.py --episodes 128 --seed 128 \
    --checkpoint kick_amp_it7200_policy.pt --foot-collision box
python3 deploy/sim2sim_kick_amp.py --episodes 128 --seed 128 \
    --checkpoint kick_amp_it7200_policy.pt --foot-collision mesh
```

JSON 产物（本机 `/tmp/`）：
- `kick_amp_xfrcfix_e32_seed123_it7200.json`
- `kick_amp_footAB_e128_seed128_box.json`
- `kick_amp_footAB_e128_seed128_mesh.json`
