# deploy/ — K1 kick_amp MuJoCo sim2sim（booster_train 内副本）

本目录是 `booster_deploy` 仓库中 kick_amp sim2sim 移植的**副本**（2026-09-20），
让训练仓库自包含可交付。原始实现仍在 `/data/rl_robot/BoosterRobotics/booster_deploy/tasks/kick_amp/`
（两边文件相同；本副本的场景路径改为随文件位置解析）。

## 运行（任意目录，conda base python 即可，CPU）

```bash
python3 deploy/sim2sim_kick_amp.py --episodes 32 --checkpoint kick_amp_it7200_policy.pt
python3 deploy/sim2sim_kick_amp.py --view --checkpoint kick_amp_it7200_policy.pt   # 交互观看
```

依赖：mujoco、torch（conda base 已有）；`booster_deploy` 控制器框架经 sys.path
从兄弟仓库导入；`booster_assets` 未安装时自动回落到源码路径。

## 内容

| 文件 | 说明 |
|---|---|
| `sim2sim_kick_amp.py` | 入口（headless 批跑 / `--view` 交互 / `--perception virtual`） |
| `tasks/kick_amp/kick_amp.py` | 79 维观测 1:1 复刻 + 虚拟感知移植 + 50 帧归一化历史栈 |
| `tasks/kick_amp/kick_amp_mujoco.py` | 按名寻址控制器 + T-N 转速曲线限矩 + 子步延迟 + 回合协议 |
| `tasks/kick_amp/scene/k1_soccer_14x9.xml` | K1 + 14×9 球场合并场景（球门/围墙纯视觉，与训练判罚一致） |
| `tasks/kick_amp/models/` | `it6400`（76.2% 进球 ckpt）与 `it7200`（**达标 ckpt**：86.3% / 2.0% 摔倒 / 7.0% 出界，256 场景 seed123） |

## 当前转移水平与实机前置（详见 `../docs/reports/2026-09-20-obs-deployability.md`）

- it6400：MuJoCo 32 回合 摔倒 41% / 存活 16.2s；it7200：摔倒 28% / 存活 21.3s
 （IsaacSim 参照：摔倒 0-2%）。会走、会追球、能踢，**sim2sim 摔倒 gap 是实机头号问题**
 （候选：脚 box 碰撞 vs URDF convex hull、有效摩擦、踝并联近似、接触柔顺性）。
- **实机部署两个前置**：① 现有 ckpt 均为 perfect_perception 训练，需虚拟感知重训；
  ② 上面的摔倒 gap。感知输入通道（相机+YOLO+BEV）在 booster_deploy 侧尚不存在。

## 关键接口语义（曾翻车，勿改错）

1. `policy.pt` 双输入：`forward(raw_obs[1,79], stacked[1,50,79])` —— 第一参数 raw
   （obs_norm 冻在图内），栈内是**归一化后**的 obs；除回合首步（栈全零）外，
   **先压当前 obs 再推理**。
2. 延迟单位是 2ms 子步（训练 5ms）：10ms = 5 子步，回合首步无延迟。
3. 力矩限幅是 T-N 转速曲线（不是常数），参数在 `__init__.py` 的 `_K1_KICK_VMAX/VKNEE`。
