# assets/ — 本仓库自带的 K1 资产（自包含，无需外部 booster_assets）

训练 + sim2sim 需要的全部资产都在这里，**克隆这一个仓库即可**，不必再装
`https://github.com/BoosterRobotics/booster_assets`。

```text
assets/
├── robots/K1/                     # K1 22-DoF URDF + meshes（约 21 MB）
│   ├── K1_22dof.urdf              #   IsaacLab 训练用
│   ├── K1_22dof.xml               #   MuJoCo 用
│   └── meshes/                    #   52 个 STL（含 Left/Right_Foot.STL）
├── motions/K1/amp_paper_waistfix/ # AMP 参考动作 24 段（50 Hz）
│   ├── walk/  12 段 *.npz        #   全向行走（含转身、侧移）
│   └── kick/  12 段 *.npz        #   官方踢球
├── scene/                         # 14×9 m 球场（sim2sim 用）
│   ├── soccer_field_14x9.xml
│   ├── grass_14x9.png
│   ├── goal.stl
│   └── footaball.stl
└── booster_assets/                # import 兼容包（替代外部 booster_assets）
    ├── __init__.py                #   BOOSTER_ASSETS_DIR = 本目录的父目录
    └── motions.py                 #   K1_JOINT_NAMES / T1_JOINT_NAMES
```

## 路径怎么解析的

`booster_train/__init__.py` 在 import 时把本目录插到 `sys.path[0]`，
于是所有既有代码里的

```python
from booster_assets import BOOSTER_ASSETS_DIR
from booster_assets.motions import K1_JOINT_NAMES
```

会解析到这里的 `booster_assets/` shim，`BOOSTER_ASSETS_DIR` 指向 `assets/`，
拼出的路径（`robots/K1/...`、`motions/K1/amp_paper_waistfix`）都在本目录下。

deploy 侧的场景 XML 也用相对路径（`../../../../assets/...`），不依赖绝对路径。

手动验证：

```bash
python -c "import sys; sys.path.insert(0,'assets'); import booster_assets; print(booster_assets.BOOSTER_ASSETS_DIR)"
# 期望输出 .../booster_train/assets
```

## 不包含什么

上游 `booster_assets` 还有 T1 / T2 机器人模型、其他任务的 motion
（`k1_kick_full.npz`、`k1_fight_001.npz` 等），本任务用不到，故未收录。
若要跑 `kick_mimic` / `beyond_mimic` 等其它任务，仍需外部 `booster_assets`；
shim 会在本目录找不到时由调用方自行回落。

数据来源：论文 2511.03996 自带重定向 CSV → `scripts/convert_paper_data.py`
（含 T1 腰 yaw 修正，故名 `waistfix`）。转换一次后产物即为本目录内容。
