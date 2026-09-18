# 当前代码、ACCAD 数据与 model_5800 复核

日期：2026-09-18。继 `2026-09-17-amp-paper-review.md` 后重新读取当前文件并复测。当前默认数据已改为 `accad_markers`；不能继续用上次对旧文件的结论代替当前检查。证据及独立探针保存在 `logs/audit_followup_20260918/`。本轮未改训练实现、未覆盖任何动作数据；四元数对照只在审计目录使用临时配置。

## 已确认修复

- GP：runner 内部乘10，当前基类及 ppo_cfg 系数均为5，有效50。之前的有效500已消除。
- amp_paper：24段、5314帧。独立读取URDF逐关节方向性检查：角度硬超限0、速度硬超限0，全部有限，FK最大误差2.42e-7m。现有验收0 HARD/0 SOFT。中性手臂已改为nominal。
- 进球50帧触发重置时：真实Isaac四环境复测，原来的约-39至-105任务负奖励变为0，success标记仍为1。大负奖励已修复，但状态同步未完整修复，见下。
- 默认pos_still的配置已显式指定penalize_pos=0.7；上次发现的函数默认0.1不再是当前soccer入口的有效阈值。approach仍有独立覆盖，课程仍存在。

## P1：新 accad_markers 的四元数偏移写反，目标朝向翻转180度

`/data/rl_robot/GMR/general_motion_retargeting/ik_configs/marker_to_k1.json` 给各body的rot_offset填写 `[0,0,0,1]`，声明为identity。`scripts/accad_markers_to_k1.py:474-486`沿用这些偏移，并给wrist覆盖同一个数组。

实际GMR `motion_retarget.py:125-126`明确调用 `R.from_quat(rot_offset, scalar_first=True)`，第275行同样按wxyz合成。wxyz单位四元数应为`[1,0,0,0]`，现在配置是绕Z轴180度。标记生成端orientation本来已经使用`as_quat(scalar_first=True)`，因此不能再施加这个翻转。

**单变量复现**：同一段B3 walk1的前2秒，120个60Hz IK输入帧，全部参数和输入相同，只将两张IK表的偏移改为wxyz identity：

| 指标 | 当前偏移 | 修正identity的临时配置 |
|---|---:|---:|
| 根朝向相对人体目标的平均误差 | 179.8416° | 0.03065° |
| 左hip roll均值 | -0.2863 rad | -0.07231 rad |
| 右hip roll均值 | +0.3657 rad | +0.07414 rad |
| 左ankle roll均值 | +0.3370 rad | +0.13656 rad |
| 右ankle roll均值 | -0.3200 rad | -0.12454 rad |
| 后处理需裁剪的关节样本数 | 377 | 111 |

磁盘上现有B3数据的根朝向与错误版本重算结果平均只差0.0092°，最大0.1392°，证明这不是仅存在于尚未运行的脚本中的问题。整个配置对目标施加180度偏移；上述数值对照只测了一个片段，不宣称一处修改已经让整个数据集动力学可行。IK报告的综合error没有随朝向修正下降，不能仅靠这个混合指标验收。

建议修复坐标约定并增加identity/人体朝向回归检查，再生成新目录复核；不要直接在已训练/正在训练的数据目录覆盖文件，也不要通过增大root旋转权重去压制错误目标。

证据：`retarget_probe.py/.json/.log`、`current_ik_config.json`、`identity_wxyz_ik_config.json`。

## P1：新专家姿态与现有软限位奖励存在大规模冲突

独立URDF检查：accad_markers的79段22640帧硬角度/速度超限均为0、全部有限，FK最大误差2.20e-7m。但硬限位合格不代表与训练奖励相容。将保存的姿态代入当前软限位罚项（0.9范围、权重-100、dt=0.02），按帧加权：

| 数据/子集 | 平均软限位奖励/步 | 其中手臂贡献 | 有软限位罚项的帧比例 |
|---|---:|---:|---:|
| amp_paper/walk | -0.0201 | 0 | 19.87% |
| accad_markers/walk | -1.2075 | -0.7963 | 100% |
| accad_markers/kick | -1.6371 | -1.1010 | 100% |

作为量级对照，当前AMP单步上限为0.6，survival为0.06。这里是**专家姿态的奖励核算**，不是策略实际训练奖励，也不能忽略其他奖励后宣称全局最优一定站桩。

新walk数据右hip roll有89.99%的帧超软限位，左ankle roll为91.48%，右ankle roll为87.54%；左右shoulder roll约98%。这与上面的180度目标错误共同指向重定向质量问题。手臂虽被AMP特征排除，仍进入参考复位、物理动力学、关节限位和动作变化惩罚，因此报告中“手臂唯一变差项不影响训练”不成立。

还需注意任务匹配：当前kick目录来自武术侧踢/回旋踢等，按帧采样只占10.55%（amp_paper为28.13%）。新kick有8.84%帧的摆动脚最低鞋底高于球顶0.22m，最高0.486m。它们不自动等于论文中的地面足弓踢足球先验；应明确筛选和采样权重。这是数据任务适配风险，不单独证明不能学踢球。

证据：`data_probe.py/.json`、`soft_limit_breakdown.json`。

## P1/P2：球重置仍污染默认critic、decoder和摩擦力

`commands.py`先更新relative_ball_pos和球速度，之后才重置球；重置分支只同步部分位置和goal flag。`ball_obs_true`及privileged target无条件读这些缓存，不受perfect_perception开关保护。

真实Isaac四环境复现，**将perfect_perception切回False**后：critic真实球位置、decoder监督球位置仍分别误差10.73/9.13/4.02/6.39m。故MightReason“只有perfect_perception受影响，默认训练不受影响”的判断错误。默认actor本就有设计延迟，应区别于critic/decoder的非预期真值错误。

另将重置前速度设为(1,2)m/s：球实际已经归零速度，但cmd.ball_vel_xy仍为(1,2)，并给静止的新球设置(-0.0894,-0.1789)N的阻力。建议统一管理物理转移结算与新状态刷新，包括位置、高度、速度、relative位置、外力及感知采样；保留设计的感知延迟语义。

证据：`runtime_extended.json`，可运行`runtime_probe.py --headless --output <路径>`。

## P2：普通每一步的踢球塑形奖励也混用了两个时刻

本机Isaac ManagerBasedRLEnv.step顺序为：physics → reward_manager.compute → reset → command_manager.compute → observations。kick_ball、side_kick_ball和camera角度奖励使用当前足/头状态，但球位置取上次command更新的缓存。

在实际env.step中以3m/s移动球，奖励计算时真实球x=2.059884m，而踢球奖励读取x=2.0m，落后约5.99cm。随后group0在command里才使用更新位置。同一transition的两组奖励由此使用不同时间状态，快速踢球时会错判脚球距离。应让RewardManager的即时几何奖励读取当前物理球状态，历史缓存只承担明确的差分和感知功能。

证据：`runtime_extended.json/reward_step_timing`。这不是仅凭代码顺序猜测。

## 仍未解决的已知项

- hip roll运行时effort_limit仍43Nm、effort_limit_sim为76Nm；动作scale按76生成，T-N限矩用43。需要核对模型/硬件权威值，不能直接提高扭矩。
- nominal膝/踝左右不对称，线性镜像遗漏0.03/0.04rad偏移。
- 验收脚本仍未检查finite和关节位置限位；本轮使用独立URDF检查补充，因此上述0硬超限结论不依赖该漏检工具。
- 课程、参考复位比例、动作硬裁剪、KL策略等仍不等价于原论文实现。论文baseline和实验课程需要分别保存完整有效配置。

## 当前训练来源与评估

最近训练目录为`logs/rsl_rl/k1_kick_amp/2026-09-17_21-35-38`，最新保存checkpoint为model_5800.pt，TensorBoard记录到5806。env_cfg快照、源码快照以及启动日志`24 clips, 5314 frames`共同确认：这次训练使用amp_paper，**不是当前默认accad_markers**。改磁盘上的默认路径不会更换已加载进内存的数据。

最后100条日志中：平均平面速度0.684m/s，朝球速度0.368m/s，单支撑比例0.654。说明相较旧approach站桩已有进展，但这些训练均值不等于独立踢球成功率。训练打印的success=6.7e-4是窗口统计，不能按比赛成功率解释。

评估采用当前环境实现，使用固定默认姿态复位、关闭AMP训练；加载的当前motion_dir不参与评估策略动作或参考动作复位。因此这是amp_paper训练checkpoint的行为评估，不是accad_markers的数据有效性证明。

独立64环境、30秒固定首轮队列，seed123，均值动作、默认虚拟感知（含噪声与延迟）、nominal物理且固定10ms电机延迟；关闭推搡、随机踢球、传送及球自动重置。结果：63/64保留至终点、0跌倒、1出界、13个触球代理、1个验证进球、0球位置跳变。平均机器人净位移1.966m，平均接近球进度0.943m，平均稳定左右支撑切换13.66次。

进球env61存活30秒，34次支撑切换，最近球距离0.195m，接触后球位移6.139m。触球基于脚球距离及球移动的几何代理，未安装逐对接触传感器；进球需有先前触球代理且整个球在门宽/横梁内越线。单种子1/64不是可靠比赛成功率，nominal评估也不代表完整随机化或真机性能。证据`model5800_soccer.json/.log`。

## 验证范围

两套数据完整CPU数值检查、120帧单变量IK对照、真实Isaac4环境边界/时序探针。已有39项CPU测试通过（禁用ROS自动加载的pytest插件后运行）；这些测试尚未覆盖本轮坐标和时序缺陷。`git diff --check`通过。

## 第二随机种子评估（历史参考）

seed456：62/64满30秒，0跌倒、2出界、19个触球代理、3个验证进球。两种子合计128场景：125个满30秒，0跌倒、3出界、32个触球代理、4个进球（3.125%）。用户随后明确优先检查新K1数据和当前代码，可从零重训；旧模型不作为后续训练前提。
