# K1 自主足球：代码审计与训练验证

目标：根据足球和球门的位置自主移动、接近并将球踢入球门。当前保留 AMP + 历史观测 + 双 critic PPO 路线。修复代码不等于已完成足球任务；以下区分可复现错误、评估证据和仍待训练验证的设计。

## 已证实并修复的错误

1. **踢球参考关节串列**：HumanoidSoccer NPZ 存 PhysX 广度优先关节顺序；原转换按 CSV 的逐条肢体顺序读取。用源 G1 模型实测重建，正确顺序 body position 最大误差 1.23e-7 m。修复后按名称映射，未知无 metadata 的 NPZ 必须指定经过验证的 legacy layout。
2. **接触传感器索引错位**：robot.body_names 与 contact sensor.body_names 顺序不同。左脚为 sensor index16、robot index21；原诊断误称 sensor16 为右肘，奖励把左脚落地当作非法接触。32环境物理审计左脚均值96N，右肘0N。修复按传感器名称解析，真实上肢碰撞仍保留惩罚。
3. **偏航角两处公式错误**：base yaw 分母混用 quaternion x/y，180°朝向会变成接近0°；foot yaw 实际求了roll。统一 wxyz yaw 公式，覆盖 ±90°/180°测试。
4. **训练重建目标错一个时间步**：用 s_t 历史预测的 decoder 被要求拟合 step 后 s_(t+1)，摔倒时甚至是重置后状态。改为 rollout 中保存 s_t 的 critic privileged slice。
5. **模型导出破坏训练归一化**：Deploy 引用 live model/normalizer，调用 eval() 后 normalizer.update 永久失效。改为深拷贝导出。
6. **归一化与镜像不交换**：一般运行均值不具左右对称性，直接镜像归一化观测会改变物理含义。统计使用原始观测及其物理镜像成对更新，验证 N(M(x))=M(N(x))。
7. **误导性评估**：原 `1 - terminated.mean()` 是当步未摔比例，不能证明连续站稳。新评估在自动重置之前采集物理状态，首次终止后退出统计 cohort。
8. **罚站诱导短回合**：原 pos_still=-100，dt=.02 后每步-2，比 survival+.06与AMP理论最大+.6之和还大；长期站立回报低于早死。审计修复时降低为-1。随后另一轮BC实验将其改为-10（还乘task课程），当前保留该实验配置；不能把不同配置的日志混作单一训练曲线。
9. **9月17日新增的行走课程错误**：feet_clearance=exp(-sum(height_error*speed)/std)，静止speed=0直接满分1；贴地起步反而减分。改为指数外乘摆动速度与离地gate，静止0，合格摆腿正分。
10. **课程权重被标准化抵消**：整组task reward乘c后逐组优势标准化，normalize(c*A)≈normalize(A)。task critic学习原始奖励，c在PPO合并标准化优势时施加。

## 数据修复及局限

新数据位于 `/data/rl_robot/BoosterRobotics/booster_assets/motions/K1/amp_corrected_v2`：24段、19,682帧、50Hz，11段步行+13段踢球。原始amp、amp_slow、amp_corrected未覆盖。修复中还发现 stylized 来源目录名误写导致漏3段，已补全并增加缺失/空来源的失败检查。

数据由名称映射、K1中立上肢、MuJoCo正运动学、脚底四角高度校准和速度重新差分生成。K1 MuJoCo与Isaac正运动学抽样匹配误差<2.86e-7m。步行根XY按支撑脚约束修正，平均支撑脚水平速度从0.371降到0.155m/s，但滑步没有完全消除；它仍是近似角度迁移，不是完整形态/动力学约束重定向。动作在物理仿真中的可实现性必须继续检验，不能凭回放流畅判定。构建报告记录来源、哈希和每段统计。

## 实际评估（3221迭代模型，修复行走课程之前）

模型：`logs/rsl_rl/k1_kick_amp/2026-09-17_06-30-26/model_3221.pt`

命令：
```bash
/home/xcj/miniconda3/envs/env_isaaclab/bin/python scripts/evaluate_kick_amp.py \
  --headless --checkpoint logs/rsl_rl/k1_kick_amp/2026-09-17_06-30-26/model_3221.pt \
  --scenario soccer --num_envs 64 --steps 1500 --seed 123 \
  --output logs/audit_20260917/eval_3221_soccer.json
```

固定初始姿态、多个机器人朝向与球方位；关闭外力、球随机踢动/传送和球重置、固定10ms电机延迟，保留虚拟球感知。64/64连续存活30秒，平均根净位移0.0388m、平均累计路径0.183m、最低躯干高度均值0.537m，触球代理0/64、进球0/64。这证明该受控条件下已能站立，尚未实现有效接近/踢球；不是随机化条件下的鲁棒性成功证明。

## 训练与可复现记录

原训练已通过SIGTERM在完整迭代结束保存model_3221。新训练从该权重继续，独立实验目录 `logs/rsl_rl/k1_kick_amp_repaired/2026-09-17_09-28-14`，初始500次追加迭代，1024环境，100迭代保存。启用修复的clearance/课程，使用完整v2数据、无物理随机扰动、真实球坐标与0.8–2m前方球生成，先隔离控制能力。此阶段不声称复现完整视觉噪声条件；达到移动/踢球后需恢复感知噪声和随机化并重测。

精确启动参数在 `logs/audit_20260917/train_launch.json` 及运行目录 `launch_args.json`；日志 `logs/audit_20260917/train_repaired.log`。原源码与日志备份 `logs/audit_20260916_before`；9月17日再次留档 `logs/audit_20260917/before_resume.diff`。

旧cron巡检包含过时指标和自动改码行为，`logs/kick_amp_manual_control` 存在时跳过旧watchdog，避免恢复不兼容checkpoint或覆盖当前实验。删除标记会重新启用旧巡检，不建议在此次审计训练期间删除。

## 验收标准及待完成工作

- 回归检查：31项CPU测试通过；`git diff --check`通过。运行 `/home/xcj/miniconda3/envs/env_isaaclab/bin/python -m unittest discover -s tests -v`。
- 待验证：修复后训练是否能产生持续移动、向球接近及触球。独立评估报告首次摔倒、真实位移和严格门线穿越，避免计入球随机干预。
- 触球当前为脚球距离+后续球位移的保守代理，尚无专用pair contact传感器证明。
- 后续需要多seed、多个球位置/朝向、真实感知与打乱球观测对照、含扰动评估；最终必须验证稳定接近与进球，不能以style reward或程序持续运行代替。
- 论文原始代码目录README与论文题目及方法一致，作者的Dryad页面列出code.zip与预训练模型，但目前未完成本地文件与作者发布包的逐字节校验：https://datadryad.org/dataset/doi:10.5061/dryad.jwstqjqps 。

## 9月17日下午追加审计

上午500次追加更新已完成并保存model_3721。下午发现另一轮从离线BC初始化的2048环境训练，保存并暂停于 `logs/rsl_rl/k1_kick_amp/2026-09-17_13-15-33/model_4635.pt`。

对4635使用64场景、30秒、seed123、真实球观测、固定默认站姿评估：64/64存活，平均净位移0.03663m，累计路径0.18194m，最佳接近量均值0.02381m，有效触球代理0，进球0。证据：`logs/audit_20260917/eval_4635_soccer_perfect.json`。站立已得到验证，行走和足球任务未达成。

新增BC脚本发现并修复：
- 四元数heading实际计算roll，导致球门坐标与朝向错误；±90°、180°测试已复现。
- 离线归一化使用sqrt(var+1e-8)，在线实际为std+0.01；小方差关节输入偏差尤其大。现在直接调用同一EmpiricalNormalization实现。
- 上一动作错一帧。以a_t=(q_(t+1)-offset)/scale生成标签时，obs_t的上一动作必须对应q_t。
- 常量导出启动了随机关节偏置，不能代表固定机器人配置。新导出关闭随机化，并分别记录观察默认角与动作offset。
- 离线训练后清除旧actor优化器动量；保存的验证误差重新用最佳权重计算。
- 原bc_eval混合重置前后数据，4个环境曾累计跌倒140回合，平均速度不能作为步态证据。入口现转用固定首次回合评估。

新增模型输出到 `logs/bc/model_bc_corrected.pt`，训练日志 `logs/audit_20260917/bc_corrected.log`，seed42、200epochs、24段数据。这是运动学目标初始化实验：q_(t+1)不是由逆动力学得到的电机指令，即使误差很小也不能证明真实行走可行。必须先闭环评估，再决定是否接续PPO。

### 动作单位边界冲突

参考T1配置 `control.action_scale=1`、`normalization.clip_actions=1`。原移植runner仍在raw action超过±1时施加bound_coef=100的损失，但K1逐关节scale仅0.268–0.887，使惩罚的实际角度范围缩水。完整v2数据中，41.70%的步行帧和81.70%的踢球帧至少一关节目标触发旧单位边界。这是统计冲突，不能单独证明训练失败完全由它造成。

runner现对 `action_mean * resolved_action_scale`（关节目标相对默认角的弧度差）计算1rad边界损失。保留原系数，运行时解析真实action term的scale，加入物理目标相同但raw action尺度不同的等价性测试。修改不影响已保存模型的确定性评估动作，影响后续PPO更新。全部31项CPU回归通过；尚需实际训练验证学习效果。

### 克隆修复的闭环结论与下一轮训练

200epoch离线拟合完成，最佳验证MSE约0.00126、平均目标角误差0.0085rad。但固定64场景全部在首次回合跌倒，平均首次跌倒时间1.004s，无触球或进球。报告 `logs/audit_20260917/eval_bc_corrected_soccer_perfect.json`。因此不将该克隆模型投入接续PPO；此结果否定了“离线角度拟合好就已经学会行走”的判断，并未单独判定失败一定来自重定向、冷启动历史还是缺少动力学补偿。

上午3721同协议评估也完成：64/64存活、净位移均值0.03205m、累计路径0.10953m、接近量0.03195m、触球/进球均0。与4635都属于静止解。

下一轮初始模型选4635，采用1024环境、seed42、nominal_physics/perfect_perception/near_ball，启用弧度动作边界。保留当前AMP系数1与pos_still=-10，不声称是相对上午配置的严格单变量实验。调度 `scripts/run_kick_amp_stages.py` 设置3阶段，每阶段500更新，再执行64场景30秒固定评估；串行使用GPU。运行状态与每阶段物理指标写入 `logs/audit_20260917/radian_stages/status.json`，源码变更或子进程失败时停止，阶段完成只标记requires_review，不把存活或进程完成宣称为足球成功。

完整31项CPU测试与git diff --check通过。当前目标仍未完成：需要观察修正边界后能否学会持续接近球；若仍冻结，应继续检查动态可实现的参考动作与探索，而不是无限延长同一失败配置。

## 17:18 已启动独立行走/接近球阶段

旧阶段调度收到SIGINT并保存后退出。选择已经独立评估的model_5635作为初始actor；旧第三轮检查点保留，不覆盖。

新实现位于 `tasks/manager_based/kick_amp/curriculum.py`，由 `--training_phase approach` 显式开启：

- 仅11段walk参考，运行日志确认16,489帧；从默认站姿开始，reference reset比例固定0。
- c固定0，不按累计迭代衰减行走奖励；足球任务奖励组置零，策略优势权重(0,1)。踢法与头部朝球奖励关闭。
- 目标速度0.4m/s、速度跟踪系数8、std=0.25；存活1，非超时终止-100，停滞-1，AMP系数0.3；其余关节和碰撞约束保留。
- 停滞位置阈值从0.7m降为0.1m；测试确认正常0.4m/s行走不再被判停滞。
- 保留actor/encoder/观测归一化，重置critic和PPO优化器，首次进入阶段将探索sigma恢复到0.15。后续阶段内续训不重复重置。
- nominal_physics + perfect_perception，球位于前方1–2m并有方位变化。

启动目录：`logs/rsl_rl/k1_kick_amp_approach/2026-09-17_17-17-59`；状态：`logs/audit_20260917/approach_stages/status.json`；进程启动时调度PID290343、训练PID290349。每500更新进行独立30秒评估，共3批；累计迭代标号继承5635。

评估新增approach场景，覆盖四种机器人朝向、前方左右球方位。首次回合计分；交替支撑须连续60ms确认，跌倒后不继续累计。单环境通过需连续存活、净位移>=0.5m、到球最小距离<=0.55m、至少4次左右稳定单支撑切换。整组门槛存活>=90%、通过>=80%，seed123通过后还须seed456复测。通过后调度标记需进入足球阶段审核；未通过继续当前行走阶段，不按迭代自动升级。

39项CPU回归通过，git diff --check通过。实际GPU训练已完成首批23次更新；观测到task_weight=0、数据11段与新奖励表，所读标量均有限。早期训练中单支撑比例和速度上升，但包含探索和跌倒，不能据此认定学会行走。新阶段独立评估尚未完成，自主足球目标仍未达成。
