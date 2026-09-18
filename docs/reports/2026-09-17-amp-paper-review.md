# amp_paper 与论文对齐复核

审计时间：2026-09-17 21:00 CST。依据：当前工作树、`/home/xcj/MightReason.md`、论文 `2511.03996v2.pdf`、本地 `code/` 参考实现及现有 `amp_paper` 文件。检查包含 CPU 数值复现和 Isaac Lab 4 环境运行时探针。本文没有将本地 `code/` 与作者下载包做逐字节认证。

结论：改用论文来源的动作覆盖了缺失的全向行走/转向，但“全部可配置差距已对齐，仅剩机器性能和形态差异”不成立。存在确定的计算错误、数据硬限位冲突和实现语义差异。目前没有训练进程运行，也没有新 amp_paper 训练成功证据。此前 approach 三次评估全部为0个通过，使用的是旧数据/配置，不能作为新数据的验证。

## P1：梯度惩罚被放大十倍

`rsl_rl/amp/runner.py:523` 内部 `grad_pen_loss=10*E[(||grad D||-1)^2]`；第541行再乘配置。新 `ppo_cfg.py:39` 设置50，实际系数500。参考 `code/utils/runner.py:283` 内部也乘10，`T1.yaml:50` 的5产生有效50，和论文Table 1、Eq.3一致。因此将5解读为旧版参数再改50是错误的。

CPU复现令梯度范数2，未加权罚项1，当前总罚项500，参考50。建议只保留一处有效系数定义（例如函数返回未加权罚项，配置50），增加有效损失测试。

证据：`logs/audit_paper_current/algorithm_repro.json`、`algorithm_audit.md`。

## P1：新数据17/24个片段含K1硬关节限位外动作

T1踝roll允许±0.44rad，K1只有±0.345rad。转换脚本 `scripts/convert_paper_data.py:93-94` 直接复制腿角，没有处理限位。24段中5段walk、全部12段kick出现角度超限。例：male1_side_step_right右踝12.86%的帧超限，峰值0.439142；side_step_left为10.87%，峰值0.439703。策略在K1物理关节限制内不能原样匹配这种专家姿态。

另外左右kick_5膝速峰值13.125rad/s，各0.8%帧超脚本12.6rad/s阈值（URDF精确12.57）。现有数据验收本身返回exit1，“验收已过”不成立。

这需要受K1关节/接触约束的重定向及必要的时间调整。简单裁剪关节角会改变脚轨迹、接触和速度，不能裁剪后不重建就视为修复。

证据：`data_numerical.json`、`data_mapping.json`、`data_gate.json`。

## P1：完成进球触发虚假大负奖励，球观察仍指向重置前位置

`mdp/commands.py:338-354` 在第50个进球帧先重置球、把current goal flag清零，并将last_ball_pos改成新球位置；但last_ball_in_goal仍为true。随后第427-431行：last_potential=0，current_potential=新球到球门距离，差值被当作球倒退，产生大额负奖励。

真实Isaac Lab 4环境控制复现：固定机器人，球在(7.3,0)，之前goal_cnt=49，last_ball_in_goal=true，执行实际_update_command。四环境success均为1，但任务奖励分别为-104.56、-88.39、-38.73、-61.47。默认survival每步才+0.06，该错误会严重歪曲进球结果。

同一次调用中，真实球已随机重置，但perfect_perception的actor_ball_xy仍为(7.3,0)，与实际相对球位置误差4.02–10.73m。这来自relative_ball_pos/感知buffer在重置前更新，之后未同步。

修复应将物理转移奖励和重置后的下一观察明确分开：先按重置前状态结算进球，再同步新球位置/潜势/感知历史；不能仅把某个goal flag清零。

可复现命令：
```bash
/home/xcj/miniconda3/envs/env_isaaclab/bin/python scripts/audit_paper_runtime.py --headless
```
证据：`runtime.json`、`runtime.log`。这是人工设置的边界条件复现，不是策略自主进球成功。

## P1/P2：髋roll的力矩限制与动作尺度不一致

`assets/robots/actuator.py:183-207` 设置effort_limit_sim，却未显式设置effort_limit；本机Isaac Lab在effort_limit=None时从URDF取得软件限矩。K1 hip_roll URDF是43Nm，motor cfg E4315配置76Nm。T-N曲线用软件限矩，K1_ACTION_SCALE按effort_limit_sim计算。

已用真实Isaac环境确认左右髋roll：effort_limit=43、effort_limit_sim=76、Kp=21.44796、scale=0.885865。不是单纯文档不一致，而是参与动作与力矩计算的两套值不同。

应先核实K1硬件/模型应采用的限制，再统一显式执行器、仿真限制、动作尺度。不能直接把43提高到76并宣称更正确。

证据：`runtime_actuators.json`。

## P2：T1到K1仍需要实质重定向

已核对12个腿关节轴向、静态零点旋转，未发现符号反转。但是杆长、限位不同；T1腰yaw位于腿的上游，`convert_paper_data.py:58` 将它丢弃，同时保留原trunk姿态。T1自身FK对照只将腰置零，kick_3脚位置最多改变7.47cm、朝向24.62°，转身片段脚位置变化最大12.19cm。因此“同名同构，映射平凡”不成立。GMR不是唯一工具，但仍需定义K1的重定向约束。

转换器第32行还读取了旧 `logs/bc_constants.json` 随机化样本，而不是nominal常量。冻结head_yaw=0.04459、肩roll=-1.33478/+1.33578等偏差被写进全部新数据。偏差较小，不能单独归因为训练失败。

anchor也改变了源根轨迹，例如一个左转片段平均根速度0.780→0.377m/s。新文件是T1数据来源的K1近似转换，不是原论文运动学/动力学原样保留。

## P2：数据验收有漏检

`validate_motion_dataset.py` 不检查关节位置限位与isfinite，损坏四元数仅记SOFT。仅在/tmp副本进行负测：实际踝超限片段、所有运动数组为NaN的片段均exit0/HARD0；零四元数也exit0。当前真实数据没有NaN，但验收工具无法防止该类错误，不能以“门禁通过”证明安全可训练。

证据：`data_gate_probe.json`。应加入必需字段/形状/有限值/四元数/模型解析出的角度及速度约束、FK/导数一致性检查。

## P2：镜像损失的坐标原点不一致

`mirror.py:89-109` 对relative joint position和action做线性左右交换，但默认膝角左右0.50/0.53、踝pitch -0.18/-0.22不对称。正确相对位置镜像应为 M(q-q0)+(Mq0-q0)。当前遗漏项分别为0.03/0.04rad，即便关闭随机化也存在。成对更新normalizer不能补上物理原点差。可通过对称nominal或完整仿射镜像解决；需要兼顾随机motor bias，不能只改一张矩阵。

## 尚未完全对齐，但不应直接称为根因的部分

- 课程仍按16000次更新线性改变任务权重；soccer reference reset最高50%，approach强制0%，原参考reference init不同。pos_still虽然系数改-100，阈值仍0.1m而参考0.7m，还乘课程c，语义并未对齐。
- approach入口覆盖奖励、任务优势权重与数据子集；nominal_physics把电机延迟固定2步（10ms）。修改默认值并不意味着这些入口运行时采用论文值。应建立独立、完整、可快照的paper与approach配置。
- 当前动作没有原T1的±1rad硬限幅；bound loss仅约束策略均值，Gaussian采样仍无界。按物理偏移限幅才与参考语义一致，不能把K1 raw action简单裁到±1。
- 延迟名义范围0–20ms，但本地各执行器组独立采样{0,5,10,15,20}ms；原全身共享{0,2,...,18}ms。reset后空buffer的首次新目标立即生效，原版先保持当前关节姿态。这是实际语义差异。
- K1额外使用按绝对转速同时衰减正负扭矩的T-N模型，达到速度上限时反向制动力矩也归零。该制动象限需验证，不能当作普通参数变化；暂未证明它导致了训练失败。
- KL调整频率、阈值和LR下限、std硬范围、history reset填充仍与参考不同。属于可配置/实现差异，不是“只剩算力”。
- 地形、推搡恢复可能影响学习，但目前没有对照证明“关推搡导致迈步不被奖励”是主因；真实任务本身也应能奖励向球移动。不要用这个假设替代确定bug的处理。

## 已确认正确、不要反向修改的部分

- 新数据12个walk为3814帧，即按T/50计算76.28s；总24段5314帧/106.28s。
- 全部真实NPZ有限；K1 FK最大误差2.42e-7m，线速度/关节速度与保存位置差分一致，最低脚底高度校准也数值自洽。这些不等于动态可行。
- 当前熵系数-0.01在loss+=coef*entropy下鼓励探索。MightReason后部“惩罚熵”的文字与前部正确说明冲突。
- decoder target已经取step前状态；GAE CPU对照参考误差0，不应继续归因于此前已修复的时序错误。
- 论文Eq.4的AMP负号与其判别器目标及参考代码存在矛盾；不要为了字面对齐而反转当前+1+tanh奖励。
- 39项已有CPU测试通过；它们未覆盖此次新发现的全部问题，不能据此宣称论文复现完成。

优先顺序：纠正有效GP与进球重置边界；完善数据验收并进行K1约束重定向；统一实际执行器限矩/动作单位/镜像；分离论文复现配置和实验课程；之后再做有对照的训练。当前审计产物保存在 `logs/audit_paper_current/`，本轮未修改现有训练实现或覆盖数据集。
