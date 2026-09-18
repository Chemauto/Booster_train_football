#!/bin/bash
# K1 kick_amp 训练巡检：crontab 每 2h 拉起。
# 第一层：bash 看门狗（无 AI 依赖，API 挂了也能保活）
# 第二层：headless claude 诊断（趋势判读 + 单旋钮代码修复）
export PATH=/home/xcj/.npm-global/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export HOME=/home/xcj
cd /data/rl_robot/BoosterRobotics/booster_train || exit 1

# Explicit ownership handoff during audited training/evaluation. The old
# watchdog must not load incompatible checkpoints or change live experiments.
[[ -f logs/kick_amp_manual_control ]] && exit 0

exec 9>/tmp/kick_amp_monitor.lock
flock -n 9 || exit 0   # 上一次还没跑完，跳过防叠

# ---- 第一层：bash 看门狗 ----
# 训练进程死 + 日志 20 分钟没更新 → 直接从最新 checkpoint 重启（不问 AI）
if [ "$(pgrep -fc 'train_kick_amp.py' || true)" = "0" ] && \
   [ -n "$(find logs/train_kick_amp.log -mmin +20 2>/dev/null)" ]; then
    RUN=$(ls -td logs/rsl_rl/k1_kick_amp/*/ 2>/dev/null | while read d; do
        ls "$d"model_*.pt >/dev/null 2>&1 && { echo "$d"; break; }
    done)
    if [ -n "$RUN" ]; then
        CKPT=$(ls "$RUN"model_*.pt | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
        echo "$(date '+%F %H:%M') | 【bash看门狗重启】进程死+日志停滞20min，从 ${RUN}model_${CKPT} 恢复" >> logs/monitor.log
        setsid nohup /home/xcj/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train_kick_amp.py \
            --task Booster-K1-KickAMP-v0 --headless --num_envs 2048 --max_iterations 20000 \
            --checkpoint "${RUN}model_${CKPT}.pt" > logs/train_kick_amp.log 2>&1 &
    else
        echo "$(date '+%F %H:%M') | 【bash看门狗: 需人工】找不到任何 checkpoint" >> logs/monitor.log
    fi
fi

# ---- 第二层：AI 诊断（API 不可用时看门狗已兜底，此层失败仅记日志） ----
timeout 1500 claude -p "$(cat scripts/monitor_kick_amp_prompt.txt)" \
  --dangerously-skip-permissions \
  >> logs/monitor_cron.log 2>&1 || \
  echo "$(date '+%F %H:%M') | 【AI层失败: claude退出码$?，看门狗已兜底】" >> logs/monitor.log
