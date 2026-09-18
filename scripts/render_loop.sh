#!/bin/bash
# 训练视频常驻渲染循环：按渲染点逐个渲染最新 checkpoint 到 mp4。
#
# 串行流程（8GB 单卡无法同时跑训练 + 渲染）：
#   等待 model_<N>.pt 出现 → 持锁 → kill 训练 → 渲染 → resume 训练 → 释放锁
# 与看门狗 monitor_kick_amp.sh 共用 /tmp/kick_amp_monitor.lock：
#   本脚本 flock 阻塞等待（渲染是必须完成的任务），看门狗 flock -n 非阻塞跳过，
#   保证渲染期间看门狗不会误判"训练死了"而重启、与渲染实例互踩。
#
# 渲染点 = 200（热身期采样，看"最开始的乱摔"）+ 每 1000 轮（1000~20000）。
# checkpoint 每 200 轮存一次，渲染点都是 200 的倍数 → 与 checkpoint 精确对齐。
export PATH=/home/xcj/.npm-global/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export HOME=/home/xcj
cd /data/rl_robot/BoosterRobotics/booster_train || exit 1

LOCK=/tmp/kick_amp_monitor.lock
RENDERED_MARK=/tmp/kick_amp_rendered.txt
START=1000
INTERVAL=1000
MAX_ITER=20000
PY=/home/xcj/miniconda3/envs/env_isaaclab/bin/python
LOG=logs/render_loop.log
touch "$RENDERED_MARK"

exec 9>"$LOCK"          # 打开锁 fd（循环内 flock 9 / flock -u 9）

latest_run() {
    ls -td logs/rsl_rl/k1_kick_amp/*/ 2>/dev/null | while read -r d; do
        ls "$d"model_*.pt >/dev/null 2>&1 && { echo "$d"; break; }
    done
}

# 渲染点列表：200 + 1000 的倍数（逐个按顺序渲染，已渲染的跳过）
POINTS="200 $(seq "$START" "$INTERVAL" "$MAX_ITER")"

while true; do
    RUN=$(latest_run)
    if [ -n "$RUN" ]; then
        # 找「第一个」未渲染、且 checkpoint 已保存的点（按顺序，不跳点）
        TARGET=""
        for n in $POINTS; do
            grep -qx "$n" "$RENDERED_MARK" && continue   # 已渲染 → 跳过
            if [ -f "$RUN"model_$n.pt ]; then
                TARGET=$n
                break
            else
                break   # 更早的渲染点还没到，后面的更不可能到，停止
            fi
        done
        if [ -n "$TARGET" ]; then
            flock 9   # 阻塞等看门狗释放锁
            # ---- 1. 停训练（TERM → 等退出 → 兜底 KILL）----
            pkill -TERM -f 'train_kick_amp.py' 2>/dev/null
            for _ in $(seq 1 30); do
                pgrep -f 'train_kick_amp.py' >/dev/null 2>&1 || break
                sleep 2
            done
            if pgrep -f 'train_kick_amp.py' >/dev/null 2>&1; then
                pkill -KILL -f 'train_kick_amp.py' 2>/dev/null
                sleep 3
            fi
            # ---- 2. 渲染 TARGET ----
            OUT="logs/render_iter_${TARGET}.mp4"
            echo "$(date '+%F %H:%M') | 【渲染】iter $TARGET → $OUT" >> logs/monitor.log
            timeout 900 "$PY" scripts/render_kick_amp.py \
                --checkpoint "$RUN"model_$TARGET.pt --out "$OUT" \
                --headless --enable_cameras >> "$LOG" 2>&1
            RC=$?
            # ---- 3. 无论渲染成败都恢复训练（从 kill 前最新 checkpoint，不重跑）----
            CKPT=$(ls "$RUN"model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
            echo "$(date '+%F %H:%M') | 【渲染${RC}后恢复】从 model_$CKPT 续训" >> logs/monitor.log
            setsid nohup "$PY" scripts/rsl_rl/train_kick_amp.py \
                --task Booster-K1-KickAMP-v0 --headless --num_envs 2048 --max_iterations "$MAX_ITER" \
                --checkpoint "$RUN"model_$CKPT.pt > logs/train_kick_amp.log 2>&1 &
            echo "$TARGET" >> "$RENDERED_MARK"
            flock -u 9
        fi
    fi
    sleep 60
done
