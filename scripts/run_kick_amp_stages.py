"""Run bounded, sequential train/evaluate stages; never edit code or promote success.

Each training process must exit and save before evaluation uses the GPU.
The status JSON records commands, processes, checkpoints and physical metrics.
SIGINT/SIGTERM forwards a graceful-stop request to the active child.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def source_digest():
    digest = hashlib.sha256()
    for folder in (ROOT / "source", ROOT / "scripts"):
        for path in sorted(folder.rglob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def motion_digest(directory):
    if directory is None:
        return None
    files = sorted(directory.rglob('*.npz'))
    if not files:
        raise ValueError(f'No motion files in {directory}')
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--training_phase", choices=("soccer", "approach"), default="soccer")
    parser.add_argument("--task_weight_floor", type=float, default=0.0)
    parser.add_argument("--boundary_weight", type=float, default=0.0,
                        help="Dense field-edge penalty, <= 0 (0 = off); see train_kick_amp.py")
    parser.add_argument("--boundary_outward_weight", type=float, default=0.0,
                        help="Outward-velocity edge penalty, <= 0 (0 = off); see train_kick_amp.py")
    parser.add_argument("--ball_spawn_bearing_deg", type=float, default=None,
                        help="Spawn half-range in degrees; 180 places the ball anywhere on the circle (see train_kick_amp.py)")
    parser.add_argument("--ball_lateral_weight", type=float, default=0.0,
                        help="Lateral ball-speed penalty, <= 0 (0 = off); see train_kick_amp.py")
    parser.add_argument("--reset_optimization", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training_profile", choices=("benchmark", "default"), default="benchmark",
                        help="default preserves configured randomization/perception; benchmark uses nominal physics and perfect perception")
    parser.add_argument("--motion_dir", type=Path, help="Pin and hash the dataset for all stages")
    parser.add_argument("--eval_envs", type=int, default=64,
                        help="Evaluation cohort. 64 is the established protocol and keeps historical "
                             "comparisons valid; raise it (e.g. 256) when the decision turns on a count "
                             "of 1-2 goals, where 64 scenes cannot separate signal from Poisson noise.")
    args = parser.parse_args()
    if not math.isfinite(args.task_weight_floor) or not 0.0 <= args.task_weight_floor <= 1.0:
        parser.error("--task_weight_floor must be finite and in [0, 1]")
    if not math.isfinite(args.boundary_weight) or args.boundary_weight > 0.0:
        parser.error("--boundary_weight must be finite and <= 0")
    if not math.isfinite(args.boundary_outward_weight) or args.boundary_outward_weight > 0.0:
        parser.error("--boundary_outward_weight must be finite and <= 0")
    if not math.isfinite(args.ball_lateral_weight) or args.ball_lateral_weight > 0.0:
        parser.error("--ball_lateral_weight must be finite and <= 0")
    if args.ball_spawn_bearing_deg is not None and not 0.0 < args.ball_spawn_bearing_deg <= 180.0:
        parser.error("--ball_spawn_bearing_deg must be in (0, 180]")
    if min(args.stages, args.updates, args.num_envs, args.eval_envs) <= 0:
        parser.error("stage, update, environment and evaluation counts must be positive")
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.exists():
        parser.error("output already contains a status.json; choose a new experiment directory")
    source_hash = source_digest()
    motion_dir = args.motion_dir.resolve(strict=True) if args.motion_dir else None
    motion_hash = motion_digest(motion_dir)
    state = {"pid": os.getpid(), "started": time.time(), "status": "starting",
             "source_sha256": source_hash, "initial_checkpoint": str(checkpoint), "training_phase": args.training_phase,
             "task_weight_floor": args.task_weight_floor, "boundary_weight": args.boundary_weight,
             "boundary_outward_weight": args.boundary_outward_weight,
             "ball_spawn_bearing_deg": args.ball_spawn_bearing_deg,
             "ball_lateral_weight": args.ball_lateral_weight,
             "eval_envs": args.eval_envs,
             "training_profile": args.training_profile, "motion_dir": str(motion_dir) if motion_dir else None,
             "motion_sha256": motion_hash, "stages": []}
    child = None
    stopping = False

    def save():
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(status_path)

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    def run(command, log, phase):
        nonlocal child
        if stopping:
            raise InterruptedError("graceful stop requested")
        state.update(status=phase, command=command, log=str(log))
        with log.open("w") as stream:
            environment = dict(os.environ, PYTHONUNBUFFERED="1", KICK_AMP_MINIMAL="0")
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            save()
            returncode = child.wait()
        child = None
        state.pop("child_pid", None)
        if stopping:
            raise InterruptedError("graceful stop requested")
        if returncode:
            raise RuntimeError(f"{phase} exited {returncode}; see {log}")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    save()
    try:
        for stage in range(1, args.stages + 1):
            if source_digest() != source_hash:
                raise RuntimeError("source changed during experiment; review before resuming")
            if motion_digest(motion_dir) != motion_hash:
                raise RuntimeError("motion data changed during experiment; review before resuming")
            train_log = output / f"stage_{stage}_train.log"
            completion_file = output / f"stage_{stage}_training.json"
            if completion_file.exists():
                raise RuntimeError(f"training completion file already exists: {completion_file}")
            command = [sys.executable, "scripts/rsl_rl/train_kick_amp.py", "--headless",
                       "--checkpoint", str(checkpoint), "--num_envs", str(args.num_envs),
                       "--completion_file", str(completion_file),
                       "--max_iterations", str(args.updates), "--save_interval", "100",
                       "--experiment_name", "k1_kick_amp_" + args.training_phase,
                       "--task_weight_floor", str(args.task_weight_floor),
                       "--boundary_weight", str(args.boundary_weight),
                       "--boundary_outward_weight", str(args.boundary_outward_weight),
                       "--ball_lateral_weight", str(args.ball_lateral_weight),
                       "--training_phase", args.training_phase, "--seed", str(args.seed)]
            if args.training_profile == "benchmark":
                command.extend(["--nominal_physics", "--perfect_perception", "--near_ball"])
            if motion_dir:
                command.extend(["--motion_dir", str(motion_dir)])
            if args.ball_spawn_bearing_deg is not None:
                command.extend(["--ball_spawn_bearing_deg", str(args.ball_spawn_bearing_deg)])
            if stage == 1 and args.reset_optimization:
                command.append("--reset_optimization")
            run(command, train_log, f"stage_{stage}_training")
            if not completion_file.is_file():
                raise RuntimeError(f"training exited without a completion report; see {train_log}")
            training = json.loads(completion_file.read_text())
            counters = ("start_iteration", "requested_updates", "requested_end_iteration",
                        "end_iteration", "completed_updates")
            if (any(type(training.get(key)) is not int for key in counters)
                    or training.get("status") != "completed"
                    or training["start_iteration"] < 0
                    or training["requested_updates"] != args.updates
                    or training["requested_end_iteration"] != training["start_iteration"] + args.updates
                    or training["end_iteration"] != training["requested_end_iteration"]
                    or training["completed_updates"] != args.updates):
                raise RuntimeError(f"training incomplete: {training}; see {train_log}")
            if state["stages"] and training["start_iteration"] != state["stages"][-1]["training"]["end_iteration"]:
                raise RuntimeError("training completion counters do not continue the preceding stage")
            checkpoint = Path(training["checkpoint"])
            if not checkpoint.is_absolute() or not checkpoint.is_file():
                raise RuntimeError("training completion checkpoint does not exist at an absolute path")
            if source_digest() != source_hash:
                raise RuntimeError("source changed during training; evaluation would be incomparable")
            if motion_digest(motion_dir) != motion_hash:
                raise RuntimeError("motion data changed during training; review before evaluation")
            report = output / f"stage_{stage}_evaluation.json"
            run([sys.executable, "scripts/evaluate_kick_amp.py", "--headless", "--checkpoint", str(checkpoint),
                 "--scenario", args.training_phase if args.training_phase == "approach" else "soccer", "--num_envs", str(args.eval_envs), "--steps", "1500", "--seed", "123",
                 *(["--perfect_perception"] if args.training_profile == "benchmark" else []), "--output", str(report)],
                output / f"stage_{stage}_evaluation.log", f"stage_{stage}_evaluating")
            result = json.loads(report.read_text())
            n = result["cohort_size"]
            summary = {k: result[k] for k in ("first_episode_survival_fraction", "fall_count", "contact_proxy_count", "validated_goal_count")}
            for metric in ("robot_displacement_m", "robot_path_length_m", "approach_progress_m"):
                summary["mean_" + metric] = sum(e[metric] for e in result["per_env"]) / n
            state["stages"].append({"stage": stage, "checkpoint": str(checkpoint), "training": training,
                                    "evaluation": str(report), **summary})
            if args.training_phase == "approach":
                from evaluate_kick_amp import assess_approach
                gate = assess_approach(result)
                state["stages"][-1]["approach_gate"] = gate
                if gate["passed"]:
                    confirmation = output / f"stage_{stage}_confirmation.json"
                    run([sys.executable, "scripts/evaluate_kick_amp.py", "--headless", "--checkpoint", str(checkpoint),
                         "--scenario", "approach", "--num_envs", str(args.eval_envs), "--steps", "1500", "--seed", "456",
                         *(["--perfect_perception"] if args.training_profile == "benchmark" else []), "--output", str(confirmation)],
                        output / f"stage_{stage}_confirmation.log", f"stage_{stage}_confirming")
                    confirm_gate = assess_approach(json.loads(confirmation.read_text()))
                    state["stages"][-1]["confirmation_gate"] = confirm_gate
                    if confirm_gate["passed"]:
                        state["status"] = "approach_gate_passed_requires_soccer_training"
                        save()
                        return
            save()
            print(json.dumps(state["stages"][-1]), flush=True)
        state["status"] = "stages_finished_requires_review"
    except InterruptedError as exc:
        state.update(status="stopped", reason=str(exc))
    except Exception as exc:
        state.update(status="failed", reason=str(exc))
        raise
    finally:
        state["updated"] = time.time()
        save()


if __name__ == "__main__":
    main()
