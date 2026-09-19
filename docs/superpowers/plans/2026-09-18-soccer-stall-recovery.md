# K1 Soccer Stall Recovery Implementation Plan

**Goal:** Diagnose and reverse the loss of ball interaction after iteration 2000, with reproducible physical evaluations and reliable training completion checks.

**Architecture:** Preserve the current dataset and model architecture. First collect physical distance, speed and weighted reward evidence from the frozen 3000 checkpoint. Introduce at most one reward/curriculum control in the first experiment; run matched continuations with/without that control from the same checkpoint. Treat improved survival alone as insufficient. Explicit completion records distinguish a completed update budget from a checkpoint saved before a failure.

**Tech Stack:** Isaac Lab, PyTorch, CPU pytest, fixed 64-environment first-episode evaluation.

User authorized diagnosis, changes and training; no additional approval needed. Keep current workspace and unrelated work intact.

- [ ] Capture 30 s mean-policy diagnostics for model_3000 with the original reward configuration and training curriculum counter. Use a disposable probe under logs/audit_soccer_stall_20260918; report final distance, stationary time near the ball and actual per-step reward contributions. Do not change the fixed evaluation definition.
- [ ] Fix stage completion verification and redundant history allocation, with regression tests for a child exiting zero after only saving a partial checkpoint. Independent agent owns runner.py, train_kick_amp.py, run_kick_amp_stages.py and its tests.
- [ ] Compare task-vs-auxiliary policy weighting to the local paper code. If the long task ramp suppresses the demonstrated soccer signal, expose an explicit bounded task-weight floor (default zero preserves existing runs) and test its range, approach-phase isolation and post-normalization application. Use only this curriculum variable for the first matched comparison; do not simultaneously change motion data, toe-kick penalties or dynamics.
- [ ] Run all CPU tests and git diff --check. Run two bounded continuations from model_3000 with identical seed, environment count, motion data and update budget. Baseline floor=0, treatment floor=1. Keep GPU runs sequential. Record each launch and independent evaluation.
- [ ] Compare falls, out-of-field events, approach, contact proxy, directed ball progress and valid goals. Continue the better supported branch only when it improves soccer behavior without an unacceptable fall rate; otherwise return to diagnosis. Never mark full autonomous soccer achieved from training reward or a single goal.
