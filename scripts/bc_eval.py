"""Compatibility entry point for the fixed-cohort AMP evaluation.

The former evaluator averaged repeated falls and resets, which made falling
look like locomotion. Use --output to select a report, --scenario soccer for
balanced headings/bearings, and --perfect_perception to isolate control.
"""
from evaluate_kick_amp import main

if __name__ == "__main__":
    main()
