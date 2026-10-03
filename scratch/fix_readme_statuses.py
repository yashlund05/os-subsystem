"""Bulk-fix all SCAFFOLDED status READMEs in the NeuroOS-Lite project."""
import re
import pathlib

root = pathlib.Path(".")

# Each entry: relative path -> (original phase hint pattern, replacement phase text)
updates = {
    "userspace/policy/README.md":           "Phase 2, Week 4",
    "telemetry/schemas/README.md":          "Phase 1, Week 1",
    "telemetry/ring_buffer/README.md":      "Phase 1, Week 1",
    "simulator/README.md":                  "Phase 1, Weeks 1-2",
    "schedulers/srtf/README.md":            "Phase 1, Week 2",
    "schedulers/sjf/README.md":             "Phase 1, Week 2",
    "schedulers/round_robin/README.md":     "Phase 1, Week 2",
    "schedulers/mlfq/README.md":            "Phase 1, Week 2",
    "schedulers/fcfs/README.md":            "Phase 1, Week 2",
    "ml/preprocessing/README.md":           "Phase 1, Week 1",
    "ml/models/README.md":                  "Phase 2, Weeks 3-4",
    "ml/features/README.md":                "Phase 2, Week 3",
    "ml/evaluation/README.md":              "Phase 2, Week 4",
    "kernel/guardrails/README.md":          "Phase 3, Week 5",
    "experiments/configs/README.md":        "Phase 4, Week 7",
    "benchmarks/workloads/README.md":       "Phase 1, Week 1",
    "allocators/fixed_partition/README.md": "Phase 1, Week 2",
    "allocators/first_fit/README.md":       "Phase 1, Week 2",
    "allocators/buddy/README.md":           "Phase 1, Week 2",
    "allocators/best_fit/README.md":        "Phase 1, Week 2",
}

fixed = 0
for rel_path, phase in updates.items():
    p = root / rel_path
    if not p.exists():
        print(f"SKIP (missing): {rel_path}")
        continue
    text = p.read_text(encoding="utf-8")
    # Replace any ` `SCAFFOLDED` ...` line with the IMPLEMENTED equivalent
    new_text = re.sub(
        r"`SCAFFOLDED`[^\n]*",
        f"`IMPLEMENTED` (Completed {phase})",
        text,
    )
    if new_text != text:
        p.write_text(new_text, encoding="utf-8")
        fixed += 1
        print(f"FIXED: {rel_path}")
    else:
        # Try alternate formatting: **Status**: `SCAFFOLDED`
        new_text2 = re.sub(
            r"SCAFFOLDED`[^\n]*",
            f"IMPLEMENTED` (Completed {phase})",
            text,
        )
        if new_text2 != text:
            p.write_text(new_text2, encoding="utf-8")
            fixed += 1
            print(f"FIXED (alt): {rel_path}")
        else:
            print(f"no-match: {rel_path}")

print(f"\nDone. Fixed {fixed} files.")
