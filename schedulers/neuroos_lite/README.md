# NeuroOS-Lite Learned Scheduler (Simulator Mirror)

Simulator-faithful mirror of `kernel/sched_ext/neuroos_sched.c` (Phase 3, Week 5).

- 16-D observation encoding, quantized 16->8->1 scoring, dynamic quantum `[1000, 50000]` us.
- Guardrails: queue depth `> 1024` or drift `> 3.0 sigma` -> MLFQ fallback.
- No oracle burst access.
- **Status**: `IMPLEMENTED` (Completed Phase 3, Week 5)

