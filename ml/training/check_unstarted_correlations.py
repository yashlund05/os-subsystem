import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import scipy.stats as stats

from ml.training.policy import FEATURE_NAMES
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.env import SchedulerEnv


def audit_unstarted_features():
    env = SchedulerEnv(workload_generator=lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.8), top_k=16)

    feature_vals = {name: [] for name in FEATURE_NAMES}
    true_bursts = []

    for s in range(50000, 50030):
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            mask = obs["action_mask"]
            valid = np.where(mask == 1)[0]
            cands = []
            if env.running_task is not None:
                cands.append(env.running_task)
            cands.extend(env.ready_queue[: env.top_k - len(cands)])

            for idx in valid:
                t = cands[idx]
                if t.executed_burst_us == 0:  # UNSTARTED CANDIDATE
                    true_bursts.append(t.total_burst_us)
                    for f_idx, f_name in enumerate(FEATURE_NAMES):
                        feature_vals[f_name].append(obs["candidates"][idx, f_idx])
            obs, _, term, trunc, _ = env.step(0)
            done = term or trunc

    b = np.array(true_bursts)
    print(f"Total unstarted candidate observations: {len(b)}")
    print(f"{'Feature Name':<25} | {'Pearson r':<12} | {'Spearman rho':<12} | {'Status':<25}")
    print("-" * 80)
    for f_name in FEATURE_NAMES:
        arr = np.array(feature_vals[f_name])
        if np.std(arr) == 0:
            print(f"{f_name:<25} | {'0.0000':<12} | {'0.0000':<12} | {'Zero variance (Clean)':<25}")
        else:
            r, _ = stats.pearsonr(b, arr)
            rho, _ = stats.spearmanr(b, arr)
            stat = "Kernel estimate" if f_name in ["pred_burst_norm", "burst_ratio"] else ("LEAK!" if abs(r) > 0.1 else "Clean (~0)")
            print(f"{f_name:<25} | {r:<12.4f} | {rho:<12.4f} | {stat:<25}")


if __name__ == "__main__":
    audit_unstarted_features()
