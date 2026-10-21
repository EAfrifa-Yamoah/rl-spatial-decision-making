"""Compute every number reported in the manuscript and supplement.

Reads results/raw/problem_{a,b,c,d}.json and writes results/manifest.json.
Each reported quantity has a stable dotted key (e.g. "C.heur.ucb.rmse");
the consistency checker (tools/check_consistency.py) compares the documents
against these keys, and the figure scripts read the same file.

Conventions (stated in Supplementary Appendix B):
  * continuous seed- or field-level metrics: median with a 95% percentile
    bootstrap CI (10,000 resamples) and the interquartile range;
  * success rates averaged over evaluation rollouts: mean across seeds with
    a 95% percentile bootstrap CI;
  * counts of seeds: k/30 with an exact Clopper-Pearson 95% CI;
  * independent training runs: two-sided Mann-Whitney U, Cliff's delta, or
    Fisher's exact test for counts;
  * comparisons on the same test fields (Problem C): two-sided Wilcoxon
    signed-rank, matched-pairs rank-biserial r, median paired difference;
  * Holm adjustment within each named family.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils import (summarise, proportion, paired_test, unpaired_test, fisher_test,
                       holm_family)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"


def load(name):
    return json.loads((RAW / f"problem_{name}.json").read_text())


# ------------------------------------------------------------------ A
def build_a():
    d = load("a")
    out = {"config": d["config"]}
    a1 = {}
    for rw in ("sparse", "step_cost", "shaped", "pbrs", "random_dense"):
        runs = [r for r in d["A1"] if r["reward"] == rw]
        fg = [r["first_goal"] for r in runs if r["first_goal"] is not None]
        greedy_first = []
        for r in runs:
            hits = [i for i, v in enumerate(r["greedy_curve"]) if v]
            greedy_first.append((hits[0] + 1) * 10 if hits else None)
        gf = [x for x in greedy_first if x is not None]
        a1[rw] = {
            "discovered": proportion([r["first_goal"] is not None for r in runs]),
            "first_goal": summarise(fg) if len(fg) >= 5 else {"n": len(fg)},
            "eval_from_start": proportion([r["eval_from_start"] for r in runs]),
            "multistart": summarise([r["eval_multistart"] for r in runs]),
            "first_greedy_success_ep": summarise(gf) if len(gf) >= 5 else {"n": len(gf)},
            "n_greedy_success_any": len(gf),
            "final_return": summarise([float(np.mean(r["returns"][-50:])) for r in runs]),
        }
    out["A1"] = a1
    if d.get("A1_tiebreak_first"):
        tb = {}
        for mode in ("first_toward", "first_away"):
          for rw in ("sparse", "step_cost", "shaped", "pbrs", "random_dense"):
            runs = [r for r in d["A1_tiebreak_first"] if r["reward"] == rw and r["tiebreak"] == mode]
            tb[f"{mode}.{rw}"] = {
                "discovered": proportion([r["first_goal"] is not None for r in runs]),
                "eval_from_start": proportion([r["eval_from_start"] for r in runs]),
                "multistart": summarise([r["eval_multistart"] for r in runs]),
            }
        out["A1_tiebreak_first"] = tb
    base = [r for r in d["A1"] if r["reward"] == "shaped"]
    tests = {}
    for rw in ("sparse", "step_cost", "pbrs", "random_dense"):
        other = [r for r in d["A1"] if r["reward"] == rw]
        tests[f"shaped_vs_{rw}"] = unpaired_test([r["eval_multistart"] for r in base],
                                                 [r["eval_multistart"] for r in other])
    out["A1_tests_multistart"] = holm_family(tests)
    ftests = {}
    for rw in ("sparse", "step_cost", "pbrs", "random_dense"):
        other = [r for r in d["A1"] if r["reward"] == rw]
        ftests[f"shaped_vs_{rw}"] = fisher_test(sum(r["eval_from_start"] for r in base), 30,
                                                sum(r["eval_from_start"] for r in other), 30)
    out["A1_tests_from_start"] = holm_family(ftests)

    a2 = {}
    for rep in ("tabular", "rowcol", "window", "hand"):
        runs = [r for r in d["A2"] if r["rep"] == rep]
        fg = [r["first_goal"] for r in runs if r["first_goal"] is not None]
        a2[rep] = {
            "alpha": d["A2_best_alpha"][rep],
            "diverged": sum(r["diverged"] for r in runs),
            "discovered": proportion([r["first_goal"] is not None for r in runs]),
            "first_goal": summarise(fg) if len(fg) >= 5 else {"n": len(fg)},
            "eval_from_start": proportion([r["eval_from_start"] for r in runs]),
            "multistart": summarise([r["eval_multistart"] for r in runs]),
            "n_multistart_above_0.8": int(sum(r["eval_multistart"] > 0.8 for r in runs)),
            "n_multistart_0.3_to_0.6": int(sum(0.3 <= r["eval_multistart"] <= 0.6 for r in runs)),
        }
    out["A2"] = a2
    out["A2_tuning"] = d["A2_tuning"]
    tab = [r["eval_multistart"] for r in d["A2"] if r["rep"] == "tabular"]
    tests = {f"tabular_vs_{rep}": unpaired_test(tab, [r["eval_multistart"] for r in d["A2"] if r["rep"] == rep])
             for rep in ("rowcol", "window", "hand")}
    out["A2_tests_multistart"] = holm_family(tests)
    return out


# ------------------------------------------------------------------ B
def build_b():
    d = load("b")
    out = {"config": d["config"]}
    per = {}
    for l in ("iql_own", "iql_joint", "jal"):
        runs = [r for r in d["runs"] if r["learner"] == l]
        mk = [r["eval_makespan_median"] for r in runs if r["eval_makespan_median"] is not None]
        per[l] = {
            "eval_success": summarise([r["eval_success"] for r in runs], stat="mean"),
            "eval_success_median": summarise([r["eval_success"] for r in runs]),
            "seeds_all_success": proportion([r["eval_success"] == 1.0 for r in runs]),
            "seeds_any_success": proportion([r["eval_success"] > 0 for r in runs]),
            "eval_collisions": summarise([r["eval_collisions"] for r in runs], stat="mean"),
            "makespan": summarise(mk) if len(mk) >= 5 else {"n": len(mk)},
            "canonical_success": proportion([r["canonical"]["success"] for r in runs]),
            "train_success_last200": summarise([float(np.mean(r["train_success"][-200:])) for r in runs], stat="mean"),
            "train_collisions_last200": summarise([float(np.mean(r["train_collisions"][-200:])) for r in runs], stat="mean"),
        }
    out["learners"] = per
    s = {l: [r["eval_success"] for r in d["runs"] if r["learner"] == l] for l in per}
    out["tests_success"] = holm_family({
        "own_vs_joint_state": unpaired_test(s["iql_joint"], s["iql_own"]),
        "independent_vs_central": unpaired_test(s["jal"], s["iql_joint"]),
    })
    return out


# ------------------------------------------------------------------ C
def build_c():
    d = load("c")
    out = {"config": d["config"], "fixed_designs": d["fixed_designs"]}
    learners = sorted({l["learner"] for l in d["learned"]})
    for set_name in ("standard", "shift"):
        H = d[f"heuristics_{set_name}"]
        block = {"heur": {}, "learned": {}}
        field_rmse, field_recall = {}, {}
        for rule in ("random", "greedy", "ucb", "maxvar", "ivr", "spacefill"):
            rm = [o["rmse"] for o in H[rule]]
            rc = [o["recall"] for o in H[rule]]
            field_rmse[rule], field_recall[rule] = rm, rc
            block["heur"][rule] = {"rmse": summarise(rm), "recall": summarise(rc)}
        block["prior_rmse"] = summarise([o["rmse_traj"][0] for o in H["random"]])
        for name in learners:
            runs = [l for l in d["learned"] if l["learner"] == name]
            M = np.array([[o["rmse"] for o in r["eval"][set_name]] for r in runs])   # runs x fields
            R = np.array([[o["recall"] for o in r["eval"][set_name]] for r in runs])
            field_rmse[name] = M.mean(0).tolist()
            field_recall[name] = R.mean(0).tolist()
            block["learned"][name] = {
                "rmse": summarise(M.mean(0)),                 # unit = test field (mean over runs)
                "recall": summarise(R.mean(0)),
                "run_rmse": summarise(M.mean(1)),             # unit = training run (mean over fields)
                "run_recall": summarise(R.mean(1)),
                "params_median": np.median(np.array([r["params"] for r in runs]), axis=0).tolist(),
                "params_q25": np.quantile(np.array([r["params"] for r in runs]), 0.25, axis=0).tolist(),
                "params_q75": np.quantile(np.array([r["params"] for r in runs]), 0.75, axis=0).tolist(),
            }
        # kappa sweep and action-selection sensitivity
        block["kappa"] = {k.split("_", 1)[1]: summarise(H[k]) for k in H if k.startswith("kappa_")}
        block["kappa_tests_vs_0"] = holm_family({
            k.split("_", 1)[1]: paired_test(H[k], H["kappa_0.0"]) for k in H
            if k.startswith("kappa_") and k != "kappa_0.0"})
        ucb = field_rmse["ucb"]
        block["selection"] = {"argmax": summarise(ucb),
                              "softmax": summarise(H["select_softmax"]),
                              "epsilon": summarise(H["select_epsilon"])}
        block["selection_tests"] = holm_family({
            "argmax_vs_softmax": paired_test(ucb, H["select_softmax"]),
            "argmax_vs_epsilon": paired_test(ucb, H["select_epsilon"]),
            "softmax_vs_epsilon": paired_test(H["select_softmax"], H["select_epsilon"]),
        })
        # factorial tests on the learned policies (paired by test field)
        f = field_rmse
        block["factorial_tests_rmse"] = holm_family({
            "reward_effect_reinforce": paired_test(f["reinforce_conc"], f["reinforce_rmse"]),
            "reward_effect_qlearn": paired_test(f["qlearn_conc"], f["qlearn_rmse"]),
            "algorithm_effect_rmse_reward": paired_test(f["qlearn_rmse"], f["reinforce_rmse"]),
            "algorithm_effect_conc_reward": paired_test(f["qlearn_conc"], f["reinforce_conc"]),
        })
        fr = field_recall
        block["factorial_tests_recall"] = holm_family({
            "reward_effect_reinforce": paired_test(fr["reinforce_conc"], fr["reinforce_rmse"]),
            "reward_effect_qlearn": paired_test(fr["qlearn_conc"], fr["qlearn_rmse"]),
            "algorithm_effect_rmse_reward": paired_test(fr["qlearn_rmse"], fr["reinforce_rmse"]),
            "algorithm_effect_conc_reward": paired_test(fr["qlearn_conc"], fr["reinforce_conc"]),
        })
        lv = {}
        for name in ("reinforce_rmse", "qlearn_rmse"):
            for rule in ("random", "ucb", "ivr", "spacefill"):
                lv[f"{name}_vs_{rule}"] = paired_test(f[name], f[rule])
        block["learned_vs_heuristic_rmse"] = holm_family(lv)
        rules = ("random", "greedy", "ucb", "maxvar", "ivr", "spacefill")
        hp = {}
        for i, a in enumerate(rules):
            for b in rules[i + 1:]:
                hp[f"{a}_vs_{b}"] = paired_test(f[a], f[b])
        block["heuristic_pairwise_rmse"] = holm_family(hp)
        # each method against the zero-sensor prior mean on the same field
        prior = np.array([o["rmse_traj"][0] for o in H["random"]])
        block["vs_prior"] = {}
        for name, vals in field_rmse.items():
            dlt = np.asarray(vals) - prior
            block["vs_prior"][name] = {"n_below_prior": int((dlt < 0).sum()), "n": int(len(dlt)),
                                       "median_change": float(np.median(dlt))}
        block["field_rmse"] = field_rmse
        block["field_recall"] = field_recall
        out[set_name] = block
    return out


# ------------------------------------------------------------------ D
def build_d():
    d = load("d")
    cfg = d["config"]
    out = {"config": cfg}
    conds = {}
    for st in ("blind", "aware"):
        for rg in ("fixed", "varied"):
            runs = [r for r in d["runs"] if r["state"] == st and r["regime"] == rg]
            train_rows = cfg["train_rows"][rg]
            held = [str(r_) for r_ in cfg["test_rows"] if r_ not in train_rows]
            indist = [str(r_) for r_ in train_rows]
            per_row = {row: summarise([r["eval_rows"][row] for r in runs], stat="mean")
                       for row in map(str, cfg["test_rows"])}
            ind = [float(np.mean([r["eval_rows"][x] for x in indist])) for r in runs]
            ho = [float(np.mean([r["eval_rows"][x] for x in held])) for r in runs]
            ho156 = [float(np.mean([r["eval_rows"][x] for x in ("1", "5", "6")])) for r in runs]
            # episodes until rolling-100 training success first reaches 0.95
            reach = []
            for r in runs:
                s = np.convolve(np.array(r["train_success"], float), np.ones(100) / 100, mode="valid")
                hit = np.flatnonzero(s >= 0.95)
                reach.append(int(hit[0] + 100) if len(hit) else None)
            rr = [x for x in reach if x is not None]
            conds[f"{st}_{rg}"] = {
                "per_row": per_row,
                "in_distribution": summarise(ind, stat="mean"),
                "held_out": summarise(ho, stat="mean"),
                "held_out_rows_156": summarise(ho156, stat="mean"),
                "two_obstacles": summarise([r["eval_two_obstacles"] for r in runs], stat="mean"),
                "episodes_to_95": summarise(rr) if len(rr) >= 5 else {"n": len(rr)},
                "n_reached_95": len(rr),
                "held_rows": held, "train_rows": list(train_rows),
            }
            conds[f"{st}_{rg}"]["_ho156"] = ho156
            conds[f"{st}_{rg}"]["_ind"] = ind
    out["conditions"] = conds
    c = conds
    out["tests"] = holm_family({
        "varied_blind_vs_aware_heldout156": unpaired_test(c["blind_varied"]["_ho156"], c["aware_varied"]["_ho156"]),
        "fixed_blind_vs_aware_heldout156": unpaired_test(c["blind_fixed"]["_ho156"], c["aware_fixed"]["_ho156"]),
        "blind_varied_vs_fixed_heldout156": unpaired_test(c["blind_varied"]["_ho156"], c["blind_fixed"]["_ho156"]),
    })
    for k in conds:
        conds[k].pop("_ho156")
        conds[k].pop("_ind")
    return out


def main():
    manifest = {}
    for key, fn in (("A", build_a), ("B", build_b), ("C", build_c), ("D", build_d)):
        if (RAW / f"problem_{key.lower()}.json").exists():
            manifest[key] = fn()
            print(f"built {key}")
    (ROOT / "results" / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
