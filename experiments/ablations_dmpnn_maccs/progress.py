"""Live progress monitor for dmpnn__x_maccs ablations.

Displays completed runs, active runs with current epoch and metrics, and overall progress.

Usage:
    python -m experiments.ablations_dmpnn_maccs.progress          # print status table and exit
    python -m experiments.ablations_dmpnn_maccs.progress --watch  # live refresh every 5s
    python -m experiments.ablations_dmpnn_maccs.progress --brief  # single-line status summary
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.ablations_dmpnn_maccs import variants as V

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "results", "ablations_dmpnn_maccs")
RUNS_ROOT = os.path.join(RESULTS_ROOT, "runs")
LOG_DIR = os.path.join(RESULTS_ROOT, "logs")
SEEDS = [42, 43, 44, 45, 46]


def parse_log(path: str) -> dict:
    """Extract current epoch and metrics from an active run log."""
    if not os.path.exists(path):
        return {"status": "pending", "info": ""}
    try:
        with open(path, "r", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception:
        return {"status": "running", "info": "reading log..."}

    if not lines:
        return {"status": "starting", "info": "initializing..."}

    # Check for failure
    for line in reversed(lines[-20:]):
        if "Traceback" in line or "Error" in line or "FAILED" in line:
            return {"status": "failed", "info": line[:80]}

    # Check for completion
    for line in reversed(lines[-5:]):
        if "DONE" in line or "val PR-AUC" in line:
            m = re.search(r"val PR-AUC\s+([0-9.]+)", line)
            pr = m.group(1) if m else ""
            m_mcc = re.search(r"val MCC\s+([0-9.]+)", line)
            mcc = m_mcc.group(1) if m_mcc else ""
            info = f"PR-AUC {pr}" + (f", MCC {mcc}" if mcc else "")
            return {"status": "complete", "info": info}

    # Check for epoch line
    epoch_re = re.compile(r"PGB Epoch\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([0-9.]+)\s+\|\s+Val PR-AUC:\s+([0-9.]+)")
    for line in reversed(lines):
        m = epoch_re.search(line)
        if m:
            ep, max_ep, loss, pr = m.groups()
            return {"status": "running", "info": f"Epoch {ep}/{max_ep} (loss {loss}, PR-AUC {pr})"}

    return {"status": "running", "info": lines[-1][:60]}


def get_status(target_runs=None, seeds=None) -> dict:
    target_seeds = seeds or SEEDS
    all_runs = [(a, s) for a in (target_runs or V.PLAN) for s in target_seeds]
    completed = []
    running = []
    failed = []
    pending = []

    for a, s in all_runs:
        name = f"{a}_seed{s}"
        sentinel = os.path.join(RUNS_ROOT, name, "COMPLETE")
        log_path = os.path.join(LOG_DIR, f"{name}.txt")

        if os.path.exists(sentinel):
            completed.append(name)
        elif os.path.exists(log_path):
            st = parse_log(log_path)
            if st["status"] == "complete":
                completed.append(name)
            elif st["status"] == "failed":
                failed.append((name, st["info"]))
            else:
                running.append((name, st["info"]))
        else:
            pending.append(name)

    return {
        "total": len(all_runs),
        "completed": completed,
        "running": running,
        "failed": failed,
        "pending": pending,
    }


def render_brief(status: dict) -> str:
    n_done = len(status["completed"])
    n_tot = status["total"]
    pct = (n_done / n_tot * 100) if n_tot else 0.0
    now = datetime.now().strftime("%H:%M:%S")

    act_str = ""
    if status["running"]:
        items = [f"{name} [{info}]" for name, info in status["running"][:3]]
        act_str = " | Active: " + "; ".join(items)
        if len(status["running"]) > 3:
            act_str += f" (+{len(status['running']) - 3} more)"
    elif status["failed"]:
        act_str = f" | {len(status['failed'])} FAILED"
    else:
        act_str = " | Waiting..."

    return f"[{now}] Progress: {n_done}/{n_tot} ({pct:.1f}%){act_str}"


def render_table(status: dict):
    n_done = len(status["completed"])
    n_tot = status["total"]
    pct = (n_done / n_tot * 100) if n_tot else 0.0

    print("=" * 80)
    print(f"  dmpnn__x_maccs Ablations Progress: {n_done}/{n_tot} ({pct:.1f}%) Completed")
    print("=" * 80)

    if status["running"]:
        print("\n  Currently Running:")
        for name, info in status["running"]:
            print(f"    * {name:30s} -> {info}")

    if status["failed"]:
        print("\n  Failed Runs:")
        for name, err in status["failed"]:
            print(f"    ! {name:30s} -> {err}")

    if status["completed"]:
        print(f"\n  Completed: {len(status['completed'])} runs")

    if status["pending"]:
        print(f"  Pending:   {len(status['pending'])} runs")
    print("=" * 80)


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def monitor_sweep(target_runs=None, target_seeds=None, pids=None, interval=3):
    target_seeds = target_seeds or SEEDS
    all_runs = [(a, s) for a in (target_runs or V.PLAN) for s in target_seeds]
    total_runs = len(all_runs)

    seen_completed = set()
    seen_failed = set()

    for a, s in all_runs:
        name = f"{a}_seed{s}"
        if os.path.exists(os.path.join(RUNS_ROOT, name, "COMPLETE")):
            seen_completed.add(name)

    already_done = len(seen_completed)
    now_str = datetime.now().strftime("%H:%M:%S")
    print(f"[{now_str}] Live Progress Monitor: {total_runs} runs in scope ({already_done} already done, {total_runs - already_done} to run)")

    last_line = ""
    is_tty = sys.stdout.isatty()

    try:
        while True:
            if pids:
                any_alive = any(is_pid_alive(p) for p in pids)
            else:
                any_alive = True

            st = get_status(target_runs, target_seeds)

            # Announce newly completed runs
            for name in st["completed"]:
                if name not in seen_completed:
                    seen_completed.add(name)
                    if is_tty and last_line:
                        sys.stdout.write("\r" + " " * len(last_line) + "\r")
                    m_str = ""
                    val_metrics_p = os.path.join(RUNS_ROOT, name, "val_metrics.json")
                    run_info_p = os.path.join(RUNS_ROOT, name, "run_info.json")
                    if os.path.exists(val_metrics_p):
                        try:
                            with open(val_metrics_p) as f:
                                m = json.load(f)
                            m_str = f"val PR-AUC {m.get('pr_auc', 0):.4f} | val MCC {m.get('mcc', 0):.4f}"
                        except Exception:
                            pass
                    mins_str = ""
                    if os.path.exists(run_info_p):
                        try:
                            with open(run_info_p) as f:
                                info = json.load(f)
                            mins_str = f" in {info.get('minutes', 0)} min"
                        except Exception:
                            pass
                    done_cnt = len(seen_completed)
                    t_str = datetime.now().strftime("%H:%M:%S")
                    print(f"[{t_str}]  ✔ [{done_cnt}/{total_runs} COMPLETE] {name}{mins_str} | {m_str}", flush=True)

            # Announce newly failed runs
            for name, err in st["failed"]:
                if name not in seen_failed:
                    seen_failed.add(name)
                    if is_tty and last_line:
                        sys.stdout.write("\r" + " " * len(last_line) + "\r")
                    t_str = datetime.now().strftime("%H:%M:%S")
                    log_file = os.path.join(LOG_DIR, f"{name}.txt")
                    print(f"[{t_str}]  ✖ [{name} FAILED]: {err} (see {log_file})", flush=True)

            if (len(seen_completed) + len(seen_failed)) >= total_runs:
                break
            if pids and not any_alive:
                break

            brief = render_brief(st)
            if is_tty:
                sys.stdout.write("\r" + brief[:120].ljust(120))
                sys.stdout.flush()
                last_line = brief[:120]
            else:
                print(brief, flush=True)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[monitor] Monitoring stopped by user.", flush=True)
        return

    if is_tty and last_line:
        sys.stdout.write("\r" + " " * len(last_line) + "\r")
        sys.stdout.flush()
    t_str = datetime.now().strftime("%H:%M:%S")
    n_done = len(seen_completed)
    n_fail = len(seen_failed)
    print(f"[{t_str}] Sweep batch finished: {n_done}/{total_runs} completed ({n_fail} failed).", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--watch", action="store_true", help="continuous watch mode")
    p.add_argument("--monitor", action="store_true", help="stream live progress in foreground")
    p.add_argument("--pids", type=int, nargs="+", default=None, help="PIDs to monitor")
    p.add_argument("--interval", type=int, default=3, help="refresh interval in seconds")
    p.add_argument("--brief", action="store_true", help="single line output")
    p.add_argument("--ablations", nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    args, _ = p.parse_known_args()

    if args.monitor:
        monitor_sweep(args.ablations, args.seeds, args.pids, args.interval)
        return

    if args.brief:
        st = get_status(args.ablations, args.seeds)
        print(render_brief(st), flush=True)
        return

    if not args.watch:
        st = get_status(args.ablations, args.seeds)
        render_table(st)
        return

    print("Watching progress (Ctrl+C to stop)...")
    try:
        while True:
            print("\033[2J\033[H", end="")
            st = get_status(args.ablations, args.seeds)
            render_table(st)
            if len(st["completed"]) == st["total"] and st["total"] > 0:
                print("\nAll runs complete!")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
