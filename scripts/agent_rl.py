from __future__ import annotations

import argparse
from pathlib import Path

from nanovlm.train.agent import LocalSandbox, tool_success_reward
from nanovlm.train.common import default_base_dir, print0
from nanovlm.train.report import MetricsLogger, write_html_report


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic sandbox smoke trainer/evaluator.")
    p.add_argument("--out-dir", default=str(default_base_dir() / "runs" / "agent_rl"))
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--timeout-s", type=float, default=5.0)
    args = p.parse_args()
    out = Path(args.out_dir)
    logger = MetricsLogger(out, "agent_rl")
    for step in range(args.steps):
        sandbox = LocalSandbox(log_path=out / f"trajectory_{step:06d}.jsonl", timeout_s=args.timeout_s)
        results = []
        try:
            # The actual training loop can replace this policy with model tool-call rollouts.
            results.append(sandbox.write_file("solve.py", "print(40 + 2)\n"))
            results.append(sandbox.run(["python", "solve.py"], tool="python"))
            success = tool_success_reward(results)
            reward = success if "42" in results[-1].stdout else 0.0
        finally:
            sandbox.close()
        logger.log(step=step, tool_success=success, reward=reward, sandbox_failures=1.0 - success)
        print0(f"step {step:05d} tool_success={success:.2f} reward={reward:.2f}")
    print0(f"report: {write_html_report(out)}")


if __name__ == "__main__":
    main()
