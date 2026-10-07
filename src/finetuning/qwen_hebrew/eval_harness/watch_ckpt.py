"""Watch a fine-tune's hub checkpoint pushes (any series repo; CLI args).

Logs one line per new checkpoint (fresh eval losses pulled from the pushed
trainer_state.json — W&B heartbeats are unreliable across Colab resumes),
an ALERT line on eval-loss regression (> best + 0.05, or > 0.85) or a
stalled run (no new push for > 3.2h; cadence is ~1h49m/100 steps), and DONE
at step >= 2000. Run under nohup from the repo — never from /private/tmp,
which is wiped on every reboot.
"""
import json
import os
import re
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv("/Users/isaac/Documents/GitHub/historical-document-analysis/.env")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

import argparse
_ap = argparse.ArgumentParser(description="Watch a hub checkpoint repo for pushes (eval loss, stale, done).")
_ap.add_argument("--repo", default="isaacmg/qwen3-vl-8b-hebrew-v20b-ckpt")
_ap.add_argument("--from-step", type=int, default=200)
_ap.add_argument("--best", type=float, default=0.7496)
_ap.add_argument("--max-steps", type=int, default=2000)
ARGS = _ap.parse_args()
REPO_ID = ARGS.repo
POLL_S = 900
STALE_H = 3.2
api = HfApi(token=os.environ["HF1_TOKEN"])


def log(msg: str) -> None:
    """Print a timestamped watcher line.

    :param msg: Message body to append after the timestamp.
    """
    print(f"{datetime.now().strftime('%m-%d %H:%M')} {msg}", flush=True)


last_step = ARGS.from_step
last_new_time = time.time()
stale_alerted = False
min_eval = ARGS.best
log(f"watching {REPO_ID} from step {last_step} (best eval {min_eval})")
while True:
    try:
        commits = api.list_repo_commits(REPO_ID)
        ck = next(c for c in commits if "checkpoint" in c.title)
        m = re.search(r"step (\d+)", ck.title)
        step = int(m.group(1)) if m else 0
        if step > last_step:
            p = hf_hub_download(
                REPO_ID, "last-checkpoint/trainer_state.json",
                token=os.environ["HF1_TOKEN"], revision=ck.commit_id,
                force_download=True)
            st = json.load(open(p))
            for h in st["log_history"]:
                if "eval_loss" not in h or h["step"] <= last_step:
                    continue
                s, l = h["step"], h["eval_loss"]
                trend = "new best" if l < min_eval else f"+{l - min_eval:.4f} vs best"
                log(f"CKPT step={s} eval_loss={l:.4f} ({trend})")
                min_eval = min(min_eval, l)
                if l > min_eval + 0.05 or l > 0.85:
                    log(f"ALERT eval-loss regression at step {s}: {l:.4f} "
                        f"(best {min_eval:.4f})")
            last_step = step
            last_new_time = time.time()
            stale_alerted = False
        if step >= ARGS.max_steps:
            log(f"DONE run reached step {step} — training complete")
            break
        age_h = (time.time() - last_new_time) / 3600
        if age_h > STALE_H and not stale_alerted:
            log(f"ALERT stale: no new checkpoint for {age_h:.1f}h "
                f"(last step {last_step}) — Colab likely disconnected")
            stale_alerted = True
    except Exception as e:  # network blips must not kill a day-long watch
        log(f"poll error: {type(e).__name__}: {e}")
    time.sleep(POLL_S)
