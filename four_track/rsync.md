# rsync — skynet → deepthought backup

Split out of `new_plan.md` 2026-05-09. The rsync work is independent of
the ML pipeline and was holding the plan hostage.

## Status (2026-05-09 ~14:35 local)

**Both rsync attempts FAILED. No rsync currently running.**

### Attempt 1 — PID 4054989

- Wrapper script:
  `four_track/scripts/rsync_to_deepthought_throttled.sh`
- Log:
  `four_track/log/rsync_to_deepthought_20260508_145345.log`
- Started 2026-05-08 14:53. Was last seen at 23:30 in
  `devstral/.git/lfs/objects/`.
- **Killed by SIGINT/SIGTERM/SIGHUP at 2026-05-09 02:23:19**
  (rsync exit code 20). Partial dest files kept in place for resume.

### Attempt 2 — auto-restart

- Log:
  `four_track/log/rsync_to_deepthought_20260509_022429.log`
- Started 02:24:29 (≈70 s after attempt 1 died).
- Ran ~6 h 22 min, traversed past `phi4/figures/` into `phi4/speech-lora/`.
- **Died 2026-05-09 08:47:03** with:
  ```
  rsync: [receiver] write failed on
    "/mnt/mypassport/backup_ai_skynet/phi4/speech-lora/adapter_model.safetensors":
    Input/output error (5)
  rsync error: error in file IO (code 11) at receiver.c(381)
  [2026-05-09T08:47:03-04:00] attempt 1 exit=11
  [2026-05-09T08:47:03-04:00] non-retryable exit 11; aborting
  ```

### What this implies

`Input/output error (5)` from the kernel on the receiver mount is a
**hardware-class signal from the MyPassport drive**, not a transient
network/ssh issue. The wrapper correctly classified exit=11 as
non-retryable and aborted instead of retrying into the same bad block.

## Drive health

`/mnt/mypassport` (deepthought) is the receiving mount. The error
hit `phi4/speech-lora/adapter_model.safetensors` — a single sector
on a single file, but no SMART check has been run since the failure.

`/mnt/mytoshiba` (deepthought) is healthy and is where the actual
project data lives (`MachineLearning/kaggle/BirdCLEF/...` and
`MachineLearning/_runon/...` via `~/work/MachineLearning` symlink).
The pretrain pipeline does **not** depend on `/mnt/mypassport`.

## kaggle/ move state

Independent of the rsync, the user requested 2026-05-08 ~21:30:
move `/home/swatson/work/MachineLearning/kaggle/` →
`/home/swatson/work/kaggle/` on skynet, with deepthought mirror at
`/mnt/mypassport/backup_kaggle_skynet/`.

### Skynet side — DONE

Verified 2026-05-09 14:30:
- `/home/swatson/work/kaggle/` is the populated tree
  (BirdCLEF/, configs/, etc.)
- `/home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track/`
  is an empty leftover shell (no files)
- All current scripts under `four_track/` reference the new path;
  only `*.bak` files contain pre-sed paths.

### Deepthought side — NOT moved (per plan step 4d which was optional)

- `/home/swatson/work/MachineLearning/` symlinks to
  `/mnt/mytoshiba/MachineLearning/`. Project lives under
  `MachineLearning/kaggle/BirdCLEF/four_track/`.
- `/home/swatson/work/kaggle/` does not exist on deepthought.
- `/mnt/mypassport/backup_kaggle_skynet/` was pre-created (per
  former plan §14.22.11.2 step 3) but is **empty** — the failed
  attempt 2 was rsyncing to `/mnt/mypassport/backup_ai_skynet/`,
  not the kaggle dest.

This split path layout is fine for the iNat pretrain dispatch —
runon resolves paths via `__file__` and `BIRDCLEF_ROOT`, and the
iNat data is on `/mnt/mytoshiba/...` which is unaffected.

## Next steps

Sequence when rsync work resumes:

1. **SMART + dmesg on deepthought MyPassport**.
   ```bash
   ssh deepthought 'sudo smartctl -a /dev/disk/by-label/<mypassport-label>'
   ssh deepthought 'sudo dmesg -T | grep -iE "i/o error|sd[a-z]"' | tail -50
   ```
   Decide based on result whether MyPassport is recoverable, needs
   `badblocks -wsv` re-test, or should be retired.

2. **If MyPassport survives**, restart the manual rsync. The wrapper
   already supports resume via `--inplace --append-verify`. Watchdog
   PID 4082806 still holds `/tmp/backup_ai_skynet.lock` — release it
   first (or let the lock guardrail deal with it):
   ```bash
   fuser -k /tmp/backup_ai_skynet.lock 2>/dev/null
   ls -la /tmp/backup_ai_skynet.lock
   ```

3. **If MyPassport is bad**, redirect the backup to
   `/mnt/mytoshiba/backup_<...>/` (5.5 T total, 4.4 T avail) and
   update both `~/bin/backup` and
   `four_track/scripts/rsync_to_deepthought_throttled.sh` to point
   at the new dest. Reformat MyPassport before reuse.

4. **kaggle/ move on deepthought** (former plan §14.22.11.2 step 4d
   — optional). If desired, mirror the skynet layout:
   ```bash
   ssh deepthought 'mv /home/swatson/work/MachineLearning/kaggle \
                        /home/swatson/work/kaggle'
   ```
   This makes the iNat data path identical on both machines
   (`/home/swatson/work/kaggle/...`). Runon mirror at
   `_runon/BirdCLEF` is unaffected. Currently NOT done; the pretrain
   uses the Machine Learning-rooted path.

5. **Restore the kaggle backup entry in `~/bin/backup`** if it was
   added in anticipation of the move and now needs path-correction.

## Hold-over context

- 06:00 nightly cron has the shared-lock guardrail in `~/bin/backup`
  + `rsync_to_deepthought_throttled.sh`. Cron's `ai` segment skips
  cleanly if any rsync to `/mnt/mypassport/backup_ai_skynet/` is in
  flight (dev rsync still runs, email still sent).
- Watchdog PID 4082806 (was holding the lock for attempt 1's
  resume) — verify it's released before re-running.
- `/mnt/MachineLearning` ext4 live on skynet (7.3 T avail);
  designated cold-archive home for `xenocanto_bulk` per former plan
  §14.21.7 step 4e. Not yet populated.
