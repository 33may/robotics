---
name: vbti-user-account
description: "Second Linux user 'vbti' on the dev machine for employee data collection — setup details"
metadata: 
  node_type: memory
  type: project
  originSessionId: 934fc31d-8d22-4972-8251-ba505270e8e5
---

Created 2026-05-18 so employees can collect teleop data on the local dev machine while may33 is away for a week.

- **Account**: `vbti` (uid 1001), standard user, NOT in `wheel` (no sudo). Groups: `vbti, dialout, video`. GECOS / display name: `vbti`. GDM login password: `vbti` (set 2026-05-18).
- **Hardware access**: `dialout` → servo ports `/dev/ttyACM0/1`; `video` → `/dev/video*` (incl. `/dev/cam_*` symlinks). RealSense USB control also relies on logind `uaccess`, granted when logged in at the GNOME seat.
- **Conda**: shares may33's install read-only. `setfacl -m u:vbti:x /home/may33` opens traversal (only `/home/may33` was 700; all subdirs already 755). `/home/vbti/.bashrc` sources `/home/may33/miniconda3/etc/profile.d/conda.sh` then `conda activate lerobot`.
- **Calibration**: copied (symlinks dereferenced) to `/home/vbti/.cache/huggingface/lerobot/calibration/` — robot ids incl. `may-sim`, teleop incl. `frodo-test`.
- **Recorded data** lands locally in `/home/vbti/.cache/huggingface/lerobot/<repo_id>/`; may33 can also retrieve via `sudo`.
- **HF Hub push**: may33's token (account `eternalmay33`, fine-grained, write) was copied to `/home/vbti/.cache/huggingface/token` (mode 600). `lerobot-record --dataset.push_to_hub=true` works — datasets upload to the `eternalmay33/` namespace.
- Data collection = the standard `lerobot-record` command (robot/cameras/teleop/dataset are all flags).
- **Employee guide**: `/home/vbti/ROBOT_GUIDE.md` (vbti-owned, 644) — collect data / run inference / check datasets / servo utils / file locations & fixes. Living doc, refined with may33; editing it from may33 needs `sudo` (/home/vbti is 700).
- **Claude Code**: usable under vbti — shared binary via PATH (`/home/may33/.local/bin` added to vbti `.bashrc`), may33's `~/.claude/.credentials.json` copied to `/home/vbti/.claude/` (mode 600). Runs on may33's Claude account; headless `claude -p` auth test passed 2026-05-18. vbti's `~/.claude` is otherwise fresh (no CLAUDE.md/skills/history).

Related: [[camera_udev_setup]] [[servo-leader-follower-voltage]]
