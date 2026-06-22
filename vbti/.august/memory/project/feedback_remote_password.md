---
name: remote SSH password — read remote_machine.md, don't guess
description: Don't guess SSH password from CLAUDE.md sudo password. Always check remote_machine.md memory first.
type: feedback
originSessionId: 92f50957-8ae3-4bb7-a2e5-c27a13fabe74
---
When SSHing to the remote (`vbti@10.11.100.151`), the password is **`vbti25robot`** (already in `remote_machine.md`). Not `vbti` (that's just the username), not `may` (that's CLAUDE.md sudo password).

**Why:** I conflated the local sudo password from CLAUDE.md (`may`) and the SSH username (`vbti`) and tried both. Wasted an SSH-attempt round (which can lock the account if too many failures) and asked the user to debug. The right password was already in memory at `remote_machine.md:10`.

**How to apply:** Before any SSH command to the remote, check `remote_machine.md` for current creds. If a "password rejected" error happens, recheck memory and the user's last `--password=` style hint, never guess.
