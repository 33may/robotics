# Global Memory - may33

## User Profile
- Shell: zsh (`~/.zshrc` for persistent env vars, aliases, PATH)
- sudo password: may (as noted in CLAUDE.md)
- OS: Fedora (Linux)
- GPU: NVIDIA (uses Isaac Sim, CUDA)

## User Preferences
- Wants automated scripts for everything, not manual file creation
- Prefers one clear solution path, not "here are 5 options" style
- Wants explanations of what code does but doesn't want to write boilerplate
- Values understanding the system deeply - explain changes as you make them
- Willing to sacrifice tokens for better memory - save everything important

## Environment Variables (persistent in ~/.zshrc)
- `LEISAAC_ASSETS_ROOT=/home/may33/projects/ml_portfolio/robotics/leisaac/assets`

## Active Projects
- **August** (`~/projects/august`) - Downstream OpenCode harness with API-first real-model validation. Project memory: `~/.claude/projects/-home-may33-projects-august/memory/`.
- **robotics** (`~/projects/ml_portfolio/robotics/`) - Main project
  - Isaac Sim robotics with SO-101 arm
  - LeIsaac framework for task management
  - SmolVLA policy training and inference
  - VBTI table scene with GS background, cup, duck objects
  - Project-specific memory: `vbti/.august/memory/project/`

## Common Pitfalls
- Claude Code is routed through Codex via `ANTHROPIC_BASE_URL=http://127.0.0.1:8317`; use Codex-native MCP tools `codex_web_search`/`codex_web_fetch` for web research. Built-in `WebSearch`/`WebFetch` are denied in `~/.claude/settings.local.json`.
- leisaac `ASSETS_ROOT` defaults to `<git_root>/assets` — breaks when leisaac is nested in another repo. Always need `LEISAAC_ASSETS_ROOT` env var.
- Isaac Sim uses radians, training datasets use degrees — always convert
- Policy outputs need postprocessor (denormalization) before use
