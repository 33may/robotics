# action/ — PolicyRunner (intent → executable actions)

`PolicyRunner` is the **HOW**: it turns `intent` (`PolicyIn`) into joint actions (`PolicyOut`) by running the ONNX policy, and owns all action-execution timing. It runs as its **own process**.

## Rules

- **Invariance boundary (hard):** import **neither** `isaacsim` **nor** `limxsdk`. Part of the deployment-invariant core (`brain` pytest marker).
- **Owns the buffer + two clocks:** the slow clock runs the policy → refills the buffer; the fast clock drains one action per `tick` at control rate R, emitting `PolicyOut`. The `tick` comes from the World (via Comm) — do not invent a wall-clock in sim.
- **Owns the model bank.** Reason selects intent; PolicyRunner runs it. No perception / planning here.
- **Never talk to a World directly** — emit `PolicyOut` onto the bus; Comm translates it to `CMD`.

See `docs/architecture/architecture.md` §6, §10.
