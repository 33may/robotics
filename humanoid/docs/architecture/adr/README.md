# Architecture Decision Records

Short records of the load-bearing decisions behind [`../architecture.md`](../architecture.md). Later ADRs supersede earlier ones where noted.

| ADR | Decision |
|---|---|
| [0001](0001-world-robot-invariance.md) | World/Robot invariance — the Robot never knows its World |
| [0002](0002-inner-dataflow-bus.md) | Inner comm is a brokerless dataflow bus (bus impl TBD) |
| [0003](0003-policyrunner-own-process.md) | PolicyRunner is its own process (buffer + two clocks) |
