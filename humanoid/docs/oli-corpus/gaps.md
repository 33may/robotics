# Oli Corpus Gaps

Gaps are tracked instead of silently inferred. Categories: `missing-source`, `unclear-source`, `chinese-only`, `blocked`.

## Chinese-Only Survey

- category: `chinese-only`
- status: none found
- scope: `sources/*.md`
- evidence: survey found isolated Chinese strings in otherwise English SDK/user-manual sections, including wake-word examples and short prompt labels, but no non-empty section that is predominantly Chinese-only.

## playstation-controller-locomotion-policy

- category: `missing-source`
- question: What does the PlayStation controller currently send into the locomotion policy?
- source-map entry: `source_map.md#4-what-does-the-playstation-controller-currently-send-into-the-locomotion-policy`
- status: no official section found that describes the PlayStation controller signal schema or its direct connection to the locomotion policy.

## open-confirmations-from-limx

- category: `missing-source`
- question: What is still blocked, unclear, or needs confirmation from LimX?
- source-map entry: `source_map.md#9-what-is-still-blocked-unclear-or-needs-confirmation-from-limx`
- status: this is a meta-question. The official docs provide interface references but do not enumerate project-specific unresolved confirmations.

## remote-controller-details

- category: `unclear-source`
- citation: `oli-corpus://user-manual#2`
- status: the user manual covers remote-controller operation, but does not explicitly connect controller inputs to the locomotion-policy command pathway.
