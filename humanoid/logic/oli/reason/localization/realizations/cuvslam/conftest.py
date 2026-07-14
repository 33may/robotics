"""Keep pytest out of the vendored upstream — its test suites (nanobind bindings, tools)
need deps and data we deliberately don't ship; the candidate's gate is test_contract.py."""

collect_ignore = ["vendor"]
