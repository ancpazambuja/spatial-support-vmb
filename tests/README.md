# Tests

Recommended tests:
- model can be instantiated from each supported configuration family;
- expected input/output shapes are preserved;
- preprocessing handles valid synthetic input and rejects invalid input clearly;
- metric implementations agree with small hand-checkable examples;
- inference returns finite values on synthetic input;
- configuration parsing and CLI help work.

Keep tests small enough to run in the documented CPU smoke-test environment where feasible.
