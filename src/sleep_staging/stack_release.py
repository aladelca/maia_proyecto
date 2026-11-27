"""Public identity of the immutable five-stage EDF release.

scripts/promote_stack_candidate.py rewrites this file when it installs a new candidate.
"""

MODEL = "stack_five"
RUN_ID = "a64415fc2bd649b39c6fd158994b0479"
VERSION = "stack-five-" + RUN_ID[:12]
EXAMPLE_SHA256 = "71ce757e390a5c2d67e56b53c908edb565512a76fe133952c68cb16a54ab9037"
CHECKPOINTS = {
    "stack_five.joblib": "4a7021f0b6d147a1c64b86204489230aac1a469982d15780ca1189b4e431fb02",
    "six_direct.pt": "4e5171ee040d42bfd389964150241d7ba618a73aec50573251ada67c76addfa3",
}
