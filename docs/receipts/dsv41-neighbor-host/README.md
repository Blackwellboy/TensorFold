# TP4 neighbor transport: host checks

Local checks on 2026-10-20 UTC, macOS arm64, Zig 0.17.0. This change is based on
`bertholomus/TensorFold` branch `deepseek-v41-zig-tp4` at
`59402086381a42c74d53053443224eec2790b4c5`.

The transport is opt-in through `TF_DS_NEIGHBOR_RDMA=1`; see the
[configuration and dispatch contract](../../recipes/deepseek-v41-neighbor.md).
No hardware or model qualification was performed for this branch.

| Check | Result |
| --- | --- |
| `zig build test-dsv41-neighbor --summary all` | PASS: 14 tests, 0 failures, 0 skips; no GPU/verbs execution |
| `zig build test test-golden -Dcpu=apple_m1 --summary all` | Inherited compile failure at `dsv41/link.zig:23`: macOS `std.c.SO` has no `BUSY_POLL`; reproduced on the pinned base and candidate |
| Frozen server corpus, from the same command | PASS on base and candidate: 312/319 equal, 7 known differences, 0 unexpected differences |
| `zig build native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=fast --summary all` | PASS: Linux arm64 cross-build, without CUDA fatbins; not a runnable model qualification |
| Linux arm64 `dsv41` host-test module | PASS: semantic compilation with `zig test -fno-emit-bin -Odebug -target aarch64-linux-gnu.2.28` and the build's CUDA/core/lanes/fabric modules; not executed |
| `python3 tools/zig/lean_check.py` | Base: 848 files, 1,511 problems. Candidate: 849 files, 1,511 problems. Zero added findings; checker remains nonzero |
| `zig fmt --check` on changed Zig files; `git diff --check` | PASS |

The base host run reported 420/422 tests passed and two skips before the existing
compile failure. The candidate reused unchanged successful test steps from that
run and reported five newly run passes and one skip, with the same compile
failure. The golden runner also logs an inherited allocator leak on both runs;
its build step exits successfully. Neither complete host command is reported as
a pass.

The focused tests exercise host-side preconditions, startup-mode comparison,
actual NCCL forwarding call sequencing with a recording API, and abort-word
propagation. They do not validate device output, RDMA queue pairs, rail selection,
CUDA graph execution, startup collectives on four machines, throughput, or model
tokens. GPU/model gates in CONTRIBUTING.md remain pending, including byte/layout
and graph checks, plain/drafted and concurrent/solo equality, prompt reuse,
long-context behavior and matched performance. No fault, restart, or reboot gate
was run.

The earlier experimental implementation used source
`b65c8b92d2403dfdefccf15a3b23ba609a4553e2`, a descendant of this pinned base, plus
separate patches. Its device and serving results support the transport design;
they are not measurements of this consolidated branch or its startup agreement.
No unrelated model, server, forced-tool, kernel or deployment changes were ported.
