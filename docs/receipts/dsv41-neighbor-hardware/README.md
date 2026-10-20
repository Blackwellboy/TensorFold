# TP4 neighbor transport: four-host hardware qualification

Tested 2026-10-20 UTC at code commit
`660bf9aba19b7ac42ea7ca1b4e26c50b4850bce1`, based on
`deepseek-v41-zig-tp4` at `b65c8b92d2403dfdefccf15a3b23ba609a4553e2`.
Subsequent receipt-only changes do not change the tested engine code.

We do not have a network switch. Four GB10 hosts form a physical ring, with two
PCI-domain RoCE rails per adjacent edge and GID 3. This was the fastest ring
implementation selected in our experiments. These results are **not a direct
performance comparison with the target branch or with a switched/full-mesh
fabric**. The attempted target-branch comparison was discontinued; its partial
measurements are not included here.

## Build and runtime

Both native executables were built with Zig 0.17.0, Linux/aarch64 glibc 2.28,
fast optimization, SM121 and the existing CUDA fatbins. The build used the pinned
NGC image `nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3`:

```sh
zig build native neighbor-lanes neighbor-probe \
  -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=fast -Dsm=121 -j2 -Dfatbins=/fatbins
```

`neighbor-lanes` and `neighbor-probe` were external qualification build targets,
not release targets added to this branch. The latter exposes a C ABI around the
production code for a weight-free Python/CUDA driver. Only build target additions
and the new adapter were applied to the source archive; all 1,200 tracked files
under `zig/src` were verified byte-identical to the tested commit. The native
executables have no test-path substitutions. Binary digests and per-rank OS,
driver and kernel versions are in [summary.json](summary.json).

Model: DeepSeek-V4.1-Flash-1017 EXL3 revision
`a2f91e2794923d0beebebe0d267fb3468eaf0b65`, original Engram revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, DSpark.
Effective server arguments: `--context 1048576 --parallel 16 --max-tokens 32768`,
shared window 2097152, round rows 48, graphs enabled, served-depth policy `solo`.
`TF_DS_NEIGHBOR_RDMA=1` and `TF_DS_NEIGHBOR_GID_INDEX=3` on all four ranks
(the GID default is 5); no per-optimization switches or stock
`TF_RDMA_DEVICES`. NCCL 2.30.7 with the existing switchless-routing patches,
IB network, Ring algorithm, four channels, and LL128 excluded. No forced-tool,
model arithmetic, or server patches were added to the tested commit.

## Passed checks

- **56 rank/shape cases:** independent NCCL all-gather byte oracle, output guards,
  changing-input CUDA graph replay, and mixed operation graphs on all four ranks.
  The adapter calls production pair/quarter/diagonal operations and the full
  gather helper used by Markov. Sizes include the 512 KiB cutoff and alignment
  fallbacks. Four-rank startup mode agreement also executed successfully.
- **9 short model replies:** three prompts through serial, drafted and concurrent
  paths, 64 generated tokens each; exact token IDs and UTF-8 agree. All nine
  also match the original commit’s replies for identical requests.
- **24 benchmark replies:** four requests × six timed bursts, 256 tokens each;
  UTF-8 agrees across repetitions for each request.
- **11 longer-prompt replies:** four approximately 8K-token prompts (requested
  lengths 8192), plus a seeded-sampling case, burst/solo replay and one serial
  reference. All 5/5 burst/solo cases match exact token IDs; drafting matches
  serial. All four recall markers were found in both greedy paths. Solo replays
  reported 6144 cached tokens each. This exercises prompt chunking and retained
  prefixes, not the full configured context.
- **Cancellation:** disconnect after an active content chunk, then exact equality
  of the next serial and drafted 64-token replies.
- **Sustained serving:** 90-second four-stream run, 62 requests, no worker errors.

Transport timings below are complete-operation candidate latencies: slowest
rank's mean of two measurement rounds. They are not a target-branch comparison.

| Diagonal payload | Latency |
|---|---:|
| 8.75 KiB | 18.34 µs |
| 52.5 KiB | 27.91 µs |
| 140 KiB | 44.79 µs |
| 420 KiB | 82.12 µs |

[transport.json](transport.json) contains every candidate rank/shape result.

## Serving measurements

The recorded workload is in [workloads.json](workloads.json). Greedy decoding,
four streams and 256-token maximum outputs were used. Set-a is the same public
prose/code/structured-output workload used during transport selection. One warmup
preceded six measured bursts; steady admission used set-a for 30 seconds. The
separate 90-second sustained run used set-b. Different workloads are not directly
comparable, and these short measurements are not confidence estimates.

| Workload | Aggregate tokens/s | Detail |
|---|---:|---|
| Set-a four-stream bursts | **133.11** | Mean of six; range 132.49–133.52 |
| Set-a steady admission | **149.00** | 30 seconds, 19 requests |
| Set-b sustained | **170.20** | 90 seconds, 62 requests; TTFT p50 0.166 s, p95 0.262 s |

The suite repeats the original commit’s workloads on the same ring. Historical
burst/steady/sustained rates were 132.63/148.83/169.77 tok/s; these were not
alternating A/B runs, and the sub-percent differences do not establish a speedup.

## Host checks and limits

A fresh `zig build test-dsv41-neighbor --summary all`
passed **15/15**. `zig build test test-golden -Dcpu=apple_m1 --summary all` again
reported the inherited macOS `std.c.SO.BUSY_POLL` compile blocker; the frozen
corpus remained 309/319 equal, ten known differences, zero unexpected. The full
host command is not reported as a pass. See the [host receipt](../dsv41-neighbor-host/README.md)
for the Linux node build, local CI checks and inherited failures.

Full-1M quality, 160K cold prefill, 16-way sustained load, fault injection,
restart/reboot and long-duration reliability were not tested on this consolidated
branch. Unequal-width diagonal forwarding has host call-sequence coverage; this
GPU probe does not claim exhaustive model shapes or distributed failure handling.
No target-branch speedup is claimed. The previously deployed build was resumed
after successful bounded qualification; the PR was not deployed as a replacement.
