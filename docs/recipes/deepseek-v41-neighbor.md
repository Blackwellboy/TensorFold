# DeepSeek-V4.1 TP4 neighbor transport

The native server and lane runner can opt into adjacent-edge RDMA on a four-node
physical ring numbered `0 → 1 → 2 → 3 → 0`. It reuses the existing RDMA kernels
and NCCL; it changes byte movement, not reductions or drafting policy.

Set `TF_DS_NEIGHBOR_RDMA=1` on **every rank**. Unset or `0` retains the existing
NCCL or `TF_RDMA_DEVICES` transport. There are no separate optimization switches.
Other values fail startup. The neighbor mode requires TP4 and cannot be combined
with `TF_RDMA_DEVICES`.

For each rank, set both comma-separated device lists:

| Setting | Physical neighbor | Devices |
| --- | --- | --- |
| `TF_DS_NEIGHBOR_FIRST` | rank xor 1 | Two rails to that neighbor |
| `TF_DS_NEIGHBOR_SECOND` | rank xor 3 | Two rails to that neighbor |

Each list must contain two distinct names; the lists cannot overlap. List rails
in the same physical order at both ends of each edge (for example, by PCI domain).
Set `TF_DS_NEIGHBOR_GID_INDEX` to the local RoCE GID table index (decimal 0–255).
It defaults to 5, matching upstream's rings; sites using GID 3 must explicitly set
`TF_DS_NEIGHBOR_GID_INDEX=3`. This is separate from NCCL's GID setting and is only
read when neighbor mode is enabled. The existing MTU and queue settings are retained.
It does not detect or configure networking. NCCL must also reach adjacent ranks
and support its ordinary collectives on this fabric.

The existing startup `Settings` exchange checks each follower’s neighbor mode
against rank 0 before that follower enters NCCL communicator creation. This is
the upstream per-follower handshake, not a global rejection broadcast. No separate mode all-gather
runs, and disabled mode creates no neighbor RDMA resources. Use the same binary
on every rank: the settings frame now includes the neighbor flag. Device and GID
selection stay local to each rank; they must describe compatible physical links.
All settings stay fixed for the lifetime of the model and its captured graphs.

## Operations

Aligned, positive shards through 512 KiB use the following paths:

- Full gathers, including Markov best scores: xor1, then xor3, in global rank order.
- Pair gathers: xor1 only.
- Equal-width quarters with 16-byte-aligned rows: collect directly into the final layout.
- Parity diagonal exchange: remote-only xor1, then xor3, yielding rank xor2.

Larger or unaligned shards retain NCCL. Unequal-width quarters retain padded
gather and layout copies. In neighbor mode, diagonal NCCL traffic is forwarded
over two physical edges; it never requires a direct diagonal link. This includes
the Engram exchange, whose existing arithmetic and order are retained.

Each rank owns two existing RDMA ring instances, two proxy threads, and 1 MiB
of device scratch in addition to the rings' registered host buffers. Operations
use the model stream in order; scratch is not shared across streams. A detected
local stage failure aborts both stages' device waits, and model synchronization
checks the failure before returning results. This adds no watchdog, retry,
recovery policy, or distributed failure guarantee.

## Validation

Use the pinned Zig 0.17.0 and the existing host checks:

```sh
zig build test-dsv41-neighbor
zig build test test-golden -Dcpu=apple_m1
python3 tools/zig/lean_check.py
```

The focused host target checks default-off mode and GID parsing, bounded dispatch, device-list
validation, relay preconditions, the production NCCL forwarding calls at every
rank with unequal counts, and local abort propagation. It also compiles the
GPU/verbs declarations. The family host suite checks startup mode disagreement,
including both directions of mixed enablement over the existing settings link.
The host tests do not execute GPU kernels, RDMA traffic, captured
graphs, or a model.

Before hardware acceptance, compare all output bytes with an independent NCCL
reference at all four ranks, including pair/full/quarter/diagonal operations,
changing-input graph replay, cutoff and unequal-width fallbacks, and mixed
operation sequences. Then apply CONTRIBUTING.md's plain/drafted,
concurrent/solo and resumed/fresh token gates and matched serving benchmarks.
Prototype measurements on a different source revision do not qualify this branch.

The [consolidated-branch hardware receipt](../receipts/dsv41-neighbor-hardware/README.md)
records the tested code commit, ring setup, byte/graph/token checks and serving
measurements. It does not claim a direct comparison with a switched fabric or the
target branch.

The [v0.2.0 update receipt](../receipts/dsv41-neighbor-v020/README.md) records
the rebased code checks separately from the historical hardware measurements.

The [v0.2.0 hardware repeat](../receipts/dsv41-neighbor-v020-hardware/README.md)
records fresh byte/graph/token checks and the same serving workloads on the rebased code.
