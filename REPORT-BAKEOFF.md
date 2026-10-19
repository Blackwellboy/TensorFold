# REPORT-BAKEOFF — DeepSeek-V4.1-Flash-1017 on the public TP2 Zig line

Thread: BAKEOFF-1019, DeepSeek-V4.1 side (the other thread runs the same brief on GLM-5.3 Flash).
Worktree `/tank/projects/dsv41-tensorfold/port1017-ds-wt`, branch `tp2/port1017-ds`, base public
`deepseek-v41-zig` 3de79d83. Port commit **84096b4d56098cfd8b20e987dd751b6ec1d7f0b2** (author and committer
`bertholomus <311723197+bertholomus@users.noreply.github.com>`, no trailers). Nothing pushed, no nodes touched; only
read-only ssh to A5/A6 for pack headers.

## What the port needed (read from the TP4 line, not guessed)

- TP4 v0.2.0 serves 1017 with `tools/serving_1017.py` (a `serving-1017.safetensors` shim: markov head renamed to the
  engine's names, Engram q/k widened to F32) plus `chat/chat_template.jinja`; its engine reads the engine's names/F32
  already. The image routing bias is `kit-1017/vision/gate_bias_vl.safetensors` (4745da4b...), made from the original
  1017 checkpoint by `make_bias_vl.py`.
- TP2's Zig `Shards.open` already maps every `*.safetensors` in name order with a later file winning (the same rule
  as the Python line's `Shards`), so a shim is visible to both. What TP2 lacked:
  - `checkpoint.source()` only looks up the engine's names, so raw 1017 packs (markov_w1/w2) fail with MissingTensor;
  - `checkpoint.read()` converts f16 -> f32 only, so the 1017 packs' bf16 Engram q/k fail with UnsupportedConversion;
  - the 1017 checkpoint's own markov head is bf16 (our converter casts it to f16; the A6 community 2.8 bpw pack keeps
    bf16) while the markov kernel reads fp16;
  - the line had no 1017 prep tool and no 1017 chat template.
- The A5 prepared 2.9 bpw pack (`~/models/deepseek-v41-1017-exl3-2.9bpw/model`) and the September packs already carry
  the engine's names and dtypes (F32 Engram, F16 head), so they must keep loading exactly as before.

## Changes

| File | What |
|---|---|
| `zig/src/families/dsv41/checkpoint.zig` | `converterName()`: engine markov name -> converter markov_w1/w2 name. `source()` falls back to it **only when the engine's name is absent** (September and prepared packs never reach it). `read()`: exact bf16 -> f32 widening for a plain key that asks f32 (Engram q/k). `narrowBfloat()` (pub): exact bf16 -> f16 for the markov head. Tests: converter-name mapping, bf16->f16 exactness, a synthetic 1017 shard, a pack with the engine's own names (canonical wins; f32 copied), the real headers of five packs (`fixtures/1017-headers.json`), an env-gated real-pack read (`TF_DSV41_PACK`) and an env-gated planned-entry digest (`TF_DSV41_DIGEST_PACK`). |
| `zig/src/families/dsv41/weights.zig` | `Source.dtype()`; `Loader.markovHead()` converts a bf16 markov head to f16 exactly at upload (an f16 head is uploaded as it is). The plan/cache key stays the checkpoint's dtype, so no rank cache is invalidated. |
| `zig/src/families/dsv41/fixtures/1017-headers.json` | The real tensor names/dtypes/shapes of the 1017 / September packs' markov and Engram entries (offsets rebased for the fixture). |
| `tools/dsv41-zig/serving_1017.py` | Port of the TP4 prep tool for this line: writes the rename shim (byte copies of `markov_w1`/`markov_w2` under `embed.weight`/`head.weight`, only when the pack has the converter's names), installs `serving/chat_template.jinja`, `--check`, and notes which tensors the engine converts at load. |
| `tools/dsv41-zig/serving/chat_template.jinja` | Our 1017 template (sha256 `b581d19d...`), from the TP4 line where it was checked 71/71 against DeepSeek's reference renderings. |
| `tools/dsv41-zig/serving/README.md` | How a 1017 pack is served on this line: what the engine handles at load, what the tool adds, the image bias and the Engram tables. |
| `zig/tests/dsv41/template_check.zig` | Test root: the template module compiles and renders the 1017 template. |

## What was verified (commands and results)

**1. Host tests, base vs branch.** On FrankenNAS, Zig 0.17.0 (`/tank/projects/glm53-tensorfold/toolchain/zig-x86_64-linux-0.17.0/zig`):

- base 3de79d83, clean worktree: `zig build test` -> rc=0; the dsv41 module's test binary: **113/113 passed**.
- branch: `zig build test` -> rc=0; dsv41 module test binary: **118 passed, 2 skipped, 0 failed** (the two env-gated
  tests skip without their variables); with `TF_DSV41_PACK` set: **119 passed, 1 skipped, 0 failed**. No new failures.

**2. aarch64 node build.** In `nvcr.io/nvidia/pytorch:26.07-py3` (Zig mounted read-only, no GPU):

```
zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast \
  -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
```

-> rc=0, no errors; `file zig-out/native/bin/tensorfold-native` -> `ELF 64-bit LSB executable, ARM aarch64`;
`zig-out/bin/tf-dsv41-lanes` likewise aarch64. Re-run after the final source edit (artifacts 17:32-17:33Z).

**3. The 1017 loader paths, from real pack headers.** Packs read: tank
`publish/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw` (ours; markov_w1/w2 + bf16 Engram), tank
`work/dsv41-1017-exl3-2.9bpw` (same), A5 `~/models/deepseek-v41-1017-exl3-2.9bpw/model` (prepared: engine names, f32
Engram, f16 head), A6 `~/ai/models/jayleaton-...-2.8bpw` (engine names, bf16 Engram and bf16 head), tank
`banked/DeepSeek-V4.1-Flash-EXL3-2.9bpw--64ba41b6` (September served pack). Only headers were read over ssh on A5/A6.

- `zig build test` includes `1017 packs from their real headers...`: builds a shard from each pack's real header
  entries and asserts, per pack, that the engine's keys resolve and read as expected (converter names, bf16 -> f32,
  canonical names, f32 copied). Passes for all five.
- Env-gated real-pack read on full packs (`TF_DSV41_PACK=... <dsv41 test binary>`):

```
TF_DSV41_PACK /tank/models/llm/publish/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw: the markov head names and the Engram q/k read under the engine's keys
TF_DSV41_PACK /tank/models/llm/work/dsv41-1017-exl3-2.9bpw: ...
TF_DSV41_PACK /tank/models/llm/banked/DeepSeek-V4.1-Flash-EXL3-2.9bpw--64ba41b6: ...
TF_DSV41_PACK /tmp/pack-a5: ...   (rebuilt from A5's real header, pattern body)
TF_DSV41_PACK /tmp/pack-a6: ...   (rebuilt from A6's real header, pattern body)
```

  all with "All 119 tests passed"; a bad path fails with FileNotFound, so the variable does reach the test.

**4. The prep tool on the real packs.** `tools/dsv41-zig/serving_1017.py` (temp dirs, packs read-only/symlinked):

- ours 4.0 bpw (real shards): writes `serving-1017.safetensors` with `mtp.2.markov_head.embed.weight` (BF16) and
  `.head.weight` (F16), **byte-identical to the pack's `markov_w1` / `markov_w2`**; shim sha256
  `3c0161031b91c3c51206a4a42d59c7f8fa0ec591ec38ba743529962e22dc7c94`; `--check` rc=1 before the write (the files are absent),
  rc=0 after; notes name the four bf16 Engram q/k the engine widens at load.
- A6 jayleaton 2.8 bpw (real header): no shim needed; notes name the bf16 markov head and the bf16 Engram; `--check`
  rc=0.
- A5 2.9 bpw prepared and September served packs: no shim needed; `--check` rc=0.
- A tampered `chat_template.jinja` makes `--check` exit 1.

**5. September weights unchanged.** The change is additive: the alias is tried only after the engine's name misses,
bf16 -> f32 only when a key asks f32 and the source is bf16, and the head conversion only when the source is bf16.
Byte-level check: a digest over 3,663 planned entries (embedding, head, all DSpark tensors, Engram q/k and layer 0's
non-expert tensors; 5,452,069,472 bytes) of the September served pack, read through the same loader on base and branch:

```
base 3de79d83 : TF_DSV41_DIGEST_PACK ...: 3663 entries, 5452069472 bytes, sha256 dbcf9abc306142ae22915b101d457aa5ea4c4e8ec3e99136aaeb5ee9e2ad1491
branch 84096b4d: TF_DSV41_DIGEST_PACK ...: 3663 entries, 5452069472 bytes, sha256 dbcf9abc306142ae22915b101d457aa5ea4c4e8ec3e99136aaeb5ee9e2ad1491
```

**6. Chat template on this engine.** `zig test -lc --dep template -Mroot=zig/tests/dsv41/template_check.zig
-Mtemplate=zig/src/core/template/template.zig` with `TPL=tools/dsv41-zig/serving/chat_template.jinja` -> the TP2
template module compiles and renders it (170 bytes for the sample; header system/user spans present).

**7. Image routing bias.** `python3 tools/dsv41-zig/kit/make_bias_vl.py
/tank/models/llm/banked/DeepSeek-V4.1-Flash-1017--1bae0e10 /tmp/bias-1017.safetensors` -> 43 tensors, 67192 bytes,
sha256 `4745da4b5f24c67170ee24e2c95652e45ba13c0ca0abc2be3cc7a2a3545792d0`, identical to the `kit-1017` MANIFEST entry.
So the existing kit tool covers the 1017 bias; no code change.

## What I could not verify (off-node limits)

- No GPU, container, service or node run: no end-to-end serve, no replies, no speed, no MTP acceptance, no real
  rank-cache build. The engine's loader paths are proven on host; the kernel paths are not exercised.
- The TP2 server requires `TF_DS_RANK_CACHE`, written by the Python line's engine (`deepseek-v41-tp2`, `weights.py`).
  I read that loader (it maps all `*.safetensors` in name order, later wins, and `.to(torch.float32)` handles bf16),
  and the tool's shim is built for it, but I did not run it (needs a GPU).
- The jayleaton 2.8 bpw pack has no Engram tables (needs the original 1017 shards 47/48 as `--engram`) and ships its
  own chat template; serving it needs ours installed and the engram dir.
- The bf16 -> f16 markov-head conversion is unit-tested exact but not exercised on a GPU.
- Template correctness beyond compile/render is the TP4 line's 71/71 result; here only parse/render on this engine.
- The 4.0 bpw and work 2.9 bpw packs contain each Engram q/k twice (byte-identical; the index maps the later shard,
  and the loaders' last-file-wins picks the same one). Noted, not a problem.
- A5/A6 packs: headers only (as allowed); full-tensor reads were done on the tank copies.

## Time spent

08:35 - 10:45 AM PDT, 2026-10-19: about 2 h 10 min (recon 20 min, loader + tests 55 min, tool + template 25 min,
builds and dry checks 25 min, report and commit 10 min).

BAKEOFF-DONE 84096b4d56098cfd8b20e987dd751b6ec1d7f0b2
