//! TF_DS_IDX_SPLIT: the keyed indexer layers' long rows split by columns over a run's nodes (prompt.zig indexer). The
//! layers before the candidate source take each row's k largest int64 keys (index_score's: a score's bits above its
//! inverted column) and their columns ascending (topk_select_pruned). Every node holds the same index keys and runs
//! the same chunk, so each one scored every column. Split, node g of W scores whole 64-column tiles [c0, c0 + n) of
//! each row with the same launches on the same row blocks (index_score of a column range: its columns' own score bits,
//! each key's column counted from c0), keeps their k largest keys (topk_select_pruned's tile step, then torch's top-k)
//! and moves them to the row's columns (ops.keysRebase: the keys the served scores hold there); the nodes all-gather
//! those keys and topk_select takes the k largest of the W k. Each of a row's k largest keys is among its node's k
//! largest (fewer than k keys of that node's columns lie above it), so the same k keys, and the same columns out, as
//! the served selection over the whole row.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("tri.zig");
const tri_index = @import("tri_index.zig");
const ops = @import("ops.zig");

/// TF_DS_IDX_SPLIT=1's least row width (a keyed layer's compressed entries): from it the scores a node leaves to the
/// others outweigh the all-gather and the merge (tf-dsv41-split-bench).
pub const default_min: usize = 32768;

/// The columns a node scores: [c0, c0 + n).
pub const Span = struct { c0: usize, n: usize };

/// Node g of `world`'s columns of a row of n: whole 64-column tiles from the row's start, the last node the rest.
pub fn span(n: usize, world: usize, g: usize) Span {
    const w = tri.cdiv(tri.cdiv(n, world), tri_index.tile) * tri_index.tile;
    const c0 = @min(n, g * w);
    return .{ .c0 = c0, .n = @min(n, c0 + w) - c0 };
}

/// A row's visible columns among a node's (vis: the row's).
pub fn nodeVis(vis: usize, sp: Span) i64 {
    return @intCast(@min(vis -| sp.c0, sp.n));
}

/// The index keys of a node's columns: the cache's rows from c0 (FP4: codes ID/2 bytes a row, scales ID/32).
pub fn nodeKeys(k: tri_index.IndexK, c0: usize, id: usize) tri_index.IndexK {
    return switch (k) {
        .fp4 => |f| .{ .fp4 = .{ .codes = f.codes + c0 * id / 2, .scales = f.scales + c0 * id / 32 } },
        .bf16 => |b| .{ .bf16 = b + c0 * id * 2 },
    };
}

/// A node's part of a chunk's rows (q [rows, IH, ID] fp4_qd's bytes, w [rows, IH], vis [rows] the node's: nodeVis):
/// index_score's keys and tile maxima of its columns in the served row blocks of rb rows (the same blocks, so a row
/// takes the same kernel as served), each block's k largest keys (tri_index.topKeysPruned) into top [rows, k], then
/// moved to the row's columns. keys and tmax: a row block's scratch, as the served path's; vis_blk (int64 [rb]): a
/// block's vis when its view is not 16-byte aligned (prompt.zig visBlock: the kit's variants take VIS aligned).
pub fn nodeTop(t: tri.Tri, o: *const ops.Ops, s: cuda.Stream, q: u64, k: tri_index.IndexK, w: u64, vis: u64, vis_blk: u64, sp: Span, keys: u64, tmax: u64, bufs: tri_index.PrunedBufs, top: tri_index.TopK, out: u64, rows: usize, rb: usize, kk: usize, ih: usize, id: usize) !void {
    if (sp.n < kk) return error.SplitTooNarrow;
    const kx = nodeKeys(k, sp.c0, id);
    var r0: usize = 0;
    while (r0 < rows) : (r0 += rb) {
        const m = @min(rows, r0 + rb) - r0;
        const view = vis + r0 * 8;
        const vr = if (view % 16 == 0) view else blk: {
            try o.copyRows(s, view, m * 8, vis_blk, m * 8, m * 8, 1);
            break :blk vis_blk;
        };
        try tri_index.indexScore(t, q + r0 * ih * id * 2, kx, w + r0 * ih * 2, vr, sp.n, keys, null, true, tmax, false, m, ih, id);
        try tri_index.topKeysPruned(t, keys, sp.n, tmax, kk, out + r0 * kk * 8, bufs, top, m, sp.n);
    }
    try o.keysRebase(s, out, kk, kk, sp.n, sp.c0, rows);
}

/// The k largest of the nodes' keys: parts [world, rows, k] int64 (node after node, as the all-gather lands them) into
/// merged [rows, world * k], then topk_select's torch steps (torch's top-k and _topk_finish, as the served selection
/// ends) -> out [rows, k]: the k largest keys' columns ascending, -1 at or past vis (the row's).
pub fn merge(t: tri.Tri, o: *const ops.Ops, s: cuda.Stream, parts: u64, merged: u64, world: usize, rows: usize, kk: usize, vis: u64, out: u64, top: tri_index.TopK) !void {
    for (0..world) |g| try o.copyRows(s, parts + g * rows * kk * 8, kk * 8, merged + g * kk * 8, world * kk * 8, kk * 8, rows);
    try tri_index.topkSelectTorch(t, merged, world * kk, kk, vis, out, top, rows, world * kk);
}

/// tf_ds_keys_rebase_kernel's arithmetic on one key (the tests' host model): a node's key of its column j (from c0,
/// below n) as the row's, its score word kept; a pad as it was.
pub fn rebaseKey(key: i64, n: usize, c0: usize) i64 {
    const u: u64 = @bitCast(key);
    if (0xFFFFFFFF - (u & 0xFFFFFFFF) >= n) return key;
    return @bitCast(u - @as(u64, c0));
}

// -- tests: the merge's argument on the host --------------------------------------------------------------------

/// index_score's key of score x at column j (tf_ds_score_key, signed).
fn scoreKey(x: f32, j: usize) i64 {
    var bits: u32 = @bitCast(x);
    if (bits == 0x80000000) bits = 0;
    const sb: i32 = @bitCast(bits);
    const ordered: i32 = if (sb < 0) sb ^ 0x7FFFFFFF else sb;
    return @bitCast((@as(u64, @as(u32, @bitCast(ordered))) << 32) | (0xFFFFFFFF - @as(u64, j)));
}

/// topk_select on the host: the k largest keys' columns ascending, -1 at or past vis.
fn select(a: std.mem.Allocator, keys: []const i64, k: usize, vis: usize) ![]i64 {
    const sorted = try a.dupe(i64, keys);
    defer a.free(sorted);
    std.mem.sort(i64, sorted, {}, std.sort.desc(i64));
    const cols = try a.alloc(i64, k);
    for (sorted[0..k], cols) |key, *c| c.* = @intCast(0xFFFFFFFF - (@as(u64, @bitCast(key)) & 0xFFFFFFFF));
    std.mem.sort(i64, cols, {}, std.sort.asc(i64));
    for (cols) |*c| {
        if (c.* >= @as(i64, @intCast(vis))) c.* = -1;
    }
    return cols;
}

test "a node's columns: whole tiles, every column once, the last node the rest" {
    for ([_]usize{ 32768, 32769, 100_000, 524_287, 4096 }) |n| {
        for ([_]usize{ 2, 4 }) |world| {
            var next: usize = 0;
            for (0..world) |g| {
                const sp = span(n, world, g);
                try std.testing.expectEqual(next, sp.c0);
                try std.testing.expectEqual(@as(usize, 0), sp.c0 % tri_index.tile);
                next = sp.c0 + sp.n;
            }
            try std.testing.expectEqual(n, next);
        }
    }
    try std.testing.expectEqual(Span{ .c0 = 24576, .n = 8192 }, span(32768, 4, 3));
    try std.testing.expectEqual(Span{ .c0 = 16384, .n = 16384 }, span(32768, 2, 1));
    try std.testing.expectEqual(@as(i64, 0), nodeVis(1000, .{ .c0 = 2048, .n = 2048 }));
    try std.testing.expectEqual(@as(i64, 952), nodeVis(3000, .{ .c0 = 2048, .n = 2048 }));
    try std.testing.expectEqual(@as(i64, 2048), nodeVis(9000, .{ .c0 = 2048, .n = 2048 }));
}

test "a node's keys moved to the row's columns: the row's keys there, pads as they were" {
    const c0: usize = 8192;
    const n: usize = 4096;
    try std.testing.expectEqual(scoreKey(1.5, c0 + 7), rebaseKey(scoreKey(1.5, 7), n, c0));
    try std.testing.expectEqual(scoreKey(-std.math.inf(f32), c0 + 150), rebaseKey(scoreKey(-std.math.inf(f32), 150), n, c0));
    try std.testing.expectEqual(scoreKey(-0.0, c0 + n - 1), rebaseKey(scoreKey(-0.0, n - 1), n, c0));
    // the largest score word over the smallest inverted column: no borrow into it
    try std.testing.expectEqual(scoreKey(std.math.inf(f32), c0 + n - 1), rebaseKey(scoreKey(std.math.inf(f32), n - 1), n, c0));
    // no column of the node's: as it was
    try std.testing.expectEqual(scoreKey(2.0, n), rebaseKey(scoreKey(2.0, n), n, c0));
    try std.testing.expectEqual(@as(i64, std.math.minInt(i64)), rebaseKey(std.math.minInt(i64), n, c0));
}

test "every node's k largest keys, moved and merged, select the row's columns as topk_select over the whole row" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1d5_9117);
    const rng = prng.random();
    for ([_]usize{ 2, 4 }) |world| {
        for ([_]usize{ 4096, 5000, 12_345 }) |n| {
            for (0..16) |case| try mergeCase(a, rng, world, n, 64, case);
        }
    }
}

/// One row of n keys: topk_select's columns over the whole row against the nodes' merged k largest.
fn mergeCase(a: std.mem.Allocator, rng: std.Random, world: usize, n: usize, k: usize, case: usize) !void {
    // scores with ties (a few values), -0 and -inf; rows that see all, most, barely k or fewer than k of the columns;
    // the columns past vis -inf keys, or in odd cases a pad (no column) through each tile wholly past vis
    const vis: usize = switch (case % 4) {
        0 => n,
        1 => n - n / 3,
        2 => k + case,
        else => k / 2 + case,
    };
    const pads = case % 2 == 1;
    const pad: i64 = std.math.minInt(i64);
    const keys = try a.alloc(i64, n);
    defer a.free(keys);
    for (keys, 0..) |*key, j| {
        const x: f32 = switch (rng.uintLessThan(u8, 8)) {
            0 => @floatFromInt(rng.uintLessThan(u8, 4)),
            1 => -0.0,
            2 => -std.math.inf(f32),
            else => rng.floatNorm(f32),
        };
        key.* = if (pads and j / tri_index.tile * tri_index.tile >= vis) pad else scoreKey(if (j < vis) x else -std.math.inf(f32), j);
    }
    const want = try select(a, keys, k, vis);
    defer a.free(want);
    var merged: std.ArrayList(i64) = .empty;
    defer merged.deinit(a);
    for (0..world) |g| {
        // the node's keys: the row's score words above its own inverted columns (pads as they are), its k largest
        const sp = span(n, world, g);
        const local = try a.alloc(i64, sp.n);
        defer a.free(local);
        for (local, 0..) |*key, j| {
            const row = keys[sp.c0 + j];
            key.* = if (row == pad) pad else (row & ~@as(i64, 0xFFFFFFFF)) | @as(i64, @intCast(0xFFFFFFFF - j));
        }
        std.mem.sort(i64, local, {}, std.sort.desc(i64));
        for (local[0..k]) |key| {
            const moved = rebaseKey(key, sp.n, sp.c0);
            // the key the row holds at its column
            const j: usize = @intCast(0xFFFFFFFF - (@as(u64, @bitCast(key)) & 0xFFFFFFFF));
            if (key != pad) try std.testing.expectEqual(keys[sp.c0 + j], moved);
            try merged.append(a, moved);
        }
    }
    const got = try select(a, merged.items, k, vis);
    defer a.free(got);
    try std.testing.expectEqualSlices(i64, want, got);
}
