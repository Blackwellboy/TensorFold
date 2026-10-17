//! A stream that sends nothing for Config.sse_keepalive_ms (a long prompt's prefill) sends SSE comments until its
//! reply comes; with the setting off its bytes are as before.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const clock = @import("clock.zig");
const Conn = @import("http_conn.zig").Conn;

/// Bytes as tokens; every prompt renders as "prompt" (6 tokens).
const TestText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine that answers "hi" `delay_ms` after the submit, from a thread of its own (a long prefill).
const SlowEngine = struct {
    delay_ms: i64,
    thread: ?std.Thread = null,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 64 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *SlowEngine = @ptrCast(@alignCast(ctx));
        e.thread = std.Thread.spawn(.{}, answer, .{ e.delay_ms, id, sink }) catch return error.Closed;
    }

    fn answer(delay_ms: i64, id: api.Id, sink: api.Sink) void {
        std.Io.sleep(std.testing.io, .fromMilliseconds(delay_ms), .awake) catch {};
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'h', 'i' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The stream as its client saw it, in order: 'e' a data event (null: [DONE]), 'c' a comment; and how long it took.
const Seen = struct {
    order: [64]u8 = undefined,
    n: usize = 0,
    ms: i96 = 0,

    fn out(c: *Seen) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply, .comment = comment } };
    }

    fn push(c: *Seen, k: u8) void {
        if (c.n == c.order.len) return;
        c.order[c.n] = k;
        c.n += 1;
    }

    fn open(_: *anyopaque) error{Closed}!void {}

    fn event(ctx: *anyopaque, _: ?json.Value) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.push('e');
    }

    fn comment(ctx: *anyopaque) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.push('c');
    }

    fn reply(_: *anyopaque, _: u16, _: json.Value) void {}

    fn seen(c: *const Seen) []const u8 {
        return c.order[0..c.n];
    }
};

/// A streamed chat whose reply comes after `delay_ms`, through a server whose keepalive is `keepalive_ms` (0: off).
fn stream(keepalive_ms: u32, delay_ms: i64) !Seen {
    var text: TestText = .{};
    var backend: SlowEngine = .{ .delay_ms = delay_ms };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
        .sse_keepalive_ms = keepalive_ms,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":2}")).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{};
    const start = clock.nowNs(std.testing.io);
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, raw);
    seen.ms = @divTrunc(clock.nowNs(std.testing.io) - start, std.time.ns_per_ms);
    if (backend.thread) |t| t.join();
    return seen;
}

test "a stream silent past the keepalive sends comments after its role chunk until the reply, and none when off" {
    const on = try stream(100, 450);
    const comments = std.mem.count(u8, on.seen(), "c");
    // one per 100 ms of the 450 ms silence (the generation looks every 50 ms), and never two within 100 ms
    try std.testing.expect(comments >= 2 and comments <= @divTrunc(on.ms, 100));
    // the role chunk, then every comment, then the reply's own events
    try std.testing.expectEqual(@as(?usize, 1), std.mem.indexOfScalar(u8, on.seen(), 'c'));
    try std.testing.expectEqual(@as(?usize, comments), std.mem.lastIndexOfScalar(u8, on.seen(), 'c'));
    // off: no comment, and the same events as the stream with them
    const off = try stream(0, 450);
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, off.seen(), "c"));
    var plain: [64]u8 = undefined;
    var k: usize = 0;
    for (on.seen()) |x| if (x != 'c') {
        plain[k] = x;
        k += 1;
    };
    try std.testing.expectEqualStrings(off.seen(), plain[0..k]);
    try std.testing.expect(off.n >= 3); // the role, the text, the last chunk
}
