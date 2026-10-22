//! The served RDMA ring's settings from the environment: its GID index and its queue pairs' ACK timeout.
const std = @import("std");

/// The served ring has its own GID selection, independent of NCCL's.
pub fn parseGidIndex(text: ?[]const u8) error{InvalidRdmaGidIndex}!u8 {
    const value = text orelse return 5;
    if (value.len == 0) return error.InvalidRdmaGidIndex;
    for (value) |c| if (c < '0' or c > '9') return error.InvalidRdmaGidIndex;
    return std.fmt.parseInt(u8, value, 10) catch error.InvalidRdmaGidIndex;
}

pub fn servedGidIndex() error{InvalidRdmaGidIndex}!u8 {
    const text = if (std.c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null;
    return parseGidIndex(text) catch |err| {
        std.debug.print("TF_RDMA_GID_INDEX: expected a decimal GID index in 0..255\n", .{});
        return err;
    };
}

/// The queue pairs' local ACK timeout exponent when TF_RDMA_TIMEOUT is unset: 4.096 us * 2^14 a try.
pub const default_ack_timeout: u8 = 14;

/// TF_RDMA_TIMEOUT: the local ACK timeout exponent, 1 to 31; unset or empty keeps 14, and 0 (no timeout) is refused.
pub fn parseAckTimeout(text: ?[]const u8) error{InvalidRdmaTimeout}!u8 {
    const value = text orelse return default_ack_timeout;
    if (value.len == 0) return default_ack_timeout;
    for (value) |c| if (c < '0' or c > '9') return error.InvalidRdmaTimeout;
    const t = std.fmt.parseInt(u8, value, 10) catch return error.InvalidRdmaTimeout;
    return if (t < 1 or t > 31) error.InvalidRdmaTimeout else t;
}

pub fn servedAckTimeout() error{InvalidRdmaTimeout}!u8 {
    const text = if (std.c.getenv("TF_RDMA_TIMEOUT")) |v| std.mem.span(v) else null;
    return parseAckTimeout(text) catch |err| {
        std.debug.print("TF_RDMA_TIMEOUT: expected a decimal ACK timeout exponent in 1..31\n", .{});
        return err;
    };
}

test "served GID defaults to 5 and accepts explicit 3 without silently accepting malformed settings" {
    try std.testing.expectEqual(@as(u8, 5), try parseGidIndex(null));
    try std.testing.expectEqual(@as(u8, 3), try parseGidIndex("3"));
    try std.testing.expectEqual(@as(u8, 0), try parseGidIndex("0"));
    try std.testing.expectEqual(@as(u8, 255), try parseGidIndex("255"));
    for ([_][]const u8{ "", "-1", "+3", " 3", "3 ", "0x3", "3_0", "256", "invalid" }) |value|
        try std.testing.expectError(error.InvalidRdmaGidIndex, parseGidIndex(value));
}
test "TF_RDMA_TIMEOUT: unset or empty keeps 14, 1 to 31 are taken, anything else is refused" {
    try std.testing.expectEqual(@as(u8, 14), try parseAckTimeout(null));
    try std.testing.expectEqual(@as(u8, 14), try parseAckTimeout(""));
    try std.testing.expectEqual(@as(u8, 1), try parseAckTimeout("1"));
    try std.testing.expectEqual(@as(u8, 20), try parseAckTimeout("20"));
    try std.testing.expectEqual(@as(u8, 31), try parseAckTimeout("31"));
    for ([_][]const u8{ "0", "32", "-1", "+20", " 20", "20 ", "0x14", "1_4", "256", "x" }) |bad|
        try std.testing.expectError(error.InvalidRdmaTimeout, parseAckTimeout(bad));
}
