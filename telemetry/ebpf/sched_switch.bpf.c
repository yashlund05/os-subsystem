#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

/* eBPF Map for Ring Buffer using BPF_MAP_TYPE_RINGBUF */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024); /* 256 KB ringbuf */
} telemetry_ringbuf SEC(".maps");

/* Compact 16-Byte Telemetry Struct per Context Switch */
struct task_telemetry {
    __u32 pid;
    __u16 elapsed_us;
    __u16 cache_misses_delta;
    __u16 branch_mispred_delta;
    __u16 mem_footprint_kb;
    __u16 flags;
} __attribute__((packed));

/* Telemetry flag bit definitions */
#define TELEMETRY_FLAG_NONE           0x0000
#define TELEMETRY_FLAG_PREEMPTED      0x0008

SEC("tracepoint/sched/sched_switch")
int handle_sched_switch(struct trace_event_raw_sched_switch *ctx)
{
    /* Previous task that is being switched out */
    pid_t prev_pid = ctx->prev_pid;
    
    /* We only care about user-space tasks, skip swapper/idle (pid 0) */
    if (prev_pid == 0)
        return 0;

    struct task_telemetry *event;
    
    /* Reserve space in the ring buffer */
    event = bpf_ringbuf_reserve(&telemetry_ringbuf, sizeof(*event), 0);
    if (!event)
        return 0; /* Ring buffer full, drop event */

    /* Populate the 16-byte event struct */
    event->pid = prev_pid;
    
    /* In a real implementation we would fetch these from PMU counters and task_struct.
       For the scaffold, we populate with dummy values or basic info */
    event->elapsed_us = 1000; /* Dummy value */
    event->cache_misses_delta = 0;
    event->branch_mispred_delta = 0;
    event->mem_footprint_kb = 0;
    
    /* Check if the task was preempted (TASK_RUNNING state but switched out) */
    long prev_state = ctx->prev_state;
    event->flags = TELEMETRY_FLAG_NONE;
    if (prev_state == 0) {
        event->flags |= TELEMETRY_FLAG_PREEMPTED;
    }

    /* Submit the event to user-space */
    bpf_ringbuf_submit(event, 0);

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
