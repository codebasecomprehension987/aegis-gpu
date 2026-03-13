// aegis-gpu/benches/scheduler_bench.rs
// Criterion benchmarks for the Aegis scheduler hot path.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use aegis_gpu::{
    hypervisor::GuestId,
    scheduler::{FairScheduler, ScheduleDecision},
};

fn bench_scheduler_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler");

    for n_guests in [2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("next_context", n_guests),
            &n_guests,
            |b, &n| {
                let mut sched = FairScheduler::new(n);
                for _ in 0..n {
                    sched.register(GuestId::new(), 8);
                }
                b.iter(|| {
                    let _ = std::hint::black_box(sched.next_context());
                });
            },
        );
    }
    group.finish();
}

fn bench_scheduler_register_deregister(c: &mut Criterion) {
    c.bench_function("register_deregister", |b| {
        b.iter(|| {
            let mut sched = FairScheduler::new(16);
            let ids: Vec<GuestId> = (0..16).map(|_| GuestId::new()).collect();
            for &id in &ids {
                sched.register(id, 8);
            }
            for &id in &ids {
                std::hint::black_box(sched.deregister(id));
            }
        });
    });
}

criterion_group!(benches, bench_scheduler_tick, bench_scheduler_register_deregister);
criterion_main!(benches);
