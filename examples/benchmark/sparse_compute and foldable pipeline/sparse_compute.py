from _bench_common import run_speedup

### Benchmark: sparse compute ###
# Measures the speedup of activity-sparse compute (``sparse_compute=True`` -- read
# only the weight rows of source neurons that spiked) over the dense folded compute,
# on the ExampleNetwork (CPU and GPU).
if __name__ == "__main__":
    run_speedup(baseline_mode="fold", test_mode="fold_sparse", technique="Sparse compute")
