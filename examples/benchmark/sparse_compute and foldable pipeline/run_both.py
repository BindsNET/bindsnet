from _bench_common import run_speedup

# Run both sparse_compute.py and foldable_pipelines.py
if __name__ == "__main__":
    run_speedup(
        baseline_mode="expansion",
        test_mode="fold_sparse",
        technique="Foldable pipelines + sparse compute (combined)",
    )
