from _bench_common import run_speedup

### Benchmark: foldable pipelines ###
# Measures the speedup of the folded MultiCompartmentConnection compute (``out = s @ A + B.sum(0)``) over
# the pre-fold ``[batch, src, tgt]`` expansion path, on the ExampleNetwork
if __name__ == "__main__":
    run_speedup(baseline_mode="expansion", test_mode="fold", technique="Foldable pipelines")
