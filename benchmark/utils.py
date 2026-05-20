import functools
import numpy as np


def create_benchmark_test(
    benchmark,
    reader_func,
    file_path,
    name,
    ref_array,
    memory_profile,
    sum_strands=True,
    ref_reader_func=None,
):
    timed_reader = functools.partial(reader_func, file_path, sum_strands=sum_strands)

    if memory_profile:
        from memory_profiler import memory_usage

        def run_with_memory_profile():
            def read_file():
                return timed_reader()

            mem_usage, result = memory_usage(
                (read_file, [], {}),
                retval=True,
                interval=0.1,
                include_children=True
            )
            return result, max(mem_usage)

        # Run benchmark with memory profiling
        result, max_mem = benchmark.pedantic(
            run_with_memory_profile,
            rounds=1,
            iterations=1,
        )
        benchmark.extra_info['max_memory_mb'] = max_mem

    else:
        # Use fixed rounds so slow readers do not perform extra full-file
        # calibration reads before the reported measurements.
        result = benchmark.pedantic(
            timed_reader,
            rounds=3,
            iterations=1,
        )
        benchmark.extra_info['max_memory_mb'] = None

    # Verify output (not timed)
    if ref_array is None:
        if ref_reader_func is None:
            raise ValueError("ref_reader_func is required when ref_array is None.")
        ref_array = ref_reader_func(file_path, sum_strands=sum_strands)
    assert np.array_equal(result, ref_array), f"Output does not match reference for {name}"
