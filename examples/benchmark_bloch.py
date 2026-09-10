#!/usr/bin/env python3
"""Benchmark Bloch simulations. For use to compare performance across different OpenMP thread counts."""

import argparse
import time

import numpy as np
from bloch import bloch, threads


def build_problem():

    # Slice-selective since pulse with rewinding gradient (same in benchmark_bloch.m and benchmark_bloch.py)
    dur = 2e-3
    dt = 10e-6
    ntime = round(dur / dt)
    flip = np.pi / 2
    slthk = 5e-3
    TBWP = 8
    t = (np.arange(1, ntime + 1) - 0.5) * dt - (dur / 2)
    alpha = 0.5
    window = (1 - alpha) + alpha * np.cos(2 * np.pi * t / dur)
    b1 = np.multiply(window, np.sinc(t * TBWP / dur))
    b1 *= flip / (2 * np.pi * np.sum(b1) * dt)
    b1 = np.concatenate([b1, np.zeros(ntime // 2)])
    gz = np.concatenate(
        [
            np.ones(ntime) * TBWP / dur / slthk,
            np.ones(ntime // 2) * -TBWP / dur / slthk,
        ]
    )
    dp = np.linspace(-slthk, slthk, 100)
    df = np.linspace(-100, 100, 100)
    dv = np.linspace(-0.5, 0.5, 100)
    return b1, gz, dt, dp, df, dv


def benchmark_once(b1, gz, dt, dp, df, dv):
    t0 = time.perf_counter()
    _, _, _ = bloch(b1, gz, dt, dp=dp, df=df, dv=dv, mode=0)
    elapsed = time.perf_counter() - t0
    return elapsed


def run_benchmark(repeats, threads_list):

    original_threads = max(1, threads())

    b1, gz, dt, dp, df, dv = build_problem()

    print(f"Problem size: dp={dp.size}, df={df.size}, dv={dv.size}, total= {dp.size * df.size * dv.size}")
    print(f"Runs per thread count: {repeats}")

    try:
        for nthreads in threads_list:
            threads(nthreads)

            times = []
            for _ in range(repeats):
                elapsed = benchmark_once(b1, gz, dt, dp, df, dv)
                times.append(elapsed)

            print(f"=== OpenMP threads = {nthreads} ===")
            print(f"Benchmark run time results: mean={np.mean(times):.3f} s"
                f", best={np.min(times):.3f} s"
                f", worst={np.max(times):.3f} s\n")

    finally:
        print(f"Restoring original thread count (threads={original_threads})")
        threads(original_threads)

def main():
    parser = argparse.ArgumentParser(description="Benchmark Bloch simulation speed-up across thread counts.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per thread count (default: 3)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=None,
        help="List of thread counts to benchmark (default: 1, current max threads/2, current max threads).",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("Repeats must be a positive integer.")

    if args.threads is None:
        args.threads = [1, round(threads()/2), threads()]

    run_benchmark(args.repeats, args.threads)


if __name__ == "__main__":
    main()
