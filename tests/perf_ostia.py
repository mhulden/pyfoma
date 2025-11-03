import math
from pyfoma.algorithms import ostia
from pyfoma.fst import FST
import time

def time_ostia(n: int = 5):
    target_fst = FST.re("'':'' | (a):(bb) | ((aa):(bb))+(((a):(bb))|('':c))")
    all_examples = [
        symbols for _, symbols in target_fst.words_nbest(n)
    ]
    all_examples = [
        ("".join(s[0] for s in ex), "".join(s[1] for s in ex)) for ex in all_examples
    ]
    time_start = time.time()
    fst = ostia(all_examples, merging_order="lex")
    return time.time() - time_start

def run_timing():
    import matplotlib.pyplot as plt
    sizes = list(range(12))
    times = []
    for x in sizes:
        times.append(time_ostia(n=2 ** x))
    plt.plot([f"{2**x}" for x in sizes], times, "-o")
    plt.xlabel("# examples")
    plt.yscale('log')
    plt.ylabel("Total time (log s)")
    plt.show()

    print(f"Average time: {times[-1] / sizes[-1]}s/ex")

if __name__ == "__main__":
    run_timing()
