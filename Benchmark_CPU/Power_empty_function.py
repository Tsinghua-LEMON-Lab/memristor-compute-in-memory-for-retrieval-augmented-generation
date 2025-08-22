import os, time, numpy as np
from contextlib import contextmanager

# --------------- 1. 环境与参数 ---------------
# os.environ["OMP_NUM_THREADS"] = "16"   # BLAS 线程
K = 100                         # Top‑k

# ---------------- 2. 批量计时循环 ------------------
BATCH = 128                              # 64 条 query 一起跑
NUM_RUNS = BATCH * 300                         # 实验用查询总数
d  = 256
N  = 65536

np.random.seed(0)
base  = np.random.rand(N, d).astype(np.float32)
query = np.random.rand(NUM_RUNS, d).astype(np.float32)

# --------------- 2. ONE‑OFF: 索引构建 ----------------
build_start = time.perf_counter()
b_norm2 = np.sum(base * base, axis=1)       # 假装的 build 工作
build_time  = time.perf_counter() - build_start
print(f"[build]  耗时 {build_time*1000:.1f} ms")

def run_query_batch(q_batch: np.ndarray, k: int = K):
    i = 1
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.069:
        i=1
    # time.sleep(0.0892)
    # return

for off in range(0, NUM_RUNS, BATCH):
    q_batch = query[off : off + BATCH]
    if q_batch.size == 0:
        break                           # 末尾不够一个 batch
    run_query_batch(q_batch)







