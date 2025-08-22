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



# ---------------- 1. 批量版检索函数 -----------------
def run_query_batch(q_batch: np.ndarray, k: int = K):
    """
    q_batch.shape == (Qb, d)
    返回 idx_sorted:  (Qb, k)
          scores:     (Qb, k)
    """
    sim = q_batch @ base.T                 # (Qb, N)

    # ----- 只关心内积 Top‑k（相似度大→小） -----
    idx = np.argpartition(-sim, k-1, axis=1)[:, :k]   # (Qb, k)
    part = np.take_along_axis(sim, idx, axis=1)       # (Qb, k)
    order = np.argsort(-part, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    part = np.take_along_axis(part, order, axis=1)
    return idx, part

for off in range(0, NUM_RUNS, BATCH):
    q_batch = query[off : off + BATCH]
    if q_batch.size == 0:
        break                           # 末尾不够一个 batch
    run_query_batch(q_batch)
