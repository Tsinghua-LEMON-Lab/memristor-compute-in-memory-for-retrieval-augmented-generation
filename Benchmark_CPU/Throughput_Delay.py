import os, time, numpy as np
from contextlib import contextmanager

# --------------- 1. 环境与参数 ---------------
os.environ["OMP_NUM_THREADS"] = "16"   # BLAS 线程
K        = 100                         # Top‑k
NUM_RUNS = 1500                        # 统计用查询条数
WARMUP   = 10                         # 预热次数

d  = 256
N  = 65536
Q  = NUM_RUNS + WARMUP

np.random.seed(0)
base  = np.random.rand(N, d).astype(np.float32)
query = np.random.rand(Q, d).astype(np.float32)

# --------------- 2. ONE‑OFF: 索引构建 ----------------
# 暴力扫描不需要，但这里示例：可预计算 b2
build_start = time.perf_counter()
b_norm2 = np.sum(base * base, axis=1)       # 假装的 build 工作
build_time  = time.perf_counter() - build_start
print(f"[build]  耗时 {build_time*1000:.1f} ms")

# --------------- 3. 检索函数 ------------------------
def run_query(q_vec: np.ndarray, k: int = K):
    """返回 (idx, score)"""
    sim = q_vec @ base.T                     # 1×N

    idx = np.argpartition(-sim, k-1)[:k]     # Top‑k 未排序, 只关心少量 topk
    scores = sim[idx]
    order = np.argsort(-scores)
    return idx[order], scores[order]

# --------------- 4. 计时工具 ------------------------
@contextmanager
def timer_ns():
    t0 = time.perf_counter_ns()              # 纳秒级，单调
    yield lambda: (time.perf_counter_ns() - t0)

# --------------- 5. WARM‑UP -------------------------
for i in range(WARMUP):
    run_query(query[i])

# --------------- 6. 正式计时 ------------------------
latencies = np.empty(NUM_RUNS, dtype=np.int64)  # ns
for i in range(NUM_RUNS):
    # print(i)
    with timer_ns() as elapsed:
        run_query(query[WARMUP + i])
    latencies[i] = elapsed()

# --------------- 7. 统计 ----------------------------
lat_ms = latencies / 1e6
print(f"[query]  p50={np.percentile(lat_ms,50):.2f} ms  "
      f"p95={np.percentile(lat_ms,95):.2f} ms  "
      f"平均 {lat_ms.mean():.2f} ms  QPS={1000/lat_ms.mean():.1f}")
