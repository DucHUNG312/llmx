import math
import os
import sys

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD = os.path.join(_ROOT, "build")

lib = load(
    name="llmx_py",
    sources=[os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "llmx_bind.cpp")],
    extra_include_paths=[
        os.path.join(_ROOT, "include"),
        os.path.join(_ROOT, "3rdparty"),
        os.path.join(_ROOT, "3rdparty/cutlass/include"),
    ],
    extra_ldflags=[
        f"-L{os.path.join(_BUILD, 'src')} -lllmx",
        f"-L{os.path.join(_BUILD, 'src/cuda/attention')} -lllmx_cu_attention",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++20"],
    verbose=True,
)


def ref_attention(
    q,                      # [q_len,  n_heads,    head_dim]
    k,                      # [kv_len, n_kv_heads, head_dim]
    v,                      # [kv_len, n_kv_heads, head_dim]
    logits_soft_cap=0.0,
    sliding_window=-1,
    alibi_slopes=None,      # [n_heads] float32, optional
):
    q_len,  n_heads,    head_dim = q.shape
    kv_len, n_kv_heads, _ = k.shape
    assert kv_len >= q_len

    orig_dtype = q.dtype
    sm_scale = 1.0 / math.sqrt(head_dim)

    q = q.float()   # [q_len,  n_heads,    head_dim]
    k = k.float()   # [kv_len, n_kv_heads, head_dim]
    v = v.float()

    # GQA: expand K/V along the heads dim (dim=1)
    if n_heads != n_kv_heads:
        group = n_heads // n_kv_heads
        k = k.repeat_interleave(group, dim=1)   # [kv_len, n_heads, head_dim]
        v = v.repeat_interleave(group, dim=1)

    # [q_len, n_heads, kv_len]
    scores = torch.einsum("qhd,khd->qhk", q, k) * sm_scale

    # Soft-cap: tanh(s / cap) * cap
    if logits_soft_cap != 0.0:
        scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap

    if alibi_slopes is not None:
        distance = torch.arange(kv_len, dtype=torch.float32, device=q.device)
        # [1, n_heads, kv_len]
        bias = distance.view(1, 1, kv_len) * alibi_slopes.view(1, n_heads, 1)
        scores = scores + bias

    # ---- Causal mask (+ optional sliding window) ----
    # tril(diagonal = kv_len - q_len): q_i attends to k_j where j <= i + (kv_len - q_len)
    # triu(diagonal = kv_len - q_len - window): k_j >= i + (kv_len - q_len) - window
    mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device)
    if sliding_window >= 0:
        mask = torch.triu(mask, diagonal=kv_len - q_len - sliding_window)
    mask = torch.tril(mask, diagonal=kv_len - q_len)   # causal

    # broadcast mask over n_heads: [q_len, 1, kv_len]
    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

    attn = torch.softmax(scores, dim=-1)

    # [q_len, n_heads, head_dim]
    return torch.einsum("qhk,khd->qhd", attn, v).to(orig_dtype)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

PASS_COUNT = 0
FAIL_COUNT = 0


def run_test(tag, n_heads, n_kv_heads, q_len, kv_len, head_dim,
             dtype, sliding_window=-1, logits_soft_cap=0.0,
             atol=1e-2, rtol=1e-3):
    global PASS_COUNT, FAIL_COUNT

    torch.manual_seed(42)
    q = torch.randn(q_len,  n_heads,    head_dim, dtype=dtype, device="cuda")
    k = torch.randn(kv_len, n_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(kv_len, n_kv_heads, head_dim, dtype=dtype, device="cuda")

    ref = ref_attention(q, k, v,
                        logits_soft_cap=logits_soft_cap,
                        sliding_window=sliding_window)
    out = torch.empty_like(q)
    lib.single_mha_attention_half(out, q, k, v,
                                  sm_scale=1.0 / math.sqrt(head_dim),
                                  sliding_window=sliding_window,
                                  logits_soft_cap=logits_soft_cap)

    ref_f = ref.float()
    out_f = out.float()
    max_diff = (out_f - ref_f).abs().max().item()
    mean_diff = (out_f - ref_f).abs().mean().item()

    ok = max_diff <= atol and mean_diff <= rtol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {tag:<52}  max={max_diff:.3e}  mean={mean_diff:.3e}")
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
        print(f"          atol={atol}, rtol={rtol}")
    return ok


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        sys.exit("No CUDA device found.")

    print(f"Device : {torch.cuda.get_device_name(0)}\n")

    # ---- standard MHA (n_heads == n_kv_heads) ----
    print("MHA (full attention, n_heads == n_kv_heads)")
    for dtype, dt in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
        for hd in [64, 128]:
            run_test(f"{dt}  h=4   q=256  kv=256  hd={hd}",
                     n_heads=4, n_kv_heads=4,
                     q_len=256, kv_len=256, head_dim=hd, dtype=dtype)

    # ---- GQA (n_kv_heads < n_heads) ----
    print("\nGQA")
    for dtype, dt in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
        run_test(f"{dt}  h=4  kv_h=2  q=256  kv=256  hd=128",
                 n_heads=4, n_kv_heads=2,
                 q_len=256, kv_len=256, head_dim=128, dtype=dtype)
        run_test(f"{dt}  h=8  kv_h=2  q=256  kv=256  hd=128",
                 n_heads=8, n_kv_heads=2,
                 q_len=256, kv_len=256, head_dim=128, dtype=dtype)

    # ---- Decode (q_len < kv_len) ----
    print("\nDecode (q_len < kv_len)")
    for dtype, dt in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
        run_test(f"{dt}  h=4  kv_h=2  q=1   kv=256  hd=128",
                 n_heads=4, n_kv_heads=2,
                 q_len=1, kv_len=256, head_dim=128, dtype=dtype)
        run_test(f"{dt}  h=4  kv_h=2  q=64  kv=256  hd=128",
                 n_heads=4, n_kv_heads=2,
                 q_len=64, kv_len=256, head_dim=128, dtype=dtype)

    # ---- Sliding-window ----
    print("\nSliding window")
    for window in [32, 64, 128]:
        run_test(f"fp16  h=4  kv_h=2  q=256  kv=256  win={window}",
                 n_heads=4, n_kv_heads=2,
                 q_len=256, kv_len=256, head_dim=128,
                 dtype=torch.float16, sliding_window=window)

    # ---- Soft-cap ----
    print("\nSoft-cap")
    for cap in [10.0, 50.0]:
        run_test(f"fp16  h=4  kv_h=2  q=256  kv=256  cap={cap}",
                 n_heads=4, n_kv_heads=2,
                 q_len=256, kv_len=256, head_dim=128,
                 dtype=torch.float16, logits_soft_cap=cap)

    print()
    print("=" * 70)
    total = PASS_COUNT + FAIL_COUNT
    print(f"Result: {PASS_COUNT}/{total} passed")
    if FAIL_COUNT:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL PASS")


if __name__ == "__main__":
    main()
