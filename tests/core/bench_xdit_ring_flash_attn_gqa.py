#!/usr/bin/env python3

# Example (FA3 + GQA):
# torchrun --standalone --nproc_per_node=2 tests/core/bench_xdit_ring_flash_attn_gqa.py \
#   --S-list 8192 --B 2 --Hq 16 --Hkv 8 --D 128 --dtype fp16 --warmup 10 --iters 50 --check --attn-type fa3

import os
import argparse
import torch
import torch.distributed as dist
import torch.cuda.nvtx as nvtx

from xfuser.core.long_ctx_attention.ring.ring_flash_attn import xdit_ring_flash_attn_func

# Try to use xDiT's init helper if present (optional)
try:
    from xfuser.core.distributed import init_distributed_environment as xdit_init_dist
except Exception:
    xdit_init_dist = None

# AttnType is defined in yunchang
try:
    from yunchang.kernels import AttnType
except Exception:
    AttnType = None


def clean_print(msg: str, rank: int, print_once: bool = True) -> None:
    if (not print_once) or rank == 0:
        print(msg, flush=True)


def init_dist() -> tuple[int, int, int]:
    """
    Returns (rank, world_size, local_rank).
    Uses torchrun env vars.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)

    # Prefer xDiT init if available; otherwise use standard torch init.
    if xdit_init_dist is not None:
        xdit_init_dist(rank=rank, world_size=world_size)
        # Some wrappers still rely on dist.* being initialized; assert it
        if not dist.is_initialized():
            # fallback
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank

def dist_barrier(local_rank: int) -> None:
    # Explicit device_ids silences: "barrier(): using the device under current context"
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[local_rank])


@torch.no_grad()
def benchmark_no_l2_clear(fn, num_warmup_iters: int, num_iters: int, local_rank: int, profile_nvtx: bool) -> float:
    # Warmup
    if profile_nvtx:
        nvtx.range_push("warmup")
    for i in range(num_warmup_iters):
        if profile_nvtx:
            nvtx.range_push(f"warmup_iter_{i}")
        fn()
        if profile_nvtx:
            nvtx.range_pop()
    if profile_nvtx:
        nvtx.range_pop()

    dist_barrier(local_rank)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if profile_nvtx:
        nvtx.range_push("timed_region")

    start.record()
    for i in range(num_iters):
        if profile_nvtx:
            nvtx.range_push(f"iter_{i}")
        fn()
        if profile_nvtx:
            nvtx.range_pop()
    end.record()

    if profile_nvtx:
        nvtx.range_pop()

    torch.cuda.synchronize()
    dist_barrier(local_rank)

    total_ms = start.elapsed_time(end)
    return total_ms / float(num_iters)



def make_local_qkv_gqa(
    B: int, S: int, Hq: int, Hkv: int, D: int,
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
    rank: int,
    check_correctness: bool,
):
    """
    GQA shapes:
      Q:  (B, S, Hq,  D)
      K/V:(B, S, Hkv, D)
    Shard along S (dim=1) => local tensors have shape (B, S/world_size, heads, D)
    """

    # sequence sharding requirement
    assert S % world_size == 0, f"S={S} must be divisible by world_size={world_size}"

    # GQA semantic requirement
    assert Hq % Hkv == 0, f"GQA requires Hq % Hkv == 0. Got Hq={Hq}, Hkv={Hkv}"

    if check_correctness:
        if rank == 0:
            q = torch.randn((B, S, Hq, D), device=device, dtype=dtype)
            k = torch.randn((B, S, Hkv, D), device=device, dtype=dtype)
            v = torch.randn((B, S, Hkv, D), device=device, dtype=dtype)
        else:
            q = torch.empty((B, S, Hq, D), device=device, dtype=dtype)
            k = torch.empty((B, S, Hkv, D), device=device, dtype=dtype)
            v = torch.empty((B, S, Hkv, D), device=device, dtype=dtype)

        dist.broadcast(q, src=0)
        dist.broadcast(k, src=0)
        dist.broadcast(v, src=0)

        local_q = q.chunk(world_size, dim=1)[rank].contiguous()
        local_k = k.chunk(world_size, dim=1)[rank].contiguous()
        local_v = v.chunk(world_size, dim=1)[rank].contiguous()
        return local_q, local_k, local_v, (q, k, v)
    else:
        local_S = S // world_size
        local_q = torch.randn((B, local_S, Hq, D), device=device, dtype=dtype)
        local_k = torch.randn((B, local_S, Hkv, D), device=device, dtype=dtype)
        local_v = torch.randn((B, local_S, Hkv, D), device=device, dtype=dtype)
        return local_q, local_k, local_v, None


def parse_attn_type(attn_type_str: str):
    """
    Maps string to AttnType enum (if available).
    Defaults to AttnType.FA when possible.
    """
    if AttnType is None:
        return None
    s = attn_type_str.lower()
    if s in ("fa", "fa2", "flash", "flashattn"):
        return AttnType.FA
    if s in ("fa3", "flash3", "flashattn3"):
        return AttnType.FA3
    if s in ("sparse_sage", "sage"):
        return AttnType.SPARSE_SAGE
    return AttnType.FA


def run_one(
    B: int, Hq: int, Hkv: int, S: int, D: int,
    dtype: torch.dtype,
    causal: bool,
    attn_type_str: str,
    num_warmup_iters: int,
    num_iters: int,
    check_correctness: bool,
    rank: int,
    world_size: int,
    local_rank: int,
    nvtx_profile: bool = False,
) -> None:
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{local_rank}")

    attn_type = parse_attn_type(attn_type_str)

    local_q, local_k, local_v, full = make_local_qkv_gqa(
        B=B, S=S, Hq=Hq, Hkv=Hkv, D=D,
        dtype=dtype,
        device=device,
        world_size=world_size,
        rank=rank,
        check_correctness=check_correctness,
    )

    # Ring-attn forward (returns local output)
    def xdit_run():
        return xdit_ring_flash_attn_func(
            q=local_q,
            k=local_k,
            v=local_v,
            dropout_p=0.0,
            softmax_scale=None,
            causal=causal,
            window_size=(-1, -1),
            group=dist.group.WORLD,
            attn_type=attn_type,
        )

    # Optional correctness check (WARNING: full flash-attn ref is O(S^2) memory; keep S small)
    if check_correctness:
        (q, k, v) = full  # type: ignore[misc]
        if attn_type is not None and attn_type == AttnType.FA3:
            # FA3 (hopper) reference supports GQA
            from flash_attn_interface import flash_attn_func as flash3_flash_attn_func
            ref = flash3_flash_attn_func(
                q, k, v,
                # dropout_p=0.0,
                causal=causal,
                window_size=(-1, -1),
            )
        else:
            # FA2 reference supports GQA as long as Hq % Hkv == 0
            from flash_attn import flash_attn_func as fa2_flash_attn_func
            ref = fa2_flash_attn_func(
                q, k, v,
                dropout_p=0.0,
                causal=causal,
                window_size=(-1, -1),
            )

        # ref has shape (B, S, Hq, D)
        ref_local = ref.chunk(world_size, dim=1)[rank].contiguous()

        out = xdit_run()
        torch.testing.assert_close(ref_local, out, rtol=1e-3, atol=1e-3)
        dist_barrier(local_rank)
        clean_print(f"[OK] correctness passed for GQA: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}", rank)

    dist_barrier(local_rank)
    torch.cuda.synchronize()

    avg_ms = benchmark_no_l2_clear(xdit_run, num_warmup_iters, num_iters, local_rank, nvtx_profile)

    # FLOPs per GPU: local queries length (S/world_size), global keys length S, query heads = Hq
    total_flops = 4 * B * Hq * (S // world_size) * S * D
    tflops = (total_flops * 1e-12) / (avg_ms * 1e-3)

    clean_print("===============================================================================", rank)
    clean_print(
        f"<xDiT Ring FlashAttn GQA | world_size={world_size} | B {B} S {S} Hq {Hq} Hkv {Hkv} D {D} "
        f"| dtype={str(dtype).replace('torch.', '')} | causal={causal} | attn_type={attn_type_str}>",
        rank
    )
    clean_print(f"xDiT Ring (GQA): {avg_ms:.3f} ms | {tflops:.2f} TFLOP/s", rank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--Hq", type=int, default=16, help="Query heads")
    ap.add_argument("--Hkv", type=int, default=8, help="KV heads (<= Hq). Use Hkv==Hq for MHA.")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--causal", action="store_true", help="Use causal attention")
    ap.add_argument("--attn-type", type=str, default="fa3", help="fa | fa3 | sparse_sage (if supported)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check", action="store_true", help="Run correctness vs flash_attn_func (keep S small!)")
    ap.add_argument("--S-list", type=str, default="", help="Comma-separated global sequence lengths, e.g. 12288,24576,...")
    ap.add_argument("--S-base", type=int, default=768, help="If --S-list not set: S = world_size * S-base * 2^i")
    ap.add_argument("--S-pow-min", type=int, default=1)
    ap.add_argument("--S-pow-max", type=int, default=6)
    ap.add_argument("--profile-nvtx", action="store_true",
                    help="Insert NVTX ranges around iterations (use with nsys/ncu). Adds overhead; don't use for final timing."
                    )

    args = ap.parse_args()

    rank, world_size, local_rank = init_dist()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # Safety checks
    if args.Hkv > args.Hq:
        raise ValueError(f"Hkv must be <= Hq. Got Hq={args.Hq}, Hkv={args.Hkv}")
    if args.Hq % args.Hkv != 0:
        raise ValueError(f"GQA requires Hq % Hkv == 0. Got Hq={args.Hq}, Hkv={args.Hkv}")

    if args.S_list.strip():
        S_values = [int(x) for x in args.S_list.split(",") if x.strip()]
    else:
        S_values = [world_size * args.S_base * (2 ** i) for i in range(args.S_pow_min, args.S_pow_max + 1)]

    clean_print(f"Running xDiT ring-attn GQA benchmark on {world_size} ranks | S_values={S_values}", rank)

    for S in S_values:
        run_one(
            B=args.B, Hq=args.Hq, Hkv=args.Hkv, S=S, D=args.D,
            dtype=dtype,
            causal=args.causal,
            attn_type_str=args.attn_type,
            num_warmup_iters=args.warmup,
            num_iters=args.iters,
            check_correctness=args.check,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            nvtx_profile=args.profile_nvtx,
        )

    dist_barrier(local_rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
