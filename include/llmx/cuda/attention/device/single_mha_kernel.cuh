#pragma once

#include "cute/layout.hpp"
#include "llmx/cuda/attention/params.h"
#include "llmx/cuda/common/fast_cast.h"
#include "llmx/cuda/common/gather_tensor.h"
#include "llmx/cuda/common/layout_convertor.h"
#include "llmx/cuda/common/mask.h"
#include "llmx/cuda/common/safe_copy.h"
#include "llmx/cuda/common/selector.h"
#include "llmx/cuda/common/softmax.h"
#include "llmx/utility.h"

namespace llmx {

using namespace cute;

CUTE_DEVICE bool is_first() {
  return (threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) &&
         (blockIdx.z == 0);
}

template <typename Kernel, typename... Args>
__global__ void device_kernel(Args... args) {
  extern __shared__ char smem[];
  Kernel kernel{};
  kernel(args..., smem);
}

template <typename Element_, typename TileShape_, const bool kEvenK_,
          const bool kAlibi_, const bool kSoftCap_, const bool kLocal_>
struct SingleMHACollectiveMainloop {
  using Element = Element_;
  using TileShape = TileShape_;

  static constexpr bool kEvenK = kEvenK_;
  static constexpr bool kAlibi = kAlibi_;
  static constexpr bool kSoftCap = kSoftCap_;
  static constexpr bool kLocal = kLocal_;

  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockN = get<1>(TileShape{});
  static constexpr int kBlockK = get<2>(TileShape{});

  using BLK_M = Int<kBlockM>;
  using BLK_N = Int<kBlockN>;
  using BLK_K = Int<kBlockK>;

  static constexpr int kRowsPerMMA = 2;
  static constexpr int kNThreads = 128;

  using MMA_Atom_ =
      std::conditional_t<std::is_same_v<Element, cute::half_t>,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using ValTile = Tile<_64, _16, _16>;
  using LayoutConvertor = LayoutConvertor<ValTile>;

  using TiledMma = TiledMMA<MMA_Atom_, Layout<Shape<_4, _1, _1>>, ValTile>;

  using SmemLayoutAtom =
      decltype(smem_layout_atom_selector<Element, kBlockK>());

  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtom{}, Shape<BLK_M, BLK_K>{}));
  using SmemLayoutK =
      decltype(tile_to_shape(SmemLayoutAtom{}, Shape<BLK_N, BLK_K>{}));
  using SmemLayoutV =
      decltype(tile_to_shape(SmemLayoutAtom{}, Shape<BLK_N, BLK_K>{}));
  using SmemLayoutVt = decltype(select<1, 0>(SmemLayoutV{}));

  using GmemTiledCopyQ =
      decltype(gmem_tiled_copy_selector<Element, kNThreads, kBlockK>(
          Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{}));

  using GmemTiledCopyKV = GmemTiledCopyQ;

  using SmemTiledCopyQ = decltype(make_tiled_copy_A(
      Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, TiledMma{}));
  using SmemTiledCopyK = decltype(make_tiled_copy_B(
      Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, TiledMma{}));

  using SmemTiledCopyVt = decltype(make_tiled_copy_B(
      Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, TiledMma{}));

  struct SharedStorage : cute::aligned_struct<128> {
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
      struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        union {
          cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
          cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>> smem_vt;
        };
      };
    };
  };

  template <typename TensorQ, typename TensorCQ, typename TensorK,
            typename TensorV, typename TensorCKV, typename TensorCMN,
            typename FrgTensor, typename Softmax, typename Mask,
            typename ResidueMNK>
  CUTE_DEVICE void
  operator()(const TensorQ &gQ,         // (BLK_M, BLK_K)
             const TensorCQ &cQ,        // (BLK_M, BLK_K) => (M, K)
             const TensorK &gK,         // (BLK_N, BLK_K, n)
             const TensorV &gV,         // (BLK_N, BLK_K, n)
             const TensorCKV &cKV,      // (BLK_N, BLK_K, n) => (N, K)
             const TensorCMN &tScMN_mn, // ((2, MMA_M), (2, MMA_N), n) => (M, N)
             FrgTensor &tOrO,           // (MMA, MMA_M, MMA_N)
             Softmax &softmax, const Mask &mask, float logits_soft_cap,
             int tidx,
             const ResidueMNK &residue_mnk, // (M, N, K)
             int n_block_min, int n_block_max, char *smem) {
    auto &ss = *reinterpret_cast<SharedStorage *>(smem);

    // (BLK_M, BLK_K)
    Tensor sQ = make_tensor(make_smem_ptr(ss.smem_q.data()), SmemLayoutQ{});
    // (BLK_N, BLK_K)
    Tensor sK = make_tensor(make_smem_ptr(ss.smem_k.data()), SmemLayoutK{});
    // (BLK_N, BLK_K)
    Tensor sV = make_tensor(make_smem_ptr(ss.smem_v.data()), SmemLayoutV{});
    // (BLK_K, BLK_N)
    Tensor sVt = make_tensor(make_smem_ptr(ss.smem_vt.data()), SmemLayoutVt{});

    // g2s tiled copy for qkv
    GmemTiledCopyQ gmem_tiled_copy_Q;
    GmemTiledCopyKV gmem_tiled_copy_KV;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    // (CPY, CPY_N, CPY_K, n) => (N, K)
    Tensor tGcKV = gmem_thr_copy_KV.partition_S(cKV);
    // (CPY, CPY_N, CPY_K, n)
    Tensor tGgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tGgV = gmem_thr_copy_KV.partition_S(gV);

    // (CPY, CPY_N, CPY_K)
    Tensor tGsK = gmem_thr_copy_KV.partition_D(sK);
    Tensor tGsV = gmem_thr_copy_KV.partition_D(sV);

    const auto residue_mk = select<0, 2>(residue_mnk);
    const auto residue_nk = select<1, 2>(residue_mnk);

    auto produce_query = [&]() {
      auto tGcQ = gmem_thr_copy_Q.partition_S(cQ);
      auto tGgQ = gmem_thr_copy_Q.partition_S(gQ);
      auto tGsQ = gmem_thr_copy_Q.partition_D(sQ);
      safe_copy</*EVEN_MN=*/false, kEvenK, /*ZFILL_MN=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy_Q, tGgQ, tGsQ, tGcQ, residue_mk);
    };

    auto produce_key = [&](int ni) {
      // skip ZFILL_MN for key since Mask will mask out oob with -inf
      safe_copy</*EVEN_N=*/false, kEvenK, /*ZFILL_N=*/false, /*ZFILL_K=*/true>(
          gmem_tiled_copy_KV, tGgK(_, _, _, ni), tGsK, tGcKV(_, _, _, ni),
          residue_nk);
    };

    // produce key without oob handling
    auto produce_key_no_oob = [&](int ni) {
      safe_copy</*EVEN_N=*/true, kEvenK, /*ZFILL_N=*/false, /*ZFILL_K=*/false>(
          gmem_tiled_copy_KV, tGgK(_, _, _, ni), tGsK, tGcKV(_, _, _, ni),
          residue_nk);
    };

    auto produce_value = [&](int ni) {
      // skipping ZFILL_MN for v may cause nan issue
      safe_copy</*EVEN_N=*/false, kEvenK, /*ZFILL_N=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy_KV, tGgV(_, _, _, ni), tGsV, tGcKV(_, _, _, ni),
          residue_nk);
    };

    // produce value without oob handling
    auto produce_value_no_oob = [&](int ni) {
      safe_copy</*EVEN_N=*/true, kEvenK, /*ZFILL_N=*/false, /*ZFILL_K=*/false>(
          gmem_tiled_copy_KV, tGgV(_, _, _, ni), tGsV, tGcKV(_, _, _, ni),
          residue_nk);
    };

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);
    // GEMM-I: S = Q@K.T
    auto tSrQ = thr_mma.partition_fragment_A(sQ); // (MMA,MMA_M,MMA_K)
    auto tSrK = thr_mma.partition_fragment_B(sK); // (MMA,MMA_N,MMA_K)

    // s2r tiled copy for qkv
    // copy query to rmem
    SmemTiledCopyQ smem_tiled_copy_Q;
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    SmemTiledCopyK smem_tiled_copy_K;
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    auto tSsK = smem_thr_copy_K.partition_S(sK);
    auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

    // S = Q@K.T
    // tSrAccS: (MMA,MMA_M,MMA_N)
    auto compute_qk = [&](auto &tSrAccS) {
      // prefetch key
      cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}),
                 tSrK_copy_view(_, _, _0{}));

      CUTE_UNROLL
      for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
        // prefetch next key
        if (ki != size<2>(tSrQ) - 1) {
          const auto next_ki = ki + 1;
          cute::copy(smem_tiled_copy_K, tSsK(_, _, next_ki),
                     tSrK_copy_view(_, _, next_ki));
        }
        cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), tSrAccS);
      }
    };

    // GEMM-II: O = softmax(S)@V
    auto tOrVt = thr_mma.partition_fragment_B(sVt); // (MMA,MMA_K,MMA_N)

    SmemTiledCopyVt smem_tiled_copy_Vt;
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(tidx);
    auto tOsVt = smem_thr_copy_Vt.partition_S(sVt);
    auto tOrVt_copy_view = smem_thr_copy_Vt.retile_D(tOrVt);

    // O = softmax(S)*V
    // tSrAccS: (MMA,MMA_M,MMA_N)
    // tOrAccO: (MMA,MMA_M,MMA_K)
    auto compute_sv = [&](const auto &tSrAccS, auto &tOrAccO) {
      // cast scores from Accumulator to Element
      auto tSrS = make_tensor_like<Element>(tSrAccS);
      fast_cast(tSrAccS, tSrS);

      // convert layout from gemm-I C to gemm-II A
      auto tOrS =
          make_tensor(tSrS.data(), LayoutConvertor::to_mma_a(tSrS.layout()));

      // prefetch V^t
      cute::copy(smem_tiled_copy_Vt, tOsVt(_, _, _0{}),
                 tOrVt_copy_view(_, _, _0{}));
      CUTE_UNROLL
      for (int ki = 0; ki < size<2>(tOrS); ++ki) {
        // prefetch next V^t
        if (ki != size<2>(tOrS) - 1) {
          const auto next_ki = ki + 1;
          cute::copy(smem_tiled_copy_Vt, tOsVt(_, _, next_ki),
                     tOrVt_copy_view(_, _, next_ki));
        }
        cute::gemm(tiled_mma, tOrS(_, _, ki), tOrVt(_, _, ki), tOrAccO);
      }
    };

    auto tOrO_mn =
        make_tensor(tOrO.data(), LayoutConvertor::to_mn(tOrO.layout()));

    auto apply_logits_soft_cap = [&](auto &tSrAccS) {
      if constexpr (kSoftCap) {
        CUTE_UNROLL
        for (int i = 0; i < size(tSrAccS); ++i) {
          tSrAccS(i) = tanh(tSrAccS(i) * logits_soft_cap);
        }
      }
    };

    // ###############  Prologue  ###############
    // produce query: [] => [q]
    produce_query();
    cp_async_fence();

    // wait g2s copy done for query
    cp_async_wait<0>();
    __syncthreads();

    // copy query from smem to rmem
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    // wait s2r copy done for query before producing key
    __syncthreads();

    // produce key: [q] => [q, k]
    produce_key(n_block_max - 1);
    cp_async_fence();

    // ###############  Mainloop  ###############
    constexpr int n_oob_mask = cute::ceil_div(kBlockM, kBlockN) + 1;
    const int n_blocks = n_block_max - n_block_min;

    // attention score accumulator, (MMA, MMA_M, MMA_N)
    auto tSrS = partition_fragment_C(tiled_mma, Shape<BLK_M, BLK_N>{});
    // ((2, MMA_M), (2, MMA_N))
    auto tSrS_mn =
        make_tensor(tSrS.data(), LayoutConvertor::to_mn(tSrS.layout()));

    CUTE_NO_UNROLL
    for (int i = 0; i < n_blocks; ++i) {
      const int ni = n_block_max - 1 - i;
      clear(tSrS);

      // wait key, queue: [q, k] => []
      cp_async_wait<0>();
      __syncthreads();

      // produce value, [] => [v]
      if (i == 0) {
        produce_value(ni);
      } else {
        produce_value_no_oob(ni);
      }
      cp_async_fence();

      // 1> S = Q@K.T
      compute_qk(tSrS);

      // wait value, [v] => []
      cp_async_wait<0>();
      __syncthreads();

      if constexpr (kSoftCap) {
        apply_logits_soft_cap(tSrS);
      }

      // apply mask
      // ((2, MMA_M), (2, MMA_N)) => (M, N)
      const auto tScS_mn = tScMN_mn(_, _, ni);
      if (i < n_oob_mask) {
        mask.template apply</*OOB_MASK=*/true>(tSrS_mn, tScS_mn);
      } else {
        mask.template apply</*OOB_MASK=*/false>(tSrS_mn, tScS_mn);
      }
      softmax.rescale(tSrS_mn, tOrO_mn);

      // produce next key: [] => [k]
      if (ni > n_block_min) {
        produce_key_no_oob(ni - 1);
      }
      cp_async_fence();

      // 2> O = softmax(S)*V
      compute_sv(tSrS, tOrO);
    }

    // normalize output: o /= rowsum
    softmax.finalize(tOrO_mn);
  }
};

template <typename Element_, typename TileShape_, const bool kEvenK_>
struct SingleMHACollectiveEpilogue {
  using TileShape = TileShape_;
  using Element = Element_;

  static constexpr int kThreads = 128;
  static constexpr bool kEvenK = kEvenK_;

  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockK = get<2>(TileShape{});

  using BLK_M = Int<kBlockM>;
  using BLK_K = Int<kBlockK>;

  using SmemLayoutAtom_ =
      decltype(smem_layout_atom_selector<Element, kBlockK>());

  // Q smem: (BLK_M, BLK_K)
  using SmemLayoutO =
      decltype(tile_to_shape(SmemLayoutAtom_{}, Shape<BLK_M, BLK_K>{}));

  // use 128-bit vectorizing copy
  using VectorizingCopy_ = AutoVectorizingCopyWithAssumedAlignment<128>;

  // r2s copy atom for O
  using SmemCopyAtom_ = Copy_Atom<VectorizingCopy_, Element>;

  // s2g tiled copy for O
  using GmemTiledCopyO =
      decltype(gmem_tiled_copy_selector<Element, kThreads, kBlockK>(
          Copy_Atom<VectorizingCopy_, Element>{}));

  struct SharedStorage : cute::aligned_struct<128> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;
  };

  template <class FrgTensor, class TiledMma, class TensorO, class TensorCO,
            class ResidueMNK>
  CUTE_DEVICE void operator()(const FrgTensor &tOrAccO, // (MMA, MMA_M, MMA_N)
                              TiledMma tiled_mma,
                              TensorO &gO,        // (BLK_M, BLK_K)
                              const TensorCO &cO, // (BLK_M, BLK_K) => (M, K)
                              int tidx, const ResidueMNK &residue_mnk,
                              char *smem) {
    static constexpr int kBlockM = get<0>(TileShape{});

    // Smem
    auto &ss = *reinterpret_cast<SharedStorage *>(smem);
    // (BLK_M, BLK_K)
    Tensor sO = make_tensor(make_smem_ptr(ss.smem_o.data()), SmemLayoutO{});

    // 1. cast output from ElementAccumulator to Element
    auto tOrO = make_tensor_like<Element>(tOrAccO);
    fast_cast(tOrAccO, tOrO);

    // 2. copy output from reg to smem
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtom_{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    auto tSrO = smem_thr_copy_O.retile_S(tOrO);
    auto tSsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, tSrO, tSsO);

    // 3. copy output from smem to gmem
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);

    auto tOsO = gmem_thr_copy_O.partition_S(sO); // (CPY,CPY_M,CPY_K)
    auto tOgO = gmem_thr_copy_O.partition_D(gO); // (CPY,CPY_M,CPY_K)
    // (CPY,CPY_M,CPY_K) -> (blk_m, head_dim)
    auto tOcO = gmem_thr_copy_O.partition_D(cO);

    // wait for smem copy done before gmem copy
    __syncthreads();

    const auto residue_mk = select<0, 2>(residue_mnk);
    safe_copy</*EVEN_M=*/false, kEvenK, /*ZFILL_M=*/false, /*ZFILL_K=*/false>(
        gmem_tiled_copy_O, tOsO, tOgO, tOcO, residue_mk);
  }
};

template <typename Params> struct MHATiler;

template <> struct MHATiler<MHAParams> {
  CUTE_DEVICE MHATiler(int batch_idx, int kv_head_idx)
      : batch_idx_(batch_idx), kv_head_idx_(kv_head_idx) {}

  template <typename Element>
  CUTE_DEVICE auto get_qo_tile(const MHAParams &params) {
    auto packed_to_idx = [&](int packed_idx) {
      return make_tuple(packed_idx / params.group_size,
                        packed_idx % params.group_size);
    };
    const auto q_offset =
        batch_idx_ * get<0>(params.q_stride) +
        kv_head_idx_ * params.group_size * get<2>(params.q_stride);
    Tensor q = make_gather_tensor(
        (Element *)params.q_ptr + q_offset,
        make_shape(params.q_len * params.group_size, params.head_dim),
        make_stride(select<1, 2>(params.q_stride), get<3>(params.q_stride)),
        packed_to_idx);
    const auto o_offset =
        batch_idx_ * get<0>(params.o_stride) +
        kv_head_idx_ * params.group_size * get<2>(params.o_stride);
    Tensor o = make_gather_tensor(
        (Element *)params.o_ptr + o_offset,
        make_shape(params.q_len * params.group_size, params.head_dim),
        make_stride(select<1, 2>(params.o_stride), get<3>(params.o_stride)),
        packed_to_idx);

    return make_tuple(q, o);
  }

  template <typename Element>
  CUTE_DEVICE auto get_kv_tile(const MHAParams &params) {
    const auto k_offset = batch_idx_ * get<0>(params.k_stride) +
                          kv_head_idx_ * get<2>(params.k_stride);
    Tensor k = make_tensor((Element *)params.k_ptr + k_offset,
                           make_shape(params.kv_len, params.head_dim),
                           select<1, 3>(params.k_stride));
    const auto v_offset = batch_idx_ * get<0>(params.v_stride) +
                          kv_head_idx_ * get<2>(params.v_stride);
    Tensor v = make_tensor((Element *)params.v_ptr + v_offset,
                           make_shape(params.kv_len, params.head_dim),
                           select<1, 3>(params.v_stride));

    return make_tuple(k, v);
  }

  int batch_idx_;
  int kv_head_idx_;
};

template <typename CollectiveMainloop_, typename CollectiveEpilogue_>
struct SingleMHAKernel {
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using Element = typename CollectiveMainloop::Element;

  static constexpr int kBlockM = CollectiveMainloop::kBlockM;
  static constexpr int kBlockN = CollectiveMainloop::kBlockN;
  static constexpr int kBlockK = CollectiveMainloop::kBlockK;

  using BLK_M = typename CollectiveMainloop::BLK_M;
  using BLK_N = typename CollectiveMainloop::BLK_N;
  using BLK_K = typename CollectiveMainloop::BLK_K;

  static constexpr bool kEvenK = CollectiveMainloop::kEvenK;
  static constexpr bool kAlibi = CollectiveMainloop::kAlibi;
  static constexpr bool kSoftCap = CollectiveMainloop::kSoftCap;
  static constexpr bool kLocal = CollectiveMainloop::kLocal;

  using LayoutConvertor = typename CollectiveMainloop::LayoutConvertor;
  using TiledMma = typename CollectiveMainloop::TiledMma;

  static constexpr int kSharedStorageSize =
      sizeof(typename CollectiveMainloop::SharedStorage);
  static constexpr int kRowsPerMMA = CollectiveMainloop::kRowsPerMMA;
  static constexpr int kNThreads = CollectiveMainloop::kNThreads;

  CUTE_DEVICE void operator()(MHAParams params, char *smem) {
    // NVCC can't resolve static constexpr members of template classes in device
    // code; use local constexpr copies instead.
    constexpr int kBlockM = CollectiveMainloop::kBlockM;
    constexpr int kBlockN = CollectiveMainloop::kBlockN;

    const auto block_idx = blockIdx.x;
    const auto kv_head_idx = blockIdx.y;
    const auto tidx = threadIdx.x;

    CollectiveMainloop mha;
    CollectiveEpilogue epilogue;

    MHATiler<MHAParams> tiler(0, kv_head_idx); // single batch
    // (q_packed_len, head_dim)
    auto [Q, O] = tiler.get_qo_tile<Element>(params);
    // (kv_len, head_dim)
    auto [K, V] = tiler.get_kv_tile<Element>(params);

    // problem shape
    const int q_packed_len = size<0>(Q);
    const int kv_len = size<0>(K);
    const int m_block_base = block_idx * kBlockM;
    if (m_block_base >= q_packed_len) {
      // m out of bound
      return;
    }

    const int q_idx = m_block_base / params.group_size;
    const auto residue_mnk = make_tuple(q_packed_len, kv_len, params.head_dim);
    const int sliding_window = kLocal ? params.sliding_window : kv_len;
    const int q_len = q_packed_len / params.group_size;
    const int diagonal = q_idx + kv_len - q_len;
    const int kv_idx_min = std::max(0, diagonal - sliding_window);
    const int kv_idx_max = std::min(kv_len, diagonal + kBlockM);
    const int n_block_min = kLocal ? kv_idx_min / kBlockN : 0;
    const int n_block_max = cute::ceil_div(kv_idx_max, kBlockN);

    // (BLK_M, BLK_K)
    auto gQ = local_tile(Q, Shape<BLK_M, BLK_K>{}, make_coord(block_idx, _0{}));
    auto gO = local_tile(O, Shape<BLK_M, BLK_K>{}, make_coord(block_idx, _0{}));
    // (BLK_M, BLK_K) => (M, K)
    Tensor cQ = local_tile(make_identity_tensor(Q.shape()),
                           Shape<BLK_M, BLK_K>{}, make_coord(block_idx, _0{}));

    // (BLK_N, BLK_K, n)
    auto gK = local_tile(K, Shape<BLK_N, BLK_K>{}, make_coord(_, _0{}));
    auto gV = local_tile(V, Shape<BLK_N, BLK_K>{}, make_coord(_, _0{}));
    // (BLK_N, BLK_K, n) => (N, K)
    Tensor cKV = local_tile(make_identity_tensor(K.shape()),
                            Shape<BLK_N, BLK_K>{}, make_coord(_, _0{}));
    // (BLK_M, BLK_N, n) => (M, N)
    Tensor cMN =
        local_tile(make_identity_tensor(make_shape(q_packed_len, kv_len)),
                   Shape<BLK_M, BLK_N>{}, make_coord(block_idx, _));

    TiledMma tiled_mma;
    // accumulator: (MMA,MMA_M,MMA_K)
    auto tOrAccO = partition_fragment_C(tiled_mma, Shape<BLK_M, BLK_K>{});
    clear(tOrAccO);

    auto thr_mma = tiled_mma.get_slice(tidx);
    // (MMA, MMA_M, MMA_N, n) => (M, N)
    auto tScMN = thr_mma.partition_C(cMN);
    // ((2, MMA_M), (2, MMA_N), n) => (M, N)
    auto tScMN_mn =
        make_tensor(tScMN.data(), LayoutConvertor::to_mn(tScMN.layout()));

    constexpr int kRowsPerThr = kRowsPerMMA * size<1>(tOrAccO);
    // Create softmax and mask
    OnlineSoftmax<kRowsPerThr> softmax(params.sm_scale * M_LOG2E);
    Mask<kRowsPerThr, kAlibi, kLocal> mask(q_len, kv_len, params.group_size,
                                           sliding_window);
    if constexpr (kAlibi) {
      const auto tScS_mn = tScMN_mn(_, _, _0{});
      mask.init_alibi(tScS_mn, kv_head_idx, params.sm_scale,
                      params.alibi_slopes_ptr);
    }

    if (n_block_min < n_block_max) {
      mha(gQ, cQ, gK, gV, cKV, tScMN_mn, tOrAccO, softmax, mask,
          params.logits_soft_cap, tidx, residue_mnk, n_block_min, n_block_max,
          smem);
    }

    epilogue(tOrAccO, tiled_mma, gO, cQ, tidx, residue_mnk, smem);
  }
};

template <typename Element, const int kHeadDim, const bool kEvenK,
          const bool kAlibi, const bool kSoftCap, const bool kLocal>
void single_mha_attention_launcher(const MHAParams &params,
                                   cudaStream_t stream = 0) {
  constexpr int BLK_M = 64;
  constexpr int BLK_N = 64;

  using TileShape = Shape<Int<BLK_M>, Int<BLK_N>, Int<kHeadDim>>;
  using CollectiveMainloop =
      SingleMHACollectiveMainloop<Element, TileShape, kEvenK, kAlibi, kSoftCap,
                                  kLocal>;
  using CollectiveEpilogue =
      SingleMHACollectiveEpilogue<Element, TileShape, kEvenK>;
  using Kernel = SingleMHAKernel<CollectiveMainloop, CollectiveEpilogue>;

  static constexpr int kNThreads = CollectiveMainloop::kNThreads;

  auto mha_kernel = device_kernel<Kernel, MHAParams>;

  const auto smem_size = Kernel::kSharedStorageSize;

  int device_id = 0;
  LLMX_CUDA_CHECK(cudaGetDevice(&device_id));
  cudaDeviceProp prop;
  LLMX_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  LLMX_CHECK(static_cast<size_t>(smem_size) <= prop.sharedMemPerBlockOptin,
             "SingleMHAKernel requires "
                 << smem_size << " bytes of shared memory but device "
                 << device_id << " supports at most "
                 << prop.sharedMemPerBlockOptin << " bytes");
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
        mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }

  const auto q_packed_len = params.q_len * params.group_size;
  dim3 grid(cute::ceil_div(q_packed_len, BLK_M), params.n_kv_heads);
  dim3 block(kNThreads);

  mha_kernel<<<grid, block, smem_size, stream>>>(params);
}

} // namespace llmx