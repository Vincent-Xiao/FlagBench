# FlagGems 和 CUDA 算子库性能对比分析

## CUDA kernel（按总时间排序）

| 框架算子名 | 执行算子名 | 调用次数 | 总时间(ms) | 平均时间(μs) | 占比 |
|---|---|---|---|---|---|
| aten::all_reduce | multimem_all_reduce_kernel | 90296 | 86054.117 | 953.022 | 48.56% |
| aten::reduce | cross_device_reduce_1stage, ncclDevKernel_AllReduce_Sum_bf16_RING_LL, ncclDevKernel_AllReduce_Sum_f32_RING_LL, reduce_1Block_kernel, reduce_kernel, splitKreduce_kernel | 273957 | 25746.309 | 93.979 | 14.53% |
| vllm::fused_moe | fused_moe_kernel | 108480 | 23957.863 | 220.851 | 13.52% |
| aten::mm | nvjet_tst_h_bz_TNN, nvjet_tst_h_bz_TNT, nvjet_tst_h_bz_coopA_bias_TNN, nvjet_tst_h_bz_coopB_TNT, nvjet_tst_h_bz_coopB_bias_TNN, nvjet_tst_h_bz_coopB_bias_TNT, nvjet_tst_h_bz_splitK_TNN, nvjet_tst_h_bz_splitK_TNT, nvjet_tst_v_bz_TNN, nvjet_tst_v_bz_TNT, nvjet_tst_v_bz_coopA_TNN, nvjet_tst_v_bz_coopA_TNT, nvjet_tst_v_bz_coopA_bias_TNN, nvjet_tst_v_bz_coopB_TNN, nvjet_tst_v_bz_coopB_TNT, nvjet_tst_v_bz_coopB_bias_TNN, nvjet_tst_v_bz_splitK_TNN, nvjet_tst_v_bz_splitK_TNT, sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize_cgasize1x1x1_warpgroupsize1x1x1_aligna4_alignc4_execute_segment_k_off_kernel__5x_cublas | 365016 | 23269.142 | 63.748 | 13.13% |
| vllm::device_kernel | device_kernel | 17256 | 6378.472 | 369.638 | 3.60% |
| aten::elementwise | distribution_elementwise_grid_stride_kernel, elementwise_kernel, elementwise_kernel_with_index, index_elementwise_kernel, unrolled_elementwise_kernel, vectorized_elementwise_kernel | 352714 | 3307.507 | 9.377 | 1.87% |
| triton::fused_reduction | triton_red_fused_1, triton_red_fused_7, triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_3, triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_5, triton_red_fused__to_copy_add_mean_moe_forward_shared_mul_pow_rsqrt_1, triton_red_fused__to_copy_add_mean_moe_forward_shared_mul_pow_rsqrt_3 | 122944 | 1957.852 | 15.925 | 1.10% |
| vllm::fused_recurrent_gated_delta_rule_fwd | fused_recurrent_gated_delta_rule_fwd_kernel | 18360 | 1139.899 | 62.086 | 0.64% |
| triton::fused_pointwise | triton_poi_fused_0, triton_poi_fused_2, triton_poi_fused_4, triton_poi_fused_6, triton_poi_fused_8, triton_poi_fused__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2, triton_poi_fused_add_all_reduce_2, triton_poi_fused_add_all_reduce_4, triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0, triton_poi_fused_mul_sigmoid_view_0, triton_poi_fused_mul_silu_slice_0 | 273192 | 1088.307 | 3.984 | 0.61% |
| vllm::topk_gating | topkGating | 54240 | 784.390 | 14.461 | 0.44% |
| aten::mul | act_and_mul_kernel | 54240 | 701.006 | 12.924 | 0.40% |
| vllm::moe_align_block_size | moe_align_block_size_kernel | 54240 | 481.256 | 8.873 | 0.27% |
| aten::sort | DeviceSegmentedRadixSortKernel | 96 | 389.430 | 4056.558 | 0.22% |
| aten::conv2d | _causal_conv1d_fwd_kernel, _causal_conv1d_update_kernel | 21240 | 265.346 | 12.493 | 0.15% |
| aten::exp | count_and_sort_expert_tokens_kernel | 54240 | 240.968 | 4.443 | 0.14% |
| aten::all_gather | ncclDevKernel_AllGather_RING_LL | 1104 | 220.422 | 199.657 | 0.12% |
| custom::kernel2 | Kernel2 | 216 | 218.524 | 1011.685 | 0.12% |
| triton::fused_persistent | triton_per_fused_1 | 40680 | 203.276 | 4.997 | 0.11% |
| aten::softmax | cunn_SoftMaxForward | 1120 | 186.013 | 166.083 | 0.10% |
| vllm::reshape_and_cache | reshape_and_cache_flash_kernel | 19560 | 144.461 | 7.386 | 0.08% |
| aten::layer_norm | vectorized_layer_norm_kernel | 440 | 121.786 | 276.786 | 0.07% |
| aten::gather | vectorized_gather_kernel | 6672 | 67.281 | 10.084 | 0.04% |
| aten::scatter | _scatter_gather_elementwise_kernel | 64 | 61.823 | 965.986 | 0.03% |
| vllm::fused_gdn_gating | fused_gdn_gating_kernel | 21240 | 57.677 | 2.715 | 0.03% |
| aten::broadcast | ncclDevKernel_Broadcast_RING_LL | 48 | 57.626 | 1200.537 | 0.03% |
| aten::scan | tensor_kernel_scan_innermost_dim | 16 | 37.962 | 2372.643 | 0.02% |
| aten::linalg_vector_norm | l2norm_fwd_kernel2 | 5760 | 31.836 | 5.527 | 0.02% |
| aten::unique | fill_reverse_indices_kernel | 16 | 24.515 | 1532.212 | 0.01% |
| vllm::rotary_embedding | rotary_kernel | 216 | 13.849 | 64.114 | 0.01% |
| aten::dot | dot_kernel | 2880 | 5.965 | 2.071 | 0.00% |
| vllm::prepare_varlen_num_blocks | prepare_varlen_num_blocks_kernel | 2120 | 4.407 | 2.079 | 0.00% |
| aten::copy_ | CatArrayBatchedCopy, CatArrayBatchedCopy_alignedK_contig, CatArrayBatchedCopy_vectorized | 40 | 1.296 | 32.399 | 0.00% |
| aten::index_select | indexSelectSmallIndex | 16 | 0.051 | 3.196 | 0.00% |
| custom::lamport_initialize | lamport_initialize_kernel | 8 | 0.039 | 4.884 | 0.00% |

## FlagGems kernel（按总时间排序）

| 框架算子名 | 执行算子名 | 调用次数 | 总时间(ms) | 平均时间(μs) | 占比 |
|---|---|---|---|---|---|
| aten::all_reduce | multimem_all_reduce_kernel | 90296 | 149523.959 | 1655.931 | 60.84% |
| aten::mm | addmm_kernel, mm_kernel_general, mm_kernel_general_host_tma | 313872 | 24313.064 | 77.462 | 9.89% |
| vllm::fused_moe | fused_moe_kernel | 108480 | 24062.654 | 221.816 | 9.79% |
| aten::reduce | cross_device_reduce_1stage, ncclDevKernel_AllReduce_Sum_bf16_RING_LL, ncclDevKernel_AllReduce_Sum_f32_RING_LL, reduce_then_scan_block_scan_kernel_row, reduce_then_scan_block_sum_kernel_row, reduce_then_scan_root_scan_kernel_row | 11352 | 20985.617 | 1848.627 | 8.54% |
| vllm::device_kernel | device_kernel | 17256 | 6375.470 | 369.464 | 2.59% |
| vllm::sweep | sweep | 128 | 2480.150 | 19376.172 | 1.01% |
| aten::copy_ | _copy_kernel_kernel_rank_1, _copy_kernel_kernel_rank_2, _copy_kernel_kernel_rank_3, _copy_kernel_kernel_rank_4, _copy_kernel_kernel_rank_5, _to_copy_func_kernel_rank_1, cat_copy_func_kernel_4, stack_copy_func_kernel_4 | 378160 | 2254.252 | 5.961 | 0.92% |
| aten::sum | sum_dim_kernel_non_inner, sum_kernel_1, sum_kernel_2 | 54290 | 1988.510 | 36.628 | 0.81% |
| triton::fused_reduction | triton_red_fused_1, triton_red_fused_7, triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_3, triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_5, triton_red_fused__to_copy_add_mean_moe_forward_shared_mul_pow_rsqrt_1, triton_red_fused__to_copy_add_mean_moe_forward_shared_mul_pow_rsqrt_3 | 122944 | 1956.101 | 15.911 | 0.80% |
| aten::fill_ | fill_scalar_func_kernel_rank_1 | 77910 | 1850.287 | 23.749 | 0.75% |
| aten::mul | act_and_mul_kernel, mul_func_kernel_rank_2, mul_func_kernel_rank_3, mul_func_scalar_kernel_rank_1 | 108536 | 1766.639 | 16.277 | 0.72% |
| vllm::fused_recurrent_gated_delta_rule_fwd | fused_recurrent_gated_delta_rule_fwd_kernel | 18360 | 1140.217 | 62.103 | 0.46% |
| triton::fused_pointwise | triton_poi_fused_0, triton_poi_fused_2, triton_poi_fused_4, triton_poi_fused_6, triton_poi_fused_8, triton_poi_fused__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2, triton_poi_fused_add_all_reduce_2, triton_poi_fused_add_all_reduce_4, triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0, triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0, triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1, triton_poi_fused_mul_sigmoid_view_0, triton_poi_fused_mul_silu_slice_0 | 273200 | 1085.720 | 3.974 | 0.44% |
| aten::layer_norm | layer_norm_persistent_kernel | 3804 | 931.313 | 244.825 | 0.38% |
| aten::mv | gemv_kernel | 54240 | 778.565 | 14.354 | 0.32% |
| vllm::topk_gating | topkGating | 54240 | 773.335 | 14.258 | 0.31% |
| vllm::moe_align_block_size | moe_align_block_size_kernel | 54240 | 488.596 | 9.008 | 0.20% |
| aten::all_gather | ncclDevKernel_AllGather_RING_LL | 1104 | 363.054 | 328.853 | 0.15% |
| aten::zeros | zeros_kernel | 55792 | 346.309 | 6.207 | 0.14% |
| aten::exp | count_and_sort_expert_tokens_kernel, exp_func_kernel_rank_1, fused_exponential_kernel_f32 | 58224 | 323.122 | 5.550 | 0.13% |
| aten::conv2d | _causal_conv1d_fwd_kernel, _causal_conv1d_update_kernel | 21240 | 263.541 | 12.408 | 0.11% |
| aten::histogram | compute_global_hist_kernel | 16 | 258.387 | 16149.217 | 0.11% |
| triton::fused_persistent | triton_per_fused_1 | 40680 | 202.698 | 4.983 | 0.08% |
| aten::div | true_div_func_kernel_rank_1, true_div_func_kernel_rank_2, true_div_func_tensor_scalar_kernel_rank_1 | 2224 | 181.115 | 81.437 | 0.07% |
| vllm::reshape_and_cache | reshape_and_cache_flash_kernel | 19560 | 144.039 | 7.364 | 0.06% |
| aten::softmax | softmax_kernel_inner | 1120 | 140.021 | 125.018 | 0.06% |
| aten::add | add_func_kernel_rank_1, add_func_tensor_scalar_kernel_rank_1 | 506 | 99.986 | 197.602 | 0.04% |
| aten::max | argmax_kernel_inner, clamp_func_max_kernel_rank_1 | 1136 | 89.539 | 78.820 | 0.04% |
| aten::broadcast | ncclDevKernel_Broadcast_RING_LL | 48 | 79.590 | 1658.126 | 0.03% |
| aten::sigmoid | sigmoid_forward_kernel_rank_1 | 54240 | 78.892 | 1.454 | 0.03% |
| aten::index_put | _index_put_jit_function | 5760 | 58.823 | 10.212 | 0.02% |
| aten::scatter | _scatter_jit_function | 48 | 57.973 | 1207.773 | 0.02% |
| vllm::fused_gdn_gating | fused_gdn_gating_kernel | 21240 | 57.911 | 2.726 | 0.02% |
| aten::index | _index_jit_function | 5672 | 56.599 | 9.979 | 0.02% |
| aten::gelu | gelu_none_kernel_rank_1, gelu_tanh_kernel_rank_1 | 224 | 45.658 | 203.832 | 0.02% |
| aten::masked_fill | masked_fill_kernel_kernel_rank_1, masked_fill_kernel_kernel_rank_2 | 1120 | 34.727 | 31.007 | 0.01% |
| aten::linalg_vector_norm | l2norm_fwd_kernel2 | 5760 | 29.544 | 5.129 | 0.01% |
| aten::sub | sub_func_kernel_rank_1, sub_func_kernel_rank_2, sub_func_scalar_tensor_kernel_rank_1, sub_func_scalar_tensor_kernel_rank_2, sub_func_tensor_scalar_kernel_rank_1 | 15056 | 16.905 | 1.123 | 0.01% |
| vllm::rotary_embedding | rotary_kernel | 216 | 13.840 | 64.073 | 0.01% |
| aten::lt | lt_func_kernel_rank_2, lt_func_scalar_kernel_rank_1 | 48 | 11.854 | 246.965 | 0.00% |
| aten::le | le_func_kernel_rank_2 | 16 | 11.806 | 737.872 | 0.00% |
| aten::full | full_func_scalar_kernel_rank_1 | 3360 | 9.755 | 2.903 | 0.00% |
| aten::embedding | embedding_kernel | 1096 | 5.963 | 5.441 | 0.00% |
| aten::nonzero | nonzero_kernel | 2880 | 5.611 | 1.948 | 0.00% |
| vllm::prepare_varlen_num_blocks | prepare_varlen_num_blocks_kernel | 2120 | 4.415 | 2.083 | 0.00% |
| aten::bitwise_not | bitwise_not_func_kernel_rank_1 | 2880 | 4.242 | 1.473 | 0.00% |
| aten::cos | cos_func_kernel_rank_1 | 16 | 2.710 | 169.406 | 0.00% |
| aten::sin | sin_func_kernel_rank_1 | 16 | 2.693 | 168.298 | 0.00% |
| aten::ones | ones_kernel | 360 | 0.347 | 0.965 | 0.00% |
| aten::gt | gt_func_scalar_kernel_rank_1 | 192 | 0.242 | 1.263 | 0.00% |
| aten::rand | rand_kernel | 16 | 0.175 | 10.912 | 0.00% |
| aten::elementwise | distribution_elementwise_grid_stride_kernel, vectorized_elementwise_kernel | 32 | 0.155 | 4.850 | 0.00% |
| aten::arange | arange_func | 48 | 0.122 | 2.543 | 0.00% |
| aten::where | where_inner_kernel_rank_1 | 32 | 0.053 | 1.654 | 0.00% |
| aten::gather | _gather_flaggems_jit_function | 16 | 0.041 | 2.566 | 0.00% |
| custom::lamport_initialize | lamport_initialize_kernel | 8 | 0.038 | 4.796 | 0.00% |
| aten::linspace | linspace_kernel | 16 | 0.024 | 1.496 | 0.00% |
| aten::pow | pow_func_scalar_tensor_kernel_rank_1 | 16 | 0.022 | 1.374 | 0.00% |
| aten::reciprocal | reciprocal_func_kernel_rank_1 | 16 | 0.018 | 1.110 | 0.00% |

## CUDA 和 FlagGems kernel 对比（按 CUDA 总时间排序）

| 框架算子名 | 执行算子名 | CUDA调用次数 | CUDA总时间(ms) | CUDA占比 | FlagGems调用次数 | FlagGems总时间(ms) | FlagGems占比 | 加速比(CUDA/FlagGems) |
|---|---|---|---|---|---|---|---|---|
| aten::all_reduce | multimem_all_reduce_kernel | 90296 | 86054.117 | 48.56% | 90296 | 149523.959 | 60.84% | 0.576 |
| aten::reduce | cross_device_reduce_1stage, ncclDevKernel_AllReduce_Sum_bf16_RING_LL, ncclDevKernel_AllReduce_Sum_f32_RING_LL, reduce_1Block_kernel, reduce_kernel, reduce_then_scan_block_scan_kernel_row, reduce_then_scan_block_sum_kernel_row, reduce_then_scan_root_scan_kernel_row, splitKreduce_kernel | 273957 | 25746.309 | 14.53% | 11352 | 20985.617 | 8.54% | 1.227 |
| vllm::fused_moe | fused_moe_kernel | 108480 | 23957.863 | 13.52% | 108480 | 24062.654 | 9.79% | 0.996 |
| aten::mm | addmm_kernel, mm_kernel_general, mm_kernel_general_host_tma, nvjet_tst_h_bz_TNN, nvjet_tst_h_bz_TNT, nvjet_tst_h_bz_coopA_bias_TNN, nvjet_tst_h_bz_coopB_TNT, nvjet_tst_h_bz_coopB_bias_TNN, nvjet_tst_h_bz_coopB_bias_TNT, nvjet_tst_h_bz_splitK_TNN, nvjet_tst_h_bz_splitK_TNT, nvjet_tst_v_bz_TNN, nvjet_tst_v_bz_TNT, nvjet_tst_v_bz_coopA_TNN, nvjet_tst_v_bz_coopA_TNT, nvjet_tst_v_bz_coopA_bias_TNN, nvjet_tst_v_bz_coopB_TNN, nvjet_tst_v_bz_coopB_TNT, nvjet_tst_v_bz_coopB_bias_TNN, nvjet_tst_v_bz_splitK_TNN, nvjet_tst_v_bz_splitK_TNT, sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize_cgasize1x1x1_warpgroupsize1x1x1_aligna4_alignc4_execute_segment_k_off_kernel__5x_cublas | 365016 | 23269.142 | 13.13% | 313872 | 24313.064 | 9.89% | 0.957 |
| vllm::device_kernel | device_kernel | 17256 | 6378.472 | 3.60% | 17256 | 6375.470 | 2.59% | 1.000 |
| aten::elementwise | distribution_elementwise_grid_stride_kernel, elementwise_kernel, elementwise_kernel_with_index, index_elementwise_kernel, unrolled_elementwise_kernel, vectorized_elementwise_kernel | 352714 | 3307.507 | 1.87% | 32 | 0.155 | 0.00% | 21311.116 |
| triton::fused_reduction | triton_red_fused_1, triton_red_fused_7, triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_3, triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_5, triton_red_fused__to_copy_add_mean_moe_forward_shared_mul_pow_rsqrt_1, triton_red_fused__to_copy_add_mean_moe_forward_shared_mul_pow_rsqrt_3 | 122944 | 1957.852 | 1.10% | 122944 | 1956.101 | 0.80% | 1.001 |
| vllm::fused_recurrent_gated_delta_rule_fwd | fused_recurrent_gated_delta_rule_fwd_kernel | 18360 | 1139.899 | 0.64% | 18360 | 1140.217 | 0.46% | 1.000 |
| triton::fused_pointwise | triton_poi_fused_0, triton_poi_fused_2, triton_poi_fused_4, triton_poi_fused_6, triton_poi_fused_8, triton_poi_fused__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2, triton_poi_fused_add_all_reduce_2, triton_poi_fused_add_all_reduce_4, triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0, triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0, triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1, triton_poi_fused_mul_sigmoid_view_0, triton_poi_fused_mul_silu_slice_0 | 273192 | 1088.307 | 0.61% | 273200 | 1085.720 | 0.44% | 1.002 |
| vllm::topk_gating | topkGating | 54240 | 784.390 | 0.44% | 54240 | 773.335 | 0.31% | 1.014 |
| aten::mul | act_and_mul_kernel, mul_func_kernel_rank_2, mul_func_kernel_rank_3, mul_func_scalar_kernel_rank_1 | 54240 | 701.006 | 0.40% | 108536 | 1766.639 | 0.72% | 0.397 |
| vllm::moe_align_block_size | moe_align_block_size_kernel | 54240 | 481.256 | 0.27% | 54240 | 488.596 | 0.20% | 0.985 |
| aten::conv2d | _causal_conv1d_fwd_kernel, _causal_conv1d_update_kernel | 21240 | 265.346 | 0.15% | 21240 | 263.541 | 0.11% | 1.007 |
| aten::exp | count_and_sort_expert_tokens_kernel, exp_func_kernel_rank_1, fused_exponential_kernel_f32 | 54240 | 240.968 | 0.14% | 58224 | 323.122 | 0.13% | 0.746 |
| aten::all_gather | ncclDevKernel_AllGather_RING_LL | 1104 | 220.422 | 0.12% | 1104 | 363.054 | 0.15% | 0.607 |
| triton::fused_persistent | triton_per_fused_1 | 40680 | 203.276 | 0.11% | 40680 | 202.698 | 0.08% | 1.003 |
| aten::softmax | cunn_SoftMaxForward, softmax_kernel_inner | 1120 | 186.013 | 0.10% | 1120 | 140.021 | 0.06% | 1.328 |
| vllm::reshape_and_cache | reshape_and_cache_flash_kernel | 19560 | 144.461 | 0.08% | 19560 | 144.039 | 0.06% | 1.003 |
| aten::layer_norm | layer_norm_persistent_kernel, vectorized_layer_norm_kernel | 440 | 121.786 | 0.07% | 3804 | 931.313 | 0.38% | 0.131 |
| aten::gather | _gather_flaggems_jit_function, vectorized_gather_kernel | 6672 | 67.281 | 0.04% | 16 | 0.041 | 0.00% | 1638.760 |
| aten::scatter | _scatter_gather_elementwise_kernel, _scatter_jit_function | 64 | 61.823 | 0.03% | 48 | 57.973 | 0.02% | 1.066 |
| vllm::fused_gdn_gating | fused_gdn_gating_kernel | 21240 | 57.677 | 0.03% | 21240 | 57.911 | 0.02% | 0.996 |
| aten::broadcast | ncclDevKernel_Broadcast_RING_LL | 48 | 57.626 | 0.03% | 48 | 79.590 | 0.03% | 0.724 |
| aten::linalg_vector_norm | l2norm_fwd_kernel2 | 5760 | 31.836 | 0.02% | 5760 | 29.544 | 0.01% | 1.078 |
| vllm::rotary_embedding | rotary_kernel | 216 | 13.849 | 0.01% | 216 | 13.840 | 0.01% | 1.001 |
| vllm::prepare_varlen_num_blocks | prepare_varlen_num_blocks_kernel | 2120 | 4.407 | 0.00% | 2120 | 4.415 | 0.00% | 0.998 |
| aten::copy_ | CatArrayBatchedCopy, CatArrayBatchedCopy_alignedK_contig, CatArrayBatchedCopy_vectorized, _copy_kernel_kernel_rank_1, _copy_kernel_kernel_rank_2, _copy_kernel_kernel_rank_3, _copy_kernel_kernel_rank_4, _copy_kernel_kernel_rank_5, _to_copy_func_kernel_rank_1, cat_copy_func_kernel_4, stack_copy_func_kernel_4 | 40 | 1.296 | 0.00% | 378160 | 2254.252 | 0.92% | 0.001 |
| custom::lamport_initialize | lamport_initialize_kernel | 8 | 0.039 | 0.00% | 8 | 0.038 | 0.00% | 1.018 |

## CUDA 和 FlagGems kernel 对比（按加速比从高到低）

| 框架算子名 | 执行算子名 | CUDA调用次数 | CUDA总时间(ms) | CUDA占比 | FlagGems调用次数 | FlagGems总时间(ms) | FlagGems占比 | 加速比(CUDA/FlagGems) |
|---|---|---|---|---|---|---|---|---|
| aten::elementwise | distribution_elementwise_grid_stride_kernel, elementwise_kernel, elementwise_kernel_with_index, index_elementwise_kernel, unrolled_elementwise_kernel, vectorized_elementwise_kernel | 352714 | 3307.507 | 1.87% | 32 | 0.155 | 0.00% | 21311.116 |
| aten::gather | _gather_flaggems_jit_function, vectorized_gather_kernel | 6672 | 67.281 | 0.04% | 16 | 0.041 | 0.00% | 1638.760 |
| aten::softmax | cunn_SoftMaxForward, softmax_kernel_inner | 1120 | 186.013 | 0.10% | 1120 | 140.021 | 0.06% | 1.328 |
| aten::reduce | cross_device_reduce_1stage, ncclDevKernel_AllReduce_Sum_bf16_RING_LL, ncclDevKernel_AllReduce_Sum_f32_RING_LL, reduce_1Block_kernel, reduce_kernel, reduce_then_scan_block_scan_kernel_row, reduce_then_scan_block_sum_kernel_row, reduce_then_scan_root_scan_kernel_row, splitKreduce_kernel | 273957 | 25746.309 | 14.53% | 11352 | 20985.617 | 8.54% | 1.227 |
| aten::linalg_vector_norm | l2norm_fwd_kernel2 | 5760 | 31.836 | 0.02% | 5760 | 29.544 | 0.01% | 1.078 |
| aten::scatter | _scatter_gather_elementwise_kernel, _scatter_jit_function | 64 | 61.823 | 0.03% | 48 | 57.973 | 0.02% | 1.066 |
| custom::lamport_initialize | lamport_initialize_kernel | 8 | 0.039 | 0.00% | 8 | 0.038 | 0.00% | 1.018 |
| vllm::topk_gating | topkGating | 54240 | 784.390 | 0.44% | 54240 | 773.335 | 0.31% | 1.014 |
| aten::conv2d | _causal_conv1d_fwd_kernel, _causal_conv1d_update_kernel | 21240 | 265.346 | 0.15% | 21240 | 263.541 | 0.11% | 1.007 |
| vllm::reshape_and_cache | reshape_and_cache_flash_kernel | 19560 | 144.461 | 0.08% | 19560 | 144.039 | 0.06% | 1.003 |
| triton::fused_persistent | triton_per_fused_1 | 40680 | 203.276 | 0.11% | 40680 | 202.698 | 0.08% | 1.003 |
| triton::fused_pointwise | triton_poi_fused_0, triton_poi_fused_2, triton_poi_fused_4, triton_poi_fused_6, triton_poi_fused_8, triton_poi_fused__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_2, triton_poi_fused_add_all_reduce_2, triton_poi_fused_add_all_reduce_4, triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0, triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0, triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1, triton_poi_fused_mul_sigmoid_view_0, triton_poi_fused_mul_silu_slice_0 | 273192 | 1088.307 | 0.61% | 273200 | 1085.720 | 0.44% | 1.002 |
| triton::fused_reduction | triton_red_fused_1, triton_red_fused_7, triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_3, triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_5, triton_red_fused__to_copy_add_mean_moe_forward_shared_mul_pow_rsqrt_1, triton_red_fused__to_copy_add_mean_moe_forward_shared_mul_pow_rsqrt_3 | 122944 | 1957.852 | 1.10% | 122944 | 1956.101 | 0.80% | 1.001 |
| vllm::rotary_embedding | rotary_kernel | 216 | 13.849 | 0.01% | 216 | 13.840 | 0.01% | 1.001 |
| vllm::device_kernel | device_kernel | 17256 | 6378.472 | 3.60% | 17256 | 6375.470 | 2.59% | 1.000 |
| vllm::fused_recurrent_gated_delta_rule_fwd | fused_recurrent_gated_delta_rule_fwd_kernel | 18360 | 1139.899 | 0.64% | 18360 | 1140.217 | 0.46% | 1.000 |
| vllm::prepare_varlen_num_blocks | prepare_varlen_num_blocks_kernel | 2120 | 4.407 | 0.00% | 2120 | 4.415 | 0.00% | 0.998 |
| vllm::fused_gdn_gating | fused_gdn_gating_kernel | 21240 | 57.677 | 0.03% | 21240 | 57.911 | 0.02% | 0.996 |
| vllm::fused_moe | fused_moe_kernel | 108480 | 23957.863 | 13.52% | 108480 | 24062.654 | 9.79% | 0.996 |
| vllm::moe_align_block_size | moe_align_block_size_kernel | 54240 | 481.256 | 0.27% | 54240 | 488.596 | 0.20% | 0.985 |
| aten::mm | addmm_kernel, mm_kernel_general, mm_kernel_general_host_tma, nvjet_tst_h_bz_TNN, nvjet_tst_h_bz_TNT, nvjet_tst_h_bz_coopA_bias_TNN, nvjet_tst_h_bz_coopB_TNT, nvjet_tst_h_bz_coopB_bias_TNN, nvjet_tst_h_bz_coopB_bias_TNT, nvjet_tst_h_bz_splitK_TNN, nvjet_tst_h_bz_splitK_TNT, nvjet_tst_v_bz_TNN, nvjet_tst_v_bz_TNT, nvjet_tst_v_bz_coopA_TNN, nvjet_tst_v_bz_coopA_TNT, nvjet_tst_v_bz_coopA_bias_TNN, nvjet_tst_v_bz_coopB_TNN, nvjet_tst_v_bz_coopB_TNT, nvjet_tst_v_bz_coopB_bias_TNN, nvjet_tst_v_bz_splitK_TNN, nvjet_tst_v_bz_splitK_TNT, sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize_cgasize1x1x1_warpgroupsize1x1x1_aligna4_alignc4_execute_segment_k_off_kernel__5x_cublas | 365016 | 23269.142 | 13.13% | 313872 | 24313.064 | 9.89% | 0.957 |
| aten::exp | count_and_sort_expert_tokens_kernel, exp_func_kernel_rank_1, fused_exponential_kernel_f32 | 54240 | 240.968 | 0.14% | 58224 | 323.122 | 0.13% | 0.746 |
| aten::broadcast | ncclDevKernel_Broadcast_RING_LL | 48 | 57.626 | 0.03% | 48 | 79.590 | 0.03% | 0.724 |
| aten::all_gather | ncclDevKernel_AllGather_RING_LL | 1104 | 220.422 | 0.12% | 1104 | 363.054 | 0.15% | 0.607 |
| aten::all_reduce | multimem_all_reduce_kernel | 90296 | 86054.117 | 48.56% | 90296 | 149523.959 | 60.84% | 0.576 |
| aten::mul | act_and_mul_kernel, mul_func_kernel_rank_2, mul_func_kernel_rank_3, mul_func_scalar_kernel_rank_1 | 54240 | 701.006 | 0.40% | 108536 | 1766.639 | 0.72% | 0.397 |
| aten::layer_norm | layer_norm_persistent_kernel, vectorized_layer_norm_kernel | 440 | 121.786 | 0.07% | 3804 | 931.313 | 0.38% | 0.131 |
| aten::copy_ | CatArrayBatchedCopy, CatArrayBatchedCopy_alignedK_contig, CatArrayBatchedCopy_vectorized, _copy_kernel_kernel_rank_1, _copy_kernel_kernel_rank_2, _copy_kernel_kernel_rank_3, _copy_kernel_kernel_rank_4, _copy_kernel_kernel_rank_5, _to_copy_func_kernel_rank_1, cat_copy_func_kernel_4, stack_copy_func_kernel_4 | 40 | 1.296 | 0.00% | 378160 | 2254.252 | 0.92% | 0.001 |
