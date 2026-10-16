#include <ATen/native/sparse/cuda/cuSPARSELtOps.h>
#include <unordered_map>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <c10/util/StringUtil.h>
#if AT_CUSPARSELT_ENABLED()

namespace at::native {

// Ideally we would use the same DeviceThreadHandlePool mechanism as used in
// aten/src/ATen/cuda/CuSparseHandlePool.cpp which would handle this for us.
// However, the cuSPARSELt handle signature is different from that of
// cuSPARSE/cuBLAS, so it's not possible to reuse the existing pooling
// mechanism. Instead we have to handle our handles ourselves, which is why
// these variables are thread local. Once cuSPARSELt updates their handle
// signature to be consistent with the rest of CUDA, we can switch to using
// DeviceThreadHandlePool.
thread_local cusparseLtHandle_t handle;
thread_local bool handle_initialized = false;

#ifdef USE_ROCM
// Single global flag for platform-wide hipSparseLt support
c10::once_flag g_hipSparseLtSupportInitFlag;
static bool g_hipSparseLtSupported = false;

static const std::vector<std::string>& hipSparseLtSupportedArchs() {
#if ROCM_VERSION >= 71400
  static const std::vector<std::string> archs = {"gfx950", "gfx942", "gfx1250", "gfx1013"};
#elif ROCM_VERSION >= 71200
  static const std::vector<std::string> archs = {"gfx950", "gfx942", "gfx1013"};
#else
  static const std::vector<std::string> archs = {};
#endif
  return archs;
}

// Initialize the hipSparseLt support status once for the platform
static void initHipSparseLtSupport() {
  // Default to not supported
  g_hipSparseLtSupported = false;

  // Check only the first available device
  try {
    if (at::cuda::device_count() > 0) {
      g_hipSparseLtSupported = at::detail::getCUDAHooks().isGPUArch(
          hipSparseLtSupportedArchs(), 0);
    }
  } catch (const std::exception&) {
    // If an exception occurs during device property check, we assume
    // hipSparseLt is not supported This could happen due to driver issues,
    // device access problems, or other runtime errors
     g_hipSparseLtSupported = false;
    TORCH_WARN(
        "Exception occurred while checking hipSparseLt support. Assuming not supported.");
  }
}

static bool isHipSparseLtSupported() {
    // Initialize support check only once
    c10::call_once(g_hipSparseLtSupportInitFlag, initHipSparseLtSupport);

  // Return cached result (platform-wide)
  if (!g_hipSparseLtSupported) {
    TORCH_CHECK(
        false,
        "hipSparseLt not supported on this device. Supported architectures: ",
        c10::Join(", ", hipSparseLtSupportedArchs()),
        ". hipSparseLt on ROCm requires ROCm 7.12 or newer.");
  }
  return g_hipSparseLtSupported;
}
#endif

static bool usesGfx1013ScalarSparseLt() {
#ifdef USE_ROCM
  const std::string_view arch(at::cuda::getCurrentDeviceProperties()->gcnArchName);
  return arch == "gfx1013" || arch.substr(0, 8) == "gfx1013:";
#else
  return false;
#endif
}

// Native packs store retained values followed by byte-sized 1:2/2:4 indices.
static std::tuple<Tensor, Tensor> unpackGfx1013(const Tensor& packed, int64_t k) {
  TORCH_CHECK(usesGfx1013ScalarSparseLt(), "Sparse Lt backward/unpacking requires gfx1013");
  TORCH_CHECK(packed.dim() == 2 && packed.is_contiguous() && k > 0,
      "Invalid native sparse pack");
  const auto dtype = packed.scalar_type();
  TORCH_CHECK(dtype == kFloat || dtype == kHalf || dtype == kBFloat16,
      "Sparse Lt backward/unpacking requires floating-point storage");
  const int64_t group = dtype == kFloat ? 2 : 4;
  TORCH_CHECK(k % group == 0, "Invalid native sparse contraction size");
  const auto m = packed.size(0), groups = k / group, kept = group / 2;
  const auto value_count = m * k / 2;
  auto raw = packed.view(kByte).flatten();
  TORCH_CHECK(raw.numel() >= value_count * packed.itemsize() + m * groups,
      "Native sparse pack is too small");
  auto tags = raw.narrow(0, value_count * packed.itemsize(), m * groups)
                  .to(kLong).view({m, groups, 1});
  auto shifts = at::arange(kept, packed.options().dtype(kLong)).mul(2);
  auto indices = at::bitwise_and(at::bitwise_right_shift(tags, shifts), 3);
  TORCH_CHECK(indices.lt(group).all().item<bool>(), "Invalid native sparse metadata");
  return {packed.flatten().narrow(0, 0, value_count).view({m, groups, kept}), indices};
}

Tensor _cslt_to_dense(const Tensor& packed, int64_t k) {
  auto [values, indices] = unpackGfx1013(packed, k);
  const int64_t m = packed.size(0), group = packed.scalar_type() == kFloat ? 2 : 4;
  return at::zeros({m, k / group, group}, packed.options())
      .scatter(2, indices, values).view({m, k});
}

Tensor _cslt_compress_backward(const Tensor& grad, const Tensor& packed, int64_t k) {
  auto [values, indices] = unpackGfx1013(packed, k);
  const int64_t m = packed.size(0), group = packed.scalar_type() == kFloat ? 2 : 4;
  auto retained_grad = grad.flatten().narrow(0, 0, m * k / 2).view(values.sizes());
  return at::zeros({m, k / group, group}, packed.options())
      .scatter(2, indices, retained_grad).view({m, k});
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _cslt_sparse_mm_backward(
    const Tensor& grad, const Tensor& packed, const Tensor& dense,
    const std::optional<Tensor>& bias, const std::optional<Tensor>& alpha,
    bool transpose_result, std::array<bool, 4> output_mask) {
  auto a = at::native::_cslt_to_dense(packed, dense.size(0)).to(kFloat);
  auto b = dense.to(kFloat);
  auto g = (transpose_result ? grad.t() : grad).to(kFloat);
  const bool has_alpha = alpha.has_value() && alpha->defined();
  const bool has_bias = bias.has_value() && bias->defined();
  auto scaled = has_alpha
      ? g.mul(alpha->numel() == 1 ? alpha->to(kFloat) : alpha->to(kFloat).view({-1, 1}))
      : g;
  Tensor dp, db, dbias, dalpha;
  if (output_mask[0]) {
    auto da = at::mm(scaled, b.t());
    auto [values, indices] = unpackGfx1013(packed, dense.size(0));
    const auto group = packed.scalar_type() == kFloat ? 2 : 4;
    auto retained = da.view({packed.size(0), dense.size(0) / group, group})
                        .gather(2, indices).to(packed.scalar_type());
    dp = at::zeros_like(packed);
    dp.flatten().narrow(0, 0, retained.numel()).copy_(retained.flatten());
  }
  if (output_mask[1]) {
    db = at::mm(a.t(), scaled).to(dense.scalar_type());
  }
  if (output_mask[2] && has_bias) {
    dbias = g.sum(1).to(bias->scalar_type());
  }
  if (output_mask[3] && has_alpha) {
    auto product = g.mul(at::mm(a, b));
    dalpha = (alpha->numel() == 1 ? product.sum() : product.sum(1))
                 .to(alpha->scalar_type()).view(alpha->sizes());
  }
  return {dp, db, dbias, dalpha};
}

at::Tensor _cslt_compress(const Tensor& sparse_input) {
  const bool native_gfx1013 = usesGfx1013ScalarSparseLt();
  if (native_gfx1013) {
    TORCH_CHECK(sparse_input.is_cuda() && sparse_input.dim() == 2 &&
        sparse_input.is_contiguous(), "gfx1013 sparse compression requires a contiguous GPU matrix");
    TORCH_CHECK(sparse_input.size(0) > 0 && sparse_input.size(1) > 0,
        "gfx1013 sparse compression requires nonempty dimensions");
  }
  if (!handle_initialized) {
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
    handle_initialized = true;
  }
  // create sparse descriptor, dtype
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  cudaDataType type;

  #ifdef USE_ROCM
  TORCH_CHECK(isHipSparseLtSupported());
  #endif

  switch (sparse_input.scalar_type()) {
    case at::ScalarType::Char:
      type = CUDA_R_8I;
      break;
    case at::ScalarType::Half:
      type = CUDA_R_16F;
      break;
    case at::ScalarType::BFloat16:
      type = CUDA_R_16BF;
      break;
    case at::ScalarType::Float:
#ifdef USE_ROCM
      TORCH_CHECK(native_gfx1013, "FP32 hipSPARSELt compression requires the native gfx1013 scalar backend");
#endif
      type = CUDA_R_32F;
      break;
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602 || defined(USE_ROCM)
    case at::ScalarType::Float8_e4m3fn:
      type = CUDA_R_8F_E4M3;
      break;
#endif
    default:
      TORCH_CHECK(false, "Unsupported dtype for cuSPARSELt/hipSparseLt compressed matrix");
      break;
  }

  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      sparse_input.size(0),
      sparse_input.size(1),
      sparse_input.size(1),
      16,
      type,
      CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT));

  // compress input
  //--------------------------------------------------------------------------
  size_t compressed_size, compressed_buffer_size;
  TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompressedSize2(
      &handle,
      &sparse_input_descriptor,
      &compressed_size,
      &compressed_buffer_size));

  // create a new compressed tensor with the same dtype as the input,
  // and with packed data/metadata stored in an array with original
  // number of rows, and sufficient columns to provide compressed_size
  // buffer (in bytes)
  size_t orig_m = sparse_input.size(0);
  size_t div = orig_m * sparse_input.itemsize();
  size_t new_n = (compressed_size + div - 1) / div; // ceil(s,d) = (s+d-1)/d
  auto compressed_tensor = sparse_input.new_empty({(int64_t)orig_m, (int64_t)new_n});

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto compressedBufferPtr = allocator.allocate(compressed_buffer_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompress2(
      &handle,
      &sparse_input_descriptor,
      true,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      sparse_input.data_ptr(),
      compressed_tensor.data_ptr(),
      compressedBufferPtr.get(),
      stream));

  return compressed_tensor;
}

std::tuple<at::Tensor, int64_t, int64_t, int64_t, int64_t> _cslt_sparse_mm_impl(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int alg_id,
    int split_k,
    int split_k_mode,
    bool search_alg_id) {
  const bool native_gfx1013 = usesGfx1013ScalarSparseLt();
  if (native_gfx1013) {
    TORCH_CHECK(compressed_A.is_cuda() && dense_B.is_cuda() &&
        compressed_A.device() == dense_B.device(), "gfx1013 sparse matmul inputs must share a GPU");
    TORCH_CHECK(compressed_A.dim() == 2 && compressed_A.is_contiguous() &&
        dense_B.dim() == 2 && (dense_B.is_contiguous() || dense_B.t().is_contiguous()),
        "gfx1013 sparse matmul requires a contiguous pack and row/column-contiguous dense matrix");
    TORCH_CHECK(compressed_A.scalar_type() == dense_B.scalar_type(),
        "gfx1013 sparse matmul input dtypes must match");
    TORCH_CHECK(compressed_A.size(0) > 0 && dense_B.size(0) > 0 && dense_B.size(1) > 0,
        "gfx1013 sparse matmul requires nonempty dimensions");
    TORCH_CHECK(!out_dtype_opt.has_value(),
        "gfx1013 scalar hipSPARSELt supports same-dtype output only; omit out_dtype");
    if (bias_opt.has_value()) {
      const auto& bias = *bias_opt;
      TORCH_CHECK(bias.is_cuda() && bias.device() == dense_B.device() &&
          bias.dim() == 1 && bias.numel() == compressed_A.size(0) && bias.is_contiguous(),
          "gfx1013 sparse bias must be a contiguous GPU vector with one value per row");
      TORCH_CHECK(bias.scalar_type() == at::kFloat ||
          (dense_B.scalar_type() != at::kChar && bias.scalar_type() == dense_B.scalar_type()),
          "gfx1013 sparse bias must be FP32 or match the floating input dtype");
    }
    if (alpha_opt.has_value() && alpha_opt->numel() != 1) {
      TORCH_CHECK(alpha_opt->is_cuda() && alpha_opt->device() == dense_B.device() &&
          alpha_opt->scalar_type() == at::kFloat && alpha_opt->is_contiguous() &&
          alpha_opt->dim() == 1 && alpha_opt->numel() == compressed_A.size(0),
          "gfx1013 sparse alpha must be a scalar or a contiguous FP32 GPU vector with one value per row");
    }
  }
  if (!handle_initialized) {
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
    handle_initialized = true;
  }
  // cuSPARSELt constructs
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;

  int tensor_alpha_mode = 0;
  float alpha = 1.0;
  float beta = 0.0;
  cudaDataType input_type;
  cudaDataType output_type;
  cudaDataType C_type;
  cusparseComputeType compute_type;

  #ifdef USE_ROCM
  TORCH_CHECK(isHipSparseLtSupported());
  #endif

  switch (compressed_A.scalar_type()) {
    case at::ScalarType::Char:
      input_type = CUDA_R_8I;
      output_type = CUDA_R_8I;
      C_type = CUDA_R_8I;
      compute_type = CUSPARSE_COMPUTE_32I;
      break;

// cuSPARSELt v0.5.2 onwards changes CUSPARSE_COMPUTE_TF32, CUSPARSE_COMPUT_16F
// to CUSPARSE_COMPUTE_32F
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 502 || defined(USE_ROCM)
    case at::ScalarType::Half:
      input_type = CUDA_R_16F;
      output_type = CUDA_R_16F;
      C_type = CUDA_R_16F;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
    case at::ScalarType::BFloat16:
      input_type = CUDA_R_16BF;
      output_type = CUDA_R_16BF;
      C_type = CUDA_R_16BF;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
    case at::ScalarType::Float:
#ifdef USE_ROCM
      TORCH_CHECK(native_gfx1013, "FP32 hipSPARSELt matmul requires the native gfx1013 scalar backend");
#endif
      input_type = CUDA_R_32F;
      output_type = CUDA_R_32F;
      C_type = CUDA_R_32F;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
// cuSPARSELt >= 0.6.2 or hipSparseLt: add Float8 support
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602 || defined(USE_ROCM)
    case at::ScalarType::Float8_e4m3fn:
      input_type = CUDA_R_8F_E4M3;
#ifdef USE_ROCM
      // hipSparseLt 0.2.7: FP8 input only supports FP32 output
      output_type = CUDA_R_32F;
      C_type = CUDA_R_32F;
#else
      output_type = CUDA_R_8F_E4M3;
      C_type = CUDA_R_16F;
#endif
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
#endif
// cuSPARSELt <= v0.5.2 uses CUSPARSE_COMPUTE_TF32, CUSPARSE_COMPUTE_16F
#else
    case at::ScalarType::Half:
      input_type = CUDA_R_16F;
      output_type = CUDA_R_16F;
      C_type = CUDA_R_16F;
      compute_type = CUSPARSE_COMPUTE_16F;
      break;
    case at::ScalarType::BFloat16:
      input_type = CUDA_R_16BF;
      output_type = CUDA_R_16BF;
      C_type = CUDA_R_16BF;
      compute_type = CUSPARSE_COMPUTE_16F;
      break;
    case at::ScalarType::Float:
      input_type = CUDA_R_32F;
      output_type = CUDA_R_32F;
      C_type = CUDA_R_32F;
      compute_type = CUSPARSE_COMPUTE_TF32;
      break;
#endif
    default:
      TORCH_CHECK(
          false,
          "Unsupported dtype for cuSPARSELt compressed matrix multiplication.");
      break;
  }
  ScalarType out_dtype = dense_B.scalar_type();
  // special check for mixed dtype support for 8 bit dtypes
  // cslt 0.5.2+: int8 int8 -> {fp16, bf16, int32} support
  if (out_dtype_opt.has_value() && !native_gfx1013) {
    out_dtype = out_dtype_opt.value();
    if (input_type == CUDA_R_8I) {
      switch (out_dtype) {
        case at::ScalarType::Half:
          C_type = CUDA_R_16F;
          output_type = CUDA_R_16F;
          break;
        case at::ScalarType::BFloat16:
          C_type = CUDA_R_16BF;
          output_type = CUDA_R_16BF;
          break;
        case at::ScalarType::Int:
          C_type = CUDA_R_32I;
          output_type = CUDA_R_32I;
          break;
        default:
          TORCH_CHECK(
              false,
              "Unsupported out_dtype passed, must be one of {fp16, bf16, int32} for int8 inputs");
          break;
      }
    }
// cslt 0.6.2+ or hipSparseLt: fp8 output dtype support
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602 || defined(USE_ROCM)
    else if (input_type == CUDA_R_8F_E4M3) {
      switch (out_dtype) {
#ifndef USE_ROCM
        case at::ScalarType::Float8_e4m3fn:
          output_type = CUDA_R_8F_E4M3;
          C_type = CUDA_R_16F;
          break;
        case at::ScalarType::Half:
          output_type = CUDA_R_16F;
          C_type = CUDA_R_16F;
          break;
        case at::ScalarType::BFloat16:
          output_type = CUDA_R_16BF;
          C_type = CUDA_R_16BF;
          break;
#endif
        case at::ScalarType::Float:
          output_type = CUDA_R_32F;
          C_type = CUDA_R_32F;
          break;
        default:
          TORCH_CHECK(
              false,
#ifdef USE_ROCM
              "Unsupported out_dtype passed, must be float32 for fp8 inputs on ROCm");
#else
              "Unsupported out_dtype passed, must be one of {fp8, fp16, bf16, float32} for fp8 inputs");
#endif
          break;
      }
    }
#endif
    else {
      TORCH_CHECK(
          false, "out_dtype support only available for int8/fp8 inputs");
    }
  }

  TORCH_INTERNAL_ASSERT(compressed_A.dim() == 2); // encoded M x S
  int64_t k = dense_B.size(0);
  int64_t n = dense_B.size(1);
  int64_t m = compressed_A.size(0);

  // initialize sparse descriptor
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      m,
      k,
      k,
      16,
      input_type,
      CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT));

  if (native_gfx1013) {
    size_t bytes, scratch;
    TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompressedSize2(
        &handle, &sparse_input_descriptor, &bytes, &scratch));
    const auto row_bytes = m * compressed_A.itemsize();
    TORCH_CHECK(compressed_A.size(1) == (bytes + row_bytes - 1) / row_bytes,
        "gfx1013 compressed shape does not match the dense contraction dimension");
  }

  // initialize dense input descriptor
  cusparseLtMatDescriptor_t dense_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &dense_input_descriptor,
      (dense_B.is_contiguous()) ? k : n,
      (dense_B.is_contiguous()) ? n : k,
      (dense_B.is_contiguous()) ? n : k,
      16,
      input_type,
      CUSPARSE_ORDER_ROW));

  // create result tensor
  auto res_tensor_options =
      c10::TensorOptions().dtype(out_dtype).device(dense_B.device());
  at::Tensor res = (transpose_result) ? at::empty({n, m}, res_tensor_options)
                                      : at::empty({m, n}, res_tensor_options);

  cusparseLtMatDescriptor_t res_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &res_descriptor,
      m,
      n,
      (transpose_result) ? m : n,
      16,
      output_type,
      (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

  // For float8, need fp16 C_descriptor, can't use FP8 for this matrix
  cusparseLtMatDescriptor_t C_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &C_descriptor,
      m,
      n,
      (transpose_result) ? m : n,
      16,
      C_type,
      (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

  // initialize matmul
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      (dense_B.is_contiguous()) ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                : CUSPARSE_OPERATION_TRANSPOSE,
      &sparse_input_descriptor,
      &dense_input_descriptor,
      &C_descriptor,
      &res_descriptor,
      compute_type));

  // set bias pointer for matmul, need to assign to get location
  if (bias_opt.has_value()) {
    auto& bias = bias_opt.value();
    void* dBias = bias.data_ptr();
#ifdef USE_ROCM
    if (native_gfx1013) {
      hipDataType bias_type = bias.scalar_type() == at::kFloat ? HIP_R_32F : input_type;
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
          &handle, &matmul, HIPSPARSELT_MATMUL_BIAS_TYPE, &bias_type, sizeof(bias_type)));
    }
#endif
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
        &handle,
        &matmul,
        CUSPARSELT_MATMUL_BIAS_POINTER,
        &dBias,
        sizeof(dBias)));
  }

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

  // set matmul search params
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
      &handle,
      &alg_sel,
      CUSPARSELT_MATMUL_ALG_CONFIG_ID,
      &alg_id,
      sizeof(alg_id)));

  cusparseLtSplitKMode_t splitKMode = static_cast<cusparseLtSplitKMode_t>(0);
  int max_alg_id = 0;
  if (split_k != 1) {
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_SPLIT_K,
        &split_k,
        sizeof(split_k)));

    if (split_k_mode > 0) {
      splitKMode = static_cast<cusparseLtSplitKMode_t>(split_k_mode);
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
          &handle,
          &alg_sel,
          CUSPARSELT_MATMUL_SPLIT_K_MODE,
          &splitKMode,
          sizeof(splitKMode)));
    }
  }

  // set tensor_alpha_mode and alpha pointer for matmul
  const auto alpha_tensor = alpha_opt.has_value() ? *alpha_opt : Tensor{};
  auto alpha_ptr = &alpha;
  if (alpha_opt.has_value()) {
    if (alpha_tensor.numel() == 1) {
      alpha = alpha_tensor.item<float>();
    } else {
      tensor_alpha_mode = 1;
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
          &handle,
          &matmul,
          CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,
          &tensor_alpha_mode,
          sizeof(tensor_alpha_mode)));
      alpha_ptr = static_cast<float*>(alpha_tensor.data_ptr());
    }
  }

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

  size_t workspace_size;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspacePtr = allocator.allocate(workspace_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (search_alg_id) {
    // run matmul search
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulSearch(
        &handle,
        &plan,
        alpha_ptr,
        compressed_A.data_ptr(),
        dense_B.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        workspacePtr.get(),
        // jank because of the way we want this to be an array of streams
        &stream,
        1));

    // get matmul params used
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
        &alg_id,
        sizeof(alg_id)));

#ifndef USE_ROCM
    // hipSPARSELt does not support querying SPLIT_K attributes
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_SPLIT_K,
        &split_k,
        sizeof(split_k)));

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_SPLIT_K_MODE,
        &splitKMode,
        sizeof(splitKMode)));
#endif

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
        &max_alg_id,
        sizeof(max_alg_id)));

  } else {
    // do normal matmul
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
        &handle,
        &plan,
        alpha_ptr,
        compressed_A.data_ptr(),
        dense_B.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        workspacePtr.get(),
        // jank because of the way we want this to be an array of streams
        &stream,
        1));
  }

  // destroy descriptors
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&sparse_input_descriptor));
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&dense_input_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&res_descriptor));
  // destroy plan
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(&plan));

  return {
      res,
      alg_id,
      split_k,
      static_cast<int64_t>(splitKMode),
      max_alg_id};
}

at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int64_t alg_id,
    int64_t split_k,
    int64_t split_k_mode) {
  auto result = _cslt_sparse_mm_impl(
      compressed_A,
      dense_B,
      bias_opt,
      alpha_opt,
      out_dtype_opt,
      transpose_result,
      (int)alg_id,
      (int)split_k,
      (int)split_k_mode,
      false);
  return std::get<0>(result);
}

int64_t _cslt_sparse_mm_search(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result) {
  TORCH_WARN_ONCE(
      "torch._cslt_sparse_mm_search is deprecated and will be removed in a future PyTorch release. Please use torch._C._cusparselt.mm_search instead.");
  int alg_id_int = 0;
  int split_k = 1;
  int split_k_mode = -1;
  auto result = _cslt_sparse_mm_impl(
      compressed_A,
      dense_B,
      bias_opt,
      alpha_opt,
      out_dtype_opt,
      transpose_result,
      alg_id_int,
      split_k,
      split_k_mode,
      true);
  return (int64_t)std::get<1>(result);
}

} // namespace at::native

#else // No cuSPARSELt support, throw error if these functions are called.

namespace at::native {

Tensor _cslt_to_dense(const Tensor&, int64_t) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}
Tensor _cslt_compress_backward(const Tensor&, const Tensor&, int64_t) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}
std::tuple<Tensor, Tensor, Tensor, Tensor> _cslt_sparse_mm_backward(
    const Tensor&, const Tensor&, const Tensor&, const std::optional<Tensor>&,
    const std::optional<Tensor>&, bool, std::array<bool, 4>) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

at::Tensor _cslt_compress(const Tensor& sparse_input) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype,
    bool transpose_result,
    int64_t alg_id,
    int64_t split_k,
    int64_t split_k_mode) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

int64_t _cslt_sparse_mm_search(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype,
    bool transpose_result) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

} // namespace at::native

#endif
