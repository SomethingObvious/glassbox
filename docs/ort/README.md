# The Page's Runtime Build

The two `ort-wasm-simd-threaded.asyncify` files are [onnxruntime](https://github.com/microsoft/onnxruntime) v1.31.1 (commit 6af1a93) with `rows.patch` applied, and nothing else is changed. The page loads them in place of the ones the npm package fetches.

Stock 1.31.1 only runs its fused 4-bit kernels (the MLP's gate and up, and attention's q, k and v) when a pass has one row. With two rows it falls back to separate programs that read the weights again for each row. The patch keeps those kernels fused for up to 4 rows and reads each weight once for all of them, which took Figure 1's two-row pass from 16.2 ms to 14.2 ms on an RTX 4070 laptop GPU. A pass with one row runs the same as before.

Its attention also picks a different shader for a pass of two or three rows than for one, and that shader adds the same products up in another order. The rows came out a little different from the same rows read one at a time (by up to 0.16 in a logit), which was enough for the run with guesses to pick another token where two were nearly tied. The patch uses the one-row shader up to three rows, so those rows are now the same bits, and the three-row pass went from 14.9 ms to 14.6 ms.

To build it, check out v1.31.1 with its submodules, apply the patch and run what onnxruntime's web CI runs.

```
git apply rows.patch
python tools/ci_build/build.py --config Release --cmake_generator Ninja --build_wasm --enable_wasm_simd \
  --enable_wasm_threads --disable_rtti --use_webgpu --use_webnn --target onnxruntime_webassembly \
  --enable_wasm_api_exception_catching --disable_ml_ops --disable_generation_ops \
  --disable_types string float4 float8 optional sparsetensor \
  --include_ops_by_config onnxruntime/wasm/reduced_types.config --enable_reduced_operator_type_support \
  --skip_tests
```

onnxruntime is under the MIT licence in `LICENSE`, and `ThirdPartyNotices.txt` covers what it bundles.
