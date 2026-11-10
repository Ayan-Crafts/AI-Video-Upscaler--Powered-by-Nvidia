# Replace RIFE inference-mode fix

1. Extract this ZIP into the repository root `S:\Video-Upscaler`, replacing `app\engines\rife.py`.
2. Confirm the file has `output = output.clone()` immediately after the `with torch.inference_mode():` block.
3. Run `Select-String -Path .\app\engines\rife.py -Pattern 'clamp_|output = output.clone'`.
   The fixed `interpolate_tensor` method must not contain `clamp_` outside inference mode, and must contain `output = output.clone()` after the context.
4. Retry the benchmark.

This also ensures the returned CUDA tensor is a normal tensor rather than an inference tensor, so downstream stages can safely operate on it.
