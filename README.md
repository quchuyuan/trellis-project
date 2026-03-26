# TRELLIS.2 RunPod Serverless

This repo packages the official [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) codebase into a RunPod Serverless worker.

## What Changed

- The Docker image now clones the official TRELLIS.2 repo and follows the same extension layout used by Microsoft's `setup.sh`.
- CUDA extension builds explicitly include RTX 4080 Super (`sm_89`) while still supporting common RunPod GPUs (`8.0`, `8.6`, `9.0`).
- The RunPod handler now uses the official `trellis2` pipeline API and accepts both `image` and `input_image`.
- DINOv3 local-weight support is optional at runtime; if local weights are not baked into the image, the handler falls back to Hugging Face.
- `.dockerignore` excludes the local virtualenv and local extension copies so Windows builds send a much smaller Docker context.

## Prerequisites

1. Accept the gated Hugging Face model terms:
   - [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
   - [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
2. Install Docker Desktop on Windows and enable the WSL2 backend.
3. Make sure the NVIDIA Container Toolkit integration is working if you want to test locally with GPU.

## Local Build on Windows

Build does not need to use your host GPU directly, but the image is compiled for `sm_89`, which matches an RTX 4080 Super.

```powershell
docker build -t trellis2-runpod:local .
```

If your Hugging Face access to `facebook/dinov3-vitl16-pretrain-lvd1689m` is still pending, use the published Docker Hub image tags instead of rebuilding from source.

## Local GPU Smoke Test on Windows

The container must be started with `--gpus all`, otherwise TRELLIS.2 will refuse to start.

```powershell
docker run --rm --gpus all trellis2-runpod:local python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output should include `True` and your GPU name, such as `NVIDIA GeForce RTX 4080 SUPER`.

## Local Memory Note

Full TRELLIS.2 model loading can be OOM-killed by Docker Desktop if the Linux VM memory is too low. On this machine the image built successfully and the GPU was detected correctly, but model loading was killed with `OOMKilled=true` when Docker Desktop had about `15.5 GiB` available.

If you want to run full inference locally on Windows, increase Docker Desktop memory to at least `24 GiB`, preferably `32 GiB`, then retry.

For RunPod deployment this local RAM limit does not apply the same way, but you should still choose a worker with at least `24 GB` VRAM and enough system RAM.

## RunPod Environment Variables

Set one of these in RunPod:

- `HUGGING_FACE_HUB_TOKEN`
- `HF_TOKEN`

Required for `model_url` uploads:

- `BUCKET_ENDPOINT_URL`
- `BUCKET_ACCESS_KEY_ID`
- `BUCKET_SECRET_ACCESS_KEY`
- `OUTPUT_BUCKET_NAME`

Optional:

- `TRELLIS_MODEL_ID=microsoft/TRELLIS.2-4B`
- `OUTPUT_BUCKET_PREFIX=trellis2`

For Supabase Storage S3 compatibility, the endpoint usually looks like:

```text
https://<project-ref>.storage.supabase.co/storage/v1/s3
```

## RunPod Endpoint

Replace `<YOUR_ENDPOINT_ID>` with the endpoint ID shown in the RunPod console:

```text
https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/runsync
```

For async jobs, use:

```text
https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/run
```

RunPod endpoint URL format reference:
- [Runpod endpoint overview](https://docs.runpod.io/serverless/endpoints/overview)
- [Runpod operation reference](https://docs.runpod.io/serverless/endpoints/operation-reference)

## Example Request

```json
{
  "input": {
    "input_image": "https://raw.githubusercontent.com/microsoft/TRELLIS.2/main/assets/example_image/T.png",
    "remove_bg": false
  }
}
```

## Example cURL

```bash
curl -X POST "https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "input_image": "https://raw.githubusercontent.com/microsoft/TRELLIS.2/main/assets/example_image/T.png",
      "remove_bg": false
    }
  }'
```

## Example Response

```json
{
  "status": "success",
  "model_format": "glb",
  "model_url": "https://<your-storage-url>/trellis2-<job-id>.glb",
  "resolution": "1024",
  "seed": 42,
  "model_id": "microsoft/TRELLIS.2-4B"
}
```

Notes:

- The worker now returns `model_url` by default instead of a large base64 payload.
- If you explicitly need inline data for local testing, send `return_base64: true`.
