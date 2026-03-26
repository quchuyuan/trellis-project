import json
import base64
import os
import tempfile
import traceback
from io import BytesIO

import requests
import runpod
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

import o_voxel
from runpod.serverless.utils import rp_upload
from trellis2 import models as trellis_models
from trellis2.modules import image_feature_extractor
from trellis2.pipelines import rembg as trellis_rembg
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.base import Pipeline as TrellisPipelineBase
from transformers import AutoModelForImageSegmentation

if os.path.isdir("/runpod-volume"):
    os.environ.setdefault("HF_HOME", "/runpod-volume/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/runpod-volume/hf-cache/hub")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/runpod-volume/hf-cache/hub")
    os.environ.setdefault("HF_ASSETS_CACHE", "/runpod-volume/hf-cache/assets")

# Prefer regular Hub downloads over Xet reconstruction on constrained worker disks.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = os.getenv("TRELLIS_MODEL_ID", "microsoft/TRELLIS.2-4B")
DINOV3_REPO_PATH = os.getenv("DINOV3_REPO_PATH", "/opt/repos/dinov3")
DINOV3_WEIGHTS_PATH = os.getenv(
    "DINOV3_WEIGHTS_PATH",
    "/opt/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
)
OUTPUT_BUCKET_NAME = os.getenv("OUTPUT_BUCKET_NAME") or os.getenv("BUCKET_NAME")
OUTPUT_BUCKET_PREFIX = os.getenv("OUTPUT_BUCKET_PREFIX", "trellis2")
PIPELINE_TYPES = {
    "512": "512",
    "1024": "1024_cascade",
    "1536": "1536_cascade",
}
DEFAULT_RESOLUTION = "1024"

pipeline = None
loader_patched = False
dinov3_patched = False
rembg_patched = False
last_mesh_stats = None
last_stage = None


def normalize_hf_token():
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def get_hf_token():
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )


def parse_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def looks_like_hf_model_ref(value):
    parts = value.split("/")
    return len(parts) >= 3 and not value.startswith("ckpts/")


def get_local_dinov3_hub_name(model_name):
    lowered = model_name.lower()
    if "dinov3-vitl16" in lowered:
        return "dinov3_vitl16"
    if "dinov3-vitb16" in lowered:
        return "dinov3_vitb16"
    if "dinov3-vits16plus" in lowered:
        return "dinov3_vits16plus"
    if "dinov3-vits16" in lowered:
        return "dinov3_vits16"
    if "dinov3-vith16plus" in lowered:
        return "dinov3_vith16plus"
    if "dinov3-vit7b16" in lowered:
        return "dinov3_vit7b16"
    return None


def patch_dinov3_extractor():
    global dinov3_patched
    if dinov3_patched:
        return

    class LocalAwareDinoV3FeatureExtractor:
        def __init__(self, model_name: str, image_size=512):
            self.model_name = model_name
            self.image_size = image_size
            self.transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

            hub_name = get_local_dinov3_hub_name(model_name)
            use_local = (
                hub_name is not None
                and os.path.exists(DINOV3_WEIGHTS_PATH)
                and os.path.isdir(DINOV3_REPO_PATH)
            )
            if use_local:
                print(
                    "Loading DINOv3 from local weights "
                    f"'{DINOV3_WEIGHTS_PATH}' via repo '{DINOV3_REPO_PATH}'"
                )
                self.model = torch.hub.load(
                    DINOV3_REPO_PATH,
                    hub_name,
                    source="local",
                    weights=DINOV3_WEIGHTS_PATH,
                )
            else:
                if hub_name is not None:
                    print(
                        "Local DINOv3 weights not available, falling back to Hugging Face: "
                        f"{model_name}"
                    )
                self.model = image_feature_extractor.DINOv3ViTModel.from_pretrained(model_name)
            self.model.eval()

        def to(self, device):
            self.model.to(device)

        def cuda(self):
            self.model.cuda()

        def cpu(self):
            self.model.cpu()

        @torch.no_grad()
        def __call__(self, image):
            if isinstance(image, torch.Tensor):
                assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            elif isinstance(image, list):
                assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
                image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
                image = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image]
                image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
                image = torch.stack(image).cuda()
            else:
                raise ValueError(f"Unsupported type of image: {type(image)}")

            image = self.transform(image).cuda()
            if hasattr(self.model, "embeddings") and hasattr(self.model, "rope_embeddings"):
                image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
                hidden_states = self.model.embeddings(image, bool_masked_pos=None)
                position_embeddings = self.model.rope_embeddings(image)
                for layer_module in self.model.layer:
                    hidden_states = layer_module(
                        hidden_states,
                        position_embeddings=position_embeddings,
                    )
                return F.layer_norm(hidden_states, hidden_states.shape[-1:])

            features = self.model(image, is_training=True)["x_prenorm"]
            return F.layer_norm(features, features.shape[-1:])

    image_feature_extractor.DinoV3FeatureExtractor = LocalAwareDinoV3FeatureExtractor
    dinov3_patched = True


def patch_rembg_loader():
    global rembg_patched
    if rembg_patched:
        return

    class SafeBiRefNet:
        def __init__(self, model_name: str = "briaai/RMBG-2.0"):
            self.model_name = model_name
            self.model = None
            self.device = "cpu"
            self.transform_image = transforms.Compose(
                [
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        def _ensure_loaded(self):
            if self.model is not None:
                return

            print(
                "Loading RMBG model with low_cpu_mem_usage disabled to avoid "
                "PyTorch meta-tensor initialization bugs"
            )
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                device_map=None,
            )
            self.model.eval()
            self.model.to(self.device)

        def to(self, device: str):
            self.device = str(device)
            if self.model is not None:
                self.model.to(device)

        def cuda(self):
            self.to("cuda")

        def cpu(self):
            self.to("cpu")

        def __call__(self, image: Image.Image) -> Image.Image:
            self._ensure_loaded()
            image_size = image.size
            input_images = self.transform_image(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.model(input_images)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image_size)
            image.putalpha(mask)
            return image

    trellis_rembg.BiRefNet = SafeBiRefNet
    rembg_patched = True


def patch_trellis_loader():
    global loader_patched
    if loader_patched:
        return

    def safe_from_pretrained(cls, path: str, config_file: str = "pipeline.json"):
        is_local = os.path.exists(f"{path}/{config_file}")
        if is_local:
            resolved_config = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download

            resolved_config = hf_hub_download(path, config_file, token=get_hf_token())

        with open(resolved_config, "r", encoding="utf-8") as file_obj:
            args = json.load(file_obj)["args"]

        loaded_models = {}
        for model_key, model_value in args["models"].items():
            if hasattr(cls, "model_names_to_load") and model_key not in cls.model_names_to_load:
                continue

            candidates = []
            if is_local:
                if not looks_like_hf_model_ref(model_value):
                    candidates.append(os.path.join(path, model_value))
                else:
                    candidates.append(model_value)
            else:
                if looks_like_hf_model_ref(model_value):
                    candidates.append(model_value)
                else:
                    candidates.append(f"{path}/{model_value}")

            last_error = None
            for candidate in candidates:
                try:
                    loaded_models[model_key] = trellis_models.from_pretrained(candidate)
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"Failed to load model '{model_key}' from '{candidate}': {exc}")
            else:
                raise last_error

        new_pipeline = cls(loaded_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    TrellisPipelineBase.from_pretrained = classmethod(safe_from_pretrained)
    loader_patched = True


def get_pipeline():
    global pipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. TRELLIS.2 requires an NVIDIA GPU. "
            "On Windows local testing, start the container with `--gpus all`."
        )

    if pipeline is None:
        normalize_hf_token()
        patch_dinov3_extractor()
        patch_rembg_loader()
        patch_trellis_loader()
        print(f"Loading TRELLIS.2 pipeline: {MODEL_ID}")
        model_path = snapshot_download(MODEL_ID, token=get_hf_token())
        print(f"Using local TRELLIS.2 snapshot: {model_path}")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        pipeline.cuda()

    return pipeline


def decode_image(image_input):
    if not image_input:
        raise ValueError("Missing image input")

    if image_input.startswith("http://") or image_input.startswith("https://"):
        response = requests.get(image_input, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        if "," in image_input:
            image_input = image_input.split(",", 1)[1]

        image_data = base64.b64decode(image_input)
        image = Image.open(BytesIO(image_data))

    image.load()
    if image.mode == "P":
        # Palette PNGs can hide transparency in metadata; preserve it when present.
        transparency = image.info.get("transparency")
        return image.convert("RGBA" if transparency is not None else "RGB")
    if image.mode in {"RGBA", "RGB"}:
        return image
    if "A" in image.getbands():
        return image.convert("RGBA")
    return image.convert("RGB")


def build_sampler_params(job_input, prefix, defaults):
    params = {}

    if f"{prefix}_sampling_steps" in job_input:
        params["steps"] = int(job_input[f"{prefix}_sampling_steps"])
    if f"{prefix}_guidance_strength" in job_input:
        params["guidance_strength"] = float(job_input[f"{prefix}_guidance_strength"])
    if f"{prefix}_guidance_rescale" in job_input:
        params["guidance_rescale"] = float(job_input[f"{prefix}_guidance_rescale"])
    if f"{prefix}_rescale_t" in job_input:
        params["rescale_t"] = float(job_input[f"{prefix}_rescale_t"])

    return params


def has_bucket_upload_config():
    return all(
        [
            os.getenv("BUCKET_ENDPOINT_URL"),
            os.getenv("BUCKET_ACCESS_KEY_ID"),
            os.getenv("BUCKET_SECRET_ACCESS_KEY"),
        ]
    )


def upload_glb(job_id, file_path, bucket_name=None, bucket_prefix=None):
    if not has_bucket_upload_config():
        raise RuntimeError(
            "Result upload is not configured. Set `BUCKET_ENDPOINT_URL`, "
            "`BUCKET_ACCESS_KEY_ID`, and `BUCKET_SECRET_ACCESS_KEY` on the RunPod endpoint. "
            "Optionally set `OUTPUT_BUCKET_NAME` and `OUTPUT_BUCKET_PREFIX`."
        )

    resolved_bucket_name = bucket_name or OUTPUT_BUCKET_NAME
    if not resolved_bucket_name:
        raise RuntimeError(
            "Missing bucket name for result upload. Set `OUTPUT_BUCKET_NAME` or pass "
            "`bucket_name` in the request."
        )

    file_name = f"trellis2-{job_id or 'job'}-{os.path.basename(file_path)}"
    return rp_upload.upload_file_to_bucket(
        file_name=file_name,
        file_location=file_path,
        bucket_name=resolved_bucket_name,
        prefix=bucket_prefix or OUTPUT_BUCKET_PREFIX,
        extra_args={"ContentType": "model/gltf-binary"},
    )


def describe_value(value):
    info = {
        "type": type(value).__name__,
        "shape": None,
        "numel": None,
    }

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            info["shape"] = tuple(int(dim) for dim in shape)
        except Exception:
            info["shape"] = str(shape)

    if hasattr(value, "numel"):
        try:
            info["numel"] = int(value.numel())
            return info
        except Exception:
            pass

    try:
        info["numel"] = int(np.asarray(value).size)
    except Exception:
        pass

    return info


def describe_mesh(mesh):
    return {
        "mesh_type": type(mesh).__name__,
        "vertices": describe_value(getattr(mesh, "vertices", None)),
        "faces": describe_value(getattr(mesh, "faces", None)),
        "attrs": describe_value(getattr(mesh, "attrs", None)),
        "coords": describe_value(getattr(mesh, "coords", None)),
    }


def validate_mesh(mesh):
    global last_mesh_stats
    stats = describe_mesh(mesh)
    last_mesh_stats = stats
    empty_fields = [
        field_name
        for field_name in ("vertices", "faces", "attrs", "coords")
        if stats[field_name]["numel"] in (None, 0)
    ]

    if empty_fields:
        raise RuntimeError(
            "TRELLIS produced an empty mesh output. "
            f"Empty fields: {', '.join(empty_fields)}. "
            f"Mesh stats: {json.dumps(stats, default=str)}. "
            "This usually means the input image did not produce a valid 3D reconstruction. "
            "Try a centered subject, cleaner silhouette, or a transparent-background PNG."
        )

    return stats


def mesh_to_glb(mesh, pipe, decimation_target, texture_size):
    mesh_stats = validate_mesh(mesh)
    print(f"Preparing GLB export with mesh stats: {json.dumps(mesh_stats, default=str)}")

    if hasattr(mesh, "simplify") and mesh_stats["faces"]["numel"] not in (None, 0):
        mesh.simplify(16777216)

    glb_kwargs = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": decimation_target,
        "texture_size": texture_size,
        "remesh": True,
        "remesh_band": 1,
        "remesh_project": 0,
        "verbose": True,
    }

    if hasattr(mesh, "layout"):
        glb_kwargs["attr_layout"] = mesh.layout
    else:
        glb_kwargs["attr_layout"] = pipe.pbr_attr_layout

    if hasattr(mesh, "voxel_size"):
        glb_kwargs["voxel_size"] = mesh.voxel_size
    elif hasattr(mesh, "grid_size"):
        glb_kwargs["grid_size"] = mesh.grid_size

    return o_voxel.postprocess.to_glb(**glb_kwargs)


def handler(job):
    try:
        global last_mesh_stats
        global last_stage
        last_mesh_stats = None
        last_stage = "starting"
        job_input = job.get("input", {})
        image_input = job_input.get("image") or job_input.get("input_image")
        if not image_input:
            return {"status": "error", "message": "Missing `image` or `input_image` in input"}

        seed = int(job_input.get("seed", 42))
        resolution = str(job_input.get("resolution", DEFAULT_RESOLUTION))
        pipeline_type = PIPELINE_TYPES.get(resolution, PIPELINE_TYPES[DEFAULT_RESOLUTION])
        remove_bg = parse_bool(job_input.get("remove_bg", True), default=True)
        return_base64 = parse_bool(job_input.get("return_base64", False), default=False)
        decimation_target = int(job_input.get("decimation_target", 1000000))
        texture_size = int(job_input.get("texture_size", 4096))
        bucket_name = job_input.get("bucket_name")
        bucket_prefix = job_input.get("bucket_prefix")

        image = decode_image(image_input)
        last_stage = f"image_decoded:{image.mode}:{image.size}"
        pipe = get_pipeline()
        last_stage = "pipeline_loaded"

        if remove_bg:
            image = pipe.preprocess_image(image)
            last_stage = "image_preprocessed"

        sparse_structure_sampler_params = build_sampler_params(
            job_input,
            "ss",
            {},
        )
        shape_slat_sampler_params = build_sampler_params(
            job_input,
            "shape_slat",
            {},
        )
        tex_slat_sampler_params = build_sampler_params(
            job_input,
            "tex_slat",
            {},
        )

        torch.manual_seed(seed)
        print(
            f"Starting inference for job {job.get('id', 'unknown')} on "
            f"{torch.cuda.get_device_name(0)} with pipeline_type={pipeline_type}"
        )

        last_stage = "running_inference"
        outputs = pipe.run(
            image,
            seed=seed,
            preprocess_image=False,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            shape_slat_sampler_params=shape_slat_sampler_params,
            tex_slat_sampler_params=tex_slat_sampler_params,
        )
        last_stage = "inference_completed"
        if not outputs:
            raise RuntimeError(
                "TRELLIS pipeline returned no outputs. Try a centered subject image and "
                "use resolution 512 for initial validation."
            )

        print(
            f"Pipeline output type: {type(outputs).__name__}, "
            f"length={len(outputs) if hasattr(outputs, '__len__') else 'unknown'}"
        )
        mesh = outputs[0]
        last_stage = "exporting_glb"
        glb = mesh_to_glb(mesh, pipe, decimation_target, texture_size)

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            glb.export(temp_path, extension_webp=True)
            if return_base64:
                with open(temp_path, "rb") as file_obj:
                    glb_base64 = base64.b64encode(file_obj.read()).decode("utf-8")
            else:
                model_url = upload_glb(job.get("id"), temp_path, bucket_name, bucket_prefix)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        torch.cuda.empty_cache()

        response = {
            "status": "success",
            "model_format": "glb",
            "resolution": resolution,
            "seed": seed,
            "model_id": MODEL_ID,
        }
        if return_base64:
            response["model_data"] = glb_base64
        else:
            response["model_url"] = model_url
        return response
    except Exception as exc:
        print(f"Error: {exc}")
        print(traceback.format_exc())
        if "No space left on device" in str(exc):
            return {
                "status": "error",
                "message": (
                    "Worker disk is full while downloading or reconstructing model files. "
                    "Increase RunPod worker/container disk or mount a Network Volume at "
                    "/runpod-volume so Hugging Face cache can live there."
                ),
            }
        if "Expected reduction dim to be specified for input.numel() == 0" in str(exc):
            return {
                "status": "error",
                "message": (
                    "TRELLIS hit an empty-tensor reduction during inference or export. "
                    "Check `stage` and `mesh_stats` in this response. If your image has a "
                    "transparent background, keep `remove_bg=false` so the alpha channel is preserved."
                ),
                "mesh_stats": last_mesh_stats,
                "stage": last_stage,
            }
        if str(exc).startswith("TRELLIS produced an empty mesh output."):
            return {
                "status": "error",
                "message": str(exc),
                "mesh_stats": last_mesh_stats,
                "stage": last_stage,
            }
        return {
            "status": "error",
            "message": str(exc),
            "mesh_stats": last_mesh_stats,
            "stage": last_stage,
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
