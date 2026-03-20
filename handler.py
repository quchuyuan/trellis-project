import os
import subprocess
import sys
import torch
import runpod
import base64
from PIL import Image
from io import BytesIO
import traceback
import tempfile

# --- Just-In-Time Extension Installation ---
# This ensures compilation happens on the real GPU (RunPod) instead of GitHub Actions
def setup_extensions():
    if os.path.exists("/app/.extensions_ready"):
        return
    
    print(">>> First-time setup: Building CUDA extensions on RunPod GPU...")
    env = os.environ.copy()
    env["MAX_JOBS"] = "4" # RunPod has plenty of CPU/Memory
    
    commands = [
        ["pip", "install", "git+https://github.com/NVlabs/nvdiffrast.git"],
        ["pip", "install", "flash-attn", "--no-build-isolation"],
        ["pip", "install", "./extensions/flexgemm"],
        ["pip", "install", "./extensions/cumesh"],
        ["pip", "install", "./extensions/o-voxel"],
        ["pip", "install", "-e", "."]
    ]
    
    for cmd in commands:
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
    
    with open("/app/.extensions_ready", "w") as f:
        f.write("ready")
    print(">>> All extensions built and installed successfully!")

# Ensure extensions are built before trying to import trellis
setup_extensions()

# Now it is safe to import
from trellis.pipelines import Trellis2ImageTo3DPipeline
from trellis.utils import postprocessing_utils

# --- Global Model Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        print("Loading TRELLIS.2 pipeline...")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2")
        pipeline.to(device)
    return pipeline

def decode_image(image_input):
    if image_input.startswith("http"):
        import requests
        response = requests.get(image_input)
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        if "," in image_input:
            image_input = image_input.split(",")[1]
        image_data = base64.b64decode(image_input)
        return Image.open(BytesIO(image_data)).convert("RGB")

def handler(job):
    try:
        job_input = job['input']
        image_input = job_input.get("image")
        if not image_input:
            return {"error": "Missing 'image' in input"}
            
        seed = job_input.get("seed", 42)
        image = decode_image(image_input)
        
        pipe = get_pipeline()
        torch.manual_seed(seed)
        
        print(f"Starting inference for job {job['id']}...")
        outputs = pipe(image, num_samples=1, seed=seed)
        output_asset = outputs[0]
        
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_file:
            temp_path = temp_file.name
        
        glb = postprocessing_utils.to_glb(output_asset)
        glb.export(temp_path)
        
        with open(temp_path, "rb") as f:
            glb_data = f.read()
            glb_base64 = base64.b64encode(glb_data).decode("utf-8")
            
        os.remove(temp_path)
        
        return {
            "status": "success",
            "model_format": "glb",
            "model_data": glb_base64
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
