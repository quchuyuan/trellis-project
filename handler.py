import os
import torch
import runpod
import base64
from PIL import Image
from io import BytesIO
import traceback
import tempfile

# Import trellis (Must be built in Docker)
from trellis.pipelines import Trellis2ImageTo3DPipeline
from trellis.utils import postprocessing_utils

# --- Global Model Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        print("Loading TRELLIS.2 pipeline...")
        # Note: Ensure HF_TOKEN is set in RunPod env vars
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
