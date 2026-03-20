import os
import base64
import torch
import runpod
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
import traceback

# Import TRELLIS components
# These will be available after the Docker build installs the TRELLIS.2 repo
try:
    from trellis.pipelines import Trellis2ImageTo3DPipeline
    from trellis.utils import postprocessing_utils
except ImportError:
    print("Warning: Trellis modules not found. Ensure they are installed in the Docker image.")

# --- Global Model Initialization ---
# We load the model outside the handler to take advantage of RunPod worker persistence.
# Note: Ensure you have granted access to gated models on Hugging Face (DINO v3, RMBG 2.0).
# You can set the HUGGING_FACE_HUB_TOKEN as an environment variable in RunPod.

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
    """
    Decodes image from base64 string or loads from a URL.
    """
    if image_input.startswith("http"):
        import requests
        response = requests.get(image_input)
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        # Handle potential data URI prefix
        if "," in image_input:
            image_input = image_input.split(",")[1]
        image_data = base64.b64decode(image_input)
        return Image.open(BytesIO(image_data)).convert("RGB")

def handler(job):
    """
    The main handler function for RunPod Serverless.
    """
    try:
        job_input = job['input']
        
        # 1. Input Validation
        image_input = job_input.get("image")
        if not image_input:
            return {"error": "Missing 'image' in input"}
            
        seed = job_input.get("seed", 42)
        ss_sampling_steps = job_input.get("ss_sampling_steps", 25)
        sl_sampling_steps = job_input.get("sl_sampling_steps", 10)
        ss_guidance_strength = job_input.get("ss_guidance_strength", 7.5)
        
        # 2. Process Image
        image = decode_image(image_input)
        
        # 3. Inference
        pipe = get_pipeline()
        torch.manual_seed(seed)
        
        print(f"Starting inference for job {job['id']}...")
        # Trellis.2 pipeline call (based on example.py)
        # Note: res refers to the internal resolution (e.g., 512, 1024)
        outputs = pipe(
            image,
            num_samples=1,
            seed=seed,
            # Adjust parameters based on the specific version of TRELLIS.2
            # These are typical for the flow-matching transformer
        )
        
        # 4. Post-processing to GLB
        # The output usually contains a list of objects (since num_samples can be > 1)
        output_asset = outputs[0]
        
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Extract mesh/textured asset
        # TRELLIS.2 typically returns a structured object that can be converted to mesh
        # We use the built-in utils to save as GLB
        glb = postprocessing_utils.to_glb(output_asset)
        glb.export(temp_path)
        
        # 5. Return Result
        with open(temp_path, "rb") as f:
            glb_data = f.read()
            glb_base64 = base64.b64encode(glb_data).decode("utf-8")
            
        # Cleanup
        os.remove(temp_path)
        
        return {
            "status": "success",
            "model_format": "glb",
            "model_data": glb_base64
        }

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
