import shutil
import os
import sys
import math
import base64
from pathlib import Path
from io import BytesIO

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
from PIL import Image

# Import model architecture
from deep_stego import PrepNet, HidingNet, RevealNet, Config

app = FastAPI()

# --- Configuration ---
CHECKPOINT_PATH = "./checkpoints/checkpoint_ep50.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
print(f"Loading models from {CHECKPOINT_PATH} on {DEVICE}...")

prep_net = PrepNet().to(DEVICE)
hiding_net = HidingNet().to(DEVICE)
reveal_net = RevealNet().to(DEVICE)

try:
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        prep_net.load_state_dict(checkpoint['prep_net'])
        hiding_net.load_state_dict(checkpoint['hiding_net'])
        reveal_net.load_state_dict(checkpoint['reveal_net'])
        prep_net.eval()
        hiding_net.eval()
        reveal_net.eval()
        print("Model loaded successfully!")
    else:
        print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}. Model uses random weights.")
except Exception as e:
    print(f"Error loading model: {e}")

# --- Utilities ---
# We use 256x256 for the Web Interface to ensure "HD-like" quality 
# even if the model was trained on smaller patches (Fully Convolutional transfer).
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def tensor_to_base64(tensor):
    """Converts a GPU tensor to a Base64 encoded PNG string."""
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu().clamp(0, 1))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return base64.b64encode(img_byte_arr.read()).decode('utf-8')

def calculate_psnr(img1, img2):
    """Calculates PSNR (Peak Signal-to-Noise Ratio) between two tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def calculate_mse(img1, img2):
    """Calculates Mean Squared Error."""
    return torch.mean((img1 - img2) ** 2).item()

# --- API Endpoints ---

@app.post("/api/hide")
async def hide_image(cover: UploadFile = File(...), secret: UploadFile = File(...)):
    try:
        # Load Images
        cover_img = Image.open(BytesIO(await cover.read())).convert('RGB')
        secret_img = Image.open(BytesIO(await secret.read())).convert('RGB')
        
        # Preprocess
        cover_tensor = transform(cover_img).unsqueeze(0).to(DEVICE)
        secret_tensor = transform(secret_img).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            prep_secret = prep_net(secret_tensor)
            stego = hiding_net(cover_tensor, prep_secret)
            
            # For metrics: Reveal the secret immediately from our generated stego
            revealed_check = reveal_net(stego)

        # Calculate Metrics
        # 1. Steganographic Quality (How invisible is the secret?)
        # Compare Cover vs Stego
        psnr_cover = calculate_psnr(cover_tensor, stego)
        mse_cover = calculate_mse(cover_tensor, stego)
        
        # 2. Recovery Fidelity (How well can we read the secret?)
        # Compare Original Secret vs Revealed Secret
        psnr_secret = calculate_psnr(secret_tensor, revealed_check)
        mse_secret = calculate_mse(secret_tensor, revealed_check)

        # Return Data
        return JSONResponse({
            "image": f"data:image/png;base64,{tensor_to_base64(stego)}",
            "metrics": {
                "cover_psnr": round(psnr_cover, 2),
                "cover_mse": round(mse_cover, 5),
                "secret_psnr": round(psnr_secret, 2),
                "secret_mse": round(mse_secret, 5)
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reveal")
async def reveal_image(stego: UploadFile = File(...)):
    try:
        # Load Image
        stego_img = Image.open(BytesIO(await stego.read())).convert('RGB')
        stego_tensor = transform(stego_img).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            revealed = reveal_net(stego_tensor)
        
        # Return Image only (base64)
        return JSONResponse({
            "image": f"data:image/png;base64,{tensor_to_base64(revealed)}",
            "metrics": None # Can't calculate without ground truth
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Setup Static Files (Frontend)
app.mount("/", StaticFiles(directory="public", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
