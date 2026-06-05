import os
import sys
import uvicorn
from dotenv import load_dotenv

# Add workspace directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env configurations
load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    
    print(f"[LAUNCH] Booting HyperSepsisTwin-Ω server at http://{host}:{port}")
    uvicorn.run("backend.api.app:app", host=host, port=port, reload=True)
