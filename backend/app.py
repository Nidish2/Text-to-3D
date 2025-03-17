import os
import uuid
import subprocess
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .model import generate_point_cloud
from .mesh_generation import point_cloud_to_mesh, save_mesh_as_obj
import logging
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/generate")
async def generate_model(input: TextInput):
    """
    Generate a 3D model from text and return it as a GLB file.
    
    Args:
        input (TextInput): JSON body with 'text' field.
    
    Returns:
        FileResponse: Generated GLB file.
    """
    text = input.text
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        output_dir = os.path.join(project_root, "generated_meshes")
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.join(output_dir, str(uuid.uuid4()))
        obj_file = f"{base_name}.obj"
        glb_file = f"{base_name}.glb"
        
        logger.info("Generating point cloud for prompt: '%s'", text)
        point_cloud = generate_point_cloud(text)  # Use base1B model
        logger.info("Point cloud generation completed")
        
        logger.info("Starting mesh generation...")
        # Extract colors from point cloud channels
        colors = None
        if 'R' in point_cloud.channels and 'G' in point_cloud.channels and 'B' in point_cloud.channels:
            colors = np.stack(
                [point_cloud.channels['R'], point_cloud.channels['G'], point_cloud.channels['B']],
                axis=-1
            )
        mesh = point_cloud_to_mesh(point_cloud.coords, colors)  # Pass colors to mesh
        logger.info("Mesh generation completed")
        
        logger.info("Saving mesh as OBJ to %s", obj_file)
        save_mesh_as_obj(mesh, obj_file)
        logger.info("OBJ saved successfully")
        
        if not os.path.exists(obj_file):
            raise RuntimeError(f"OBJ file not found at {obj_file} after saving")
        
        logger.info("Converting to GLB...")
        blender_script = os.path.join(current_dir, "blender_script.py")
        blender_cmd = ["blender", "-b", "-P", blender_script, "--", obj_file, glb_file]
        subprocess.run(blender_cmd, check=True, cwd=current_dir)
        logger.info("GLB conversion completed")
        
        if not os.path.exists(glb_file):
            raise RuntimeError(f"GLB file not found at {glb_file}")
        
        return FileResponse(
            glb_file,
            media_type="model/gltf-binary",
            filename=os.path.basename(glb_file)
        )
    
    except Exception as e:
        logger.error(f"Error during model generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model generation failed: {str(e)}")
    
    finally:
        if 'obj_file' in locals() and os.path.exists(obj_file):
            os.remove(obj_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)