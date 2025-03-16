try:
    import bpy
except ImportError:
    print("This script must be run from within Blender.")
    exit(1)

import sys
import os

def convert_obj_to_glb(obj_path: str, glb_path: str) -> None:
    """
    Convert an OBJ file to GLB using Blender.
    
    Args:
        obj_path (str): Path to the input OBJ file.
        glb_path (str): Path to save the output GLB file.
    """
    try:
        # Clear existing objects
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        # Import OBJ
        bpy.ops.wm.obj_import(filepath=obj_path)
        
        # Optional: Apply basic processing
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Export as GLB
        bpy.ops.export_scene.gltf(
            filepath=glb_path,
            export_format='GLB',  # Changed to GLB
            export_apply=True
        )
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        exit(1)

if __name__ == "__main__":
    # Get arguments after '--'
    try:
        args = sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        args = []
    
    if len(args) != 2:
        print("Usage: blender -b -P blender_script.py -- <input.obj> <output.glb>")
        exit(1)

    obj_file, glb_file = args
    
    if not os.path.exists(obj_file):
        print(f"Error: OBJ file {obj_file} not found.")
        exit(1)

    convert_obj_to_glb(obj_file, glb_file)
    print(f"Converted {obj_file} to {glb_file}")