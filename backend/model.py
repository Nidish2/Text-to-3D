import torch
import numpy as np
import sys
import os
import logging
import intel_extension_for_pytorch as ipex

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#os.environ["ONEAPI_DEVICE_SELECTOR"] = "*:1"


# Import Point-E modules
from point_e.diffusion.configs  import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud

class PointEModel:
    """Class to handle Point-E model for text-to-3D point cloud generation."""
    
    def __init__(self, use_base1b=False, use_upsampler=True):
        """
        Initialize the Point-E model.
        
        Args:
            use_base1b (bool): Whether to use the higher quality base1B model.
            use_upsampler (bool): Whether to use the upsampler model.
        """
        self.device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Model configuration
        self.use_base1b = use_base1b
        self.use_upsampler = use_upsampler
        self.base_name = "base1B" if use_base1b else "base40M-textvec"
        
        # Load models
        logger.info("Loading Point-E models...")
        self._load_models()

    def _load_models(self):
        """Load the text-to-point-cloud models."""
        try:
            # Load base model
            logger.info(f"Creating base model: {self.base_name}...")
            self.base_model = model_from_config(MODEL_CONFIGS[self.base_name], self.device)
            self.base_model.eval()
            self.base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[self.base_name])
            
            logger.info(f"Downloading base checkpoint: {self.base_name}...")
            self.base_model.load_state_dict(load_checkpoint(self.base_name, self.device))
            
            # Load upsampler model if requested
            if self.use_upsampler:
                logger.info("Creating upsampler model...")
                self.upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], self.device)
                self.upsampler_model.eval()
                self.upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
                
                logger.info("Downloading upsampler checkpoint...")
                self.upsampler_model.load_state_dict(load_checkpoint('upsample', self.device))
                
                # Create sampler with both models
                self.sampler = PointCloudSampler(
                    device=self.device,
                    models=[self.base_model, self.upsampler_model],
                    diffusions=[self.base_diffusion, self.upsampler_diffusion],
                    num_points=[1024, 4096 - 1024],
                    aux_channels=['R', 'G', 'B'],
                    guidance_scale=[3.0, 0.0],
                    model_kwargs_key_filter=('texts', ''),  # Do not condition the upsampler
                )
            else:
                # Create sampler with only base model
                self.sampler = PointCloudSampler(
                    device=self.device,
                    models=[self.base_model],
                    diffusions=[self.base_diffusion],
                    num_points=[4096],
                    aux_channels=['R', 'G', 'B'],
                    guidance_scale=[3.0],
                )
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load Point-E models: {str(e)}")

    def generate_point_cloud(self, text_prompt):
        """
        Generate a 3D point cloud from a text prompt.

        Args:
            text_prompt (str): Text description of the 3D object.

        Returns:
            PointCloud: Point-E PointCloud object.
        """
        try:
            logger.info(f"Generating point cloud for prompt: '{text_prompt}'")
            
            # Generate samples using the sampler
            samples = None
            for x in self.sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[text_prompt])):
                samples = x
            
            # Convert samples to point cloud
            pc = self.sampler.output_to_point_clouds(samples)[0]
            logger.info("Point cloud generation completed")
            return pc
            
        except Exception as e:
            logger.error(f"Point cloud generation failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to generate point cloud: {str(e)}")

    def save_point_cloud(self, point_cloud, output_path):
        """
        Save a point cloud to a file.

        Args:
            point_cloud (PointCloud): Point-E PointCloud object to save.
            output_path (str): Path to save the point cloud.

        Returns:
            str: Path to the saved file.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            point_cloud.save(output_path)
            logger.info(f"Saved point cloud to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save point cloud: {str(e)}")
            raise RuntimeError(f"Failed to save point cloud: {str(e)}")
    
    def convert_to_mesh(self, point_cloud, output_path=None, grid_size=32):
        """
        Convert a point cloud to a mesh using the SDF model.
        
        Args:
            point_cloud (PointCloud): Point-E PointCloud object to convert.
            output_path (str, optional): Path to save the mesh. If None, mesh is not saved.
            grid_size (int): Resolution of the marching cubes grid.
            
        Returns:
            mesh: The generated mesh object.
        """
        try:
            # Load the SDF model if not already loaded
            if not hasattr(self, 'sdf_model'):
                logger.info('Creating SDF model...')
                self.sdf_model = model_from_config(MODEL_CONFIGS['sdf'], self.device)
                self.sdf_model.eval()
                
                logger.info('Loading SDF model checkpoint...')
                self.sdf_model.load_state_dict(load_checkpoint('sdf', self.device))
            
            # Import inside the method to avoid circular imports
            from point_e.util.pc_to_mesh import marching_cubes_mesh
            
            # Generate mesh
            logger.info(f"Converting point cloud to mesh with grid size {grid_size}...")
            mesh = marching_cubes_mesh(
                pc=point_cloud,
                model=self.sdf_model,
                batch_size=4096,
                grid_size=grid_size,
                progress=True,
            )
            
            # Save mesh if output path is provided
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)  # Go up one level from backend/ to project root
            output_dir = os.path.join(project_root, "generated_meshes")
            os.makedirs(output_dir, exist_ok=True)
            
            return mesh
            
        except Exception as e:
            logger.error(f"Mesh conversion failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to convert point cloud to mesh: {str(e)}")

# Global instance of PointEModel
_point_e_model = None

def get_point_e_model(use_base1b=False, use_upsampler=True):
    """
    Get or create a global instance of PointEModel.

    Args:
        use_base1b (bool): Whether to use the higher quality base1B model.
        use_upsampler (bool): Whether to use the upsampler model.

    Returns:
        PointEModel: A global instance of the model.
    """
    global _point_e_model
    if _point_e_model is None:
        _point_e_model = PointEModel(use_base1b=use_base1b, use_upsampler=use_upsampler)
    return _point_e_model

def generate_point_cloud(text_prompt):
    """
    Generate a 3D point cloud from text using the global PointEModel.

    Args:
        text_prompt (str): Text description of the 3D object.

    Returns:
        PointCloud: Point-E PointCloud object.
    """
    model = get_point_e_model()
    return model.generate_point_cloud(text_prompt)

def generate_mesh_from_text(text_prompt, output_path=None, grid_size=32):
    """
    Generate a 3D mesh from text using the global PointEModel.

    Args:
        text_prompt (str): Text description of the 3D object.
        output_path (str, optional): Path to save the mesh. If None, mesh is not saved.
        grid_size (int): Resolution of the marching cubes grid.

    Returns:
        tuple: (PointCloud object, mesh object)
    """
    model = get_point_e_model()
    pc = model.generate_point_cloud(text_prompt)
    mesh = model.convert_to_mesh(pc, output_path, grid_size)
    return pc, mesh

if __name__ == "__main__":
    # Test the model with a sample prompt
    test_prompt = "a modern blue chair with curved armrests"
    
    try:
        # Create model instance
        model = PointEModel(use_base1b=False, use_upsampler=True)
        
        # Generate point cloud
        logger.info("Generating point cloud...")
        pc = model.generate_point_cloud(test_prompt)
        
        # Save point cloud
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(current_dir), "generated_meshes")
        os.makedirs(output_dir, exist_ok=True)
        pc_filename = f"{test_prompt.replace(' ', '_')}.npz"
        pc_path = os.path.join(output_dir, pc_filename)
        model.save_point_cloud(pc, pc_path)
        
        # Convert to mesh
        logger.info("Converting to mesh...")
        mesh_filename = f"{test_prompt.replace(' ', '_')}.ply"
        mesh_path = os.path.join(output_dir, mesh_filename)
        mesh = model.convert_to_mesh(pc, mesh_path)
        model.save_point_cloud(pc, mesh_path)
        
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
