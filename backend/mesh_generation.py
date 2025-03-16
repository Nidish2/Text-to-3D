import open3d as o3d
import numpy as np

def point_cloud_to_mesh(point_cloud: np.ndarray, depth: int = 9) -> o3d.geometry.TriangleMesh:
    """
    Convert a point cloud to a triangle mesh using Poisson surface reconstruction.
    
    Args:
        point_cloud (np.ndarray): Array of shape (N, 3) representing 3D points.
        depth (int): Depth parameter for Poisson reconstruction (default: 9).
    
    Returns:
        o3d.geometry.TriangleMesh: Generated mesh.
    """
    if not isinstance(point_cloud, np.ndarray) or point_cloud.shape[1] != 3:
        raise ValueError("Input must be a numpy array of shape (N, 3).")
    
    try:
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Perform Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        
        # Clean up the mesh by removing low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        return mesh
    
    except Exception as e:
        raise RuntimeError(f"Mesh generation failed: {str(e)}")
    
def save_mesh_as_obj(mesh: o3d.geometry.TriangleMesh, filename: str) -> None:
    """
    Save the mesh as an OBJ file.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh to save.
        filename (str): Path to save the OBJ file.
    """
    try:
        o3d.io.write_triangle_mesh(filename, mesh)
    except Exception as e:
        raise RuntimeError(f"Failed to save mesh as OBJ: {str(e)}")

if __name__ == "__main__":
    # Test with a dummy point cloud
    dummy_points = np.random.rand(1000, 3)
    mesh = point_cloud_to_mesh(dummy_points)
    print(f"Mesh vertices: {len(mesh.vertices)}")