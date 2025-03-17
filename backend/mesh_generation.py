import open3d as o3d
import numpy as np

def point_cloud_to_mesh(point_cloud: np.ndarray, colors: np.ndarray = None, depth: int = 12) -> o3d.geometry.TriangleMesh:
    """
    Convert a point cloud to a triangle mesh using Poisson surface reconstruction.
    
    Args:
        point_cloud (np.ndarray): Array of shape (N, 3) representing 3D points.
        colors (np.ndarray, optional): Array of shape (N, 3) representing RGB colors (0-1).
        depth (int): Depth parameter for Poisson reconstruction (default: 9).
    
    Returns:
        o3d.geometry.TriangleMesh: Generated mesh with colors.
    """
    if not isinstance(point_cloud, np.ndarray) or point_cloud.shape[1] != 3:
        raise ValueError("Input must be a numpy array of shape (N, 3).")
    
    try:
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        # Add colors if provided
        if colors is not None and len(colors) == len(point_cloud):
            colors = np.clip(colors, 0, 1)  # Clip colors to [0, 1]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.full((len(point_cloud), 3), [1.0, 0.5, 0.0]))
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=30)
        
        # Perform Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        
        # Clean up the mesh by removing low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Transfer colors to the mesh using closest point interpolation
        if colors is not None:
            # Create a KD-tree for the original point cloud
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            
            # For each vertex in the mesh, find the closest point in the original point cloud
            mesh_vertices = np.asarray(mesh.vertices)
            vertex_colors = np.zeros((len(mesh_vertices), 3))
            
            for i, vertex in enumerate(mesh_vertices):
                _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
                if colors is not None:
                    vertex_colors[i] = np.asarray(pcd.colors)[idx[0]]
                else:
                    vertex_colors[i] = [0.5, 0.5, 0.5]  # Default gray
            
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
        # Compute vertex normals for better rendering
        mesh.compute_vertex_normals()
        
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
        # Make sure the mesh has vertex colors
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Default gray color
        
        # Write mesh with options to include vertex colors
        success = o3d.io.write_triangle_mesh(
            filename, 
            mesh,
            write_vertex_normals=True,
            write_vertex_colors=True,
            write_triangle_uvs=True
        )
        
        if not success:
            raise RuntimeError(f"Failed to write mesh to {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to save mesh as OBJ: {str(e)}")

if __name__ == "__main__":
    # Test with a dummy point cloud and colors
    dummy_points = np.random.rand(1000, 3)
    dummy_colors = np.random.rand(1000, 3)
    mesh = point_cloud_to_mesh(dummy_points, dummy_colors)
    print(f"Mesh vertices: {len(mesh.vertices)}")
    save_mesh_as_obj(mesh, "test_mesh.obj")