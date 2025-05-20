import collections
import logging
import habitat_sim
import numpy as np
import trimesh
import os # <--- IMPORT OS ADDED HERE

from habitat.utils.visualizations import maps
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)


def dense_sampling_trimesh(triangles, density=100.0, max_points=200000):
    if not isinstance(triangles, np.ndarray) or triangles.ndim != 3 or triangles.shape[1:] != (3,3) : # Check if triangles is a Nx3x3 array
        logger.warning(f"dense_sampling_trimesh: Invalid triangles input shape: {triangles.shape if isinstance(triangles, np.ndarray) else type(triangles)}. Expected Nx3x3. Returning empty.")
        return np.array([])
    if triangles.shape[0] == 0:
        logger.warning("dense_sampling_trimesh: Received empty triangles array (0 faces).")
        return np.array([])

    t_vertices = triangles.reshape(-1, 3)
    t_faces = np.arange(0, t_vertices.shape[0]).reshape(-1, 3)
    try:
        t_mesh = trimesh.Trimesh(vertices=t_vertices, faces=t_faces, process=False) # process=False can be faster
        if t_mesh.is_empty:
            logger.warning("dense_sampling_trimesh: Created empty Trimesh object.")
            return np.array([])
        surface_area = t_mesh.area
        if abs(surface_area) < 1e-9: # Check for effectively zero area
            logger.debug(f"dense_sampling_trimesh: Mesh surface area is close to 0 ({surface_area}). Vertices: {len(t_vertices)}")
            if len(t_vertices) > 0:
                sample_size = min(len(t_vertices), int(max_points * 0.01) if max_points > 0 else 100) # Sample 1% or 100
                if sample_size > 0:
                    indices = np.random.choice(len(t_vertices), size=sample_size, replace=len(t_vertices) < sample_size)
                    return t_vertices[indices]
            return np.array([])


        n_points = min(int(surface_area * density), max_points)
        if n_points <= 0:
            logger.debug(f"dense_sampling_trimesh: Calculated n_points <= 0 (Area: {surface_area}, Density: {density}). Returning empty.")
            return np.array([])
        
        t_pts, _ = trimesh.sample.sample_surface_even(t_mesh, n_points)
        return t_pts
    except Exception as e:
        logger.error(f"dense_sampling_trimesh: Error creating/sampling Trimesh: {e}", exc_info=True)
        return np.array([])


def make_configuration(scene_path, scene_dataset_config_file=None, radius=0.18, height=0.88):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    if scene_dataset_config_file and os.path.exists(scene_dataset_config_file): # Check if file exists
        backend_cfg.scene_dataset_config_file = scene_dataset_config_file
        logger.debug(f"make_configuration: Using scene_dataset_config_file: {scene_dataset_config_file}")
    elif scene_dataset_config_file: # Path provided but not found
        logger.warning(f"make_configuration: scene_dataset_config_file '{scene_dataset_config_file}' not found. Proceeding without it.")
    else: # Not provided
        logger.debug(f"make_configuration: scene_dataset_config_file not provided for {scene_path}.")


    
    depth_sensor_cfg = habitat_sim.CameraSensorSpec()
    depth_sensor_cfg.uuid = "depth"
    depth_sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_cfg.resolution = [1080, 960]
    depth_sensor_cfg.position = [0.0, height, 0.0]

    semantic_sensor_cfg = habitat_sim.CameraSensorSpec()
    semantic_sensor_cfg.uuid = "semantic"
    semantic_sensor_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_cfg.resolution = [1080, 960]
    semantic_sensor_cfg.position = [0.0, height, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = height
    agent_cfg.radius = radius
    agent_cfg.sensor_specifications = [depth_sensor_cfg, semantic_sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def robust_load_sim(glb_path, scene_dataset_config_file=None): # Signature fixed
    logger.info(f"robust_load_sim: Attempting to load scene: {os.path.basename(glb_path)}")
    if scene_dataset_config_file:
        logger.info(f"robust_load_sim: Using scene_dataset_config_file: {scene_dataset_config_file}")

    cfg = make_configuration(glb_path, scene_dataset_config_file=scene_dataset_config_file)
    sim = None
    try:
        sim = habitat_sim.Simulator(cfg)
        if not sim.pathfinder.is_loaded:
            logger.warning(f"NavMesh not loaded for {os.path.basename(glb_path)}. Attempting to recompute.")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings, include_static_objects=True)
            if sim.pathfinder.is_loaded:
                logger.info(f"NavMesh recomputed and loaded successfully for {os.path.basename(glb_path)}.")
            else:
                logger.error(f"Failed to load or recompute NavMesh for {os.path.basename(glb_path)}.")
        else:
            logger.info(f"NavMesh loaded successfully for {os.path.basename(glb_path)}.")
        logger.info(f"Simulator created for {os.path.basename(glb_path)}")
    except Exception as e:
        logger.error(f"Error creating simulator for {os.path.basename(glb_path)}: {e}", exc_info=True)
        if sim is not None:
            sim.close()
        raise
    return sim


def get_dense_navigable_points(sim, sampling_resolution=0.05):
    logger.debug(f"get_dense_navigable_points: Sampling at resolution {sampling_resolution}")
    if not sim.pathfinder.is_loaded:
        logger.warning("get_dense_navigable_points: NavMesh is not loaded. Returning empty list.")
        return []
    
    navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
    if navmesh_vertices.size == 0:
        logger.warning("get_dense_navigable_points: NavMesh has no vertices. Returning empty list.")
        return []
        
    navmesh_vertices = navmesh_vertices.reshape(-1, 3, 3)
    navigable_points = []
    for face_idx, face in enumerate(navmesh_vertices):
        p1, p2, p3 = face[0], face[1], face[2]
        navigable_points.extend([p1, p2, p3])
        try:
            ps = dense_sampling_util(p1, p2, p3, sampling_resolution)
            navigable_points.extend(ps)
        except Exception as e:
            logger.warning(f"Error in dense_sampling_util for face {face_idx}: {e}. Points: {p1}, {p2}, {p3}")
            continue 

    logger.debug(f"get_dense_navigable_points: Sampled {len(navigable_points)} points.")
    return navigable_points

def dense_sampling_util(p1, p2, p3, sampling_resolution):
    n1 = p2 - p1
    d1 = np.linalg.norm(n1)
    if d1 < 1e-6: return [] 
    n1 = n1 / d1

    n2 = p3 - p1
    d2 = np.linalg.norm(n2)
    if d2 < 1e-6: return [] 
    n2 = n2 / d2
    
    cross_prod_norm = np.linalg.norm(np.cross(n1, n2))
    if cross_prod_norm < 1e-6: 
        return []

    dense_points = [] 
    for i_val in np.arange(0, d1, sampling_resolution):
        b = (d1 - i_val) * d2 / d1 
        js_coords = []
        for j_val in np.arange(0, b, sampling_resolution):
            js_coords.append([1.0, i_val, j_val]) 
        
        if not js_coords:
            continue

        js_arr = np.array(js_coords)
        x_basis_matrix = np.stack([p1, n1, n2], axis=1) 
        ps = np.matmul(js_arr, x_basis_matrix.T) 
        dense_points.extend(ps.tolist())
    return dense_points

def get_floor_heights(sim, sampling_resolution=0.10):
    logger.info("get_floor_heights: Starting floor height extraction.")
    nav_points_list = get_dense_navigable_points(sim, sampling_resolution=sampling_resolution)
    if not nav_points_list:
        logger.warning("get_floor_heights: No navigable points found. Cannot determine floor heights.")
        return []
    
    nav_points = np.array(nav_points_list)
    if nav_points.shape[0] < 10: 
        logger.warning(f"get_floor_heights: Too few navigable points ({nav_points.shape[0]}) for robust clustering.")
        if nav_points.shape[0] > 0:
            median_y = np.median(nav_points[:, 1])
            return [{"min": median_y - 0.1, "max": median_y + 0.1, "mean": median_y}] 
        return []

    y_coors = np.around(nav_points[:, 1], decimals=1) 
    
    y_counter = collections.Counter(y_coors)
    min_count_for_y_level = max(10, int(len(y_coors) * 0.005)) 
    filtered_y_coors_list = []
    for y_val, count in y_counter.items():
        if count >= min_count_for_y_level:
            filtered_y_coors_list.extend([y_val] * count) 
    
    if not filtered_y_coors_list:
        logger.warning("get_floor_heights: No significant Y-levels found after filtering sparse points.")
        if nav_points.shape[0] > 0:
            median_y = np.median(nav_points[:, 1])
            return [{"min": median_y - 0.1, "max": median_y + 0.1, "mean": median_y}]
        return []
    
    y_coors_for_clustering = np.array(filtered_y_coors_list)

    min_samples_dbscan = max(10, int(0.05 * len(y_coors_for_clustering))) 
    dbscan_eps = 0.15
    try:
        clustering = DBSCAN(eps=dbscan_eps, min_samples=min_samples_dbscan).fit(y_coors_for_clustering[:, np.newaxis])
    except ValueError as ve: 
        logger.error(f"ValueError during DBSCAN fitting: {ve}. y_coors_for_clustering shape: {y_coors_for_clustering.shape}")
        if nav_points.shape[0] > 0: 
            median_y = np.median(nav_points[:, 1])
            return [{"min": median_y - 0.1, "max": median_y + 0.1, "mean": median_y}]
        return []

    c_labels = clustering.labels_
    unique_labels = set(c_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    logger.info(f"get_floor_heights: Found {n_clusters} potential floors via DBSCAN.")

    floor_extents = []
    for label_idx in unique_labels:
        if label_idx == -1: continue 
        
        cluster_points_y = y_coors_for_clustering[c_labels == label_idx]
        if len(cluster_points_y) < min_samples_dbscan: 
            logger.debug(f"Skipping small cluster (label {label_idx}) with {len(cluster_points_y)} points.")
            continue

        floor_min = cluster_points_y.min().item()
        floor_max = cluster_points_y.max().item()
        floor_mean = cluster_points_y.mean().item()
        floor_extents.append({"min": floor_min, "max": floor_max, "mean": floor_mean})
    
    floor_extents = sorted(floor_extents, key=lambda x: x["mean"])
    
    clean_floor_extents = []
    if floor_extents:
        max_projected_points = 0
        floor_projected_points = []
        for fext in floor_extents:
            try:
                top_down_map = maps.get_topdown_map(sim.pathfinder, fext["mean"], map_resolution=0.1) 
                num_projected_points = np.count_nonzero(top_down_map > 0) 
                floor_projected_points.append(num_projected_points)
                if num_projected_points > max_projected_points:
                    max_projected_points = num_projected_points
            except Exception as e_tdm:
                logger.warning(f"Could not get top_down_map for floor at mean_y {fext['mean']}: {e_tdm}")
                floor_projected_points.append(0)
                continue
        
        for idx, fext in enumerate(floor_extents):
            if floor_projected_points[idx] < 0.15 * max_projected_points and max_projected_points > 0: 
                logger.info(f"Rejecting floor (mean_y: {fext['mean']:.2f}) due to small projected area: {floor_projected_points[idx]} vs max {max_projected_points}")
                continue
            clean_floor_extents.append(fext)
    
    if not clean_floor_extents and nav_points.shape[0] > 0: 
        logger.warning("All potential floors rejected by area check. Falling back to single floor based on overall median Y.")
        median_y = np.median(nav_points[:, 1])
        return [{"min": median_y - 0.1, "max": median_y + 0.1, "mean": median_y}]

    logger.info(f"get_floor_heights: Final floor extents: {clean_floor_extents}")
    return clean_floor_extents


def get_navmesh_extents_at_y(sim, y_bounds=None):
    logger.debug(f"get_navmesh_extents_at_y: Called with y_bounds: {y_bounds}")
    if not sim.pathfinder.is_loaded:
        logger.warning("get_navmesh_extents_at_y: NavMesh is not loaded. Cannot get extents.")
        return None, None

    if y_bounds is None:
        try:
            lower_bound, upper_bound = sim.pathfinder.get_bounds()
            if lower_bound is None or upper_bound is None: 
                 logger.warning("sim.pathfinder.get_bounds() returned None. Navmesh might be empty or problematic.")
                 return None, None
            logger.debug(f"Overall navmesh bounds: Lower={lower_bound}, Upper={upper_bound}")
            return lower_bound, upper_bound
        except Exception as e:
            logger.error(f"Error getting overall navmesh bounds: {e}", exc_info=True)
            return None, None
    else:
        assert len(y_bounds) == 2 and y_bounds[0] <= y_bounds[1], "y_bounds must be (min, max)"
        
        navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
        if navmesh_vertices.size == 0:
            logger.warning(f"No navmesh vertices found for y_bounds: {y_bounds}")
            return None, None
            
        mask = (y_bounds[0] <= navmesh_vertices[:, 1]) & (navmesh_vertices[:, 1] <= y_bounds[1])
        filtered_vertices = navmesh_vertices[mask]
        
        if filtered_vertices.shape[0] == 0:
            logger.warning(f"No navmesh vertices found within specified y_bounds: {y_bounds}")
            y_min_expanded, y_max_expanded = y_bounds[0] - 0.1, y_bounds[1] + 0.1
            mask_expanded = (y_min_expanded <= navmesh_vertices[:, 1]) & (navmesh_vertices[:, 1] <= y_max_expanded)
            filtered_vertices_expanded = navmesh_vertices[mask_expanded]
            if filtered_vertices_expanded.shape[0] == 0:
                logger.warning(f"Still no navmesh vertices found with expanded y_bounds: ({y_min_expanded:.2f}, {y_max_expanded:.2f})")
                return None, None
            else:
                logger.info(f"Found {filtered_vertices_expanded.shape[0]} vertices with expanded y_bounds.")
                filtered_vertices = filtered_vertices_expanded

        lower_bound = filtered_vertices.min(axis=0)
        upper_bound = filtered_vertices.max(axis=0)
        logger.debug(f"Bounds for y_range {y_bounds}: Lower={lower_bound}, Upper={upper_bound}")
        return lower_bound, upper_bound

def convert_lu_bound_to_smnet_bound(lu_bound, buf=np.array([3.0, 0.0, 3.0])):
    if lu_bound is None or lu_bound[0] is None or lu_bound[1] is None:
        logger.warning("convert_lu_bound_to_smnet_bound received None bounds.")
        return { 
            "xlo": 0, "ylo": 0, "zlo": 0,
            "xhi": 0, "yhi": 0, "zhi": 0,
            "center": [0,0,0], "sizes": [0,0,0],
        }
    lower_bound = lu_bound[0] - buf
    upper_bound = lu_bound[1] + buf
    smnet_bound = {
        "xlo": lower_bound[0].item(), "ylo": lower_bound[1].item(), "zlo": lower_bound[2].item(),
        "xhi": upper_bound[0].item(), "yhi": upper_bound[1].item(), "zhi": upper_bound[2].item(),
        "center": ((lower_bound + upper_bound) / 2.0).tolist(),
        "sizes": np.abs(upper_bound - lower_bound).tolist(),
    }
    return smnet_bound

