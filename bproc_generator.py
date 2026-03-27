import blenderproc as bproc
import sys
import json
import os
import yaml
import numpy as np

def render(features_path):
    print(f"[Stage 2: Generate] Initializing BlenderProc2...")
    bproc.init()
    
    if not os.path.exists(features_path):
        print(f"[Stage 2: Generate] Checkpoint missing: {features_path}. Run Stage 1 first.")
        sys.exit(1)
        
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"[Stage 2: Generate] Error reading config.yaml: {e}")
        config = {}
        
    with open(features_path, "r") as f:
        attrs = json.load(f)
    print(f"[Stage 2: Generate] Proceeding with Semantic Attributes: {attrs}")
    
    # Primitive mapping based on VLM semantic output
    shape = attrs.get("shape", "box").lower()
    if "cylinder" in shape:
        obj = bproc.object.create_primitive("CYLINDER")
    else:
        obj = bproc.object.create_primitive("CUBE")
        
    obj.set_location([0, 0, 1])
    obj.set_name("TargetSKU")
    obj.set_cp("category_id", 1) # Must possess a category ID to be tracked in COCO!
    
    # HDRI Lighting application (Simulating Stadium environment)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([0, 0, 4])
    light.set_energy(1000)
    
    dist_range = config.get("camera_distance_range", [1.5, 3.0])
    elev_range = config.get("camera_elevation_range", [0.1, 1.5])
    render_count = config.get("render_count", 15)
    
    for i in range(render_count):
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.random.uniform(elev_range[0], elev_range[1])
        distance = np.random.uniform(dist_range[0], dist_range[1])
        
        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation)
        
        location = [x, y, z]
        poi = bproc.object.compute_poi([obj])
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
    
    # Enable depth & native COCO annotations
    res = config.get("image_resolution", [640, 640])
    bproc.camera.set_resolution(res[0], res[1])
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["instance", "class", "name"])
    
    print("[Stage 2: Generate] Rendering physical occlusion datasets and outputting pure COCO annotations to checkpoints/synthetic_dataset/ ...")
    data = bproc.renderer.render()
    
    bproc.writer.write_coco_annotations("checkpoints/synthetic_dataset/",
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        color_file_format="JPEG")
    print("[Stage 2: Generate] Successfully wrote COCO annotations to checkpoints/synthetic_dataset/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: blenderproc run bproc_generator.py <features_path>")
        sys.exit(1)
    
    # The last argument is the one passed to the script via `--` or standalone
    render(sys.argv[-1])
