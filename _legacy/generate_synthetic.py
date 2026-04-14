import os
import sys

# Try to import bpy but graceful fail for CLI mocking without actual Blender installed
try:
    import bpy
    HAS_BENDER = True
except ImportError:
    HAS_BENDER = False
import json
import random
import math

def render_mock_dataset(attributes, num_images):
    print(f"[Procedural Gen] Blender environment not found. Mocking the execution...")
    print(f"[Procedural Gen] Generated {num_images} domain-randomized synthetic frames for {attributes['shape']} using material {attributes['material']}.")

def setup_scene(output_dir="synthetic_dataset"):
    if not HAS_BENDER: return output_dir
    bpy.ops.wm.read_factory_settings(use_empty=True)
    os.makedirs(output_dir, exist_ok=True)
    
    cam_data = bpy.data.cameras.new('Camera')
    cam = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return output_dir

def load_product_primitive(attributes):
    if not HAS_BENDER: return None
    shape = attributes.get("shape", "box").lower()
    if "cylinder" in shape:
        bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, location=(0, 0, 1))
    else:
        bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
    
    product = bpy.context.active_object
    product.name = "TargetSKU"
    
    mat = bpy.data.materials.new(name="ProductMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    if "glossy" in attributes.get("material", "").lower() or "aluminum" in attributes.get("material", "").lower():
        bsdf.inputs['Roughness'].default_value = 0.1
        bsdf.inputs['Metallic'].default_value = 0.8
    else:
        bsdf.inputs['Roughness'].default_value = 0.8
        
    product.data.materials.append(mat)
    return product

def apply_domain_randomization(product, camera):
    if not HAS_BENDER: return
    # Randomize lighting
    if 'HarshLight' in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects['HarshLight'], do_unlink=True)
        
    light_data = bpy.data.lights.new(name="HarshLight", type='AREA')
    light_data.energy = random.uniform(500, 1500)
    light_obj = bpy.data.objects.new(name="HarshLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(3, 5))
    
    # Randomize camera position
    camera.location = (random.uniform(-4, 4), random.uniform(-6, -3), random.uniform(2, 5))
    
    # Track to product
    ttc = camera.constraints.new(type='TRACK_TO')
    ttc.target = product
    ttc.track_axis = 'TRACK_NEGATIVE_Z'
    ttc.up_axis = 'UP_Y'
    
    # Introduce occlusion
    if 'Occluder' in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects['Occluder'], do_unlink=True)
        
    bpy.ops.mesh.primitive_icosphere_add(radius=0.5)
    occluder = bpy.context.active_object
    occluder.name = "Occluder"
    
    direction_vec = product.location - camera.location
    distance = direction_vec.length
    direction_vec.normalize()
    
    random_dist = random.uniform(distance * 0.3, distance * 0.8)
    occluder.location = camera.location + (direction_vec * random_dist)

def render_dataset(attributes, num_images=10):
    if not HAS_BENDER:
        render_mock_dataset(attributes, num_images)
        return
        
    out_dir = setup_scene()
    product = load_product_primitive(attributes)
    camera = bpy.data.objects['Camera']
    
    print(f"[Procedural Gen] Generating {num_images} domain-randomized synthetic frames...")
    
    for i in range(num_images):
        apply_domain_randomization(product, camera)
        filepath = os.path.join(out_dir, f"syn_{i:04d}.jpg")
        bpy.context.scene.render.filepath = filepath
        bpy.context.scene.render.resolution_x = 640
        bpy.context.scene.render.resolution_y = 640
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.ops.render.render(write_still=True)
        
    print(f"[Procedural Gen] Successfully rendered {num_images} images to {out_dir}")

if __name__ == "__main__":
    try:
        idx = sys.argv.index("--") + 1
        attr_file = sys.argv[idx]
        with open(attr_file, "r") as f:
            attrs = json.load(f)
    except (ValueError, IndexError):
        attrs = {"shape": "box", "material": "matte cardboard", "primary_colors": ["blue"]}
        
    render_dataset(attrs, num_images=10)
