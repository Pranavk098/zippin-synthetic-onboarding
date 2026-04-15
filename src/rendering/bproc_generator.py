# BlenderProc2 requires this to be the first executable line — do not move.
import blenderproc as bproc  # noqa: E402

"""
BlenderProc2 Synthetic Dataset Generator — Production-Grade Renderer

Invoked via: blenderproc run src/rendering/bproc_generator.py <features_path>

Key improvements over the v1 prototype:
  1. Texture projection: wraps the reference product image onto the 3D mesh
     surface (UVProject), dramatically closing the Sim2Real appearance gap.
  2. HDRI lighting: samples stadium/industrial HDRIs from the BlenderProc2
     haven directory for realistic inter-reflection and ambient occlusion.
  3. Multi-occlusion: spawns 3–7 varied occluder geometries at randomised
     depths/sizes, mimicking hand-reach and product-front occlusion in CVS
     and Zippin shelf environments.
  4. Environment variables: all rendering params are injectable via env vars
     for the Stage 2 orchestrator, removing hard-coded constants.
  5. Output naming: includes job_id prefix to prevent COCO collisions in
     multi-SKU parallel runs.

BlenderProc2 uses its own embedded Python environment (Blender's bpy).
This file runs INSIDE that environment — standard Python imports only.
"""

import sys
import os
import json
import numpy as np

# ---- Read runtime parameters from env (set by stage_generate orchestrator) ----
OUTPUT_DIR      = os.environ.get("BPROC_OUTPUT_DIR", "checkpoints/synthetic_dataset")
RENDER_COUNT    = int(os.environ.get("BPROC_RENDER_COUNT", "300"))
RESOLUTION_W    = int(os.environ.get("BPROC_RESOLUTION_W", "640"))
RESOLUTION_H    = int(os.environ.get("BPROC_RESOLUTION_H", "640"))
# BPROC_OCCLUSION_MODE: "standard" | "partial_stress" | "full_stress" | "all"
# "all" runs standard + both stress suites back-to-back (3× render count).
OCCLUSION_MODE  = os.environ.get("BPROC_OCCLUSION_MODE", "standard")
# BPROC_RENDER_BATCHES: number of times the scene is fully re-initialised.
# Each batch gets fresh random occluder positions/colours/shapes, so
# RENDER_COUNT=300 with RENDER_BATCHES=6 → 6 scenes × 50 poses = 300 unique
# occluder configurations instead of 1 scene seen from 300 angles.
RENDER_BATCHES  = int(os.environ.get("BPROC_RENDER_BATCHES", "6"))

# Randomisation envelopes tuned for Zippin overhead shelf-cam geometry
# 1.2–2.5m: 1.0m was occasionally too close for the larger SKU, causing the
# object to fill 85%+ of frame and lose environment context (e.g. frame 77).
CAMERA_DIST_MIN, CAMERA_DIST_MAX = 1.2, 2.5
# 0.20–1.0 rad (11°–57°): lower minimum adds more floor-context shots
# while still keeping the camera safely above the floor plane
CAMERA_ELEV_MIN, CAMERA_ELEV_MAX = 0.20, 0.65   # ~37° max — prevents extreme top-down crops
# 2–4 occluders: enough real occlusion without burying the target entirely
N_OCCLUDERS_RANGE = (2, 4)

# Occlusion stress-test coverage bands
PARTIAL_COVERAGE_RANGE = (0.30, 0.55)   # 30-55 % SKU frontal area obscured
FULL_COVERAGE_RANGE    = (0.75, 0.95)   # 75-95 % — product nearly invisible


def _build_can_mesh(name: str = "TargetSKU"):
    """
    Build a realistic soda-can mesh using bmesh and wrap it in a BlenderProc
    MeshObject.  Replaces create_primitive("CYLINDER") which produces a
    featureless tube with zero visual resemblance to a real can.

    Geometry is centred at the local origin (z ∈ [-0.5, 0.5], r_max = 0.5)
    so it behaves identically to the previous cylinder primitive under
    set_scale([0.15, 0.15, 0.30]) / set_location([0, 0, 0.30]).

    Shape cues added vs the plain cylinder:
      - Bevelled bottom rim
      - Full straight body section
      - Shoulder taper (~85 % up)
      - Neck with reduced radius
      - Flat top with mouth ring
      - Smooth shading on all faces
    """
    import bpy
    import bmesh
    import math
    from blenderproc.python.types.MeshObjectUtility import MeshObject

    # ---------- 2-D lathe profile (r, z) centred at z=0 ----------------------
    # z: -1.000 = bottom rim  →  +1.000 = top rim   ← matches Blender's default
    # cylinder primitive so set_scale / set_location downstream need no changes.
    # r:  0.000 = axis        →   0.500 = max body radius
    profile = [
        (0.455, -1.000),   # bottom outer rim
        (0.485, -0.950),   # bottom bead / chime
        (0.500, -0.880),   # body starts (max radius)
        (0.500,  0.740),   # body top (straight section ends)
        (0.492,  0.830),   # shoulder start — gentle slope
        (0.465,  0.890),   # shoulder mid
        (0.428,  0.936),   # neck
        (0.410,  0.970),   # mouth
        (0.410,  1.000),   # top rim
    ]
    N_SEG = 48   # circumference segments — enough for smooth curvature

    mesh = bpy.data.meshes.new(name + "_mesh")
    bm   = bmesh.new()

    # Build vertex rings by revolving the profile around Z
    rings = []
    for r, z in profile:
        ring = []
        for i in range(N_SEG):
            angle = 2.0 * math.pi * i / N_SEG
            v = bm.verts.new((r * math.cos(angle), r * math.sin(angle), z))
            ring.append(v)
        rings.append(ring)

    # Side quads between adjacent profile rings
    for ri in range(len(rings) - 1):
        for i in range(N_SEG):
            j = (i + 1) % N_SEG
            bm.faces.new([rings[ri][i], rings[ri][j],
                          rings[ri + 1][j], rings[ri + 1][i]])

    # Bottom cap — fan from centre point
    bc = bm.verts.new((0.0, 0.0, -1.000))
    for i in range(N_SEG):
        bm.faces.new([bc, rings[0][(i + 1) % N_SEG], rings[0][i]])

    # Top cap — fan from centre point (CCW from above → normal faces up)
    tc = bm.verts.new((0.0, 0.0,  1.000))
    for i in range(N_SEG):
        bm.faces.new([tc, rings[-1][i], rings[-1][(i + 1) % N_SEG]])

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # Smooth shading — eliminates the faceted look of the default cylinder
    for poly in mesh.polygons:
        poly.use_smooth = True

    # Wrap in a Blender object and link to the active collection
    bpy_obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(bpy_obj)

    return MeshObject(bpy_obj)


def render(features_path: str) -> None:
    """
    Multi-batch renderer: re-initialises the scene RENDER_BATCHES times so
    each batch gets fresh random occluder positions, colours, and shapes.
    RENDER_COUNT frames are split evenly across batches.

    e.g. RENDER_COUNT=300, RENDER_BATCHES=6 → 6 scenes × 50 poses each.
    BlenderProc's write_coco_annotations appends to the same OUTPUT_DIR so
    all batches produce a single merged COCO JSON.
    """
    if not os.path.exists(features_path):
        print(f"[BProc] ERROR: features checkpoint not found: {features_path}")
        sys.exit(1)

    with open(features_path, "r") as f:
        attrs = json.load(f)

    print(f"[BProc] Loaded SKU attributes: {attrs}")

    # Resolve product image path once — reused across all batches
    product_img_path = os.environ.get("BPROC_PRODUCT_IMAGE", "")
    if not product_img_path:
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        features_abs = os.path.abspath(features_path)
        search_dirs  = [
            os.path.dirname(features_abs),
            os.path.dirname(os.path.dirname(features_abs)),
            project_root,
        ]
        for d in search_dirs:
            candidate = os.path.join(d, "product.jpg")
            if os.path.exists(candidate):
                product_img_path = candidate
                break
        if not product_img_path:
            print(f"[BProc] WARNING: product.jpg not found in {search_dirs}")

    # Occluder palette — computed once, reused per batch
    OCCLUDER_PALETTE_FULL = [
        (0.80, 0.05, 0.05),   # Red
        (0.05, 0.15, 0.80),   # Blue
        (0.05, 0.65, 0.10),   # Green
        (0.90, 0.75, 0.02),   # Yellow
        (0.90, 0.38, 0.02),   # Orange
        (0.02, 0.55, 0.55),   # Teal
        (0.38, 0.18, 0.05),   # Dark brown
        (0.88, 0.88, 0.88),   # White / off-white
        (0.05, 0.05, 0.05),   # Black
        (0.55, 0.05, 0.70),   # Purple
    ]
    _hue_is = {
        "red":    lambda r, g, b: r > 0.45 and r > g + 0.35 and r > b + 0.35,
        "blue":   lambda r, g, b: b > 0.45 and b > r + 0.25 and b > g + 0.25,
        "green":  lambda r, g, b: g > 0.45 and g > r + 0.25 and g > b + 0.25,
        "yellow": lambda r, g, b: r > 0.55 and g > 0.45 and b < 0.25,
        "orange": lambda r, g, b: r > 0.65 and 0.15 < g < 0.55 and b < 0.15,
        "silver": lambda r, g, b: abs(r - g) < 0.12 and abs(g - b) < 0.12 and r > 0.45,
        "white":  lambda r, g, b: r > 0.70 and g > 0.70 and b > 0.70,
        "black":  lambda r, g, b: r < 0.12 and g < 0.12 and b < 0.12,
    }
    sku_hue      = (attrs.get("primary_colors") or [""])[0].lower()
    is_sku_col   = _hue_is.get(sku_hue, lambda r, g, b: False)
    OCCLUDER_PALETTE = [c for c in OCCLUDER_PALETTE_FULL if not is_sku_col(*c)] \
                       or OCCLUDER_PALETTE_FULL

    frames_per_batch = max(1, RENDER_COUNT // RENDER_BATCHES)

    for batch_idx in range(RENDER_BATCHES):
        frames_this_batch = (
            frames_per_batch if batch_idx < RENDER_BATCHES - 1
            else RENDER_COUNT - batch_idx * frames_per_batch
        )
        if frames_this_batch <= 0:
            break

        print(f"\n[BProc] === Batch {batch_idx + 1}/{RENDER_BATCHES} "
              f"({frames_this_batch} frames) ===")
        bproc.init()   # Full scene reset — clears objects, lights, camera poses

        import bpy

        # ---- 1. Product mesh -------------------------------------------------
        shape = attrs.get("shape", "box").lower()
        if "cylinder" in shape or "bottle" in shape or "can" in shape:
            obj = _build_can_mesh("TargetSKU")
            obj.set_scale([0.15, 0.15, 0.30])
        elif "bag" in shape:
            obj = bproc.object.create_primitive("CUBE")
            obj.set_scale([0.20, 0.08, 0.30])
        else:
            obj = bproc.object.create_primitive("CUBE")
            obj.set_scale([0.15, 0.12, 0.25])

        obj.set_location([0, 0, 0.30])
        obj.set_name("TargetSKU")
        obj.set_cp("category_id", 1)

        # ---- 2. Material: texture projection / VLM colour fallback ----------
        material_desc = attrs.get("material", "").lower()
        mat           = bproc.material.create("sku_material")
        texture_loaded = False

        if product_img_path and os.path.exists(product_img_path):
            try:
                img       = bpy.data.images.load(product_img_path)
                mat_nodes = mat.blender_obj.node_tree
                mat_nodes.nodes.clear()
                tex_coord = mat_nodes.nodes.new("ShaderNodeTexCoord")
                mapping   = mat_nodes.nodes.new("ShaderNodeMapping")
                img_tex   = mat_nodes.nodes.new("ShaderNodeTexImage")
                bsdf      = mat_nodes.nodes.new("ShaderNodeBsdfPrincipled")
                out_node  = mat_nodes.nodes.new("ShaderNodeOutputMaterial")

                img_tex.image = img
                # -90° X rotation: projects image onto side face (not top)
                mapping.inputs['Rotation'].default_value = (-1.5708, 0.0, 0.0)

                mat_nodes.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
                mat_nodes.links.new(mapping.outputs["Vector"],      img_tex.inputs["Vector"])
                mat_nodes.links.new(img_tex.outputs["Color"],       bsdf.inputs["Base Color"])
                mat_nodes.links.new(bsdf.outputs["BSDF"],           out_node.inputs["Surface"])

                # Moderate metallic/roughness — keeps label colour visible
                if "aluminum" in material_desc or "metallic" in material_desc:
                    bsdf.inputs["Roughness"].default_value = 0.40
                    bsdf.inputs["Metallic"].default_value  = 0.45
                elif "plastic" in material_desc:
                    bsdf.inputs["Roughness"].default_value = 0.30
                else:
                    bsdf.inputs["Roughness"].default_value = 0.70

                texture_loaded = True
                print(f"[BProc] Texture loaded from: {product_img_path}")
            except Exception as e:
                print(f"[BProc] Texture load failed ({e}) — falling back to VLM colour.")

        if not texture_loaded:
            if "gloss" in material_desc or "aluminum" in material_desc or "metallic" in material_desc:
                mat.set_principled_shader_value("Roughness", 0.08)
                mat.set_principled_shader_value("Metallic", 0.85 if "aluminum" in material_desc else 0.0)
                mat.set_principled_shader_value("Coat Weight", 0.3)
            elif "plastic" in material_desc:
                mat.set_principled_shader_value("Roughness", 0.25)
                mat.set_principled_shader_value("Metallic", 0.0)
            else:
                mat.set_principled_shader_value("Roughness", 0.70)

            primary_colors = attrs.get("primary_colors", [])
            if primary_colors:
                colour_map = {
                    "red": (0.8, 0.05, 0.05, 1.0), "blue": (0.05, 0.1, 0.8, 1.0),
                    "green": (0.05, 0.6, 0.1, 1.0), "yellow": (0.9, 0.8, 0.02, 1.0),
                    "silver": (0.75, 0.75, 0.75, 1.0), "white": (0.9, 0.9, 0.9, 1.0),
                    "black": (0.05, 0.05, 0.05, 1.0), "orange": (0.9, 0.4, 0.02, 1.0),
                    "purple": (0.5, 0.05, 0.7, 1.0),
                }
                rgba = colour_map.get(primary_colors[0].lower(), (0.5, 0.5, 0.5, 1.0))
                mat.set_principled_shader_value("Base Color", rgba)

        obj.replace_materials(mat)

        # ---- 3. Scene geometry: floor, walls, ceiling -----------------------
        floor = bproc.object.create_primitive("PLANE")
        floor.set_scale([8, 8, 1])
        floor.set_location([0, 0, 0])
        floor.set_cp("category_id", 0)
        floor_mat = bproc.material.create("floor_material")
        floor_mat.set_principled_shader_value("Roughness", 0.85)
        floor_mat.set_principled_shader_value("Base Color", (0.50, 0.50, 0.50, 1.0))
        floor.replace_materials(floor_mat)

        wall_mat = bproc.material.create("wall_material")
        wall_mat.set_principled_shader_value("Base Color", (0.58, 0.58, 0.58, 1.0))
        wall_mat.set_principled_shader_value("Roughness", 0.90)

        for loc, rot in [
            ([0, -8, 4], [ np.pi / 2, 0, 0]),   # Back wall
            ([0,  8, 4], [-np.pi / 2, 0, 0]),   # Front wall
        ]:
            w = bproc.object.create_primitive("PLANE")
            w.set_scale([8, 8, 1])
            w.set_location(loc)
            w.set_rotation_euler(rot)
            w.set_cp("category_id", 0)
            w.replace_materials(wall_mat)

        ceiling = bproc.object.create_primitive("PLANE")
        ceiling.set_scale([8, 8, 1])
        ceiling.set_location([0, 0, 5.0])
        ceiling.set_rotation_euler([np.pi, 0, 0])
        ceiling.set_cp("category_id", 0)
        ceiling.replace_materials(wall_mat)

        # ---- 4. Randomised occluders (fresh every batch) --------------------
        SKU_RADIUS    = 0.15
        prim_types    = ["CUBE", "SPHERE", "CYLINDER"]
        n_occluders   = np.random.randint(N_OCCLUDERS_RANGE[0], N_OCCLUDERS_RANGE[1] + 1)
        chosen_cols   = np.random.choice(len(OCCLUDER_PALETTE), size=n_occluders, replace=False)

        for i in range(n_occluders):
            ptype    = prim_types[i % len(prim_types)]
            xy_scale = np.random.uniform(0.10, 0.14)
            if ptype == "SPHERE":
                occ   = bproc.object.create_primitive("SPHERE", scale=[xy_scale] * 3)
                occ_z = xy_scale
            else:
                z_scale = np.random.uniform(0.15, 0.30)
                occ     = bproc.object.create_primitive(ptype, scale=[xy_scale, xy_scale, z_scale])
                occ_z   = z_scale

            min_dist = SKU_RADIUS + xy_scale + 0.02
            dist     = np.random.uniform(min_dist, min_dist + 0.25)
            angle    = np.random.uniform(0, 2 * np.pi)
            occ.set_location([dist * np.cos(angle), dist * np.sin(angle), occ_z])
            occ.set_cp("category_id", 0)
            occ_mat = bproc.material.create(f"occ_mat_{i}")
            r, g, b = OCCLUDER_PALETTE[chosen_cols[i]]
            occ_mat.set_principled_shader_value("Base Color", (r, g, b, 1.0))
            occ_mat.set_principled_shader_value("Roughness", np.random.uniform(0.55, 0.90))
            occ.replace_materials(occ_mat)

        # ---- 5. Lighting ----------------------------------------------------
        for pos, energy in [
            ([2.5,  2.0, 3.5], np.random.uniform(600,  1200)),
            ([-2.0, 1.5, 3.0], np.random.uniform(300,   700)),
            ([0.0, -2.5, 3.0], np.random.uniform(250,   550)),
        ]:
            lt = bproc.types.Light()
            lt.set_type("AREA")
            lt.set_location(pos)
            lt.set_energy(energy)
            t = np.random.uniform(0.0, 1.0)
            lt.set_color([1.0, 0.9 + 0.1 * t, 0.8 + 0.2 * t])

        # ---- 6. World background + camera -----------------------------------
        bpy.data.worlds["World"].use_nodes = True
        bg = bpy.data.worlds["World"].node_tree.nodes["Background"]
        bg.inputs[0].default_value = (0.55, 0.55, 0.55, 1.0)
        bg.inputs[1].default_value = 0.1   # low ambient — keeps floor grey

        bproc.camera.set_resolution(RESOLUTION_W, RESOLUTION_H)
        poi = np.array(obj.get_location())

        for _ in range(frames_this_batch):
            az  = np.random.uniform(0, 2 * np.pi)
            el  = np.random.uniform(CAMERA_ELEV_MIN, CAMERA_ELEV_MAX)
            d   = np.random.uniform(CAMERA_DIST_MIN, CAMERA_DIST_MAX)
            x   = d * np.cos(el) * np.cos(az)
            y   = d * np.cos(el) * np.sin(az)
            z   = d * np.sin(el) + poi[2]
            cam_loc = np.array([x, y, z])
            rot_mat = bproc.camera.rotation_from_forward_vec(
                poi - cam_loc, inplane_rot=np.random.uniform(-0.1, 0.1)
            )
            bproc.camera.add_camera_pose(
                bproc.math.build_transformation_mat(cam_loc, rot_mat)
            )

        # ---- 7. Render + append COCO ----------------------------------------
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_segmentation_output(
            map_by=["instance", "class", "name"],
            default_values={"category_id": 0},
        )
        bproc.renderer.set_max_amount_of_samples(128)

        print(f"[BProc] Rendering batch {batch_idx + 1} → {OUTPUT_DIR}")
        data = bproc.renderer.render()

        bproc.writer.write_coco_annotations(
            OUTPUT_DIR,
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            color_file_format="JPEG",
        )

    print(f"\n[BProc] Done. {RENDER_COUNT} frames across {RENDER_BATCHES} scenes "
          f"→ {OUTPUT_DIR}/coco_annotations.json")


def _build_hand_occluder(coverage_fraction: float, obj_half_w: float,
                          obj_half_h: float, cam_dir_xy=None) -> list:
    """
    Construct a synthetic hand-reach occluder scaled to cover `coverage_fraction`
    of the target SKU's frontal bounding area.

    Models a palm plate + 2-4 finger cylinders for partial occlusion, or a
    full double-palm stack for ≥75 % coverage.  Randomised skin-tone materials
    ensure the model sees realistic human-hand appearance rather than generic
    geometry.

    Args:
        coverage_fraction: Fraction of SKU frontal area to obscure (0–1).
        obj_half_w:  Half-width  of the target SKU mesh (metres).
        obj_half_h:  Half-height of the target SKU mesh (metres).
        cam_dir_xy:  2-element (dx, dy) unit vector pointing FROM the SKU TOWARD
                     the camera.  The hand is placed between the SKU and this
                     direction so it actually occludes the target regardless of
                     camera azimuth.  Defaults to (0, -1) for backward compat
                     (legacy caller with camera on +Y side).

    Returns:
        List of bproc objects (palm + optional fingers) — all tagged category_id=0
        so they are excluded from COCO annotations.
    """
    occluders = []

    # Camera-relative coordinate frame
    if cam_dir_xy is None:
        cam_dir_xy = np.array([0.0, -1.0])   # legacy default: camera on +Y side
    cam_dir_xy = np.array(cam_dir_xy, dtype=float)
    cam_dir_xy /= np.linalg.norm(cam_dir_xy) + 1e-8
    # Perpendicular axis — used for lateral jitter of fingers
    perp_xy = np.array([-cam_dir_xy[1], cam_dir_xy[0]])

    # --- Palm plate ---------------------------------------------------------------
    palm_w = obj_half_w * 2.0 * np.sqrt(coverage_fraction)   # Area ∝ fraction²
    palm_h = obj_half_h * 2.0 * np.sqrt(coverage_fraction)

    palm = bproc.object.create_primitive("CUBE")
    palm.set_scale([palm_w * 0.5, 0.012, palm_h * 0.5])

    # Place palm between the SKU and the camera: offset along cam_dir_xy
    depth  = np.random.uniform(0.03, 0.10)
    jitter = np.random.uniform(-obj_half_w * 0.4, obj_half_w * 0.4)
    palm.set_location([
        cam_dir_xy[0] * depth + perp_xy[0] * jitter,
        cam_dir_xy[1] * depth + perp_xy[1] * jitter,
        obj_half_h * np.random.uniform(0.0, 0.6),
    ])
    # Rotate palm face to point toward the camera direction
    face_angle = np.arctan2(cam_dir_xy[1], cam_dir_xy[0])
    palm.set_rotation_euler([0.0, np.random.uniform(-0.15, 0.15), face_angle])
    palm.set_cp("category_id", 0)

    skin_tone = [
        (0.88, 0.73, 0.54, 1.0),  # Light
        (0.75, 0.58, 0.41, 1.0),  # Medium-light
        (0.58, 0.41, 0.26, 1.0),  # Medium-dark
        (0.36, 0.24, 0.14, 1.0),  # Dark
    ][np.random.randint(4)]

    palm_mat = bproc.material.create("hand_palm")
    palm_mat.set_principled_shader_value("Base Color", skin_tone)
    palm_mat.set_principled_shader_value("Roughness", 0.62)
    palm_mat.set_principled_shader_value("Specular", 0.2)
    palm.replace_materials(palm_mat)
    occluders.append(palm)

    # --- Finger cylinders (partial occlusion only — looks like reaching hand) -----
    if coverage_fraction < 0.65:
        n_fingers = np.random.randint(2, 5)
        for i in range(n_fingers):
            finger = bproc.object.create_primitive("CYLINDER")
            finger.set_scale([0.011, 0.011, 0.042])  # ~11 mm dia, 42 mm long
            f_lateral = palm_w * (i / max(n_fingers - 1, 1) - 0.5) * 0.8
            f_depth   = np.random.uniform(0.02, 0.07)
            finger.set_location([
                cam_dir_xy[0] * f_depth + perp_xy[0] * f_lateral,
                cam_dir_xy[1] * f_depth + perp_xy[1] * f_lateral,
                obj_half_h * 0.55 + np.random.uniform(-0.015, 0.015),
            ])
            finger.set_rotation_euler([
                np.pi / 2,                         # Orient cylinder along Z
                np.random.uniform(-0.12, 0.12),
                0.0,
            ])
            finger.set_cp("category_id", 0)
            f_mat = bproc.material.create(f"hand_finger_{i}")
            f_mat.set_principled_shader_value("Base Color", skin_tone)
            f_mat.set_principled_shader_value("Roughness", 0.58)
            finger.replace_materials(f_mat)
            occluders.append(finger)

    # --- Second palm layer for ≥75 % full-cover mode ------------------------------
    if coverage_fraction >= 0.75:
        palm2 = bproc.object.create_primitive("CUBE")
        palm2.set_scale([palm_w * 0.6, 0.015, palm_h * 0.6])
        depth2  = np.random.uniform(0.06, 0.13)
        jitter2 = np.random.uniform(-obj_half_w * 0.2, obj_half_w * 0.2)
        palm2.set_location([
            cam_dir_xy[0] * depth2 + perp_xy[0] * jitter2,
            cam_dir_xy[1] * depth2 + perp_xy[1] * jitter2,
            obj_half_h * np.random.uniform(-0.2, 0.2),
        ])
        palm2.set_cp("category_id", 0)
        p2_mat = bproc.material.create("hand_palm2")
        p2_mat.set_principled_shader_value("Base Color", skin_tone)
        p2_mat.set_principled_shader_value("Roughness", 0.60)
        palm2.replace_materials(p2_mat)
        occluders.append(palm2)

    return occluders


def render_occlusion_stress_sequences(features_path: str) -> None:
    """
    Occlusion Stress-Test Generator.

    Produces two labelled synthetic sub-datasets that specifically stress-test
    the pipeline against Zippin's known production failure mode: a shopper's
    arm/hand entering the camera frame during a pick event.

    Sub-datasets written to:
        <OUTPUT_DIR>/occlusion_stress/partial/   (PARTIAL_COVERAGE_RANGE)
        <OUTPUT_DIR>/occlusion_stress/full/      (FULL_COVERAGE_RANGE)

    Each sub-dataset contains RENDER_COUNT//2 renders with separate
    coco_annotations.json so they can be used independently for:
      - Targeted YOLOv8n fine-tuning on hard occlusion cases
      - mAP degradation benchmarking at each occlusion band
      - Sensor-fusion trigger threshold calibration

    Args:
        features_path: Path to the JSON SKU attribute file from Stage 1.
    """
    if not os.path.exists(features_path):
        print(f"[BProc:Stress] ERROR: features file not found: {features_path}")
        sys.exit(1)

    with open(features_path, "r") as f:
        attrs = json.load(f)

    n_per_mode = max(RENDER_COUNT // 2, 10)

    for mode_name, cov_range in [
        ("partial", PARTIAL_COVERAGE_RANGE),
        ("full",    FULL_COVERAGE_RANGE),
    ]:
        out_dir = os.path.join(OUTPUT_DIR, "occlusion_stress", mode_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[BProc:Stress] Rendering {n_per_mode} × '{mode_name}' frames "
              f"(coverage {cov_range[0]:.0%}–{cov_range[1]:.0%}) → {out_dir}")

        bproc.init()

        # World background: must be set after every bproc.init() — matches render()
        import bpy as _bpy
        _bpy.data.worlds["World"].use_nodes = True
        _bg = _bpy.data.worlds["World"].node_tree.nodes["Background"]
        _bg.inputs[0].default_value = (0.55, 0.55, 0.55, 1.0)
        _bg.inputs[1].default_value = 0.2   # match render() — reduced to avoid floor washout

        # --- Rebuild scene — matches render() scale so training is consistent ------
        shape = attrs.get("shape", "box").lower()
        if "cylinder" in shape or "bottle" in shape or "can" in shape:
            obj = _build_can_mesh("TargetSKU")  # Same geometry as main render
            obj.set_scale([0.15, 0.15, 0.30])
        elif "bag" in shape:
            obj = bproc.object.create_primitive("CUBE")
            obj.set_scale([0.20, 0.08, 0.30])
        else:
            obj = bproc.object.create_primitive("CUBE")
            obj.set_scale([0.15, 0.12, 0.25])

        obj.set_location([0, 0, 0.30])          # Lifted by half-height, same as main
        obj.set_name("TargetSKU")
        obj.set_cp("category_id", 1)

        mat = bproc.material.create("sku_material")
        colour_map = {
            "red": (0.8, 0.05, 0.05, 1.0), "blue": (0.05, 0.1, 0.8, 1.0),
            "green": (0.05, 0.6, 0.1, 1.0), "yellow": (0.9, 0.8, 0.02, 1.0),
            "silver": (0.75, 0.75, 0.75, 1.0), "white": (0.9, 0.9, 0.9, 1.0),
            "black": (0.05, 0.05, 0.05, 1.0), "orange": (0.9, 0.4, 0.02, 1.0),
        }
        primary_colors = attrs.get("primary_colors", [])
        rgba = colour_map.get(
            primary_colors[0].lower() if primary_colors else "",
            (0.5, 0.5, 0.5, 1.0),
        )
        mat.set_principled_shader_value("Base Color", rgba)
        mat.set_principled_shader_value("Roughness", 0.70)  # Match main render — prevents domain gap
        obj.replace_materials(mat)

        floor = bproc.object.create_primitive("PLANE")
        floor.set_scale([8, 8, 1])              # Match main render floor size
        floor.set_location([0, 0, 0])
        floor_mat = bproc.material.create("floor_material")
        floor_mat.set_principled_shader_value("Roughness", 0.85)
        floor_mat.set_principled_shader_value("Base Color", (0.50, 0.50, 0.50, 1.0))  # Match main render floor
        floor.replace_materials(floor_mat)

        # --- Single hand occluder for the entire render batch -----------------
        # BlenderProc renders all frames with the same static scene.
        # Occluder is built ONCE; all n_per_mode camera poses see it from
        # different angles (no accumulation of overlapping hands).
        obj_scale = obj.get_scale()
        obj_half_w = obj_scale[0]
        obj_half_h = obj_scale[2]
        # Pick a random viewing direction for occluder placement — the hand will
        # correctly sit between the SKU and cameras near this azimuth (~50 % of
        # the batch), rather than always being stuck on the -Y side regardless
        # of camera position.
        hand_azimuth = np.random.uniform(0, 2 * np.pi)
        hand_cam_dir = (np.cos(hand_azimuth), np.sin(hand_azimuth))
        coverage = np.random.uniform(*cov_range)
        _build_hand_occluder(coverage, obj_half_w, obj_half_h, cam_dir_xy=hand_cam_dir)

        # --- Lighting: built ONCE outside frame loop --------------------------
        # Creating lights inside the loop would add 2 new lights per iteration,
        # leaving n_per_mode*2 lights in the scene and massively overexposing
        # every frame. One set of lights illuminates the entire batch.
        for pos, base_energy in [([2.5, 2.0, 4.5], 400), ([-2.5, 1.5, 3.5], 180)]:
            light = bproc.types.Light()
            light.set_type("AREA")
            light.set_location(pos)
            light.set_energy(base_energy * np.random.uniform(0.7, 1.4))
            t = np.random.uniform(0.0, 1.0)
            light.set_color([1.0, 0.9 + 0.1 * t, 0.8 + 0.2 * t])

        # Resolution and POI don't change between frames — compute once.
        bproc.camera.set_resolution(RESOLUTION_W, RESOLUTION_H)
        poi = bproc.object.compute_poi([obj])

        for frame_idx in range(n_per_mode):
            azimuth   = np.random.uniform(0, 2 * np.pi)
            elevation = np.random.uniform(CAMERA_ELEV_MIN, CAMERA_ELEV_MAX)
            distance  = np.random.uniform(1.2, 2.2)  # Closer than standard
            x = distance * np.cos(elevation) * np.cos(azimuth)
            y = distance * np.cos(elevation) * np.sin(azimuth)
            z = distance * np.sin(elevation) + poi[2]
            cam_loc = np.array([x, y, z])
            rot_mat = bproc.camera.rotation_from_forward_vec(
                poi - cam_loc,
                inplane_rot=np.random.uniform(-0.1, 0.1),
            )
            cam2world = bproc.math.build_transformation_mat(cam_loc, rot_mat)
            bproc.camera.add_camera_pose(cam2world)

        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_segmentation_output(
            map_by=["instance", "class", "name"],
            default_values={"category_id": 0},   # Hand/floor → background, not unlabelled
        )
        bproc.renderer.set_max_amount_of_samples(96)
        data = bproc.renderer.render()

        bproc.writer.write_coco_annotations(
            out_dir,
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            color_file_format="JPEG",
        )
        print(f"[BProc:Stress] '{mode_name}' suite done → {out_dir}/coco_annotations.json")

    print(f"\n[BProc:Stress] All occlusion stress suites complete.")
    print(f"  Use partial/ suite to fine-tune on reaching-arm events.")
    print(f"  Use full/    suite to calibrate sensor-fusion fallback threshold.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: blenderproc run src/rendering/bproc_generator.py <features_path>")
        print("       Set BPROC_OCCLUSION_MODE=partial_stress|full_stress|all for stress suites.")
        sys.exit(1)

    features = sys.argv[-1]

    if OCCLUSION_MODE == "standard":
        render(features)
    elif OCCLUSION_MODE in ("partial_stress", "full_stress"):
        render_occlusion_stress_sequences(features)
    elif OCCLUSION_MODE == "all":
        render(features)                       # Standard suite first
        render_occlusion_stress_sequences(features)   # Then both stress suites
    else:
        print(f"[BProc] Unknown BPROC_OCCLUSION_MODE='{OCCLUSION_MODE}'. "
              f"Valid: standard | partial_stress | full_stress | all")
        sys.exit(1)
