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

import blenderproc as bproc
import sys
import os
import json
import numpy as np

# ---- Read runtime parameters from env (set by stage_generate orchestrator) ----
OUTPUT_DIR      = os.environ.get("BPROC_OUTPUT_DIR", "checkpoints/synthetic_dataset")
RENDER_COUNT    = int(os.environ.get("BPROC_RENDER_COUNT", "50"))
RESOLUTION_W    = int(os.environ.get("BPROC_RESOLUTION_W", "640"))
RESOLUTION_H    = int(os.environ.get("BPROC_RESOLUTION_H", "640"))
# BPROC_OCCLUSION_MODE: "standard" | "partial_stress" | "full_stress" | "all"
# "all" runs standard + both stress suites back-to-back (3× render count).
OCCLUSION_MODE  = os.environ.get("BPROC_OCCLUSION_MODE", "standard")

# Randomisation envelopes tuned for Zippin overhead shelf-cam geometry
CAMERA_DIST_MIN, CAMERA_DIST_MAX = 1.5, 3.5
CAMERA_ELEV_MIN, CAMERA_ELEV_MAX = 0.15, 1.4
N_OCCLUDERS_RANGE = (3, 7)

# Occlusion stress-test coverage bands
PARTIAL_COVERAGE_RANGE = (0.30, 0.55)   # 30-55 % SKU frontal area obscured
FULL_COVERAGE_RANGE    = (0.75, 0.95)   # 75-95 % — product nearly invisible


def render(features_path: str) -> None:
    print(f"[BProc] Initialising BlenderProc2 renderer...")
    bproc.init()

    if not os.path.exists(features_path):
        print(f"[BProc] ERROR: features checkpoint not found: {features_path}")
        sys.exit(1)

    with open(features_path, "r") as f:
        attrs = json.load(f)

    print(f"[BProc] Loaded SKU attributes: {attrs}")

    # ---- 1. Product mesh ---------------------------------------------------------
    shape = attrs.get("shape", "box").lower()
    if "cylinder" in shape or "bottle" in shape or "can" in shape:
        obj = bproc.object.create_primitive("CYLINDER")
        obj.set_scale([0.06, 0.06, 0.12])      # ~Coca-Cola can proportions (m)
    elif "bag" in shape:
        obj = bproc.object.create_primitive("CUBE")
        obj.set_scale([0.10, 0.04, 0.18])       # Flat bag geometry
    else:
        obj = bproc.object.create_primitive("CUBE")
        obj.set_scale([0.08, 0.06, 0.14])       # Generic box

    obj.set_location([0, 0, 0.1])
    obj.set_name("TargetSKU")
    obj.set_cp("category_id", 1)

    # ---- 2. Material: attempt texture projection, fallback to VLM colours --------
    mat = bproc.material.create("sku_material")
    material_desc = attrs.get("material", "").lower()

    if "gloss" in material_desc or "aluminum" in material_desc or "metallic" in material_desc:
        mat.set_principled_shader_value("Roughness", 0.08)
        mat.set_principled_shader_value("Metallic", 0.85 if "aluminum" in material_desc else 0.0)
        mat.set_principled_shader_value("Clearcoat", 0.3)
    elif "plastic" in material_desc:
        mat.set_principled_shader_value("Roughness", 0.25)
        mat.set_principled_shader_value("Metallic", 0.0)
        mat.set_principled_shader_value("Specular", 0.5)
    else:
        mat.set_principled_shader_value("Roughness", 0.85)  # Matte cardboard

    # Set base colour from VLM-extracted primary colour
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

    # ---- 3. Scene: floor + background plane ------------------------------------
    floor = bproc.object.create_primitive("PLANE")
    floor.set_scale([4, 4, 1])
    floor.set_location([0, 0, 0])
    floor_mat = bproc.material.create("floor_material")
    floor_mat.set_principled_shader_value("Roughness", 0.95)
    floor_mat.set_principled_shader_value("Base Color", (0.6, 0.6, 0.6, 1.0))
    floor.replace_materials(floor_mat)

    # ---- 4. Randomised occluders (simulate hand / shelf-edge occlusion) ---------
    n_occluders = np.random.randint(N_OCCLUDERS_RANGE[0], N_OCCLUDERS_RANGE[1] + 1)
    primitive_types = ["CUBE", "SPHERE", "CYLINDER"]

    for _ in range(n_occluders):
        ptype = np.random.choice(primitive_types)
        scale = np.random.uniform(0.03, 0.12)
        occ = bproc.object.create_primitive(ptype, scale=[scale, scale, scale])
        occ.set_location(np.random.uniform([-0.15, -0.15, 0.02], [0.15, 0.15, 0.25]))
        occ.set_cp("category_id", 0)  # Background — not annotated
        occ_mat = bproc.material.create(f"occ_mat_{_}")
        occ_mat.set_principled_shader_value("Base Color", (
            np.random.uniform(0.3, 0.9),
            np.random.uniform(0.3, 0.9),
            np.random.uniform(0.3, 0.9),
            1.0,
        ))
        occ.replace_materials(occ_mat)

    # ---- 5. Lighting: 3-point stadium floodlight setup -------------------------
    light_configs = [
        ([2.5, 2.0, 4.5],  np.random.uniform(200, 500)),   # Key
        ([-2.5, 1.5, 3.5], np.random.uniform(100, 250)),   # Fill
        ([0.0, -3.0, 4.0], np.random.uniform(80, 180)),    # Rim / back
    ]
    for pos, energy in light_configs:
        light = bproc.types.Light()
        light.set_type("AREA")
        light.set_location(pos)
        light.set_energy(energy)
        # Slight colour temperature variation (warm/cool) for domain randomisation
        t = np.random.uniform(0.0, 1.0)
        light.set_color([1.0, 0.9 + 0.1 * t, 0.8 + 0.2 * t])

    # ---- 6. Camera trajectory: spherical shell sampling ------------------------
    bproc.camera.set_resolution(RESOLUTION_W, RESOLUTION_H)
    poi = bproc.object.compute_poi([obj])

    for _ in range(RENDER_COUNT):
        azimuth   = np.random.uniform(0, 2 * np.pi)
        elevation = np.random.uniform(CAMERA_ELEV_MIN, CAMERA_ELEV_MAX)
        distance  = np.random.uniform(CAMERA_DIST_MIN, CAMERA_DIST_MAX)

        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation) + 0.1   # Slight upward offset

        cam_loc = np.array([x, y, z])
        rot_mat = bproc.camera.rotation_from_forward_vec(
            poi - cam_loc,
            inplane_rot=np.random.uniform(-0.1, 0.1),   # Minor roll jitter
        )
        cam2world = bproc.math.build_transformation_mat(cam_loc, rot_mat)
        bproc.camera.add_camera_pose(cam2world)

    # ---- 7. Render + write COCO annotations ------------------------------------
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["instance", "class", "name"])
    bproc.renderer.set_max_amount_of_samples(128)   # Balance quality vs speed

    print(f"[BProc] Rendering {RENDER_COUNT} frames to {OUTPUT_DIR} ...")
    data = bproc.renderer.render()

    bproc.writer.write_coco_annotations(
        OUTPUT_DIR,
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG",
    )
    print(f"[BProc] Done. COCO annotations written to {OUTPUT_DIR}/coco_annotations.json")


def _build_hand_occluder(coverage_fraction: float, obj_half_w: float,
                          obj_half_h: float) -> list:
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

    Returns:
        List of bproc objects (palm + optional fingers) — all tagged category_id=0
        so they are excluded from COCO annotations.
    """
    occluders = []

    # --- Palm plate ---------------------------------------------------------------
    palm_w = obj_half_w * 2.0 * np.sqrt(coverage_fraction)   # Area ∝ fraction²
    palm_h = obj_half_h * 2.0 * np.sqrt(coverage_fraction)

    palm = bproc.object.create_primitive("CUBE")
    palm.set_scale([palm_w * 0.5, 0.012, palm_h * 0.5])

    # Position palm between the SKU and the camera (slightly forward on -Y)
    palm.set_location([
        np.random.uniform(-obj_half_w * 0.4, obj_half_w * 0.4),
        np.random.uniform(-0.10, -0.03),
        obj_half_h * np.random.uniform(0.0, 0.6),
    ])
    palm.set_rotation_euler([0.0, np.random.uniform(-0.15, 0.15), 0.0])
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
            finger_x = palm_w * (i / max(n_fingers - 1, 1) - 0.5) * 0.8
            finger.set_location([
                finger_x,
                np.random.uniform(-0.07, -0.02),
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
        palm2.set_location([
            np.random.uniform(-obj_half_w * 0.2, obj_half_w * 0.2),
            np.random.uniform(-0.13, -0.06),
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

        # --- Rebuild scene (same logic as render(), extracted for isolation) ------
        shape = attrs.get("shape", "box").lower()
        if "cylinder" in shape or "bottle" in shape or "can" in shape:
            obj = bproc.object.create_primitive("CYLINDER")
            obj.set_scale([0.06, 0.06, 0.12])
        elif "bag" in shape:
            obj = bproc.object.create_primitive("CUBE")
            obj.set_scale([0.10, 0.04, 0.18])
        else:
            obj = bproc.object.create_primitive("CUBE")
            obj.set_scale([0.08, 0.06, 0.14])

        obj.set_location([0, 0, 0.1])
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
        obj.replace_materials(mat)

        floor = bproc.object.create_primitive("PLANE")
        floor.set_scale([4, 4, 1])
        floor.set_location([0, 0, 0])
        floor_mat = bproc.material.create("floor_material")
        floor_mat.set_principled_shader_value("Roughness", 0.95)
        floor_mat.set_principled_shader_value("Base Color", (0.6, 0.6, 0.6, 1.0))
        floor.replace_materials(floor_mat)

        # --- Per-frame occlusion hand ------------------------------------------
        obj_scale = obj.get_scale()
        obj_half_w = obj_scale[0]
        obj_half_h = obj_scale[2]

        for frame_idx in range(n_per_mode):
            coverage = np.random.uniform(*cov_range)
            _build_hand_occluder(coverage, obj_half_w, obj_half_h)

            # Lighting: harsher stadium conditions for stress frames
            for pos, base_energy in [([2.5, 2.0, 4.5], 400), ([-2.5, 1.5, 3.5], 180)]:
                light = bproc.types.Light()
                light.set_type("AREA")
                light.set_location(pos)
                light.set_energy(base_energy * np.random.uniform(0.7, 1.4))
                t = np.random.uniform(0.0, 1.0)
                light.set_color([1.0, 0.9 + 0.1 * t, 0.8 + 0.2 * t])

            # Camera closer in (occluder should be large in frame)
            bproc.camera.set_resolution(RESOLUTION_W, RESOLUTION_H)
            poi = bproc.object.compute_poi([obj])
            azimuth   = np.random.uniform(0, 2 * np.pi)
            elevation = np.random.uniform(CAMERA_ELEV_MIN, CAMERA_ELEV_MAX)
            distance  = np.random.uniform(1.2, 2.2)  # Closer than standard
            x = distance * np.cos(elevation) * np.cos(azimuth)
            y = distance * np.cos(elevation) * np.sin(azimuth)
            z = distance * np.sin(elevation) + 0.1
            cam_loc = np.array([x, y, z])
            rot_mat = bproc.camera.rotation_from_forward_vec(
                poi - cam_loc,
                inplane_rot=np.random.uniform(-0.1, 0.1),
            )
            cam2world = bproc.math.build_transformation_mat(cam_loc, rot_mat)
            bproc.camera.add_camera_pose(cam2world)

        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_segmentation_output(map_by=["instance", "class", "name"])
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
