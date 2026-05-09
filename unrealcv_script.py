# ue_build_scene_and_preview_rig.py
# 用法：在 UE 编辑器中运行（Tools -> Execute Python Script）

import math
import unreal

FBX_FILE = r"C:/Users/ASUS/Documents/blender/steel_frame.fbx"          # 修改为你的模型路径
DEST_PATH = "/Game/ImportedDataset"        # 导入到内容浏览器的位置

OBJECT_LOCATION = unreal.Vector(0.0, 0.0, 0.0)
OBJECT_SCALE = unreal.Vector(1.0, 1.0, 1.0)

CENTER = unreal.Vector(0.0, 0.0, 50.0)    # 球心 / 目标中心（cm）
RADIUS_CM = 600.0                          # <<< 半径修改位置
NUM_CAMERAS = 64                           # <<< 相机数量修改位置
USE_CINE_CAMERA = True
NAME_PREFIX = "SphCam"

def fibonacci_sphere(n, radius, center):
    pts = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (2.0 * i) / max(1, n - 1)
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        pts.append(unreal.Vector(
            center.x + radius * x,
            center.y + radius * y,
            center.z + radius * z
        ))
    return pts

def import_asset(filename, dest_path):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", filename)
    task.set_editor_property("destination_path", dest_path)
    task.set_editor_property("automated", True)
    task.set_editor_property("save", True)
    task.set_editor_property("replace_existing", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

    imported = task.get_editor_property("imported_object_paths")
    if not imported:
        raise RuntimeError("资产导入失败，请检查路径和导入器。")

    asset = unreal.EditorAssetLibrary.load_asset(imported[0])
    if asset is None:
        raise RuntimeError(f"无法加载导入资产: {imported[0]}")
    return asset

def spawn_target_mesh(asset):
    actor_subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = actor_subsys.spawn_actor_from_object(asset, OBJECT_LOCATION, unreal.Rotator())
    actor.set_actor_label("DatasetObject")
    actor.set_actor_scale3d(OBJECT_SCALE)
    return actor

def spawn_preview_cameras(points):
    actor_subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cam_class = unreal.CineCameraActor if USE_CINE_CAMERA else unreal.CameraActor
    cams = []

    for idx, loc in enumerate(points):
        rot = unreal.MathLibrary.find_look_at_rotation(loc, CENTER)
        cam = actor_subsys.spawn_actor_from_class(cam_class, loc, rot)
        cam.set_actor_label(f"{NAME_PREFIX}_{idx:03d}")

        # CameraActor 与 CineCameraActor 都可通过 camera_component 设置基础 FOV
        comp = cam.get_editor_property("camera_component")
        comp.set_editor_property("field_of_view", 90.0)
        comp.set_editor_property("aspect_ratio", 16.0 / 9.0)
        comp.set_editor_property("post_process_blend_weight", 0.0)

        cams.append(cam)
    return cams

mesh_asset = import_asset(FBX_FILE, DEST_PATH)
mesh_actor = spawn_target_mesh(mesh_asset)
points = fibonacci_sphere(NUM_CAMERAS, RADIUS_CM, CENTER)
cams = spawn_preview_cameras(points)

print(f"导入资产: {mesh_asset.get_path_name()}")
print(f"目标 Actor: {mesh_actor.get_actor_label()}")
print(f"相机数量: {len(cams)}")
print(f"球半径(cm): {RADIUS_CM}")

