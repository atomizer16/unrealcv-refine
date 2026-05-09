"""Generate an RGB-Depth dataset from CineCameraActor cameras via UnrealCV.

Workflow:
1. Query all scene objects (``vget /objects``).
2. Keep only objects whose class is ``CineCameraActor``.
3. For each CineCameraActor, move one UnrealCV sensor camera to its pose.
4. Save RGB and Depth pairs to one sub-folder per CineCameraActor.

Example:
    python examples/generate_mono_depth_dataset.py \
        --ip 127.0.0.1 --port 9000 --output ./datasets/rgb_depth
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from unrealcv import Client
from unrealcv.util import read_npy, read_png


def _normalize_uclass_name(raw_uclass: str) -> str:
    """Normalize Unreal class strings for robust matching.

    Unreal can return class names in forms like:
    - ``CineCameraActor``
    - ``Class /Script/CinematicCamera.CineCameraActor``
    - ``/Script/CinematicCamera.CineCameraActor``
    """
    if not raw_uclass:
        return ""
    normalized = raw_uclass.strip()
    if "CineCameraActor" in normalized:
        return "CineCameraActor"
    if "." in normalized:
        normalized = normalized.split(".")[-1]
    if " " in normalized:
        normalized = normalized.split(" ")[-1]
    return normalized


def _sanitize_name(name: str) -> str:
    """Convert actor names to filesystem-safe folder names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "camera"


def _assert_unrealcv_ok(response: str, context: str) -> None:
    """Raise a clear error when UnrealCV command execution fails."""
    text = (response or "").strip().lower()
    if text.startswith("error") or text.startswith("failed"):
        raise RuntimeError(f"UnrealCV command failed in {context}: {response}")


def _list_camera_ids(client: Client) -> List[int]:
    """Return available UnrealCV camera IDs inferred from ``vget /cameras`` output."""
    cameras_raw = client.request("vget /cameras")
    _assert_unrealcv_ok(cameras_raw, "listing cameras")
    camera_names = [v for v in cameras_raw.split() if v.strip()]
    return list(range(len(camera_names)))


def _resolve_capture_camera_id(client: Client, capture_camera_id: int) -> int:
    """Resolve a usable camera ID; optionally spawn a dedicated camera when needed."""
    camera_ids = _list_camera_ids(client)
    if capture_camera_id >= 0:
        if capture_camera_id not in camera_ids:
            raise RuntimeError(
                f"Requested --capture-camera-id={capture_camera_id}, "
                f"but available camera ids are {camera_ids}."
            )
        return capture_camera_id

    # Auto mode: spawn a new camera to avoid depending on current PIE/player camera.
    before_count = len(camera_ids)
    spawn_resp = client.request("vset /cameras/spawn")
    _assert_unrealcv_ok(spawn_resp, "spawning camera")
    after_ids = _list_camera_ids(client)
    if len(after_ids) <= before_count:
        raise RuntimeError(
            "Failed to auto-create a new camera with `vset /cameras/spawn`. "
            "Please pass --capture-camera-id manually."
        )
    return len(after_ids) - 1


def get_cine_camera_actors(client: Client) -> List[str]:
    """Return scene actor names whose UClass is ``CineCameraActor``."""
    scene_objects = client.request("vget /objects").split()
    cine_cameras: List[str] = []

    for actor_name in scene_objects:
        uclass = client.request(f"vget /object/{actor_name}/uclass_name")
        if _normalize_uclass_name(uclass) == "CineCameraActor":
            cine_cameras.append(actor_name)

    return cine_cameras


def capture_rgb_depth_dataset(
    client: Client,
    output_dir: Path,
    frame_count: int,
    rgb_ext: str,
    capture_camera_id: int,
    depth_max_meters: float,
) -> int:
    """Capture RGB/Depth pairs for each CineCameraActor actor.

    Returns:
        int: Number of selected CineCameraActor cameras.
    """
    cine_cameras = get_cine_camera_actors(client)
    if not cine_cameras:
        raise RuntimeError(
            "No CineCameraActor found from `vget /objects`. "
            "Please ensure your level contains CineCameraActor instances."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_camera_id = _resolve_capture_camera_id(client, capture_camera_id)

    for actor_idx, actor_name in enumerate(cine_cameras):
        cam_folder = output_dir / f"camera_{actor_idx:03d}_{_sanitize_name(actor_name)}"
        rgb_folder = cam_folder / "rgb"
        depth_folder = cam_folder / "depth"
        rgb_folder.mkdir(parents=True, exist_ok=True)
        depth_folder.mkdir(parents=True, exist_ok=True)

        location = client.request(f"vget /object/{actor_name}/location")
        rotation = client.request(f"vget /object/{actor_name}/rotation")
        _assert_unrealcv_ok(location, f"querying location for actor={actor_name}")
        _assert_unrealcv_ok(rotation, f"querying rotation for actor={actor_name}")

        set_loc_resp = client.request(
            f"vset /camera/{resolved_camera_id}/location {location}"
        )
        set_rot_resp = client.request(
            f"vset /camera/{resolved_camera_id}/rotation {rotation}"
        )
        _assert_unrealcv_ok(set_loc_resp, f"setting camera location for actor={actor_name}")
        _assert_unrealcv_ok(set_rot_resp, f"setting camera rotation for actor={actor_name}")

        metadata = {
            "actor_name": actor_name,
            "capture_camera_id": resolved_camera_id,
            "capture_mode": "move_unrealcv_camera_to_cinecameraactor_pose",
            "location": [float(v) for v in location.split()],
            "rotation_pitch_yaw_roll": [float(v) for v in rotation.split()],
            "frame_count": frame_count,
            "depth_encoding": "png_uint16",
            "depth_max_meters": depth_max_meters,
        }
        (cam_folder / "meta.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for frame_idx in range(frame_count):
            rgb_path = rgb_folder / f"{frame_idx:06d}.{rgb_ext}"
            depth_path = depth_folder / f"{frame_idx:06d}.png"

            rgb_bytes = client.request(f"vget /camera/{resolved_camera_id}/lit png")
            depth_bytes = client.request(f"vget /camera/{resolved_camera_id}/depth npy")

            rgb_img = read_png(rgb_bytes)
            depth_arr = read_npy(depth_bytes)
            if rgb_img is None:
                raise RuntimeError(
                    f"Failed to decode RGB for actor {actor_name} at frame {frame_idx}."
                )
            if depth_arr is None:
                raise RuntimeError(
                    f"Failed to decode Depth for actor {actor_name} at frame {frame_idx}."
                )

            if rgb_ext == "png":
                Image.fromarray(rgb_img).save(rgb_path)
            else:
                Image.fromarray(rgb_img).save(rgb_path.with_suffix(".bmp"))

            # Unreal depth is in centimeters; convert to meters first.
            depth_m = depth_arr.astype(np.float32) / 100.0
            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=depth_max_meters, neginf=0.0)
            depth_m = np.clip(depth_m, 0.0, depth_max_meters)

            # Encode to 16-bit PNG (0..65535), preserving depth continuity.
            depth_u16 = np.round((depth_m / depth_max_meters) * 65535.0).astype(np.uint16)
            Image.fromarray(depth_u16, mode="I;16").save(depth_path)

        print(
            f"[OK] cine_actor={actor_name}, capture_camera={resolved_camera_id}, "
            f"frames={frame_count}, folder={cam_folder}"
        )

    return len(cine_cameras)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate monocular depth RGB-Depth dataset from UE5 CineCameraActor cameras using UnrealCV."
    )
    parser.add_argument("--ip", default="127.0.0.1", help="UnrealCV server IP")
    parser.add_argument("--port", type=int, default=9000, help="UnrealCV server port")
    parser.add_argument(
        "--output",
        default="./rgb_depth_dataset",
        help="Output root folder. Each camera gets one sub-folder.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of frames captured per camera (default: 1).",
    )
    parser.add_argument(
        "--rgb-ext",
        choices=["png", "bmp"],
        default="png",
        help="RGB image file extension used by UnrealCV save command.",
    )
    parser.add_argument(
        "--capture-camera-id",
        type=int,
        default=-1,
        help=(
            "UnrealCV camera id used for capture. Use -1 (default) to auto-spawn "
            "a dedicated camera, avoiding dependence on current PIE viewport camera."
        ),
    )
    parser.add_argument(
        "--depth-max-meters",
        type=float,
        default=15.0,
        help="Depth clipping distance for 16-bit PNG encoding. Larger values keep farther range.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be >= 1")
    if args.depth_max_meters <= 0:
        raise ValueError("--depth-max-meters must be > 0")

    client = Client((args.ip, args.port))
    client.connect()

    if not client.isconnected():
        raise RuntimeError(
            f"Failed to connect UnrealCV server at {args.ip}:{args.port}. "
            "Please make sure UE is running with UnrealCV plugin enabled."
        )

    camera_count = capture_rgb_depth_dataset(
        client=client,
        output_dir=Path(args.output),
        frame_count=args.frames,
        rgb_ext=args.rgb_ext,
        capture_camera_id=args.capture_camera_id,
        depth_max_meters=args.depth_max_meters,
    )
    print(f"Dataset generation finished. CineCameraActor count: {camera_count}")


if __name__ == "__main__":
    main()