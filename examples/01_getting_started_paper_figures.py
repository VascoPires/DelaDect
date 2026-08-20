"""Create paper-ready final-frame figures for the Getting Started example.

The 3D rendering follows the earlier PyVista specimen-view workflow from the
``DelaDect-Dev`` repository.  The 2D exports follow its high-resolution paper
overlay workflow.  Colours come from the light theme in the sibling
``my_plots`` repository, and every output has an opaque white background.

Run from the repository root after ``examples/01_getting_started.py``::

    python examples/01_getting_started_paper_figures.py

The script writes separate high-resolution PNG and TIFF versions of the
full-frame crack detection and full-frame delamination classification, plus a
high-resolution PNG 3D laminate rendering of those same detections, to
``results/01-getting-started/paper_figures``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageColor, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
MY_PLOTS_ROOT = REPO_ROOT.parent / "my_plots"
if str(MY_PLOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(MY_PLOTS_ROOT))

from plot_style import get_colors  # noqa: E402


DATA_ROOT = REPO_ROOT / "example_images" / "sample-1"
FULL_RESULTS_ROOT = REPO_ROOT / "results" / "01-getting-started-full"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "01-getting-started" / "paper_figures"

FRAME_KEY_TEMPLATE = "frame_{:04d}"
SCALE_PX_PER_MM = 31.953

# Manual paper-visualization cleanup for frame 0004. These full-frame 0-degree
# detections lie within 25 px of an upper/lower long image boundary and run
# parallel to that boundary. Cracks that merely end near a short edge remain.
# Keep the source NPZ and the 2D detection figures untouched.
LAST_FRAME_3D_EXCLUDED_CRACK_0_INDICES = (
    0,
    4,
    6,
    7,
    9,
    11,
    12,
    14,
    16,
    17,
    18,
    20,
    21,
)


def _hex_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(value) for value in ImageColor.getrgb(color))


def _rgba_float(color: str, alpha: float) -> tuple[float, float, float, float]:
    rgb = np.asarray(_hex_rgb(color), dtype=float) / 255.0
    return float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha)


def _load_npz_frame(path: Path, frame_index: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required result bundle does not exist: {path}")
    key = FRAME_KEY_TEMPLATE.format(frame_index)
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise KeyError(f"{path} does not contain {key}; available keys: {payload.files}")
        return np.asarray(payload[key])


def _frame_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as payload:
        return len([key for key in payload.files if key.startswith("frame_")])


def _resolve_frame_index(requested: int) -> int:
    combined_path = FULL_RESULTS_ROOT / "delamination" / "both" / "masks" / "combined.npz"
    count = _frame_count(combined_path)
    if count == 0:
        raise RuntimeError(f"No frame data found in {combined_path}")
    index = count - 1 if requested < 0 else requested
    if not 0 <= index < count:
        raise IndexError(f"frame_index={index} is outside the available range 0..{count - 1}")
    return index


def _sorted_full_frames() -> list[Path]:
    frames = sorted((DATA_ROOT / "full").glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {DATA_ROOT / 'full'}")
    return frames


def _load_cracks(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    crack_root = FULL_RESULTS_ROOT / "cracks"
    crack_0 = _load_npz_frame(
        crack_root / "ply_ply_0" / "data" / "01-getting-started-full_ply_0_cracks.npz",
        frame_index,
    )
    crack_90 = _load_npz_frame(
        crack_root / "ply_ply_90" / "data" / "01-getting-started-full_ply_90_cracks.npz",
        frame_index,
    )
    return crack_0.reshape((-1, 2, 2)), crack_90.reshape((-1, 2, 2))


def _load_masks(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    mask_root = FULL_RESULTS_ROOT / "delamination" / "both" / "masks"
    edge = _load_npz_frame(mask_root / "edge_exclusion.npz", frame_index).astype(bool)
    diffuse = _load_npz_frame(mask_root / "diffuse_final.npz", frame_index).astype(bool)
    return edge, diffuse


def _blend_mask(base: np.ndarray, mask: np.ndarray, color: str, alpha: float) -> None:
    rgb = np.asarray(_hex_rgb(color), dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    base[selected] = (1.0 - alpha) * base[selected] + alpha * rgb


def _draw_cracks(
    image: Image.Image,
    cracks: np.ndarray,
    *,
    color: str,
    scale: int,
    width_native_px: float,
) -> None:
    draw = ImageDraw.Draw(image, mode="RGB")
    line_width = max(1, int(round(width_native_px * scale)))
    for segment in np.asarray(cracks, dtype=float):
        row0, col0 = segment[0]
        row1, col1 = segment[1]
        draw.line(
            (
                float(col0 * scale),
                float(row0 * scale),
                float(col1 * scale),
                float(row1 * scale),
            ),
            fill=_hex_rgb(color),
            width=line_width,
        )


def _save_raster(image: Image.Image, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, dpi=(dpi, dpi), optimize=True)
    image.convert("RGB").save(
        path.with_suffix(".tiff"),
        dpi=(dpi, dpi),
        compression="tiff_lzw",
    )


def export_2d_figures(
    *,
    frame_index: int,
    output_dir: Path,
    raster_scale: int,
    dpi: int,
    colors: dict[str, str],
) -> tuple[Path, Path]:
    frame_paths = _sorted_full_frames()
    if frame_index >= len(frame_paths):
        raise IndexError(f"Only {len(frame_paths)} raw frames are available")

    raw_native = Image.open(frame_paths[frame_index]).convert("L").convert("RGB")
    native_width, native_height = raw_native.size
    size = (native_width * raster_scale, native_height * raster_scale)
    edge, diffuse = _load_masks(frame_index)
    if edge.shape != (native_height, native_width) or diffuse.shape != edge.shape:
        raise ValueError(
            "The final-frame masks must match the raw full frame: "
            f"raw={(native_height, native_width)}, edge={edge.shape}, diffuse={diffuse.shape}"
        )

    crack_0, crack_90 = _load_cracks(frame_index)

    crack_detection = raw_native.resize(size, resample=Image.Resampling.LANCZOS)
    _draw_cracks(
        crack_detection,
        crack_0,
        color="#E0B928",
        scale=raster_scale,
        width_native_px=1.5,
    )
    _draw_cracks(
        crack_detection,
        crack_90,
        color=colors["blue"],
        scale=raster_scale,
        width_native_px=1.5,
    )
    crack_path = output_dir / "getting_started_full_crack_detection.png"
    _save_raster(crack_detection, crack_path, dpi)

    base = np.asarray(raw_native, dtype=np.float32)
    _blend_mask(base, diffuse, colors["green"], alpha=0.38)
    _blend_mask(base, edge, colors["red"], alpha=0.38)
    overlay = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    overlay = overlay.resize(size, resample=Image.Resampling.LANCZOS)

    _draw_cracks(
        overlay,
        crack_0,
        color="#E0B928",
        scale=raster_scale,
        width_native_px=1.5,
    )
    _draw_cracks(
        overlay,
        crack_90,
        color=colors["blue"],
        scale=raster_scale,
        width_native_px=1.5,
    )

    overlay_path = output_dir / "getting_started_full_delamination_classification.png"
    _save_raster(overlay, overlay_path, dpi)
    return crack_path, overlay_path


def _segment_prism(
    segment: np.ndarray,
    *,
    frame_shape: tuple[int, int],
    width_mm: float,
    height_mm: float,
    z0: float,
    z1: float,
    crack_width_mm: float,
    pv: object,
) -> object:
    frame_h, frame_w = frame_shape
    row0, col0 = (float(value) for value in segment[0])
    row1, col1 = (float(value) for value in segment[1])

    def to_world(row: float, col: float) -> tuple[float, float]:
        x = (col / frame_w - 0.5) * width_mm
        y = (0.5 - row / frame_h) * height_mm
        return x, y

    x0, y0 = to_world(row0, col0)
    x1, y1 = to_world(row1, col1)
    p0 = np.asarray([x0, y0, z0], dtype=np.float32)
    p1 = np.asarray([x1, y1, z0], dtype=np.float32)
    direction = p1[:2] - p0[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        direction = np.asarray([1.0, 0.0], dtype=np.float32)
        norm = 1.0
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float32) / norm
    offset = np.append(perpendicular * (crack_width_mm / 2.0), 0.0)

    bottom = np.vstack([p0 + offset, p0 - offset, p1 - offset, p1 + offset])
    top = bottom + np.asarray([0.0, 0.0, z1 - z0], dtype=np.float32)
    points = np.vstack([bottom, top])
    faces = np.hstack(
        [
            [4, 0, 1, 2, 3],
            [4, 4, 7, 6, 5],
            [4, 0, 4, 5, 1],
            [4, 1, 5, 6, 2],
            [4, 2, 6, 7, 3],
            [4, 3, 7, 4, 0],
        ]
    )
    return pv.PolyData(points, faces)


def _mask_texture(mask: np.ndarray, color: str, alpha: float, pv: object) -> object:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgb = _hex_rgb(color)
    rgba[np.asarray(mask, dtype=bool)] = (*rgb, int(round(255 * alpha)))
    return pv.Texture(rgba)


def _add_mask_plane(
    plotter: object,
    mask: np.ndarray,
    *,
    color: str,
    alpha: float,
    width_mm: float,
    height_mm: float,
    z: float,
    pv: object,
) -> None:
    plane = pv.Plane(
        center=(0.0, 0.0, z),
        direction=(0.0, 0.0, 1.0),
        i_size=width_mm,
        j_size=height_mm,
        i_resolution=1,
        j_resolution=1,
    )
    plane.texture_map_to_plane(inplace=True)
    plotter.add_mesh(
        plane,
        texture=_mask_texture(mask, color, alpha, pv),
        lighting=False,
    )


def _finalize_3d_png(image_path: Path, *, dpi: int, transparent: bool) -> None:
    """Store the legend-free PyVista render with the requested alpha mode."""
    with Image.open(image_path) as rendered:
        finalized = rendered.convert("RGBA" if transparent else "RGB")
    finalized.save(image_path, dpi=(dpi, dpi), optimize=True)


def _camera_payload(plotter: object) -> dict[str, object]:
    position, focal_point, viewup = plotter.camera_position
    return {
        "position": [float(value) for value in position],
        "focal_point": [float(value) for value in focal_point],
        "viewup": [float(value) for value in viewup],
        "parallel_projection": bool(plotter.camera.parallel_projection),
        "parallel_scale": float(plotter.camera.parallel_scale),
        "view_angle": float(plotter.camera.view_angle),
        "clipping_range": [float(value) for value in plotter.camera.clipping_range],
    }


def _apply_camera_payload(plotter: object, payload: dict[str, object]) -> None:
    required = ("position", "focal_point", "viewup")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Camera JSON is missing required keys: {missing}")
    plotter.camera_position = (
        payload["position"],
        payload["focal_point"],
        payload["viewup"],
    )
    plotter.camera.parallel_projection = bool(payload.get("parallel_projection", True))
    if "parallel_scale" in payload:
        plotter.camera.parallel_scale = float(payload["parallel_scale"])
    if "view_angle" in payload:
        plotter.camera.view_angle = float(payload["view_angle"])
    if "clipping_range" in payload:
        plotter.camera.clipping_range = tuple(float(value) for value in payload["clipping_range"])


def export_3d_figure(
    *,
    frame_index: int,
    output_dir: Path,
    window_size: tuple[int, int],
    screenshot_scale: float,
    crack_width_scale: float,
    crack_opacity: float,
    camera_azimuth: float,
    camera_elevation: float,
    camera_json: Path | None,
    interactive_3d: bool,
    transparent_background: bool,
    dpi: int,
    colors: dict[str, str],
) -> Path:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("The 3D paper figure requires pyvista: pip install pyvista") from exc

    frame_paths = _sorted_full_frames()
    raw = Image.open(frame_paths[frame_index])
    frame_width, frame_height = raw.size
    frame_shape = (frame_height, frame_width)
    width_mm = frame_width / SCALE_PX_PER_MM
    height_mm = frame_height / SCALE_PX_PER_MM

    crack_0, crack_90 = _load_cracks(frame_index)
    if frame_index == _frame_count(
        FULL_RESULTS_ROOT / "delamination" / "both" / "masks" / "combined.npz"
    ) - 1:
        crack_0 = np.delete(
            crack_0,
            LAST_FRAME_3D_EXCLUDED_CRACK_0_INDICES,
            axis=0,
        )
    edge, diffuse = _load_masks(frame_index)

    # The physical plies are separated slightly to make the internal interface
    # visible. The gap is explicitly a display choice, not physical geometry.
    ply_thickness = 1.0
    ply_gap = 1.4
    ply_ranges = ((0.0, ply_thickness), (ply_thickness + ply_gap, 2.0 * ply_thickness + ply_gap))
    interface_z = ply_thickness + ply_gap / 2.0

    plotter = pv.Plotter(off_screen=not interactive_3d, window_size=window_size)
    plotter.set_background("white")
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    plotter.camera.parallel_projection = False

    ply_rgb = np.asarray(_hex_rgb(colors["warm"]), dtype=float) / 255.0
    for z0, z1 in ply_ranges:
        mesh = pv.Box((-width_mm / 2, width_mm / 2, -height_mm / 2, height_mm / 2, z0, z1))
        plotter.add_mesh(
            mesh,
            color=tuple(ply_rgb),
            opacity=0.22,
            show_edges=True,
            edge_color=colors["muted"],
            line_width=1.2,
            lighting=True,
            smooth_shading=False,
        )

    crack_width_mm = (14.0 / SCALE_PX_PER_MM) * max(0.01, float(crack_width_scale))
    for cracks, ply_range, color in (
        (crack_0, ply_ranges[1], "#E0B928"),
        (crack_90, ply_ranges[0], colors["blue"]),
    ):
        rgb = np.asarray(_hex_rgb(color), dtype=float) / 255.0
        for segment in cracks:
            mesh = _segment_prism(
                segment,
                frame_shape=frame_shape,
                width_mm=width_mm,
                height_mm=height_mm,
                z0=ply_range[0],
                z1=ply_range[1],
                crack_width_mm=crack_width_mm,
                pv=pv,
            )
            plotter.add_mesh(
                mesh,
                color=tuple(rgb),
                opacity=float(np.clip(crack_opacity, 0.0, 1.0)),
                lighting=False,
            )

    _add_mask_plane(
        plotter,
        diffuse,
        color=colors["green"],
        alpha=0.72,
        width_mm=width_mm,
        height_mm=height_mm,
        z=interface_z + 0.015,
        pv=pv,
    )
    _add_mask_plane(
        plotter,
        edge,
        color=colors["red"],
        alpha=0.78,
        width_mm=width_mm,
        height_mm=height_mm,
        z=interface_z + 0.030,
        pv=pv,
    )

    if camera_json is not None:
        camera_path = camera_json.resolve()
        if not camera_path.exists():
            raise FileNotFoundError(f"Camera JSON does not exist: {camera_path}")
        _apply_camera_payload(plotter, json.loads(camera_path.read_text(encoding="utf-8")))
    else:
        # A shallow perspective keeps the long specimen readable while exposing
        # both plies and the internal damage plane.
        plotter.camera_position = (
            (70.0, -82.0, 48.0),
            (0.0, 0.0, interface_z),
            (0.0, 0.0, 1.0),
        )
        if camera_azimuth:
            plotter.camera.azimuth += float(camera_azimuth)
        if camera_elevation:
            plotter.camera.elevation += float(camera_elevation)
        plotter.reset_camera()
        plotter.camera.zoom(1.08)

    output_path = output_dir / "getting_started_3d_last_frame.png"
    camera_output_path = output_dir / "getting_started_3d_camera.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def print_camera() -> None:
        print(json.dumps(_camera_payload(plotter), indent=2))

    def save_current_view() -> None:
        plotter.render()
        plotter.screenshot(
            str(output_path),
            window_size=window_size,
            scale=max(1.0, float(screenshot_scale)),
            transparent_background=transparent_background,
        )
        _finalize_3d_png(
            output_path,
            dpi=dpi,
            transparent=transparent_background,
        )
        camera_output_path.write_text(
            json.dumps(_camera_payload(plotter), indent=2),
            encoding="utf-8",
        )
        print(f"Saved 3D view: {output_path}")
        print(f"Saved camera: {camera_output_path}")

    if interactive_3d:
        plotter.add_key_event("p", print_camera)
        plotter.add_key_event("s", save_current_view)
        print("Interactive 3D controls:")
        print("  Left drag: rotate | Middle drag: pan | Wheel/right drag: zoom")
        print("  P: print camera JSON | S: save figure and camera JSON")
        print("  Closing the window also saves the current view.")
        plotter.show(title="Getting Started paper view — two plies, one interface", auto_close=False)
        save_current_view()
    else:
        plotter.show(auto_close=False)
        save_current_view()
    plotter.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Zero-based result frame index; negative selects the last frame (default).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--raster-scale",
        type=int,
        default=2,
        help="Integer enlargement for 2D paper rasters (default: 2, producing 4422x1248 px).",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--render-width", type=int, default=2400)
    parser.add_argument("--render-height", type=int, default=1350)
    parser.add_argument("--screenshot-scale", type=float, default=2.0)
    parser.add_argument(
        "--crack-width-scale",
        type=float,
        default=0.35,
        help="Scale applied to the physical 3D crack width (default: 0.35).",
    )
    parser.add_argument(
        "--crack-opacity",
        type=float,
        default=0.60,
        help="Opacity of the 3D crack prisms from 0 to 1 (default: 0.60).",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=0.0,
        help="Horizontal camera rotation in degrees around the specimen.",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=0.0,
        help="Vertical camera rotation in degrees around the specimen.",
    )
    parser.add_argument(
        "--camera-json",
        type=Path,
        default=None,
        help="Reload an exact camera state previously saved by the interactive viewer.",
    )
    parser.add_argument(
        "--interactive-3d",
        action="store_true",
        help="Open a rotatable PyVista window; P prints and S saves the camera state.",
    )
    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="Save the 3D PNG with a transparent alpha background.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raster_scale < 1:
        raise ValueError("--raster-scale must be at least 1")
    colors = get_colors("light")
    frame_index = _resolve_frame_index(args.frame_index)
    output_dir = args.output_dir.resolve()

    crack_path, classification_path = export_2d_figures(
        frame_index=frame_index,
        output_dir=output_dir,
        raster_scale=args.raster_scale,
        dpi=args.dpi,
        colors=colors,
    )
    figure_3d_path = export_3d_figure(
        frame_index=frame_index,
        output_dir=output_dir,
        window_size=(args.render_width, args.render_height),
        screenshot_scale=args.screenshot_scale,
        crack_width_scale=args.crack_width_scale,
        crack_opacity=args.crack_opacity,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        camera_json=args.camera_json,
        interactive_3d=args.interactive_3d,
        transparent_background=args.transparent_background,
        dpi=args.dpi,
        colors=colors,
    )

    print(f"Frame index: {frame_index}")
    print(f"Full-frame crack detection: {crack_path}")
    print(f"Full-frame delamination classification: {classification_path}")
    print(f"3D figure: {figure_3d_path}")


if __name__ == "__main__":
    main()
