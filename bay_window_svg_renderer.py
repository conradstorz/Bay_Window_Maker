from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class WindowSpec:
    """Represents one window unit used in a rendered bay layout.

    :param width: Nominal unit width in inches.
    :param height: Nominal unit height in inches.
    :param style: Human-readable style label.
    :param count: Number of repeated units in this face.
    """

    width: float
    height: float
    style: str
    count: int = 1


@dataclass(frozen=True)
class FaceSpec:
    """Represents one face of the bay frame.

    :param label: Face label such as 'left', 'center', or 'right'.
    :param finished_width: Overall finished face width in inches.
    :param rough_opening_width: Rough opening width within the face in inches.
    :param frame_depth: Front-to-back frame body depth in inches for visual rendering.
    :param window: Window specification carried by the face.
    """

    label: str
    finished_width: float
    rough_opening_width: float
    frame_depth: float
    window: WindowSpec


@dataclass(frozen=True)
class BayRenderSpec:
    """Complete rendering specification for one bay window candidate sheet.

    :param rank: Candidate rank from the calculator output.
    :param score: Candidate score from the calculator output.
    :param opening_width: Existing masonry passage opening width in inches.
    :param opening_height: Existing masonry passage opening height in inches.
    :param side_angle_deg: Side face angle relative to the wall in degrees.
    :param requested_projection_depth: Requested target projection from the constraints block.
    :param actual_projection_depth: Actual projection achieved by the candidate.
    :param wall_parallel_span: Overall span parallel to the wall in inches.
    :param outside_corner_to_corner_width: Sum of face lengths in inches.
    :param sill_height: Optional sill height above grade or floor for annotation.
    :param frame_body_depth: Visual frame body depth for rendering the top view.
    :param notes: Freeform notes to include in the drawing panel.
    :param left_face: Left face specification.
    :param center_face: Center face specification.
    :param right_face: Right face specification.
    """

    rank: int
    score: float
    opening_width: float
    opening_height: float
    side_angle_deg: float
    requested_projection_depth: float
    actual_projection_depth: float
    wall_parallel_span: float
    outside_corner_to_corner_width: float
    sill_height: float | None
    frame_body_depth: float
    notes: tuple[str, ...]
    left_face: FaceSpec
    center_face: FaceSpec
    right_face: FaceSpec


@dataclass(frozen=True)
class Point:
    """Simple 2D point in drawing space."""

    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    """Simple rectangle used for layout calculations."""

    x: float
    y: float
    width: float
    height: float


class SvgBuilder:
    """Small helper for constructing SVG markup."""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.elements: list[str] = []

    def add(self, element: str) -> None:
        """Append raw SVG markup."""

        self.elements.append(element)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = "#222",
        stroke_width: float = 1.5,
        dash: str | None = None,
    ) -> None:
        """Draw a line element."""

        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"{dash_attr} />'
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = "none",
        stroke: str = "#222",
        stroke_width: float = 1.5,
        rx: float = 0.0,
    ) -> None:
        """Draw a rectangle element."""

        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" rx="{rx:.2f}" />'
        )

    def polygon(
        self,
        points: Sequence[Point],
        *,
        fill: str = "none",
        stroke: str = "#222",
        stroke_width: float = 1.5,
    ) -> None:
        """Draw a polygon element."""

        pts = " ".join(f"{p.x:.2f},{p.y:.2f}" for p in points)
        self.add(
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.2f}" />'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 12.0,
        anchor: str = "start",
        weight: str = "normal",
        fill: str = "#222",
    ) -> None:
        """Draw a text element."""

        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size:.2f}" '
            f'font-weight="{weight}" fill="{fill}">{escape(value)}</text>'
        )

    def build(self) -> str:
        """Return the final SVG document."""

        joined = "\n  ".join(self.elements)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:.0f}" '
            f'height="{self.height:.0f}" viewBox="0 0 {self.width:.2f} {self.height:.2f}">\n'
            f'  <rect x="0" y="0" width="{self.width:.2f}" height="{self.height:.2f}" fill="white" />\n'
            f'  {joined}\n'
            f'</svg>\n'
        )


def load_specs(path: Path) -> list[BayRenderSpec]:
    """Load one or more rendering specifications from a calculator JSON file.

    The input is expected to be a list of candidates in the calculator's JSON format.
    Each candidate must include a `constraints` object so the renderer can operate as
    a pure rendering tool without making layout decisions.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of candidate objects.")

    return [_parse_candidate_spec(item) for item in data]


def _parse_candidate_spec(data: dict[str, Any]) -> BayRenderSpec:
    """Parse one calculator candidate into the renderer's internal spec."""

    constraints = data["constraints"]
    side_window = data["side_window"]
    center_window = data["center_window"]
    side_face = data["side_face"]
    center_face = data["center_face"]

    left_face = FaceSpec(
        label="left",
        finished_width=float(side_face["finished_face_width"]),
        rough_opening_width=float(side_face["rough_opening_width"]),
        frame_depth=float(constraints.get("frame_body_depth", 4.0)),
        window=WindowSpec(
            width=float(side_window["width"]),
            height=float(side_window["height"]),
            style=str(side_window.get("style", "double_hung")),
            count=1,
        ),
    )
    center_face_spec = FaceSpec(
        label="center",
        finished_width=float(center_face["finished_face_width"]),
        rough_opening_width=float(center_face["rough_opening_width"]),
        frame_depth=float(constraints.get("frame_body_depth", 4.0)),
        window=WindowSpec(
            width=float(center_window["width"]),
            height=float(center_window["height"]),
            style=str(center_window.get("style", "double_hung")),
            count=int(data["center_count"]),
        ),
    )
    right_face = FaceSpec(
        label="right",
        finished_width=float(side_face["finished_face_width"]),
        rough_opening_width=float(side_face["rough_opening_width"]),
        frame_depth=float(constraints.get("frame_body_depth", 4.0)),
        window=WindowSpec(
            width=float(side_window["width"]),
            height=float(side_window["height"]),
            style=str(side_window.get("style", "double_hung")),
            count=1,
        ),
    )

    return BayRenderSpec(
        rank=int(data.get("rank", 0)),
        score=float(data.get("score", 0.0)),
        opening_width=float(constraints["opening_width"]),
        opening_height=float(constraints["opening_height"]),
        side_angle_deg=float(data["side_angle_deg"]),
        requested_projection_depth=float(constraints["projection_depth"]),
        actual_projection_depth=float(data["projection_depth"]),
        wall_parallel_span=float(data["wall_parallel_span"]),
        outside_corner_to_corner_width=float(data["outside_corner_to_corner_width"]),
        sill_height=_optional_float(constraints.get("sill_height")),
        frame_body_depth=float(constraints.get("frame_body_depth", 4.0)),
        notes=tuple(str(item) for item in data.get("notes", [])),
        left_face=left_face,
        center_face=center_face_spec,
        right_face=right_face,
    )


def _optional_float(value: Any) -> float | None:
    """Convert a possibly-missing numeric value to float or None."""

    if value is None:
        return None
    return float(value)


def _parse_face_spec(data: dict[str, Any]) -> FaceSpec:
    """Parse one face specification object from JSON."""

    window_data = data["window"]
    window = WindowSpec(
        width=float(window_data["width"]),
        height=float(window_data["height"]),
        style=str(window_data.get("style", "double_hung")),
        count=int(window_data.get("count", 1)),
    )
    return FaceSpec(
        label=str(data["label"]),
        finished_width=float(data["finished_width"]),
        rough_opening_width=float(data["rough_opening_width"]),
        frame_depth=float(data.get("frame_depth", 4.0)),
        window=window,
    )


def rotate_point(point: Point, angle_deg: float) -> Point:
    """Rotate a point around the origin by angle degrees."""

    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return Point(
        x=(point.x * cos_a) - (point.y * sin_a),
        y=(point.x * sin_a) + (point.y * cos_a),
    )


def translate_points(points: Sequence[Point], dx: float, dy: float) -> tuple[Point, ...]:
    """Translate a sequence of points."""

    return tuple(Point(x=point.x + dx, y=point.y + dy) for point in points)


def face_polygon(
    length: float,
    depth: float,
    rotation_deg: float,
    anchor: Point,
    anchor_mode: str,
) -> tuple[Point, ...]:
    """Return one rotated rectangle polygon for a bay face in top view."""

    if anchor_mode == "center":
        local = (
            Point(-length / 2.0, 0.0),
            Point(length / 2.0, 0.0),
            Point(length / 2.0, depth),
            Point(-length / 2.0, depth),
        )
    elif anchor_mode == "left":
        local = (
            Point(0.0, 0.0),
            Point(length, 0.0),
            Point(length, depth),
            Point(0.0, depth),
        )
    elif anchor_mode == "right":
        local = (
            Point(-length, 0.0),
            Point(0.0, 0.0),
            Point(0.0, depth),
            Point(-length, depth),
        )
    else:
        raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")

    rotated = tuple(rotate_point(point, rotation_deg) for point in local)
    return translate_points(rotated, anchor.x, anchor.y)


def bounds_from_points(groups: Sequence[Sequence[Point]]) -> tuple[float, float, float, float]:
    """Return min_x, min_y, max_x, max_y for nested point groups."""

    xs = [point.x for group in groups for point in group]
    ys = [point.y for group in groups for point in group]
    return min(xs), min(ys), max(xs), max(ys)


def center_of_polygon(points: Sequence[Point]) -> Point:
    """Return the arithmetic center of a polygon's points."""

    return Point(
        x=sum(point.x for point in points) / len(points),
        y=sum(point.y for point in points) / len(points),
    )


def draw_dimension(
    svg: SvgBuilder,
    start: Point,
    end: Point,
    offset: float,
    label: str,
    orientation: str,
) -> None:
    """Draw a simple horizontal or vertical dimension line with arrows."""

    arrow = 6.0
    if orientation == "horizontal":
        y = start.y + offset
        svg.line(start.x, start.y, start.x, y, stroke="#666", stroke_width=1.0)
        svg.line(end.x, end.y, end.x, y, stroke="#666", stroke_width=1.0)
        svg.line(start.x, y, end.x, y, stroke="#666", stroke_width=1.0)
        svg.line(start.x, y, start.x + arrow, y - 3.0, stroke="#666", stroke_width=1.0)
        svg.line(start.x, y, start.x + arrow, y + 3.0, stroke="#666", stroke_width=1.0)
        svg.line(end.x, y, end.x - arrow, y - 3.0, stroke="#666", stroke_width=1.0)
        svg.line(end.x, y, end.x - arrow, y + 3.0, stroke="#666", stroke_width=1.0)
        svg.text((start.x + end.x) / 2.0, y - 6.0, label, anchor="middle", size=11.0)
        return

    if orientation == "vertical":
        x = start.x + offset
        svg.line(start.x, start.y, x, start.y, stroke="#666", stroke_width=1.0)
        svg.line(end.x, end.y, x, end.y, stroke="#666", stroke_width=1.0)
        svg.line(x, start.y, x, end.y, stroke="#666", stroke_width=1.0)
        svg.line(x, start.y, x - 3.0, start.y + arrow, stroke="#666", stroke_width=1.0)
        svg.line(x, start.y, x + 3.0, start.y + arrow, stroke="#666", stroke_width=1.0)
        svg.line(x, end.y, x - 3.0, end.y - arrow, stroke="#666", stroke_width=1.0)
        svg.line(x, end.y, x + 3.0, end.y - arrow, stroke="#666", stroke_width=1.0)
        svg.text(x + 8.0, (start.y + end.y) / 2.0, label, size=11.0)
        return

    raise ValueError(f"Unsupported dimension orientation: {orientation}")


def draw_top_view(svg: SvgBuilder, spec: BayRenderSpec, panel: Rect) -> None:
    """Draw the top view panel."""

    svg.rect(panel.x, panel.y, panel.width, panel.height, stroke="#888", stroke_width=1.2)
    svg.text(panel.x + 10.0, panel.y + 22.0, "Top View / Plan", weight="bold", size=16.0)

    margin = 40.0
    draw_width = panel.width - (2.0 * margin)
    draw_height = panel.height - 80.0
    base_x = panel.x + (panel.width / 2.0)
    base_y = panel.y + 75.0

    geometry_extent = max(spec.wall_parallel_span, spec.outside_corner_to_corner_width, spec.opening_width)
    geometry_depth = spec.actual_projection_depth + max(
        spec.left_face.frame_depth,
        spec.center_face.frame_depth,
        spec.right_face.frame_depth,
    ) + 12.0
    scale = min(draw_width / max(geometry_extent, 1.0), draw_height / max(geometry_depth, 1.0))

    half_center = spec.center_face.finished_width / 2.0
    center_poly = face_polygon(
        length=spec.center_face.finished_width,
        depth=spec.center_face.frame_depth,
        rotation_deg=0.0,
        anchor=Point(0.0, 0.0),
        anchor_mode="center",
    )
    left_poly = face_polygon(
        length=spec.left_face.finished_width,
        depth=spec.left_face.frame_depth,
        rotation_deg=-spec.side_angle_deg,
        anchor=Point(-half_center, 0.0),
        anchor_mode="right",
    )
    right_poly = face_polygon(
        length=spec.right_face.finished_width,
        depth=spec.right_face.frame_depth,
        rotation_deg=spec.side_angle_deg,
        anchor=Point(half_center, 0.0),
        anchor_mode="left",
    )

    min_x, min_y, max_x, max_y = bounds_from_points((center_poly, left_poly, right_poly))
    width_span = max_x - min_x
    height_span = max_y - min_y

    # Center the geometry in the panel.
    offset_x = base_x - (((min_x + max_x) / 2.0) * scale)
    offset_y = base_y - (min_y * scale)

    def tx(point: Point) -> Point:
        return Point(x=(point.x * scale) + offset_x, y=(point.y * scale) + offset_y)

    center_poly_t = tuple(tx(point) for point in center_poly)
    left_poly_t = tuple(tx(point) for point in left_poly)
    right_poly_t = tuple(tx(point) for point in right_poly)

    wall_y = offset_y - (16.0 * scale)
    opening_left = offset_x - ((spec.opening_width / 2.0) * scale)
    opening_right = offset_x + ((spec.opening_width / 2.0) * scale)

    svg.line(panel.x + 12.0, wall_y, panel.x + panel.width - 12.0, wall_y, stroke="#111", stroke_width=2.0)
    svg.line(opening_left, wall_y, opening_right, wall_y, stroke="white", stroke_width=4.0)
    svg.text((opening_left + opening_right) / 2.0, wall_y - 8.0, "Masonry opening", anchor="middle", size=11.0)

    svg.polygon(left_poly_t, fill="#f3f3f3")
    svg.polygon(center_poly_t, fill="#e5e5e5")
    svg.polygon(right_poly_t, fill="#f3f3f3")

    left_label = center_of_polygon(left_poly_t)
    center_label = center_of_polygon(center_poly_t)
    right_label = center_of_polygon(right_poly_t)
    svg.text(left_label.x, left_label.y, f"Left {spec.left_face.window.width:g}\"", anchor="middle", size=11.0)
    svg.text(
        center_label.x,
        center_label.y,
        f"Center {spec.center_face.window.count} x {spec.center_face.window.width:g}\"",
        anchor="middle",
        size=11.0,
    )
    svg.text(right_label.x, right_label.y, f"Right {spec.right_face.window.width:g}\"", anchor="middle", size=11.0)

    draw_dimension(
        svg,
        Point(opening_left, wall_y),
        Point(opening_right, wall_y),
        offset=-28.0,
        label=f'Opening {spec.opening_width:.2f}"',
        orientation="horizontal",
    )

    all_top = left_poly_t + center_poly_t + right_poly_t
    min_tx = min(point.x for point in all_top)
    max_tx = max(point.x for point in all_top)
    max_ty = max(point.y for point in all_top)
    draw_dimension(
        svg,
        Point(min_tx, max_ty),
        Point(max_tx, max_ty),
        offset=26.0,
        label=f'Span {spec.wall_parallel_span:.2f}"',
        orientation="horizontal",
    )
    draw_dimension(
        svg,
        Point(max_tx, wall_y),
        Point(max_tx, wall_y + (spec.actual_projection_depth * scale)),
        offset=22.0,
        label=f'Actual projection {spec.actual_projection_depth:.2f}"',
        orientation="vertical",
    )
    svg.text(panel.x + panel.width - 12.0, panel.y + 22.0, f"Angle {spec.side_angle_deg:g}°", anchor="end", size=12.0)
    svg.text(max_tx + 34.0, wall_y + (spec.actual_projection_depth * scale) + 18.0, f"Requested {spec.requested_projection_depth:.2f}\"", size=10.5)


def _draw_window_divisions(
    svg: SvgBuilder,
    face_rect: Rect,
    count: int,
    frame_margin: float,
    mullion_width: float,
) -> None:
    """Draw equal repeated windows within a face elevation panel."""

    clear_width = face_rect.width - (2.0 * frame_margin) - ((count - 1) * mullion_width)
    unit_width = clear_width / count
    for index in range(count):
        x = face_rect.x + frame_margin + index * (unit_width + mullion_width)
        svg.rect(x, face_rect.y + frame_margin, unit_width, face_rect.height - (2.0 * frame_margin), stroke="#333", stroke_width=1.2)
        sash_split_y = face_rect.y + (face_rect.height / 2.0)
        svg.line(x, sash_split_y, x + unit_width, sash_split_y, stroke="#666", stroke_width=1.0)


def draw_front_elevation(svg: SvgBuilder, spec: BayRenderSpec, panel: Rect) -> None:
    """Draw the front elevation panel with three face elevations."""

    svg.rect(panel.x, panel.y, panel.width, panel.height, stroke="#888", stroke_width=1.2)
    svg.text(panel.x + 10.0, panel.y + 22.0, "Front Elevation", weight="bold", size=16.0)

    margin_x = 28.0
    margin_top = 50.0
    available_width = panel.width - (2.0 * margin_x)
    available_height = panel.height - 80.0

    total_face_width = spec.left_face.finished_width + spec.center_face.finished_width + spec.right_face.finished_width
    scale = min(available_width / max(total_face_width, 1.0), available_height / max(spec.opening_height, 1.0))

    left_w = spec.left_face.finished_width * scale
    center_w = spec.center_face.finished_width * scale
    right_w = spec.right_face.finished_width * scale
    frame_h = spec.opening_height * scale

    start_x = panel.x + (panel.width - (left_w + center_w + right_w)) / 2.0
    y = panel.y + margin_top

    left_rect = Rect(start_x, y, left_w, frame_h)
    center_rect = Rect(start_x + left_w, y, center_w, frame_h)
    right_rect = Rect(start_x + left_w + center_w, y, right_w, frame_h)

    for rect in (left_rect, center_rect, right_rect):
        svg.rect(rect.x, rect.y, rect.width, rect.height, fill="#fafafa")

    frame_margin = 8.0
    mullion_width = 6.0
    _draw_window_divisions(svg, left_rect, spec.left_face.window.count, frame_margin, mullion_width)
    _draw_window_divisions(svg, center_rect, spec.center_face.window.count, frame_margin, mullion_width)
    _draw_window_divisions(svg, right_rect, spec.right_face.window.count, frame_margin, mullion_width)

    svg.text(left_rect.x + (left_rect.width / 2.0), y - 8.0, "Left face", anchor="middle", size=11.0)
    svg.text(center_rect.x + (center_rect.width / 2.0), y - 8.0, "Center face", anchor="middle", size=11.0)
    svg.text(right_rect.x + (right_rect.width / 2.0), y - 8.0, "Right face", anchor="middle", size=11.0)

    base_y = y + frame_h
    svg.line(panel.x + 16.0, base_y, panel.x + panel.width - 16.0, base_y, stroke="#111", stroke_width=1.8)

    draw_dimension(
        svg,
        Point(left_rect.x, base_y),
        Point(right_rect.x + right_rect.width, base_y),
        offset=26.0,
        label=f'Corner-to-corner {spec.outside_corner_to_corner_width:.2f}"',
        orientation="horizontal",
    )
    draw_dimension(
        svg,
        Point(right_rect.x + right_rect.width, y),
        Point(right_rect.x + right_rect.width, base_y),
        offset=20.0,
        label=f'Height {spec.opening_height:.2f}"',
        orientation="vertical",
    )

    if spec.sill_height is not None:
        svg.text(panel.x + panel.width - 12.0, panel.y + panel.height - 12.0, f'Sill height {spec.sill_height:.2f}"', anchor="end", size=11.0)


def draw_side_elevation(svg: SvgBuilder, spec: BayRenderSpec, panel: Rect) -> None:
    """Draw a simplified right-side elevation panel."""

    svg.rect(panel.x, panel.y, panel.width, panel.height, stroke="#888", stroke_width=1.2)
    svg.text(panel.x + 10.0, panel.y + 22.0, "Side Elevation", weight="bold", size=16.0)

    margin_x = 34.0
    margin_top = 50.0
    available_width = panel.width - (2.0 * margin_x)
    available_height = panel.height - 84.0

    depth_for_elevation = max(spec.actual_projection_depth, spec.right_face.frame_depth)
    scale = min(available_width / max(depth_for_elevation, 1.0), available_height / max(spec.opening_height, 1.0))

    frame_w = spec.actual_projection_depth * scale
    frame_h = spec.opening_height * scale

    x = panel.x + (panel.width - frame_w) / 2.0
    y = panel.y + margin_top

    outer = Rect(x, y, frame_w, frame_h)
    svg.rect(outer.x, outer.y, outer.width, outer.height, fill="#fafafa")

    frame_margin = 8.0
    svg.rect(
        outer.x + frame_margin,
        outer.y + frame_margin,
        max(outer.width - (2.0 * frame_margin), 1.0),
        max(outer.height - (2.0 * frame_margin), 1.0),
        stroke="#333",
        stroke_width=1.2,
    )
    sash_split_y = outer.y + (outer.height / 2.0)
    svg.line(outer.x + frame_margin, sash_split_y, outer.x + outer.width - frame_margin, sash_split_y, stroke="#666", stroke_width=1.0)

    base_y = outer.y + outer.height
    svg.line(panel.x + 16.0, base_y, panel.x + panel.width - 16.0, base_y, stroke="#111", stroke_width=1.8)

    draw_dimension(
        svg,
        Point(outer.x, base_y),
        Point(outer.x + outer.width, base_y),
        offset=24.0,
        label=f'Projection {spec.actual_projection_depth:.2f}"',
        orientation="horizontal",
    )
    draw_dimension(
        svg,
        Point(outer.x + outer.width, outer.y),
        Point(outer.x + outer.width, base_y),
        offset=20.0,
        label=f'Height {spec.opening_height:.2f}"',
        orientation="vertical",
    )
    svg.text(panel.x + (panel.width / 2.0), panel.y + panel.height - 14.0, f'Requested projection = {spec.requested_projection_depth:.2f}\" | side angle = {spec.side_angle_deg:g}°', anchor="middle", size=10.5)


def draw_notes_panel(svg: SvgBuilder, spec: BayRenderSpec, panel: Rect) -> None:
    """Draw a notes and data summary panel."""

    svg.rect(panel.x, panel.y, panel.width, panel.height, stroke="#888", stroke_width=1.2)
    svg.text(panel.x + 10.0, panel.y + 22.0, "Data / Notes", weight="bold", size=16.0)

    lines = [
        f'Rank: {spec.rank}',
        f'Score: {spec.score:.2f}',
        f'Opening: {spec.opening_width:.2f}" W x {spec.opening_height:.2f}" H',
        f'Side angle: {spec.side_angle_deg:g}°',
        f'Requested projection: {spec.requested_projection_depth:.2f}"',
        f'Actual projection: {spec.actual_projection_depth:.2f}"',
        f'Wall-parallel span: {spec.wall_parallel_span:.2f}"',
        f'Corner-to-corner width: {spec.outside_corner_to_corner_width:.2f}"',
        f'Left face width: {spec.left_face.finished_width:.2f}"',
        f'Center face width: {spec.center_face.finished_width:.2f}"',
        f'Right face width: {spec.right_face.finished_width:.2f}"',
        f'Center windows: {spec.center_face.window.count} x {spec.center_face.window.width:g}" {spec.center_face.window.style.replace("_", " ")}',
        f'Side windows: {spec.left_face.window.width:g}" {spec.left_face.window.style.replace("_", " ")}',
    ]
    if spec.sill_height is not None:
        lines.append(f'Sill height: {spec.sill_height:.2f}"')

    y = panel.y + 46.0
    for line in lines:
        svg.text(panel.x + 12.0, y, line, size=11.0)
        y += 16.0

    if spec.notes:
        y += 8.0
        svg.text(panel.x + 12.0, y, "Notes:", size=12.0, weight="bold")
        y += 16.0
        for note in spec.notes:
            svg.text(panel.x + 18.0, y, f"- {note}", size=10.8)
            y += 15.0


def render_svg(spec: BayRenderSpec, page_width: float = 1400.0, page_height: float = 1000.0) -> str:
    """Render a multi-view SVG sheet for the given bay specification."""

    svg = SvgBuilder(page_width, page_height)
    svg.text(24.0, 34.0, f"Bay Window Layout Sheet - Candidate {spec.rank}", size=24.0, weight="bold")
    svg.text(24.0, 56.0, f"Standalone SVG renderer | score {spec.score:.2f}", size=12.0, fill="#444")

    top_panel = Rect(24.0, 80.0, 780.0, 420.0)
    front_panel = Rect(824.0, 80.0, 552.0, 420.0)
    side_panel = Rect(24.0, 530.0, 520.0, 420.0)
    notes_panel = Rect(564.0, 530.0, 812.0, 420.0)

    draw_top_view(svg, spec, top_panel)
    draw_front_elevation(svg, spec, front_panel)
    draw_side_elevation(svg, spec, side_panel)
    draw_notes_panel(svg, spec, notes_panel)

    return svg.build()


def build_example_specs() -> list[BayRenderSpec]:
    """Return built-in example candidates for testing the renderer quickly."""

    return [
        BayRenderSpec(
            rank=1,
            score=93.66,
            opening_width=72.0,
            opening_height=36.0,
            side_angle_deg=30.0,
            requested_projection_depth=16.0,
            actual_projection_depth=21.0,
            wall_parallel_span=175.24613391789285,
            outside_corner_to_corner_width=186.5,
            sill_height=None,
            frame_body_depth=4.0,
            notes=(
                "Example data for renderer development.",
                "Center face uses two equal double-hung units.",
                "The calculator can later export this same JSON structure.",
            ),
            left_face=FaceSpec(
                label="left",
                finished_width=42.0,
                rough_opening_width=39.0,
                frame_depth=4.0,
                window=WindowSpec(width=38.0, height=36.0, style="double_hung", count=1),
            ),
            center_face=FaceSpec(
                label="center",
                finished_width=102.5,
                rough_opening_width=98.0,
                frame_depth=4.0,
                window=WindowSpec(width=48.0, height=36.0, style="double_hung", count=2),
            ),
            right_face=FaceSpec(
                label="right",
                finished_width=42.0,
                rough_opening_width=39.0,
                frame_depth=4.0,
                window=WindowSpec(width=38.0, height=36.0, style="double_hung", count=1),
            ),
        )
    ]


def write_example_json(path: Path) -> Path:
    """Write an example JSON candidate list for the renderer."""

    example = [
        {
            "side_window": {
                "width": 38.0,
                "height": 36.0,
                "style": "double_hung"
            },
            "center_window": {
                "width": 48.0,
                "height": 36.0,
                "style": "double_hung"
            },
            "center_count": 2,
            "side_angle_deg": 30.0,
            "side_face": {
                "window_area_width": 38.0,
                "rough_opening_width": 39.0,
                "finished_face_width": 42.0
            },
            "center_face": {
                "window_area_width": 96.0,
                "rough_opening_width": 98.0,
                "finished_face_width": 102.5
            },
            "projection_depth": 21.0,
            "wall_parallel_span": 175.24613391789285,
            "outside_corner_to_corner_width": 186.5,
            "center_face_projection_contribution": 0.0,
            "side_face_projection_contribution": 21.0,
            "passage_overlap_width": 72.0,
            "passage_overlap_fraction": 1.0,
            "score": 93.66,
            "notes": [
                "2 center units reduce the need for an extra-wide single sash.",
                "Center window units are wider than the side units, which usually looks balanced.",
                "Center face covers most of the masonry passage width."
            ],
            "rank": 1,
            "constraints": {
                "opening_width": 72.0,
                "opening_height": 36.0,
                "sill_height": null,
                "projection_depth": 16.0,
                "side_angle_deg": 30.0,
                "frame_thickness": 1.5,
                "rough_clearance": 0.5,
                "mullion_thickness": 1.5,
                "face_trim_allowance": 0.0,
                "frame_body_depth": 4.0,
                "min_passage_fraction": 0.65,
                "min_unit_width": 18.0,
                "max_single_unit_width": 48.0
            }
        }
    ]
    path.write_text(json.dumps(example, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Render one SVG sheet per bay candidate from a calculator JSON file.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Path to the calculator JSON candidate list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where timestamped SVG files should be written.",
    )
    parser.add_argument(
        "--write-example-json",
        type=Path,
        help="Write an example calculator-style JSON file to this path and exit unless --input-json is also provided.",
    )
    parser.add_argument(
        "--use-example",
        action="store_true",
        help="Render using a built-in example candidate list.",
    )
    return parser.parse_args()


def _timestamp_slug() -> str:
    """Return a timestamp string suitable for filenames."""

    from datetime import datetime

    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _candidate_output_name(spec: BayRenderSpec, timestamp_slug: str) -> str:
    """Return the SVG filename for one rendered candidate."""

    return f"bay_candidate_{spec.rank:02d}_{timestamp_slug}.svg"


def main() -> None:
    """Run the standalone renderer CLI."""

    args = parse_args()

    if args.write_example_json:
        args.write_example_json.parent.mkdir(parents=True, exist_ok=True)
        write_example_json(args.write_example_json)
        if not args.input_json and not args.use_example:
            print(f"Example JSON written to: {args.write_example_json}")
            return

    if args.use_example:
        specs = build_example_specs()
    elif args.input_json:
        specs = load_specs(args.input_json)
    else:
        raise ValueError("Provide --input-json or --use-example.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = _timestamp_slug()

    for spec in specs:
        svg_text = render_svg(spec)
        output_path = args.output_dir / _candidate_output_name(spec, timestamp_slug)
        output_path.write_text(svg_text, encoding="utf-8")
        print(f"SVG written to: {output_path}")


if __name__ == "__main__":
    main()
