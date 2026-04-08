from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class WindowUnit:
    """Represents a stock window unit.

    :param width: Nominal unit width in inches.
    :param height: Nominal unit height in inches.
    :param style: Human-readable style label.
    """

    width: float
    height: float
    style: str = "double_hung"

    def label(self) -> str:
        """Return a compact display label for the window unit."""
        return f'{self.width:g}" x {self.height:g}" {self.style.replace("_", " ")}'


@dataclass(frozen=True)
class BayConstraints:
    """Defines the search or verification constraints for the bay layout.

    Geometry model:
    - The bay sits entirely outside the building.
    - The masonry opening is treated as a passage opening behind the bay.
    - The center face is square to the building.
    - The side faces are symmetric and set at a fixed angle.
    - All window units have the same height.

    Framing model:
    - Each face has an outer frame thickness at both ends.
    - Each window unit has a rough opening allowance on both sides.
    - Adjacent center windows are separated by mullion posts.
    - A small face trim allowance can be included for practical fitting.

    :param opening_width: Existing masonry opening width in inches.
    :param opening_height: Existing masonry opening height in inches.
    :param projection_depth: Desired bay projection from the wall in inches.
    :param side_angle_deg: Side face angle in degrees relative to the building wall.
    :param frame_thickness: Thickness of outer framing members/posts in inches.
    :param rough_clearance: Rough opening allowance per side of each window in inches.
    :param mullion_thickness: Thickness of interior mullion posts between center windows.
    :param face_trim_allowance: Additional allowance per face for trim/shimming in inches.
    :param min_center_units: Minimum number of center window units to consider.
    :param max_center_units: Maximum number of center window units to consider.
    :param min_passage_fraction: Minimum fraction of masonry opening width the center face should cover.
    :param max_single_unit_width: Maximum acceptable width for a single window unit.
    :param min_unit_width: Minimum acceptable width for a single window unit.
    :param sill_height: Optional finished sill height above the floor in inches.
    :param frame_body_depth: Depth of the window frame body in inches. Used for SVG plan rendering.
    :param max_wall_overlap: Maximum allowed per-side overhang of the bay beyond the masonry opening
        in inches. ``None`` means no limit is enforced.
    """

    opening_width: float
    opening_height: float
    projection_depth: float
    side_angle_deg: float
    frame_thickness: float = 1.5
    rough_clearance: float = 0.5
    mullion_thickness: float = 1.5
    face_trim_allowance: float = 0.0
    min_center_units: int = 1
    max_center_units: int = 4
    min_passage_fraction: float = 0.65
    max_single_unit_width: float = 48.0
    min_unit_width: float = 18.0
    sill_height: float | None = None
    frame_body_depth: float = 4.0
    max_wall_overlap: float | None = None


@dataclass(frozen=True)
class FaceDimensions:
    """Calculated dimensions for one face of the bay.

    :param window_area_width: Sum of nominal window widths in the face.
    :param rough_opening_width: Width required for rough openings in the face.
    :param finished_face_width: Overall framed face width in inches.
    """

    window_area_width: float
    rough_opening_width: float
    finished_face_width: float


@dataclass(frozen=True)
class BayCandidate:
    """Represents one valid bay layout candidate."""

    side_window: WindowUnit
    center_window: WindowUnit
    center_count: int
    side_face: FaceDimensions
    center_face: FaceDimensions
    side_angle_deg: float
    projection_depth: float
    wall_parallel_span: float
    outside_corner_to_corner_width: float
    center_face_projection_contribution: float
    side_face_projection_contribution: float
    passage_overlap_width: float
    passage_overlap_fraction: float
    wall_overlap: float
    score: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FacePolygon:
    """Simple 2D polygon for one bay face in the SVG top view.

    Points are stored in drawing coordinates where +x points right and +y points down.
    """

    label: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class BaySvgLayout:
    """Holds the generated SVG text and supporting geometry data."""

    svg_text: str
    width: float
    height: float
    view_box: str
    faces: tuple[FacePolygon, ...]


def build_default_stock_windows(height: float) -> list[WindowUnit]:
    """Return a practical default list of stock double-hung windows.

    :param height: Desired common window height in inches.
    :return: List of stock window definitions.
    """

    widths = [18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
    return [WindowUnit(width=float(width), height=height) for width in widths]


def load_stock_windows_from_json(path: Path) -> list[WindowUnit]:
    """Load stock window definitions from a JSON file.

    Expected structure:
    [
      {"width": 24, "height": 36, "style": "double_hung"},
      ...
    ]

    :param path: Path to a JSON file.
    :return: Parsed list of window units.
    :raises ValueError: If the file contents are invalid.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Stock window JSON must contain a list of window definitions.")

    windows: list[WindowUnit] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Each stock window entry must be an object.")
        try:
            windows.append(
                WindowUnit(
                    width=float(entry["width"]),
                    height=float(entry["height"]),
                    style=str(entry.get("style", "double_hung")),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Missing required key in stock window entry: {exc}") from exc

    return windows


def calculate_single_window_face(
    window: WindowUnit,
    constraints: BayConstraints,
) -> FaceDimensions:
    """Calculate finished width for a one-window face.

    The formula assumes:
    - rough opening allowance on both sides of the window
    - one outer frame/post at each end of the face
    - one optional face trim allowance applied once per face

    :param window: Window unit used in the face.
    :param constraints: Bay framing constraints.
    :return: Face dimension summary.
    """

    rough_opening_width = window.width + (2.0 * constraints.rough_clearance)
    finished_face_width = (
        rough_opening_width
        + (2.0 * constraints.frame_thickness)
        + constraints.face_trim_allowance
    )
    return FaceDimensions(
        window_area_width=window.width,
        rough_opening_width=rough_opening_width,
        finished_face_width=finished_face_width,
    )


def calculate_center_face(
    window: WindowUnit,
    center_count: int,
    constraints: BayConstraints,
) -> FaceDimensions:
    """Calculate finished width for the center face.

    The formula assumes:
    - `center_count` equal-width windows in the center face
    - rough opening allowance on each side of each window
    - one outer frame/post at each end of the face
    - mullion posts between adjacent windows
    - one optional face trim allowance applied once per face

    :param window: Window unit used repeatedly in the center face.
    :param center_count: Number of center windows.
    :param constraints: Bay framing constraints.
    :return: Face dimension summary.
    """

    if center_count < 1:
        raise ValueError("Center window count must be at least 1.")

    rough_opening_width = center_count * (window.width + (2.0 * constraints.rough_clearance))
    mullion_total = max(0, center_count - 1) * constraints.mullion_thickness
    finished_face_width = (
        rough_opening_width
        + mullion_total
        + (2.0 * constraints.frame_thickness)
        + constraints.face_trim_allowance
    )
    return FaceDimensions(
        window_area_width=center_count * window.width,
        rough_opening_width=rough_opening_width,
        finished_face_width=finished_face_width,
    )


def calculate_wall_parallel_span(
    center_face_width: float,
    side_face_width: float,
    side_angle_deg: float,
) -> float:
    """Return total span parallel to the building wall.

    :param center_face_width: Center face width in inches.
    :param side_face_width: One side face width in inches.
    :param side_angle_deg: Side face angle relative to the wall.
    :return: Overall wall-parallel span in inches.
    """

    angle_rad = math.radians(side_angle_deg)
    return center_face_width + (2.0 * side_face_width * math.cos(angle_rad))


def calculate_projection_from_side_faces(side_face_width: float, side_angle_deg: float) -> float:
    """Return the projection produced by one side face.

    For this bay style, the side faces are what create the bump-out depth.
    The center face stays square to the wall and does not add depth beyond the side-derived depth.

    :param side_face_width: One side face width in inches.
    :param side_angle_deg: Side face angle relative to the wall.
    :return: Projection depth in inches.
    """

    angle_rad = math.radians(side_angle_deg)
    return side_face_width * math.sin(angle_rad)


def calculate_outside_corner_to_corner_width(center_face_width: float, side_face_width: float) -> float:
    """Return the face-to-face width measured across the front corners.

    :param center_face_width: Center face width in inches.
    :param side_face_width: One side face width in inches.
    :return: Outside corner-to-corner width along the unfolded face lengths.
    """

    return center_face_width + (2.0 * side_face_width)


def calculate_wall_overlap(wall_parallel_span: float, opening_width: float) -> float:
    """Return the per-side overhang of the bay beyond the masonry opening.

    A positive value means the bay frame extends beyond the masonry opening on each side,
    resting on the masonry wall/pier. Zero means the bay fits entirely within the opening
    width (or exactly at the edges).

    :param wall_parallel_span: Total bay span parallel to the building wall in inches.
    :param opening_width: Masonry opening width in inches.
    :return: Per-side overlap in inches (always >= 0).
    """

    return max(0.0, (wall_parallel_span - opening_width) / 2.0)


def build_notes(
    side_window: WindowUnit,
    center_window: WindowUnit,
    center_count: int,
    passage_overlap_fraction: float,
    constraints: BayConstraints,
    wall_overlap: float = 0.0,
) -> tuple[str, ...]:
    """Create explanatory notes for a candidate."""

    notes: list[str] = []

    if center_count == 1:
        notes.append("Single center unit keeps the center face visually simple.")
    else:
        notes.append(f"{center_count} center units reduce the need for an extra-wide single sash.")

    if center_window.width > side_window.width:
        notes.append("Center window units are wider than the side units, which usually looks balanced.")
    elif center_window.width == side_window.width:
        notes.append("Center and side units match in width for a very regular appearance.")
    else:
        notes.append("Center window units are narrower than the side units; verify the visual proportions.")

    if passage_overlap_fraction >= max(constraints.min_passage_fraction + 0.15, 0.90):
        notes.append("Center face covers most of the masonry passage width.")
    elif passage_overlap_fraction >= constraints.min_passage_fraction:
        notes.append("Center face gives acceptable coverage of the masonry passage width.")

    if wall_overlap > 0.0:
        notes.append(
            f"Bay extends {wall_overlap:.2f}\" beyond the masonry opening on each side."
        )
    else:
        notes.append("Bay fits within the masonry opening width.")

    return tuple(notes)


def score_candidate(
    side_window: WindowUnit,
    center_window: WindowUnit,
    center_count: int,
    center_face: FaceDimensions,
    passage_overlap_fraction: float,
    constraints: BayConstraints,
) -> float:
    """Assign a simple practical score to a candidate.

    Higher is better.
    """

    score = 0.0

    # Prefer practical passage coverage.
    score += passage_overlap_fraction * 40.0

    # Prefer wider center glass overall, but not absurdly wide single units.
    score += min(center_face.window_area_width / max(constraints.opening_width, 1.0), 1.5) * 20.0

    # Prefer moderate center unit counts.
    if center_count == 1:
        score += 12.0
    elif center_count == 2:
        score += 15.0
    elif center_count == 3:
        score += 10.0
    else:
        score += 5.0

    # Prefer center units a bit wider than side units.
    width_delta = center_window.width - side_window.width
    if 2.0 <= width_delta <= 10.0:
        score += 12.0
    elif width_delta == 0.0:
        score += 8.0
    elif width_delta > 10.0:
        score += 4.0
    else:
        score += 3.0

    # Penalize very wide single units, since you mentioned using multiple units if needed.
    if center_count == 1 and center_window.width > 36.0:
        score -= (center_window.width - 36.0) * 0.75

    return score


def is_window_usable(window: WindowUnit, constraints: BayConstraints) -> bool:
    """Check basic user-defined width limits for a window unit."""

    return constraints.min_unit_width <= window.width <= constraints.max_single_unit_width


def filter_stock_by_height(
    windows: Iterable[WindowUnit],
    target_height: float,
    tolerance: float = 0.01,
) -> list[WindowUnit]:
    """Keep only windows matching the required common height.

    :param windows: Candidate stock windows.
    :param target_height: Required height in inches.
    :param tolerance: Allowed numeric mismatch.
    :return: Filtered list.
    """

    return [window for window in windows if abs(window.height - target_height) <= tolerance]


def find_candidates(
    constraints: BayConstraints,
    stock_windows: Sequence[WindowUnit],
) -> list[BayCandidate]:
    """Search for valid bay configurations.

    Validity rules for v1:
    - All windows must match the masonry opening height.
    - Left and right side windows are identical.
    - Center windows are equal-width units.
    - Side-derived projection must be at least the requested projection.
    - Center face must cover at least `min_passage_fraction` of the masonry opening width.

    :param constraints: Bay search constraints.
    :param stock_windows: List of stock window units.
    :return: Ranked list of valid candidates.
    """

    matching_height_windows = filter_stock_by_height(stock_windows, constraints.opening_height)
    candidates: list[BayCandidate] = []

    for side_window in matching_height_windows:
        if not is_window_usable(side_window, constraints):
            continue

        side_face = calculate_single_window_face(side_window, constraints)
        actual_projection = calculate_projection_from_side_faces(
            side_face_width=side_face.finished_face_width,
            side_angle_deg=constraints.side_angle_deg,
        )

        if actual_projection < constraints.projection_depth:
            continue

        for center_window in matching_height_windows:
            if not is_window_usable(center_window, constraints):
                continue

            for center_count in range(constraints.min_center_units, constraints.max_center_units + 1):
                center_face = calculate_center_face(center_window, center_count, constraints)
                passage_overlap_width = min(center_face.finished_face_width, constraints.opening_width)
                passage_overlap_fraction = passage_overlap_width / max(constraints.opening_width, 1.0)

                if passage_overlap_fraction < constraints.min_passage_fraction:
                    continue

                wall_parallel_span = calculate_wall_parallel_span(
                    center_face_width=center_face.finished_face_width,
                    side_face_width=side_face.finished_face_width,
                    side_angle_deg=constraints.side_angle_deg,
                )
                outside_width = calculate_outside_corner_to_corner_width(
                    center_face_width=center_face.finished_face_width,
                    side_face_width=side_face.finished_face_width,
                )
                wall_overlap = calculate_wall_overlap(wall_parallel_span, constraints.opening_width)

                if (
                    constraints.max_wall_overlap is not None
                    and wall_overlap > constraints.max_wall_overlap
                ):
                    continue

                score = score_candidate(
                    side_window=side_window,
                    center_window=center_window,
                    center_count=center_count,
                    center_face=center_face,
                    passage_overlap_fraction=passage_overlap_fraction,
                    constraints=constraints,
                )
                notes = build_notes(
                    side_window=side_window,
                    center_window=center_window,
                    center_count=center_count,
                    passage_overlap_fraction=passage_overlap_fraction,
                    constraints=constraints,
                    wall_overlap=wall_overlap,
                )

                candidates.append(
                    BayCandidate(
                        side_window=side_window,
                        center_window=center_window,
                        center_count=center_count,
                        side_face=side_face,
                        center_face=center_face,
                        side_angle_deg=constraints.side_angle_deg,
                        projection_depth=actual_projection,
                        wall_parallel_span=wall_parallel_span,
                        outside_corner_to_corner_width=outside_width,
                        center_face_projection_contribution=0.0,
                        side_face_projection_contribution=actual_projection,
                        passage_overlap_width=passage_overlap_width,
                        passage_overlap_fraction=passage_overlap_fraction,
                        wall_overlap=wall_overlap,
                        score=score,
                        notes=notes,
                    )
                )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def verify_candidate(
    constraints: BayConstraints,
    side_width: float,
    center_width: float,
    center_count: int,
    style: str = "double_hung",
) -> BayCandidate:
    """Build and return a single fully calculated candidate.

    :param constraints: Bay verification constraints.
    :param side_width: Nominal side window width in inches.
    :param center_width: Nominal center window width in inches.
    :param center_count: Number of center windows.
    :param style: Window style label.
    :return: Verified candidate.
    :raises ValueError: If the candidate fails basic rules.
    """

    side_window = WindowUnit(width=side_width, height=constraints.opening_height, style=style)
    center_window = WindowUnit(width=center_width, height=constraints.opening_height, style=style)

    if not is_window_usable(side_window, constraints):
        raise ValueError(f"Side window width {side_width:g} is outside the allowed limits.")
    if not is_window_usable(center_window, constraints):
        raise ValueError(f"Center window width {center_width:g} is outside the allowed limits.")

    side_face = calculate_single_window_face(side_window, constraints)
    center_face = calculate_center_face(center_window, center_count, constraints)
    actual_projection = calculate_projection_from_side_faces(
        side_face_width=side_face.finished_face_width,
        side_angle_deg=constraints.side_angle_deg,
    )

    if actual_projection < constraints.projection_depth:
        raise ValueError(
            f"The selected side window is too narrow to reach the requested projection of "
            f"{constraints.projection_depth:g} inches. Actual projection is {actual_projection:g} inches."
        )

    passage_overlap_width = min(center_face.finished_face_width, constraints.opening_width)
    passage_overlap_fraction = passage_overlap_width / max(constraints.opening_width, 1.0)
    if passage_overlap_fraction < constraints.min_passage_fraction:
        raise ValueError(
            f"The center face covers only {passage_overlap_fraction:.1%} of the masonry opening width, "
            f"which is below the minimum {constraints.min_passage_fraction:.1%}."
        )

    wall_parallel_span = calculate_wall_parallel_span(
        center_face_width=center_face.finished_face_width,
        side_face_width=side_face.finished_face_width,
        side_angle_deg=constraints.side_angle_deg,
    )
    outside_width = calculate_outside_corner_to_corner_width(
        center_face_width=center_face.finished_face_width,
        side_face_width=side_face.finished_face_width,
    )
    wall_overlap = calculate_wall_overlap(wall_parallel_span, constraints.opening_width)

    if constraints.max_wall_overlap is not None and wall_overlap > constraints.max_wall_overlap:
        raise ValueError(
            f"Wall overlap of {wall_overlap:.2f}\" per side exceeds the maximum allowed "
            f"{constraints.max_wall_overlap:.2f}\" per side."
        )

    score = score_candidate(
        side_window=side_window,
        center_window=center_window,
        center_count=center_count,
        center_face=center_face,
        passage_overlap_fraction=passage_overlap_fraction,
        constraints=constraints,
    )
    notes = build_notes(
        side_window=side_window,
        center_window=center_window,
        center_count=center_count,
        passage_overlap_fraction=passage_overlap_fraction,
        constraints=constraints,
        wall_overlap=wall_overlap,
    )

    return BayCandidate(
        side_window=side_window,
        center_window=center_window,
        center_count=center_count,
        side_face=side_face,
        center_face=center_face,
        side_angle_deg=constraints.side_angle_deg,
        projection_depth=actual_projection,
        wall_parallel_span=wall_parallel_span,
        outside_corner_to_corner_width=outside_width,
        center_face_projection_contribution=0.0,
        side_face_projection_contribution=actual_projection,
        passage_overlap_width=passage_overlap_width,
        passage_overlap_fraction=passage_overlap_fraction,
        wall_overlap=wall_overlap,
        score=score,
        notes=notes,
    )


def format_candidate(candidate: BayCandidate, rank: int | None = None) -> str:
    """Render a candidate as human-readable text."""

    header = f"Valid candidate #{rank}" if rank is not None else "Verified candidate"
    lines = [header, "-" * len(header)]
    lines.append(f"Side windows:   {candidate.side_window.label()}")
    lines.append(
        f"Center windows: {candidate.center_count} x {candidate.center_window.label()}"
    )
    lines.append(f"Side angle:     {candidate.side_angle_deg:g}°")
    lines.append("")
    lines.append("Face widths")
    lines.append(f"  Side face finished width:   {candidate.side_face.finished_face_width:.2f}\"")
    lines.append(f"  Side rough opening width:   {candidate.side_face.rough_opening_width:.2f}\"")
    lines.append(f"  Center face finished width: {candidate.center_face.finished_face_width:.2f}\"")
    lines.append(f"  Center rough opening width: {candidate.center_face.rough_opening_width:.2f}\"")
    lines.append("")
    lines.append("Overall bay")
    lines.append(f"  Wall-parallel span:         {candidate.wall_parallel_span:.2f}\"")
    lines.append(
        f"  Outside corner-to-corner:   {candidate.outside_corner_to_corner_width:.2f}\""
    )
    lines.append(f"  Actual projection:          {candidate.projection_depth:.2f}\"")
    lines.append(f"  Wall overlap per side:      {candidate.wall_overlap:.2f}\"")
    lines.append("")
    lines.append("Masonry opening relation")
    lines.append(f"  Passage overlap width:      {candidate.passage_overlap_width:.2f}\"")
    lines.append(f"  Passage overlap fraction:   {candidate.passage_overlap_fraction:.1%}")
    lines.append("")
    lines.append(f"Score: {candidate.score:.2f}")

    if candidate.notes:
        lines.append("Notes:")
        for note in candidate.notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)


def _rotate_point(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    """Rotate a point around the origin by angle degrees."""

    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return ((x * cos_a) - (y * sin_a), (x * sin_a) + (y * cos_a))


def _translate_points(
    points: Sequence[tuple[float, float]],
    dx: float,
    dy: float,
) -> tuple[tuple[float, float], ...]:
    """Translate a sequence of points by dx and dy."""

    return tuple((x + dx, y + dy) for x, y in points)


def _face_rectangle(
    length: float,
    depth: float,
    rotation_deg: float,
    anchor_x: float,
    anchor_y: float,
    anchor_mode: str,
) -> tuple[tuple[float, float], ...]:
    """Build a rotated rectangle polygon for one bay face.

    The local rectangle is defined with its top edge on y=0 and extending downward by `depth`.
    The top edge is the face line visible in plan view.

    :param length: Face length in inches.
    :param depth: Visual drawing depth for the frame body.
    :param rotation_deg: Rotation angle in degrees.
    :param anchor_x: World x coordinate for the anchor point.
    :param anchor_y: World y coordinate for the anchor point.
    :param anchor_mode: One of `center`, `left`, or `right`, describing which point on the top edge is anchored.
    :return: Rotated and translated polygon points.
    """

    if anchor_mode == "center":
        local_points = [
            (-length / 2.0, 0.0),
            (length / 2.0, 0.0),
            (length / 2.0, depth),
            (-length / 2.0, depth),
        ]
    elif anchor_mode == "left":
        local_points = [
            (0.0, 0.0),
            (length, 0.0),
            (length, depth),
            (0.0, depth),
        ]
    elif anchor_mode == "right":
        local_points = [
            (-length, 0.0),
            (0.0, 0.0),
            (0.0, depth),
            (-length, depth),
        ]
    else:
        raise ValueError(f"Unsupported anchor mode: {anchor_mode}")

    rotated = [_rotate_point(x, y, rotation_deg) for x, y in local_points]
    return _translate_points(rotated, anchor_x, anchor_y)



def render_candidate_svg(
    candidate: BayCandidate,
    frame_body_depth: float = 4.0,
    margin: float = 24.0,
) -> BaySvgLayout:
    """Render a simple top-view SVG for a bay layout candidate.

    The SVG shows:
    - the wall line
    - the masonry opening behind the bay
    - the center and side faces as polygons
    - simple labels for each face

    :param candidate: The candidate to render.
    :param frame_body_depth: Visual thickness of the frame body in the plan drawing.
    :param margin: Margin around the drawing in SVG units.
    :return: SVG text and metadata.
    """

    center_length = candidate.center_face.finished_face_width
    side_length = candidate.side_face.finished_face_width
    angle = candidate.side_angle_deg
    opening_width = candidate.passage_overlap_width / max(candidate.passage_overlap_fraction, 1e-9)

    cx = 0.0
    cy = 0.0
    half_center = center_length / 2.0

    left_join = (-half_center, 0.0)
    right_join = (half_center, 0.0)

    center_points = _face_rectangle(
        length=center_length,
        depth=frame_body_depth,
        rotation_deg=0.0,
        anchor_x=cx,
        anchor_y=cy,
        anchor_mode="center",
    )
    left_points = _face_rectangle(
        length=side_length,
        depth=frame_body_depth,
        rotation_deg=-angle,
        anchor_x=left_join[0],
        anchor_y=left_join[1],
        anchor_mode="right",
    )
    right_points = _face_rectangle(
        length=side_length,
        depth=frame_body_depth,
        rotation_deg=angle,
        anchor_x=right_join[0],
        anchor_y=right_join[1],
        anchor_mode="left",
    )

    all_points = list(center_points) + list(left_points) + list(right_points)
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]

    min_x = min(xs + [-(opening_width / 2.0)]) - margin
    max_x = max(xs + [(opening_width / 2.0)]) + margin
    min_y = min(ys + [-frame_body_depth - 16.0]) - margin
    max_y = max(ys + [12.0]) + margin

    svg_width = max_x - min_x
    svg_height = max_y - min_y

    def to_svg(points: Sequence[tuple[float, float]]) -> str:
        transformed = [(x - min_x, y - min_y) for x, y in points]
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in transformed)

    def text_point(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
        x = sum(point[0] for point in points) / len(points)
        y = sum(point[1] for point in points) / len(points)
        return (x - min_x, y - min_y)

    center_label_x, center_label_y = text_point(center_points)
    left_label_x, left_label_y = text_point(left_points)
    right_label_x, right_label_y = text_point(right_points)

    wall_y = cy - frame_body_depth - 10.0
    svg_wall_y = wall_y - min_y
    opening_left_x = (-(opening_width / 2.0)) - min_x
    opening_right_x = ((opening_width / 2.0)) - min_x

    center_text = escape(f"Center: {candidate.center_count} x {candidate.center_window.width:g}\"")
    side_text = escape(f"Side: {candidate.side_window.width:g}\"")
    title_text = escape(
        f"Bay layout | angle {candidate.side_angle_deg:g}° | projection {candidate.projection_depth:.2f}\""
    )

    svg_text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width:.0f}" height="{svg_height:.0f}" viewBox="0 0 {svg_width:.2f} {svg_height:.2f}">
  <rect x="0" y="0" width="{svg_width:.2f}" height="{svg_height:.2f}" fill="white" />
  <text x="16" y="24" font-family="Arial, Helvetica, sans-serif" font-size="16">{title_text}</text>
  <line x1="0" y1="{svg_wall_y:.2f}" x2="{svg_width:.2f}" y2="{svg_wall_y:.2f}" stroke="black" stroke-width="2" />
  <line x1="{opening_left_x:.2f}" y1="{svg_wall_y:.2f}" x2="{opening_right_x:.2f}" y2="{svg_wall_y:.2f}" stroke="white" stroke-width="4" />
  <text x="{(opening_left_x + opening_right_x) / 2.0:.2f}" y="{svg_wall_y - 8.0:.2f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">Masonry opening</text>
  <polygon points="{to_svg(left_points)}" fill="#f2f2f2" stroke="black" stroke-width="1.5" />
  <polygon points="{to_svg(center_points)}" fill="#e6e6e6" stroke="black" stroke-width="1.5" />
  <polygon points="{to_svg(right_points)}" fill="#f2f2f2" stroke="black" stroke-width="1.5" />
  <text x="{left_label_x:.2f}" y="{left_label_y:.2f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">{side_text}</text>
  <text x="{center_label_x:.2f}" y="{center_label_y:.2f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">{center_text}</text>
  <text x="{right_label_x:.2f}" y="{right_label_y:.2f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">{side_text}</text>
</svg>
'''

    return BaySvgLayout(
        svg_text=svg_text,
        width=svg_width,
        height=svg_height,
        view_box=f"0 0 {svg_width:.2f} {svg_height:.2f}",
        faces=(
            FacePolygon(label="left", points=left_points),
            FacePolygon(label="center", points=center_points),
            FacePolygon(label="right", points=right_points),
        ),
    )


def candidate_to_dict(
    candidate: BayCandidate,
    rank: int | None = None,
    constraints: BayConstraints | None = None,
) -> dict:
    """Serialise a BayCandidate to a plain JSON-compatible dict.

    :param candidate: The candidate to serialise.
    :param rank: Optional 1-based rank to embed in the output.
    :param constraints: Optional source constraints to embed as a sub-object.
    :return: Nested dict with all calculated fields.
    """

    data: dict = {
        "side_window": {
            "width": candidate.side_window.width,
            "height": candidate.side_window.height,
            "style": candidate.side_window.style,
        },
        "center_window": {
            "width": candidate.center_window.width,
            "height": candidate.center_window.height,
            "style": candidate.center_window.style,
        },
        "center_count": candidate.center_count,
        "side_angle_deg": candidate.side_angle_deg,
        "side_face": {
            "window_area_width": candidate.side_face.window_area_width,
            "rough_opening_width": candidate.side_face.rough_opening_width,
            "finished_face_width": candidate.side_face.finished_face_width,
        },
        "center_face": {
            "window_area_width": candidate.center_face.window_area_width,
            "rough_opening_width": candidate.center_face.rough_opening_width,
            "finished_face_width": candidate.center_face.finished_face_width,
        },
        "projection_depth": candidate.projection_depth,
        "wall_parallel_span": candidate.wall_parallel_span,
        "outside_corner_to_corner_width": candidate.outside_corner_to_corner_width,
        "center_face_projection_contribution": candidate.center_face_projection_contribution,
        "side_face_projection_contribution": candidate.side_face_projection_contribution,
        "passage_overlap_width": candidate.passage_overlap_width,
        "passage_overlap_fraction": candidate.passage_overlap_fraction,
        "wall_overlap": candidate.wall_overlap,
        "score": candidate.score,
        "notes": list(candidate.notes),
    }
    if rank is not None:
        data["rank"] = rank
    if constraints is not None:
        data["constraints"] = {
            "opening_width": constraints.opening_width,
            "opening_height": constraints.opening_height,
            "sill_height": constraints.sill_height,
            "projection_depth": constraints.projection_depth,
            "side_angle_deg": constraints.side_angle_deg,
            "frame_thickness": constraints.frame_thickness,
            "rough_clearance": constraints.rough_clearance,
            "mullion_thickness": constraints.mullion_thickness,
            "face_trim_allowance": constraints.face_trim_allowance,
            "frame_body_depth": constraints.frame_body_depth,
            "min_passage_fraction": constraints.min_passage_fraction,
            "min_unit_width": constraints.min_unit_width,
            "max_single_unit_width": constraints.max_single_unit_width,
            "max_wall_overlap": constraints.max_wall_overlap,
        }
    return data


def write_candidates_json(
    candidates: Sequence[BayCandidate],
    output_path: Path,
    constraints: BayConstraints | None = None,
) -> Path:
    """Write the top ranked candidates to a JSON file as an array.

    :param candidates: Candidates to serialise (already trimmed to the desired limit).
    :param output_path: Destination file path.
    :param constraints: Optional source constraints to embed in each entry.
    :return: The path that was written.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [candidate_to_dict(c, rank=i, constraints=constraints) for i, c in enumerate(candidates, start=1)]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_candidate_json(
    candidate: BayCandidate,
    output_path: Path,
    constraints: BayConstraints | None = None,
) -> Path:
    """Write a single verified candidate to a JSON file.

    :param candidate: Candidate to serialise.
    :param output_path: Destination file path.
    :param constraints: Optional source constraints to embed in the output.
    :return: The path that was written.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(candidate_to_dict(candidate, constraints=constraints), indent=2), encoding="utf-8")
    return output_path


def write_svg_file(
    candidate: BayCandidate,
    output_path: Path,
    frame_body_depth: float = 4.0,
) -> Path:
    """Write an SVG top-view rendering for a candidate to disk."""

    layout = render_candidate_svg(candidate, frame_body_depth=frame_body_depth)
    output_path.write_text(layout.svg_text, encoding="utf-8")
    return output_path


def print_candidates(
    candidates: Sequence[BayCandidate],
    limit: int,
    svg_dir: Path | None = None,
    frame_body_depth: float = 4.0,
) -> None:
    """Print the top ranked candidates and optionally emit SVG files."""

    if not candidates:
        print("No valid candidates were found with the current constraints.")
        return

    if svg_dir is not None:
        svg_dir.mkdir(parents=True, exist_ok=True)

    for index, candidate in enumerate(candidates[:limit], start=1):
        print(format_candidate(candidate, rank=index))
        if svg_dir is not None:
            svg_path = svg_dir / f"candidate_{index:02d}.svg"
            write_svg_file(candidate, svg_path, frame_body_depth=frame_body_depth)
            print(f"SVG written to: {svg_path}")
        if index < min(limit, len(candidates)):
            print("\n")


def prompt_for_value(
    prompt: str,
    type_fn,
    hint: str = "",
    validator=None,
):
    """Read one value from stdin, retrying until it is valid.

    :param prompt: Human-readable label shown before the colon.
    :param type_fn: Callable that converts a string (e.g. ``float``, ``int``).
    :param hint: Optional short note appended in parentheses to the prompt.
    :param validator: Optional callable ``(value) -> str | None``; return an error
        message string to reject the value, or ``None`` to accept it.
    :return: The validated, converted value.
    """

    display = prompt
    if hint:
        display += f" ({hint})"
    display += ": "

    while True:
        raw = input(display).strip()
        if not raw:
            print("  Value is required, please enter a number.", file=sys.stderr)
            continue
        try:
            value = type_fn(raw)
        except (ValueError, TypeError):
            print(f"  Invalid input '{raw}'. Expected a {type_fn.__name__}.", file=sys.stderr)
            continue
        if validator is not None:
            error = validator(value)
            if error:
                print(f"  {error}", file=sys.stderr)
                continue
        return value


def interactive_fill_args(args: argparse.Namespace) -> None:
    """Prompt interactively for any required CLI argument that was not supplied.

    Required fields that are still ``None`` after argparse are filled in-place.
    If stdin is not a terminal and values are missing the function exits with a
    clear error listing the missing flags.

    :param args: Parsed argparse namespace; missing required fields are filled in-place.
    :raises SystemExit: When stdin is not a tty but required fields are missing.
    """

    _pos = lambda v: "Must be > 0." if v <= 0 else None
    _pos_int = lambda v: "Must be >= 1." if v < 1 else None

    required_common = [
        ("opening_width",   "Masonry opening width",    float, _pos),
        ("opening_height",  "Common window height",     float, _pos),
        ("projection_depth","Desired bay projection",   float, _pos),
    ]
    required_verify = [
        ("side_width",   "Side window width",          float, _pos),
        ("center_width", "Center window width",        float, _pos),
        ("center_count", "Number of center windows",   int,   _pos_int),
    ]

    fields_to_check = list(required_common)
    if getattr(args, "command", None) == "verify":
        fields_to_check.extend(required_verify)

    missing = [name for name, *_ in fields_to_check if getattr(args, name, None) is None]
    if not missing:
        return

    if not sys.stdin.isatty():
        raise SystemExit(
            "Error: the following required arguments were not provided and stdin is not a "
            "terminal, so interactive prompting is not possible:\n"
            + "\n".join(f"  --{name.replace('_', '-')}" for name in missing)
        )

    print("Bay Window Calculator — interactive input")
    print("  (supply missing values below, or re-run with full CLI flags)\n")

    for attr, prompt, type_fn, validator in fields_to_check:
        if getattr(args, attr, None) is None:
            setattr(args, attr, prompt_for_value(prompt + " in inches", type_fn, validator=validator))

    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Search and verify bay window layouts using stock double-hung windows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--opening-width",    type=float, default=None, help="Masonry opening width in inches.")
    common.add_argument("--opening-height",   type=float, default=None, help="Common window height in inches.")
    common.add_argument("--projection-depth", type=float, default=None, help="Requested bay projection in inches.")
    common.add_argument(
        "--side-angle-deg",
        type=float,
        default=30.0,
        help="Side face angle relative to the wall in degrees. Default: 30.",
    )
    common.add_argument(
        "--frame-thickness",
        type=float,
        default=1.5,
        help="Outer frame/post thickness in inches. Default: 1.5.",
    )
    common.add_argument(
        "--rough-clearance",
        type=float,
        default=0.5,
        help="Rough opening allowance per side of each window in inches. Default: 0.5.",
    )
    common.add_argument(
        "--mullion-thickness",
        type=float,
        default=1.5,
        help="Thickness of mullion posts between center windows in inches. Default: 1.5.",
    )
    common.add_argument(
        "--face-trim-allowance",
        type=float,
        default=0.0,
        help="Additional allowance per face for trim/shimming in inches. Default: 0.",
    )
    common.add_argument(
        "--min-center-units",
        type=int,
        default=1,
        help="Minimum number of center window units. Default: 1.",
    )
    common.add_argument(
        "--max-center-units",
        type=int,
        default=4,
        help="Maximum number of center window units. Default: 4.",
    )
    common.add_argument(
        "--min-passage-fraction",
        type=float,
        default=0.65,
        help="Minimum fraction of opening width the center face should cover. Default: 0.65.",
    )
    common.add_argument(
        "--min-unit-width",
        type=float,
        default=18.0,
        help="Minimum allowed stock window width in inches. Default: 18.",
    )
    common.add_argument(
        "--max-single-unit-width",
        type=float,
        default=48.0,
        help="Maximum allowed width for any single unit in inches. Default: 48.",
    )
    common.add_argument(
        "--sill-height",
        type=float,
        default=None,
        help="Optional finished sill height above the floor in inches.",
    )
    common.add_argument(
        "--frame-body-depth",
        type=float,
        default=4.0,
        help="Depth of the window frame body in inches; used for SVG plan rendering. Default: 4.",
    )
    common.add_argument(
        "--max-wall-overlap",
        type=float,
        default=None,
        help=(
            "Maximum allowed per-side bay overhang beyond the masonry opening in inches. "
            "Default: no limit."
        ),
    )
    common.add_argument(
        "--stock-json",
        type=Path,
        help="Optional JSON file containing stock window definitions.",
    )

    search_parser = subparsers.add_parser("search", parents=[common], help="Search for valid layouts.")
    search_parser.add_argument(
        "--svg-dir",
        type=Path,
        help="Optional directory where SVG renderings for the printed candidates will be written.",
    )
    search_parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON file path where the ranked candidates and their calculations will be written.",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of ranked results to print. Default: 10.",
    )

    verify_parser = subparsers.add_parser("verify", parents=[common], help="Verify one specific layout.")
    verify_parser.add_argument(
        "--svg-output",
        type=Path,
        help="Optional SVG file path for writing a top-view rendering of the verified layout.",
    )
    verify_parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON file path where the verified candidate and its calculations will be written.",
    )
    verify_parser.add_argument("--side-width",   type=float, default=None, help="Side window width in inches.")
    verify_parser.add_argument("--center-width", type=float, default=None, help="Center window width in inches.")
    verify_parser.add_argument(
        "--center-count",
        type=int,
        default=None,
        help="Number of center window units.",
    )

    return parser.parse_args()


def build_constraints_from_args(args: argparse.Namespace) -> BayConstraints:
    """Create the constraint object from parsed arguments."""

    return BayConstraints(
        opening_width=args.opening_width,
        opening_height=args.opening_height,
        projection_depth=args.projection_depth,
        side_angle_deg=args.side_angle_deg,
        frame_thickness=args.frame_thickness,
        rough_clearance=args.rough_clearance,
        mullion_thickness=args.mullion_thickness,
        face_trim_allowance=args.face_trim_allowance,
        min_center_units=args.min_center_units,
        max_center_units=args.max_center_units,
        min_passage_fraction=args.min_passage_fraction,
        max_single_unit_width=args.max_single_unit_width,
        min_unit_width=args.min_unit_width,
        sill_height=args.sill_height,
        frame_body_depth=args.frame_body_depth,
        max_wall_overlap=args.max_wall_overlap,
    )


def build_stock_windows(args: argparse.Namespace) -> list[WindowUnit]:
    """Return stock window definitions from JSON or defaults."""

    if args.stock_json:
        return load_stock_windows_from_json(args.stock_json)
    return build_default_stock_windows(height=args.opening_height)


def _prompt_for_subcommand() -> str:
    """Prompt the user to choose a subcommand when none was given on the CLI.

    :return: One of ``"search"`` or ``"verify"``.
    :raises SystemExit: When stdin is not a terminal.
    """

    if not sys.stdin.isatty():
        raise SystemExit(
            "Error: a subcommand is required.\n"
            "  Usage: bay_window_calculator_with_svg.py search --opening-width ...\n"
            "         bay_window_calculator_with_svg.py verify --opening-width ..."
        )

    print("Bay Window Calculator")
    print("Choose a mode:")
    print("  1. search  — find all valid layout candidates")
    print("  2. verify  — check one specific layout")
    while True:
        raw = input("Mode [search/verify]: ").strip().lower()
        if raw in ("search", "s", "1"):
            return "search"
        if raw in ("verify", "v", "2"):
            return "verify"
        print("  Please enter 'search' or 'verify'.", file=sys.stderr)


def main() -> None:
    """Run the bay window calculator CLI."""

    known_commands = {"search", "verify"}
    if not any(tok in known_commands for tok in sys.argv[1:]):
        sys.argv.insert(1, _prompt_for_subcommand())

    args = parse_args()
    interactive_fill_args(args)
    constraints = build_constraints_from_args(args)
    stock_windows = build_stock_windows(args)

    if args.command == "search":
        candidates = find_candidates(constraints=constraints, stock_windows=stock_windows)
        print_candidates(
            candidates=candidates,
            limit=args.limit,
            svg_dir=args.svg_dir,
            frame_body_depth=constraints.frame_body_depth,
        )
        if args.json_output:
            write_candidates_json(candidates[:args.limit], args.json_output, constraints=constraints)
            print(f"JSON written to: {args.json_output}")
        return

    if args.command == "verify":
        candidate = verify_candidate(
            constraints=constraints,
            side_width=args.side_width,
            center_width=args.center_width,
            center_count=args.center_count,
        )
        print(format_candidate(candidate))
        if args.svg_output:
            args.svg_output.parent.mkdir(parents=True, exist_ok=True)
            write_svg_file(candidate, args.svg_output, frame_body_depth=constraints.frame_body_depth)
            print(f"SVG written to: {args.svg_output}")
        if args.json_output:
            write_candidate_json(candidate, args.json_output, constraints=constraints)
            print(f"JSON written to: {args.json_output}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
