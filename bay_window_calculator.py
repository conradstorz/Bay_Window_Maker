from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


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
    score: float
    notes: tuple[str, ...]


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


def build_notes(
    side_window: WindowUnit,
    center_window: WindowUnit,
    center_count: int,
    passage_overlap_fraction: float,
    constraints: BayConstraints,
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


def print_candidates(candidates: Sequence[BayCandidate], limit: int) -> None:
    """Print the top ranked candidates."""

    if not candidates:
        print("No valid candidates were found with the current constraints.")
        return

    for index, candidate in enumerate(candidates[:limit], start=1):
        print(format_candidate(candidate, rank=index))
        if index < min(limit, len(candidates)):
            print("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Search and verify bay window layouts using stock double-hung windows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--opening-width", type=float, required=True, help="Masonry opening width in inches.")
    common.add_argument("--opening-height", type=float, required=True, help="Common window height in inches.")
    common.add_argument("--projection-depth", type=float, required=True, help="Requested bay projection in inches.")
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
        "--stock-json",
        type=Path,
        help="Optional JSON file containing stock window definitions.",
    )

    search_parser = subparsers.add_parser("search", parents=[common], help="Search for valid layouts.")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of ranked results to print. Default: 10.",
    )

    verify_parser = subparsers.add_parser("verify", parents=[common], help="Verify one specific layout.")
    verify_parser.add_argument("--side-width", type=float, required=True, help="Side window width in inches.")
    verify_parser.add_argument("--center-width", type=float, required=True, help="Center window width in inches.")
    verify_parser.add_argument(
        "--center-count",
        type=int,
        required=True,
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
    )


def build_stock_windows(args: argparse.Namespace) -> list[WindowUnit]:
    """Return stock window definitions from JSON or defaults."""

    if args.stock_json:
        return load_stock_windows_from_json(args.stock_json)
    return build_default_stock_windows(height=args.opening_height)


def main() -> None:
    """Run the bay window calculator CLI."""

    args = parse_args()
    constraints = build_constraints_from_args(args)
    stock_windows = build_stock_windows(args)

    if args.command == "search":
        candidates = find_candidates(constraints=constraints, stock_windows=stock_windows)
        print_candidates(candidates=candidates, limit=args.limit)
        return

    if args.command == "verify":
        candidate = verify_candidate(
            constraints=constraints,
            side_width=args.side_width,
            center_width=args.center_width,
            center_count=args.center_count,
        )
        print(format_candidate(candidate))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
