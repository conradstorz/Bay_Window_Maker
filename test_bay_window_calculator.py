"""Tests for bay_window_calculator_with_svg.py"""
from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import pytest

from bay_window_calculator_with_svg import (
    BayCandidate,
    BayConstraints,
    FaceDimensions,
    WindowUnit,
    _face_rectangle,
    _rotate_point,
    _translate_points,
    build_default_stock_windows,
    build_notes,
    calculate_center_face,
    calculate_outside_corner_to_corner_width,
    calculate_projection_from_side_faces,
    calculate_single_window_face,
    calculate_wall_parallel_span,
    candidate_to_dict,
    filter_stock_by_height,
    find_candidates,
    format_candidate,
    is_window_usable,
    load_stock_windows_from_json,
    render_candidate_svg,
    score_candidate,
    verify_candidate,
    write_candidate_json,
    write_candidates_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_constraints(**overrides) -> BayConstraints:
    """BayConstraints with sensible defaults for testing."""
    kwargs = dict(
        opening_width=72.0,
        opening_height=36.0,
        projection_depth=16.0,
        side_angle_deg=30.0,
    )
    kwargs.update(overrides)
    return BayConstraints(**kwargs)


# ---------------------------------------------------------------------------
# WindowUnit
# ---------------------------------------------------------------------------

class TestWindowUnit:
    def test_label_uses_g_format(self):
        w = WindowUnit(width=24.0, height=36.0)
        assert '24"' in w.label()
        assert '36"' in w.label()

    def test_label_includes_style(self):
        w = WindowUnit(width=30.0, height=36.0, style="casement")
        assert "casement" in w.label()

    def test_label_replaces_underscore(self):
        w = WindowUnit(width=30.0, height=36.0, style="double_hung")
        assert "double hung" in w.label()


# ---------------------------------------------------------------------------
# calculate_single_window_face
# ---------------------------------------------------------------------------

class TestCalculateSingleWindowFace:
    def test_rough_opening_adds_both_clearances(self):
        w = WindowUnit(width=24.0, height=36.0)
        c = default_constraints(rough_clearance=0.5)
        face = calculate_single_window_face(w, c)
        assert face.rough_opening_width == pytest.approx(25.0)

    def test_finished_width_adds_frames(self):
        w = WindowUnit(width=24.0, height=36.0)
        c = default_constraints(frame_thickness=1.5, rough_clearance=0.5)
        face = calculate_single_window_face(w, c)
        # 24 + 2*0.5 + 2*1.5 = 28.0
        assert face.finished_face_width == pytest.approx(28.0)

    def test_face_trim_allowance_is_added_once(self):
        w = WindowUnit(width=24.0, height=36.0)
        c = default_constraints(frame_thickness=1.5, rough_clearance=0.5, face_trim_allowance=1.0)
        face = calculate_single_window_face(w, c)
        assert face.finished_face_width == pytest.approx(29.0)

    def test_window_area_width_equals_window_width(self):
        w = WindowUnit(width=30.0, height=36.0)
        c = default_constraints()
        face = calculate_single_window_face(w, c)
        assert face.window_area_width == pytest.approx(30.0)

    def test_returns_face_dimensions_instance(self):
        w = WindowUnit(width=24.0, height=36.0)
        face = calculate_single_window_face(w, default_constraints())
        assert isinstance(face, FaceDimensions)


# ---------------------------------------------------------------------------
# calculate_center_face
# ---------------------------------------------------------------------------

class TestCalculateCenterFace:
    def test_single_unit_no_mullions(self):
        w = WindowUnit(width=24.0, height=36.0)
        c = default_constraints(frame_thickness=1.5, rough_clearance=0.5, mullion_thickness=1.5)
        face = calculate_center_face(w, 1, c)
        # rough: 24 + 2*0.5 = 25; mullions: 0; finished: 25 + 2*1.5 = 28
        assert face.finished_face_width == pytest.approx(28.0)
        assert face.window_area_width == pytest.approx(24.0)

    def test_two_units_one_mullion(self):
        w = WindowUnit(width=24.0, height=36.0)
        c = default_constraints(frame_thickness=1.5, rough_clearance=0.5, mullion_thickness=1.5)
        face = calculate_center_face(w, 2, c)
        # rough: 2*(24 + 1) = 50; mullions: 1*1.5 = 1.5; finished: 50 + 1.5 + 3 = 54.5
        assert face.rough_opening_width == pytest.approx(50.0)
        assert face.finished_face_width == pytest.approx(54.5)
        assert face.window_area_width == pytest.approx(48.0)

    def test_three_units_two_mullions(self):
        w = WindowUnit(width=24.0, height=36.0)
        c = default_constraints(frame_thickness=1.5, rough_clearance=0.5, mullion_thickness=1.5)
        face = calculate_center_face(w, 3, c)
        # rough: 3*(24 + 1) = 75; mullions: 2*1.5 = 3; finished: 75 + 3 + 3 = 81
        assert face.finished_face_width == pytest.approx(81.0)
        assert face.window_area_width == pytest.approx(72.0)

    def test_zero_count_raises(self):
        w = WindowUnit(width=24.0, height=36.0)
        with pytest.raises(ValueError):
            calculate_center_face(w, 0, default_constraints())


# ---------------------------------------------------------------------------
# calculate_wall_parallel_span
# ---------------------------------------------------------------------------

class TestCalculateWallParallelSpan:
    def test_90_degree_side_angle_adds_nothing(self):
        # cos(90°) == 0, so side faces don't contribute any parallel span
        span = calculate_wall_parallel_span(
            center_face_width=60.0,
            side_face_width=30.0,
            side_angle_deg=90.0,
        )
        assert span == pytest.approx(60.0, abs=1e-9)

    def test_zero_degree_side_angle_full_contribution(self):
        # cos(0°) == 1, so two side faces add their full width
        span = calculate_wall_parallel_span(
            center_face_width=60.0,
            side_face_width=30.0,
            side_angle_deg=0.0,
        )
        assert span == pytest.approx(120.0, abs=1e-9)

    def test_30_degree_known_value(self):
        # cos(30°) = sqrt(3)/2 ≈ 0.8660
        span = calculate_wall_parallel_span(
            center_face_width=60.0,
            side_face_width=30.0,
            side_angle_deg=30.0,
        )
        expected = 60.0 + 2.0 * 30.0 * math.cos(math.radians(30.0))
        assert span == pytest.approx(expected)


# ---------------------------------------------------------------------------
# calculate_projection_from_side_faces
# ---------------------------------------------------------------------------

class TestCalculateProjection:
    def test_90_degree_equals_face_width(self):
        proj = calculate_projection_from_side_faces(side_face_width=30.0, side_angle_deg=90.0)
        assert proj == pytest.approx(30.0, abs=1e-9)

    def test_zero_degree_no_projection(self):
        proj = calculate_projection_from_side_faces(side_face_width=30.0, side_angle_deg=0.0)
        assert proj == pytest.approx(0.0, abs=1e-9)

    def test_30_degree_known_value(self):
        proj = calculate_projection_from_side_faces(side_face_width=30.0, side_angle_deg=30.0)
        expected = 30.0 * math.sin(math.radians(30.0))  # = 15.0
        assert proj == pytest.approx(expected)


# ---------------------------------------------------------------------------
# calculate_outside_corner_to_corner_width
# ---------------------------------------------------------------------------

class TestOutsideCornerWidth:
    def test_basic_sum(self):
        result = calculate_outside_corner_to_corner_width(
            center_face_width=60.0,
            side_face_width=30.0,
        )
        assert result == pytest.approx(120.0)

    def test_zero_side(self):
        result = calculate_outside_corner_to_corner_width(
            center_face_width=50.0,
            side_face_width=0.0,
        )
        assert result == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# filter_stock_by_height
# ---------------------------------------------------------------------------

class TestFilterStockByHeight:
    def setup_method(self):
        self.stock = [
            WindowUnit(width=24.0, height=36.0),
            WindowUnit(width=30.0, height=36.0),
            WindowUnit(width=30.0, height=48.0),
        ]

    def test_filters_to_matching_height(self):
        result = filter_stock_by_height(self.stock, 36.0)
        assert all(w.height == 36.0 for w in result)
        assert len(result) == 2

    def test_tolerance_boundary(self):
        stock = [WindowUnit(width=24.0, height=36.005)]
        result = filter_stock_by_height(stock, 36.0, tolerance=0.01)
        assert len(result) == 1

    def test_outside_tolerance_excluded(self):
        stock = [WindowUnit(width=24.0, height=36.02)]
        result = filter_stock_by_height(stock, 36.0, tolerance=0.01)
        assert len(result) == 0

    def test_empty_input(self):
        assert filter_stock_by_height([], 36.0) == []


# ---------------------------------------------------------------------------
# is_window_usable
# ---------------------------------------------------------------------------

class TestIsWindowUsable:
    def test_within_limits_is_usable(self):
        w = WindowUnit(width=30.0, height=36.0)
        c = default_constraints(min_unit_width=18.0, max_single_unit_width=48.0)
        assert is_window_usable(w, c) is True

    def test_below_min_is_not_usable(self):
        w = WindowUnit(width=16.0, height=36.0)
        c = default_constraints(min_unit_width=18.0)
        assert is_window_usable(w, c) is False

    def test_above_max_is_not_usable(self):
        w = WindowUnit(width=50.0, height=36.0)
        c = default_constraints(max_single_unit_width=48.0)
        assert is_window_usable(w, c) is False

    def test_at_boundary_is_usable(self):
        c = default_constraints(min_unit_width=18.0, max_single_unit_width=48.0)
        assert is_window_usable(WindowUnit(width=18.0, height=36.0), c) is True
        assert is_window_usable(WindowUnit(width=48.0, height=36.0), c) is True


# ---------------------------------------------------------------------------
# build_default_stock_windows
# ---------------------------------------------------------------------------

class TestBuildDefaultStockWindows:
    def test_all_windows_share_given_height(self):
        windows = build_default_stock_windows(height=42.0)
        assert all(w.height == 42.0 for w in windows)

    def test_includes_expected_width_range(self):
        windows = build_default_stock_windows(height=36.0)
        widths = [w.width for w in windows]
        assert 18.0 in widths
        assert 48.0 in widths

    def test_returns_nonempty_list(self):
        assert len(build_default_stock_windows(36.0)) > 0


# ---------------------------------------------------------------------------
# load_stock_windows_from_json
# ---------------------------------------------------------------------------

class TestLoadStockWindowsFromJson:
    def test_loads_valid_json(self, tmp_path: Path):
        data = [{"width": 24, "height": 36, "style": "casement"}]
        f = tmp_path / "stock.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        windows = load_stock_windows_from_json(f)
        assert len(windows) == 1
        assert windows[0].width == 24.0
        assert windows[0].height == 36.0
        assert windows[0].style == "casement"

    def test_defaults_style_to_double_hung(self, tmp_path: Path):
        data = [{"width": 30, "height": 36}]
        f = tmp_path / "stock.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        windows = load_stock_windows_from_json(f)
        assert windows[0].style == "double_hung"

    def test_raises_on_non_list_json(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text('{"width": 24, "height": 36}', encoding="utf-8")
        with pytest.raises(ValueError, match="list"):
            load_stock_windows_from_json(f)

    def test_raises_on_missing_key(self, tmp_path: Path):
        data = [{"width": 24}]  # missing 'height'
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="height"):
            load_stock_windows_from_json(f)

    def test_raises_on_non_dict_entry(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text('[42]', encoding="utf-8")
        with pytest.raises(ValueError):
            load_stock_windows_from_json(f)


# ---------------------------------------------------------------------------
# build_notes
# ---------------------------------------------------------------------------

class TestBuildNotes:
    def _constraints(self, **kw):
        return default_constraints(min_passage_fraction=0.65, **kw)

    def test_single_center_note(self):
        sw = WindowUnit(width=30.0, height=36.0)
        cw = WindowUnit(width=30.0, height=36.0)
        notes = build_notes(sw, cw, center_count=1, passage_overlap_fraction=0.85,
                            constraints=self._constraints())
        assert any("Single center" in n for n in notes)

    def test_multi_center_note(self):
        sw = WindowUnit(width=30.0, height=36.0)
        cw = WindowUnit(width=30.0, height=36.0)
        notes = build_notes(sw, cw, center_count=2, passage_overlap_fraction=0.85,
                            constraints=self._constraints())
        assert any("2 center" in n for n in notes)

    def test_center_wider_than_side_note(self):
        sw = WindowUnit(width=24.0, height=36.0)
        cw = WindowUnit(width=30.0, height=36.0)
        notes = build_notes(sw, cw, center_count=1, passage_overlap_fraction=0.85,
                            constraints=self._constraints())
        assert any("wider" in n for n in notes)

    def test_matching_widths_note(self):
        sw = WindowUnit(width=30.0, height=36.0)
        cw = WindowUnit(width=30.0, height=36.0)
        notes = build_notes(sw, cw, center_count=1, passage_overlap_fraction=0.85,
                            constraints=self._constraints())
        assert any("match" in n for n in notes)

    def test_center_narrower_than_side_note(self):
        sw = WindowUnit(width=36.0, height=36.0)
        cw = WindowUnit(width=24.0, height=36.0)
        notes = build_notes(sw, cw, center_count=1, passage_overlap_fraction=0.85,
                            constraints=self._constraints())
        assert any("narrower" in n for n in notes)


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------

class TestScoreCandidate:
    def _score(self, side_w=24.0, center_w=30.0, center_count=2, overlap_fraction=0.85):
        c = default_constraints()
        side = WindowUnit(width=side_w, height=36.0)
        center = WindowUnit(width=center_w, height=36.0)
        center_face = calculate_center_face(center, center_count, c)
        return score_candidate(
            side_window=side,
            center_window=center,
            center_count=center_count,
            center_face=center_face,
            passage_overlap_fraction=overlap_fraction,
            constraints=c,
        )

    def test_returns_positive_score(self):
        assert self._score() > 0.0

    def test_higher_overlap_fraction_improves_score(self):
        low = self._score(overlap_fraction=0.70)
        high = self._score(overlap_fraction=0.95)
        assert high > low

    def test_very_wide_single_unit_penalized(self):
        normal = self._score(center_w=30.0, center_count=1)
        wide = self._score(center_w=48.0, center_count=1)
        assert normal > wide


# ---------------------------------------------------------------------------
# _rotate_point
# ---------------------------------------------------------------------------

class TestRotatePoint:
    def test_zero_degrees_unchanged(self):
        rx, ry = _rotate_point(3.0, 4.0, 0.0)
        assert rx == pytest.approx(3.0)
        assert ry == pytest.approx(4.0)

    def test_90_degrees(self):
        # (1, 0) rotated 90° CCW → (0, 1)
        rx, ry = _rotate_point(1.0, 0.0, 90.0)
        assert rx == pytest.approx(0.0, abs=1e-9)
        assert ry == pytest.approx(1.0, abs=1e-9)

    def test_180_degrees(self):
        rx, ry = _rotate_point(3.0, 4.0, 180.0)
        assert rx == pytest.approx(-3.0, abs=1e-9)
        assert ry == pytest.approx(-4.0, abs=1e-9)

    def test_360_degrees_returns_to_origin(self):
        rx, ry = _rotate_point(3.0, 4.0, 360.0)
        assert rx == pytest.approx(3.0, abs=1e-9)
        assert ry == pytest.approx(4.0, abs=1e-9)


# ---------------------------------------------------------------------------
# _translate_points
# ---------------------------------------------------------------------------

class TestTranslatePoints:
    def test_basic_translation(self):
        pts = [(0.0, 0.0), (1.0, 2.0)]
        result = _translate_points(pts, 3.0, 5.0)
        assert result == ((3.0, 5.0), (4.0, 7.0))

    def test_negative_translation(self):
        pts = [(5.0, 5.0)]
        result = _translate_points(pts, -2.0, -3.0)
        assert result == ((3.0, 2.0),)

    def test_identity_translation(self):
        pts = [(1.0, 2.0), (3.0, 4.0)]
        result = _translate_points(pts, 0.0, 0.0)
        assert result == tuple(pts)


# ---------------------------------------------------------------------------
# _face_rectangle
# ---------------------------------------------------------------------------

class TestFaceRectangle:
    def test_center_anchor_has_four_points(self):
        pts = _face_rectangle(40.0, 4.0, 0.0, 0.0, 0.0, "center")
        assert len(pts) == 4

    def test_center_anchor_symmetric_around_origin(self):
        pts = _face_rectangle(40.0, 4.0, 0.0, 0.0, 0.0, "center")
        xs = [p[0] for p in pts]
        assert min(xs) == pytest.approx(-20.0)
        assert max(xs) == pytest.approx(20.0)

    def test_left_anchor_starts_at_anchor_x(self):
        pts = _face_rectangle(40.0, 4.0, 0.0, 10.0, 0.0, "left")
        xs = [p[0] for p in pts]
        assert min(xs) == pytest.approx(10.0)
        assert max(xs) == pytest.approx(50.0)

    def test_right_anchor_ends_at_anchor_x(self):
        pts = _face_rectangle(40.0, 4.0, 0.0, 10.0, 0.0, "right")
        xs = [p[0] for p in pts]
        assert max(xs) == pytest.approx(10.0)
        assert min(xs) == pytest.approx(-30.0)

    def test_invalid_anchor_raises(self):
        with pytest.raises(ValueError):
            _face_rectangle(40.0, 4.0, 0.0, 0.0, 0.0, "invalid")


# ---------------------------------------------------------------------------
# find_candidates
# ---------------------------------------------------------------------------

class TestFindCandidates:
    def _stock(self, height=36.0):
        return build_default_stock_windows(height=height)

    def test_returns_list(self):
        c = default_constraints()
        result = find_candidates(c, self._stock())
        assert isinstance(result, list)

    def test_results_sorted_descending_by_score(self):
        c = default_constraints()
        candidates = find_candidates(c, self._stock())
        scores = [cand.score for cand in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_all_candidates_meet_projection_requirement(self):
        c = default_constraints(projection_depth=16.0)
        for cand in find_candidates(c, self._stock()):
            assert cand.projection_depth >= c.projection_depth

    def test_all_candidates_meet_passage_fraction_requirement(self):
        c = default_constraints(min_passage_fraction=0.65)
        for cand in find_candidates(c, self._stock()):
            assert cand.passage_overlap_fraction >= c.min_passage_fraction - 1e-9

    def test_impossible_projection_returns_empty(self):
        # Require 1000" projection — impossible with stock windows
        c = default_constraints(projection_depth=1000.0)
        result = find_candidates(c, self._stock())
        assert result == []

    def test_no_matching_height_returns_empty(self):
        c = default_constraints(opening_height=99.0)
        # Stock windows built for height 36" won't match height 99"
        result = find_candidates(c, self._stock(height=36.0))
        assert result == []

    def test_candidate_fields_are_consistent(self):
        c = default_constraints()
        candidates = find_candidates(c, self._stock())
        assert len(candidates) > 0
        cand = candidates[0]
        # projection stored on candidate should be from the side face
        expected_proj = calculate_projection_from_side_faces(
            cand.side_face.finished_face_width,
            cand.side_angle_deg,
        )
        assert cand.projection_depth == pytest.approx(expected_proj)


# ---------------------------------------------------------------------------
# verify_candidate
# ---------------------------------------------------------------------------

class TestVerifyCandidate:
    def test_valid_candidate_returns_bay_candidate(self):
        c = default_constraints(projection_depth=14.0, side_angle_deg=30.0)
        # 36"-wide side window at 30° with default frame gives enough projection
        cand = verify_candidate(c, side_width=36.0, center_width=30.0, center_count=2)
        assert cand.side_window.width == 36.0
        assert cand.center_window.width == 30.0
        assert cand.center_count == 2

    def test_insufficient_projection_raises(self):
        # Require 200" projection — impossible with a narrow window
        c = default_constraints(projection_depth=200.0)
        with pytest.raises(ValueError, match="projection"):
            verify_candidate(c, side_width=24.0, center_width=24.0, center_count=1)

    def test_insufficient_passage_coverage_raises(self):
        # Use a tiny center window so coverage < min_passage_fraction
        c = default_constraints(
            opening_width=200.0,
            projection_depth=1.0,
            side_angle_deg=30.0,
            min_passage_fraction=0.65,
        )
        with pytest.raises(ValueError, match="cover"):
            verify_candidate(c, side_width=48.0, center_width=18.0, center_count=1)

    def test_side_window_below_min_raises(self):
        c = default_constraints(min_unit_width=20.0)
        with pytest.raises(ValueError):
            verify_candidate(c, side_width=10.0, center_width=30.0, center_count=1)

    def test_center_window_above_max_raises(self):
        c = default_constraints(max_single_unit_width=40.0)
        with pytest.raises(ValueError):
            verify_candidate(c, side_width=30.0, center_width=50.0, center_count=1)


# ---------------------------------------------------------------------------
# format_candidate
# ---------------------------------------------------------------------------

class TestFormatCandidate:
    def _make_candidate(self):
        c = default_constraints(projection_depth=14.0)
        return verify_candidate(c, side_width=36.0, center_width=30.0, center_count=2)

    def test_contains_rank_when_given(self):
        cand = self._make_candidate()
        text = format_candidate(cand, rank=3)
        assert "#3" in text

    def test_contains_side_window_info(self):
        cand = self._make_candidate()
        text = format_candidate(cand)
        assert '36"' in text

    def test_contains_center_window_count(self):
        cand = self._make_candidate()
        text = format_candidate(cand)
        assert "2 x" in text

    def test_contains_score(self):
        cand = self._make_candidate()
        text = format_candidate(cand)
        assert "Score:" in text

    def test_contains_projection(self):
        cand = self._make_candidate()
        text = format_candidate(cand)
        assert "projection" in text.lower()


# ---------------------------------------------------------------------------
# render_candidate_svg
# ---------------------------------------------------------------------------

class TestRenderCandidateSvg:
    def _candidate(self):
        c = default_constraints(projection_depth=14.0)
        return verify_candidate(c, side_width=36.0, center_width=30.0, center_count=2)

    def test_svg_text_starts_with_svg_tag(self):
        layout = render_candidate_svg(self._candidate())
        assert layout.svg_text.strip().startswith("<svg ")

    def test_svg_text_closes_with_svg_tag(self):
        layout = render_candidate_svg(self._candidate())
        assert layout.svg_text.strip().endswith("</svg>")

    def test_dimensions_are_positive(self):
        layout = render_candidate_svg(self._candidate())
        assert layout.width > 0
        assert layout.height > 0

    def test_view_box_matches_dimensions(self):
        layout = render_candidate_svg(self._candidate())
        parts = layout.view_box.split()
        assert len(parts) == 4
        assert float(parts[2]) == pytest.approx(layout.width, rel=1e-3)
        assert float(parts[3]) == pytest.approx(layout.height, rel=1e-3)

    def test_svg_contains_three_polygons(self):
        layout = render_candidate_svg(self._candidate())
        assert layout.svg_text.count("<polygon") == 3

    def test_faces_tuple_has_three_entries(self):
        layout = render_candidate_svg(self._candidate())
        assert len(layout.faces) == 3
        labels = {f.label for f in layout.faces}
        assert labels == {"left", "center", "right"}

    def test_svg_escapes_special_characters(self):
        """SVG text must not contain raw unescaped < or > inside element content."""
        layout = render_candidate_svg(self._candidate())
        # Titles are escaped; the only < and > should be tag delimiters
        import re
        # Extract text content nodes (between > and <) and verify none contain < or >
        content_nodes = re.findall(r">([^<]+)<", layout.svg_text)
        for node in content_nodes:
            assert "<" not in node
            assert ">" not in node

    def test_larger_margin_increases_canvas_size(self):
        cand = self._candidate()
        small = render_candidate_svg(cand, margin=10.0)
        large = render_candidate_svg(cand, margin=50.0)
        assert large.width > small.width
        assert large.height > small.height


# ---------------------------------------------------------------------------
# candidate_to_dict
# ---------------------------------------------------------------------------

class TestCandidateToDict:
    def _candidate(self):
        c = default_constraints(projection_depth=14.0)
        return verify_candidate(c, side_width=36.0, center_width=30.0, center_count=2)

    def _constraints(self):
        return default_constraints(projection_depth=14.0, sill_height=18.0, frame_body_depth=5.0)

    def test_top_level_keys_present(self):
        d = candidate_to_dict(self._candidate())
        for key in ("side_window", "center_window", "center_count", "side_angle_deg",
                    "side_face", "center_face", "projection_depth", "wall_parallel_span",
                    "outside_corner_to_corner_width", "passage_overlap_width",
                    "passage_overlap_fraction", "score", "notes"):
            assert key in d, f"Missing key: {key}"

    def test_side_window_nested_keys(self):
        d = candidate_to_dict(self._candidate())
        assert set(d["side_window"]) == {"width", "height", "style"}

    def test_center_face_nested_keys(self):
        d = candidate_to_dict(self._candidate())
        assert set(d["center_face"]) == {"window_area_width", "rough_opening_width", "finished_face_width"}

    def test_notes_is_a_list(self):
        d = candidate_to_dict(self._candidate())
        assert isinstance(d["notes"], list)

    def test_rank_absent_by_default(self):
        d = candidate_to_dict(self._candidate())
        assert "rank" not in d

    def test_rank_present_when_given(self):
        d = candidate_to_dict(self._candidate(), rank=3)
        assert d["rank"] == 3

    def test_values_match_candidate_fields(self):
        cand = self._candidate()
        d = candidate_to_dict(cand)
        assert d["center_count"] == cand.center_count
        assert d["projection_depth"] == pytest.approx(cand.projection_depth)
        assert d["side_window"]["width"] == cand.side_window.width
        assert d["center_face"]["finished_face_width"] == pytest.approx(cand.center_face.finished_face_width)

    def test_result_is_json_serialisable(self):
        d = candidate_to_dict(self._candidate())
        text = json.dumps(d)
        assert isinstance(text, str)

    def test_constraints_absent_when_not_passed(self):
        d = candidate_to_dict(self._candidate())
        assert "constraints" not in d

    def test_constraints_present_when_passed(self):
        d = candidate_to_dict(self._candidate(), constraints=self._constraints())
        assert "constraints" in d

    def test_constraints_contains_opening_fields(self):
        d = candidate_to_dict(self._candidate(), constraints=self._constraints())
        c = d["constraints"]
        assert c["opening_width"] == pytest.approx(72.0)
        assert c["opening_height"] == pytest.approx(36.0)

    def test_constraints_contains_sill_height(self):
        d = candidate_to_dict(self._candidate(), constraints=self._constraints())
        assert d["constraints"]["sill_height"] == pytest.approx(18.0)

    def test_constraints_sill_height_none_when_not_set(self):
        c = default_constraints(projection_depth=14.0)  # sill_height defaults to None
        d = candidate_to_dict(self._candidate(), constraints=c)
        assert d["constraints"]["sill_height"] is None

    def test_constraints_contains_frame_body_depth(self):
        d = candidate_to_dict(self._candidate(), constraints=self._constraints())
        assert d["constraints"]["frame_body_depth"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# write_candidates_json / write_candidate_json
# ---------------------------------------------------------------------------

class TestWriteCandidatesJson:
    def _candidates(self):
        c = default_constraints(projection_depth=14.0)
        return find_candidates(c, build_default_stock_windows(height=36.0))[:3]

    def _constraints(self):
        return default_constraints(projection_depth=14.0, sill_height=12.0, frame_body_depth=4.0)

    def test_creates_file(self, tmp_path: Path):
        out = tmp_path / "results.json"
        write_candidates_json(self._candidates(), out)
        assert out.exists()

    def test_output_is_valid_json_array(self, tmp_path: Path):
        out = tmp_path / "results.json"
        write_candidates_json(self._candidates(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, list)

    def test_array_length_matches_input(self, tmp_path: Path):
        candidates = self._candidates()
        out = tmp_path / "results.json"
        write_candidates_json(candidates, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) == len(candidates)

    def test_rank_field_is_1_indexed(self, tmp_path: Path):
        out = tmp_path / "results.json"
        write_candidates_json(self._candidates(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data[0]["rank"] == 1
        assert data[-1]["rank"] == len(data)

    def test_creates_parent_directory(self, tmp_path: Path):
        out = tmp_path / "subdir" / "results.json"
        write_candidates_json(self._candidates(), out)
        assert out.exists()

    def test_returns_output_path(self, tmp_path: Path):
        out = tmp_path / "results.json"
        result = write_candidates_json(self._candidates(), out)
        assert result == out

    def test_constraints_embedded_when_passed(self, tmp_path: Path):
        out = tmp_path / "results.json"
        write_candidates_json(self._candidates(), out, constraints=self._constraints())
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "constraints" in data[0]
        assert data[0]["constraints"]["opening_width"] == pytest.approx(72.0)
        assert data[0]["constraints"]["sill_height"] == pytest.approx(12.0)
        assert data[0]["constraints"]["frame_body_depth"] == pytest.approx(4.0)

    def test_constraints_absent_when_not_passed(self, tmp_path: Path):
        out = tmp_path / "results.json"
        write_candidates_json(self._candidates(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "constraints" not in data[0]


class TestWriteCandidateJson:
    def _candidate(self):
        c = default_constraints(projection_depth=14.0)
        return verify_candidate(c, side_width=36.0, center_width=30.0, center_count=2)

    def _constraints(self):
        return default_constraints(projection_depth=14.0, sill_height=24.0, frame_body_depth=3.5)

    def test_creates_file(self, tmp_path: Path):
        out = tmp_path / "verified.json"
        write_candidate_json(self._candidate(), out)
        assert out.exists()

    def test_output_is_valid_json_object(self, tmp_path: Path):
        out = tmp_path / "verified.json"
        write_candidate_json(self._candidate(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_contains_expected_fields(self, tmp_path: Path):
        out = tmp_path / "verified.json"
        write_candidate_json(self._candidate(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "side_window" in data
        assert "center_face" in data
        assert "projection_depth" in data

    def test_no_rank_field(self, tmp_path: Path):
        out = tmp_path / "verified.json"
        write_candidate_json(self._candidate(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "rank" not in data

    def test_creates_parent_directory(self, tmp_path: Path):
        out = tmp_path / "nested" / "dir" / "verified.json"
        write_candidate_json(self._candidate(), out)
        assert out.exists()

    def test_returns_output_path(self, tmp_path: Path):
        out = tmp_path / "verified.json"
        result = write_candidate_json(self._candidate(), out)
        assert result == out

    def test_constraints_embedded_when_passed(self, tmp_path: Path):
        out = tmp_path / "verified.json"
        write_candidate_json(self._candidate(), out, constraints=self._constraints())
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "constraints" in data
        assert data["constraints"]["opening_width"] == pytest.approx(72.0)
        assert data["constraints"]["opening_height"] == pytest.approx(36.0)
        assert data["constraints"]["sill_height"] == pytest.approx(24.0)
        assert data["constraints"]["frame_body_depth"] == pytest.approx(3.5)

    def test_constraints_absent_when_not_passed(self, tmp_path: Path):
        out = tmp_path / "verified.json"
        write_candidate_json(self._candidate(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "constraints" not in data


# ---------------------------------------------------------------------------
# BayConstraints – defaults and immutability
# ---------------------------------------------------------------------------

class TestBayConstraintsDefaults:
    def _minimal(self) -> BayConstraints:
        return BayConstraints(opening_width=72.0, opening_height=36.0,
                              projection_depth=16.0, side_angle_deg=30.0)

    def test_sill_height_defaults_to_none(self):
        assert self._minimal().sill_height is None

    def test_frame_body_depth_defaults_to_4(self):
        assert self._minimal().frame_body_depth == pytest.approx(4.0)

    def test_frame_thickness_default(self):
        assert self._minimal().frame_thickness == pytest.approx(1.5)

    def test_rough_clearance_default(self):
        assert self._minimal().rough_clearance == pytest.approx(0.5)

    def test_mullion_thickness_default(self):
        assert self._minimal().mullion_thickness == pytest.approx(1.5)

    def test_min_passage_fraction_default(self):
        assert self._minimal().min_passage_fraction == pytest.approx(0.65)

    def test_min_center_units_default(self):
        assert self._minimal().min_center_units == 1

    def test_max_center_units_default(self):
        assert self._minimal().max_center_units == 4

    def test_frozen_cannot_be_mutated(self):
        c = self._minimal()
        with pytest.raises((AttributeError, TypeError)):
            c.opening_width = 100.0  # type: ignore

    def test_hashable(self):
        c = self._minimal()
        assert {c: "value"}[c] == "value"

    def test_equality_same_values(self):
        assert self._minimal() == self._minimal()

    def test_inequality_different_opening_width(self):
        c1 = self._minimal()
        c2 = BayConstraints(opening_width=80.0, opening_height=36.0,
                             projection_depth=16.0, side_angle_deg=30.0)
        assert c1 != c2


# ---------------------------------------------------------------------------
# WindowUnit – defaults and equality
# ---------------------------------------------------------------------------

class TestWindowUnitEquality:
    def test_default_style_is_double_hung(self):
        assert WindowUnit(width=24.0, height=36.0).style == "double_hung"

    def test_equal_when_same_params(self):
        assert WindowUnit(width=24.0, height=36.0) == WindowUnit(width=24.0, height=36.0)

    def test_unequal_different_width(self):
        assert WindowUnit(width=24.0, height=36.0) != WindowUnit(width=30.0, height=36.0)

    def test_hashable_deduplicated_in_set(self):
        w1 = WindowUnit(width=24.0, height=36.0)
        w2 = WindowUnit(width=24.0, height=36.0)
        assert len({w1, w2}) == 1

    def test_frozen_cannot_mutate(self):
        w = WindowUnit(width=24.0, height=36.0)
        with pytest.raises((AttributeError, TypeError)):
            w.width = 30.0  # type: ignore


# ---------------------------------------------------------------------------
# FaceDimensions – basic structural checks
# ---------------------------------------------------------------------------

class TestFaceDimensions:
    def _make(self) -> FaceDimensions:
        return FaceDimensions(window_area_width=24.0, rough_opening_width=25.0, finished_face_width=28.0)

    def test_stores_all_three_fields(self):
        face = self._make()
        assert face.window_area_width == 24.0
        assert face.rough_opening_width == 25.0
        assert face.finished_face_width == 28.0

    def test_frozen_cannot_mutate(self):
        with pytest.raises((AttributeError, TypeError)):
            self._make().window_area_width = 30.0  # type: ignore

    def test_equality(self):
        assert self._make() == self._make()

    def test_inequality_on_different_value(self):
        f1 = self._make()
        f2 = FaceDimensions(window_area_width=30.0, rough_opening_width=25.0, finished_face_width=28.0)
        assert f1 != f2


# ---------------------------------------------------------------------------
# calculate_single_window_face – edge cases
# ---------------------------------------------------------------------------

class TestCalculateSingleWindowFaceEdgeCases:
    def test_zero_clearance_zero_frame_equals_window_width(self):
        w = WindowUnit(width=30.0, height=36.0)
        c = default_constraints(rough_clearance=0.0, frame_thickness=0.0, face_trim_allowance=0.0)
        face = calculate_single_window_face(w, c)
        assert face.finished_face_width == pytest.approx(30.0)
        assert face.rough_opening_width == pytest.approx(30.0)

    def test_thicker_frame_adds_proportionally(self):
        w = WindowUnit(width=24.0, height=36.0)
        c1 = default_constraints(frame_thickness=1.0, rough_clearance=0.0, face_trim_allowance=0.0)
        c2 = default_constraints(frame_thickness=2.0, rough_clearance=0.0, face_trim_allowance=0.0)
        diff = (calculate_single_window_face(w, c2).finished_face_width
                - calculate_single_window_face(w, c1).finished_face_width)
        assert diff == pytest.approx(2.0)

    def test_large_window_width(self):
        w = WindowUnit(width=48.0, height=36.0)
        c = default_constraints(frame_thickness=1.5, rough_clearance=0.5, face_trim_allowance=0.0)
        face = calculate_single_window_face(w, c)
        # 48 + 2*0.5 + 2*1.5 = 52.0
        assert face.finished_face_width == pytest.approx(52.0)


# ---------------------------------------------------------------------------
# calculate_center_face – extended
# ---------------------------------------------------------------------------

class TestCalculateCenterFaceExtended:
    def test_four_units_three_mullions(self):
        w = WindowUnit(width=24.0, height=36.0)
        c = default_constraints(frame_thickness=1.5, rough_clearance=0.5, mullion_thickness=1.5)
        face = calculate_center_face(w, 4, c)
        # rough: 4*(24+1)=100; mullions: 3*1.5=4.5; finished: 100+4.5+3=107.5
        assert face.rough_opening_width == pytest.approx(100.0)
        assert face.finished_face_width == pytest.approx(107.5)
        assert face.window_area_width == pytest.approx(96.0)

    def test_window_area_width_always_count_times_unit_width(self):
        w = WindowUnit(width=20.0, height=36.0)
        c = default_constraints()
        for n in range(1, 5):
            assert calculate_center_face(w, n, c).window_area_width == pytest.approx(n * 20.0)

    def test_face_trim_applied_exactly_once_regardless_of_unit_count(self):
        w = WindowUnit(width=24.0, height=36.0)
        c_no_trim = default_constraints(face_trim_allowance=0.0)
        c_trim = default_constraints(face_trim_allowance=2.0)
        for n in range(1, 5):
            diff = (calculate_center_face(w, n, c_trim).finished_face_width
                    - calculate_center_face(w, n, c_no_trim).finished_face_width)
            assert diff == pytest.approx(2.0), f"trim diff wrong for n={n}"

    def test_negative_count_raises(self):
        w = WindowUnit(width=24.0, height=36.0)
        with pytest.raises(ValueError):
            calculate_center_face(w, -1, default_constraints())


# ---------------------------------------------------------------------------
# calculate_wall_parallel_span – extended
# ---------------------------------------------------------------------------

class TestCalculateWallParallelSpanExtended:
    def test_45_degree_known_value(self):
        span = calculate_wall_parallel_span(
            center_face_width=60.0, side_face_width=30.0, side_angle_deg=45.0)
        assert span == pytest.approx(60.0 + 2.0 * 30.0 * math.cos(math.radians(45.0)))

    def test_wider_center_face_increases_span(self):
        assert (calculate_wall_parallel_span(60.0, 30.0, 30.0)
                > calculate_wall_parallel_span(40.0, 30.0, 30.0))

    def test_wider_side_face_increases_span(self):
        assert (calculate_wall_parallel_span(60.0, 40.0, 30.0)
                > calculate_wall_parallel_span(60.0, 20.0, 30.0))


# ---------------------------------------------------------------------------
# calculate_projection_from_side_faces – extended
# ---------------------------------------------------------------------------

class TestCalculateProjectionExtended:
    def test_45_degree_known_value(self):
        proj = calculate_projection_from_side_faces(side_face_width=30.0, side_angle_deg=45.0)
        assert proj == pytest.approx(30.0 * math.sin(math.radians(45.0)))

    def test_60_degree_known_value(self):
        proj = calculate_projection_from_side_faces(side_face_width=30.0, side_angle_deg=60.0)
        assert proj == pytest.approx(30.0 * math.sin(math.radians(60.0)))

    def test_wider_face_increases_projection(self):
        assert (calculate_projection_from_side_faces(40.0, 30.0)
                > calculate_projection_from_side_faces(20.0, 30.0))

    def test_steeper_angle_increases_projection(self):
        assert (calculate_projection_from_side_faces(30.0, 60.0)
                > calculate_projection_from_side_faces(30.0, 30.0))


# ---------------------------------------------------------------------------
# score_candidate – extended
# ---------------------------------------------------------------------------

class TestScoreCandidateExtended:
    def _score(self, side_w=26.0, center_w=48.0, center_count=2, overlap=0.85):
        """Uses center_w=48 by default so glass-area bonus saturates for all counts,
        leaving centre-count preference as the deciding factor."""
        c = default_constraints()
        center_face = calculate_center_face(WindowUnit(width=center_w, height=36.0), center_count, c)
        return score_candidate(
            side_window=WindowUnit(width=side_w, height=36.0),
            center_window=WindowUnit(width=center_w, height=36.0),
            center_count=center_count,
            center_face=center_face,
            passage_overlap_fraction=overlap,
            constraints=c,
        )

    def test_two_units_scores_higher_than_four(self):
        assert self._score(center_count=2) > self._score(center_count=4)

    def test_two_units_scores_higher_than_one(self):
        assert self._score(center_count=2) > self._score(center_count=1)

    def test_three_units_scores_higher_than_four(self):
        assert self._score(center_count=3) > self._score(center_count=4)

    def test_center_slightly_wider_earns_bonus(self):
        # delta=4 (within 2–10" bonus range) vs equal widths; use small center_w so area bonus doesn't saturate
        c = default_constraints()
        def score(side_w, center_w):
            cf = calculate_center_face(WindowUnit(width=center_w, height=36.0), 2, c)
            return score_candidate(
                side_window=WindowUnit(width=side_w, height=36.0),
                center_window=WindowUnit(width=center_w, height=36.0),
                center_count=2,
                center_face=cf,
                passage_overlap_fraction=0.85,
                constraints=c,
            )
        bonus = score(side_w=26.0, center_w=30.0)   # delta = 4
        same  = score(side_w=30.0, center_w=30.0)   # delta = 0
        assert bonus > same

    def test_overlap_fraction_is_monotone_in_score(self):
        low = self._score(overlap=0.70)
        mid = self._score(overlap=0.85)
        high = self._score(overlap=1.00)
        assert high > mid > low


# ---------------------------------------------------------------------------
# build_notes – extended
# ---------------------------------------------------------------------------

class TestBuildNotesExtended:
    def _c(self):
        return default_constraints(min_passage_fraction=0.65)

    def test_returns_tuple(self):
        notes = build_notes(WindowUnit(30.0, 36.0), WindowUnit(30.0, 36.0),
                            center_count=1, passage_overlap_fraction=0.85, constraints=self._c())
        assert isinstance(notes, tuple)

    def test_all_notes_are_strings(self):
        notes = build_notes(WindowUnit(30.0, 36.0), WindowUnit(30.0, 36.0),
                            center_count=1, passage_overlap_fraction=0.85, constraints=self._c())
        assert all(isinstance(n, str) for n in notes)

    def test_high_coverage_triggers_covers_most_note(self):
        # min_passage_fraction=0.65; threshold=max(0.80, 0.90)=0.90
        notes = build_notes(WindowUnit(30.0, 36.0), WindowUnit(30.0, 36.0),
                            center_count=1, passage_overlap_fraction=0.95, constraints=self._c())
        assert any("most" in n for n in notes)

    def test_moderate_coverage_triggers_acceptable_note(self):
        # 0.65 <= 0.70 < 0.90 → "acceptable"
        notes = build_notes(WindowUnit(30.0, 36.0), WindowUnit(30.0, 36.0),
                            center_count=1, passage_overlap_fraction=0.70, constraints=self._c())
        assert any("acceptable" in n for n in notes)

    def test_below_min_fraction_no_passage_note(self):
        notes = build_notes(WindowUnit(30.0, 36.0), WindowUnit(30.0, 36.0),
                            center_count=1, passage_overlap_fraction=0.50, constraints=self._c())
        assert not any("passage" in n.lower() for n in notes)

    def test_four_center_units_note(self):
        notes = build_notes(WindowUnit(30.0, 36.0), WindowUnit(30.0, 36.0),
                            center_count=4, passage_overlap_fraction=0.85, constraints=self._c())
        assert any("4 center" in n for n in notes)


# ---------------------------------------------------------------------------
# filter_stock_by_height – extended
# ---------------------------------------------------------------------------

class TestFilterStockByHeightExtended:
    def test_preserves_order(self):
        stock = [
            WindowUnit(width=48.0, height=36.0),
            WindowUnit(width=24.0, height=36.0),
            WindowUnit(width=36.0, height=36.0),
        ]
        result = filter_stock_by_height(stock, 36.0)
        assert [w.width for w in result] == [48.0, 24.0, 36.0]

    def test_zero_tolerance_exact_match_only(self):
        stock = [WindowUnit(width=24.0, height=36.0), WindowUnit(width=24.0, height=36.001)]
        result = filter_stock_by_height(stock, 36.0, tolerance=0.0)
        assert len(result) == 1
        assert result[0].width == 24.0

    def test_all_matching_returned(self):
        stock = [WindowUnit(width=float(w), height=36.0) for w in range(18, 50, 2)]
        assert len(filter_stock_by_height(stock, 36.0)) == len(stock)


# ---------------------------------------------------------------------------
# build_default_stock_windows – extended
# ---------------------------------------------------------------------------

class TestBuildDefaultStockWindowsExtended:
    def test_all_windows_are_double_hung(self):
        assert all(w.style == "double_hung" for w in build_default_stock_windows(36.0))

    def test_all_widths_are_unique(self):
        widths = [w.width for w in build_default_stock_windows(36.0)]
        assert len(widths) == len(set(widths))

    def test_widths_in_ascending_order(self):
        widths = [w.width for w in build_default_stock_windows(36.0)]
        assert widths == sorted(widths)


# ---------------------------------------------------------------------------
# is_window_usable – extended
# ---------------------------------------------------------------------------

class TestIsWindowUsableExtended:
    def test_min_equals_max_only_that_width_passes(self):
        c = default_constraints(min_unit_width=24.0, max_single_unit_width=24.0)
        assert is_window_usable(WindowUnit(width=24.0, height=36.0), c) is True
        assert is_window_usable(WindowUnit(width=23.9, height=36.0), c) is False
        assert is_window_usable(WindowUnit(width=24.1, height=36.0), c) is False


# ---------------------------------------------------------------------------
# _rotate_point – extended
# ---------------------------------------------------------------------------

class TestRotatePointExtended:
    def test_45_degrees_unit_vector(self):
        rx, ry = _rotate_point(1.0, 0.0, 45.0)
        expected = math.sqrt(2.0) / 2.0
        assert rx == pytest.approx(expected, abs=1e-9)
        assert ry == pytest.approx(expected, abs=1e-9)

    def test_rotation_preserves_distance_from_origin(self):
        x, y = 3.0, 4.0
        dist = math.hypot(x, y)
        for angle in [15.0, 37.0, 90.0, 135.0, 270.0]:
            rx, ry = _rotate_point(x, y, angle)
            assert math.hypot(rx, ry) == pytest.approx(dist, abs=1e-9)

    def test_negative_angle_is_inverse_of_positive(self):
        x, y = 5.0, 2.0
        rx, ry = _rotate_point(x, y, 45.0)
        rx2, ry2 = _rotate_point(rx, ry, -45.0)
        assert rx2 == pytest.approx(x, abs=1e-9)
        assert ry2 == pytest.approx(y, abs=1e-9)


# ---------------------------------------------------------------------------
# _translate_points – extended
# ---------------------------------------------------------------------------

class TestTranslatePointsExtended:
    def test_empty_input_returns_empty_tuple(self):
        assert _translate_points([], 5.0, 5.0) == ()

    def test_round_trip_back_to_original(self):
        pts = [(1.0, 2.0), (3.0, 4.0), (-1.0, 5.0)]
        shifted = _translate_points(pts, 7.0, -3.0)
        back = _translate_points(shifted, -7.0, 3.0)
        for orig, restored in zip(pts, back):
            assert restored == pytest.approx(orig, abs=1e-9)

    def test_result_is_tuple(self):
        assert isinstance(_translate_points([(0.0, 0.0)], 1.0, 1.0), tuple)


# ---------------------------------------------------------------------------
# _face_rectangle – extended
# ---------------------------------------------------------------------------

class TestFaceRectangleExtended:
    def test_unrotated_y_values_span_zero_to_depth(self):
        depth = 6.0
        pts = _face_rectangle(40.0, depth, 0.0, 0.0, 0.0, "center")
        ys = [p[1] for p in pts]
        assert min(ys) == pytest.approx(0.0, abs=1e-9)
        assert max(ys) == pytest.approx(depth, abs=1e-9)

    def test_depth_controls_y_extent(self):
        for depth in [2.0, 5.0, 10.0]:
            pts = _face_rectangle(40.0, depth, 0.0, 0.0, 0.0, "left")
            ys = [p[1] for p in pts]
            assert max(ys) - min(ys) == pytest.approx(depth, abs=1e-9)

    def test_90_degree_rotation_swaps_length_and_depth_axes(self):
        # After 90° CCW rotation of a 40×4 rect: x-range≈4, y-range≈40
        pts = _face_rectangle(40.0, 4.0, 90.0, 0.0, 0.0, "center")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        assert max(xs) - min(xs) == pytest.approx(4.0, abs=1e-9)
        assert max(ys) - min(ys) == pytest.approx(40.0, abs=1e-9)


# ---------------------------------------------------------------------------
# render_candidate_svg – extended
# ---------------------------------------------------------------------------

class TestRenderCandidateSvgExtended:
    def _candidate(self, center_count=2):
        c = default_constraints(projection_depth=14.0)
        return verify_candidate(c, side_width=36.0, center_width=30.0, center_count=center_count)

    def test_title_contains_side_angle(self):
        layout = render_candidate_svg(self._candidate())
        assert "30" in layout.svg_text  # side_angle_deg=30

    def test_single_center_window_renders_without_error(self):
        # Use a wide center unit so passage-coverage constraint is satisfied for count=1
        c = default_constraints(projection_depth=14.0)
        cand = verify_candidate(c, side_width=36.0, center_width=48.0, center_count=1)
        layout = render_candidate_svg(cand)
        assert layout.svg_text.strip().startswith("<svg ")
        assert len(layout.faces) == 3

    def test_frame_body_depth_changes_svg_output(self):
        cand = self._candidate()
        thin = render_candidate_svg(cand, frame_body_depth=1.0)
        thick = render_candidate_svg(cand, frame_body_depth=20.0)
        assert thin.svg_text != thick.svg_text

    def test_each_face_polygon_has_four_points(self):
        layout = render_candidate_svg(self._candidate())
        for face in layout.faces:
            assert len(face.points) == 4


# ---------------------------------------------------------------------------
# find_candidates – extended
# ---------------------------------------------------------------------------

class TestFindCandidatesExtended:
    def _stock(self):
        return build_default_stock_windows(height=36.0)

    def test_respects_max_center_units(self):
        c = default_constraints(max_center_units=2)
        assert all(cand.center_count <= 2 for cand in find_candidates(c, self._stock()))

    def test_respects_min_center_units(self):
        c = default_constraints(min_center_units=2)
        assert all(cand.center_count >= 2 for cand in find_candidates(c, self._stock()))

    def test_all_window_heights_match_opening_height(self):
        c = default_constraints(opening_height=36.0)
        for cand in find_candidates(c, self._stock()):
            assert cand.side_window.height == pytest.approx(36.0)
            assert cand.center_window.height == pytest.approx(36.0)

    def test_outside_corner_width_consistent_with_formula(self):
        c = default_constraints()
        for cand in find_candidates(c, self._stock())[:10]:
            expected = calculate_outside_corner_to_corner_width(
                cand.center_face.finished_face_width,
                cand.side_face.finished_face_width,
            )
            assert cand.outside_corner_to_corner_width == pytest.approx(expected)

    def test_center_face_projection_contribution_always_zero(self):
        c = default_constraints()
        for cand in find_candidates(c, self._stock())[:10]:
            assert cand.center_face_projection_contribution == pytest.approx(0.0)

    def test_custom_stock_windows_used(self):
        custom = [WindowUnit(width=24.0, height=36.0), WindowUnit(width=36.0, height=36.0)]
        c = default_constraints(projection_depth=14.0)
        results = find_candidates(c, custom)
        widths_used = (
            {cand.side_window.width for cand in results}
            | {cand.center_window.width for cand in results}
        )
        assert widths_used.issubset({24.0, 36.0})


# ---------------------------------------------------------------------------
# verify_candidate – extended
# ---------------------------------------------------------------------------

class TestVerifyCandidateExtended:
    def test_passage_overlap_fraction_capped_at_one(self):
        # Center face wider than the opening should clamp the fraction to 1.0
        c = default_constraints(opening_width=40.0, projection_depth=12.0)
        cand = verify_candidate(c, side_width=30.0, center_width=48.0, center_count=1)
        assert cand.passage_overlap_fraction == pytest.approx(1.0)

    def test_center_face_projection_contribution_is_zero(self):
        c = default_constraints(projection_depth=14.0)
        cand = verify_candidate(c, side_width=36.0, center_width=30.0, center_count=2)
        assert cand.center_face_projection_contribution == pytest.approx(0.0)

    def test_side_face_projection_contribution_equals_projection_depth(self):
        c = default_constraints(projection_depth=14.0)
        cand = verify_candidate(c, side_width=36.0, center_width=30.0, center_count=2)
        assert cand.side_face_projection_contribution == pytest.approx(cand.projection_depth)

    def test_notes_are_non_empty(self):
        c = default_constraints(projection_depth=14.0)
        assert len(verify_candidate(c, side_width=36.0, center_width=30.0, center_count=2).notes) > 0

    def test_custom_style_is_stored(self):
        c = default_constraints(projection_depth=14.0)
        cand = verify_candidate(c, side_width=36.0, center_width=30.0, center_count=2, style="casement")
        assert cand.side_window.style == "casement"
        assert cand.center_window.style == "casement"

    def test_wall_parallel_span_consistent_with_formula(self):
        c = default_constraints(projection_depth=14.0)
        cand = verify_candidate(c, side_width=36.0, center_width=30.0, center_count=2)
        expected = calculate_wall_parallel_span(
            cand.center_face.finished_face_width,
            cand.side_face.finished_face_width,
            cand.side_angle_deg,
        )
        assert cand.wall_parallel_span == pytest.approx(expected)


# ---------------------------------------------------------------------------
# format_candidate – extended
# ---------------------------------------------------------------------------

class TestFormatCandidateExtended:
    def _candidate(self):
        c = default_constraints(projection_depth=14.0)
        return verify_candidate(c, side_width=36.0, center_width=30.0, center_count=2)

    def test_no_hash_symbol_when_rank_omitted(self):
        assert "#" not in format_candidate(self._candidate())

    def test_contains_wall_parallel_span_value(self):
        cand = self._candidate()
        assert f"{cand.wall_parallel_span:.2f}" in format_candidate(cand)

    def test_contains_side_angle(self):
        assert "30" in format_candidate(self._candidate())

    def test_contains_passage_overlap_percentage(self):
        assert "%" in format_candidate(self._candidate())

    def test_each_note_appears_in_output(self):
        cand = self._candidate()
        text = format_candidate(cand)
        for note in cand.notes:
            assert note[:30] in text


# ---------------------------------------------------------------------------
# Integration: find → verify → serialise round-trips
# ---------------------------------------------------------------------------

class TestIntegration:
    def _setup(self):
        c = default_constraints(projection_depth=14.0)
        stock = build_default_stock_windows(height=36.0)
        return c, stock

    def test_top_candidate_re_verified_matches(self):
        c, stock = self._setup()
        top = find_candidates(c, stock)[0]
        rv = verify_candidate(c, side_width=top.side_window.width,
                              center_width=top.center_window.width,
                              center_count=top.center_count)
        assert rv.projection_depth == pytest.approx(top.projection_depth)
        assert rv.wall_parallel_span == pytest.approx(top.wall_parallel_span)
        assert rv.passage_overlap_fraction == pytest.approx(top.passage_overlap_fraction)

    def test_top_five_candidates_all_re_verify(self):
        c, stock = self._setup()
        for cand in find_candidates(c, stock)[:5]:
            rv = verify_candidate(c, side_width=cand.side_window.width,
                                  center_width=cand.center_window.width,
                                  center_count=cand.center_count)
            assert rv.projection_depth == pytest.approx(cand.projection_depth)

    def test_candidate_to_dict_survives_json_round_trip(self):
        c, stock = self._setup()
        cand = find_candidates(c, stock)[0]
        d = candidate_to_dict(cand, rank=1, constraints=c)
        d2 = json.loads(json.dumps(d))
        assert d2["projection_depth"] == pytest.approx(cand.projection_depth, rel=1e-9)
        assert d2["center_count"] == cand.center_count
        assert d2["constraints"]["opening_width"] == pytest.approx(c.opening_width)

    def test_write_and_reload_candidates_json_preserves_count_and_ranks(self, tmp_path):
        c, stock = self._setup()
        candidates = find_candidates(c, stock)[:3]
        out = tmp_path / "rt.json"
        write_candidates_json(candidates, out, constraints=c)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) == 3
        for i, entry in enumerate(data, start=1):
            assert entry["rank"] == i

    def test_svg_render_does_not_raise_for_top_candidate(self):
        c, stock = self._setup()
        top = find_candidates(c, stock)[0]
        layout = render_candidate_svg(top, frame_body_depth=c.frame_body_depth)
        assert layout.width > 0
        assert layout.height > 0
