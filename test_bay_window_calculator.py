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
