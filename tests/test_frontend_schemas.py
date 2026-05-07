"""Static-analysis tests for the frontend config / help / popover wiring.

These guard the contracts between static/app.js, static/index.html, and
static/style.css that were established when we added per-detector config
filtering and the click-help popover. They don't execute JS, but they
catch the kinds of regressions that come from renaming/removing keys in
one place and forgetting another.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
STYLE_CSS = (ROOT / "static" / "style.css").read_text(encoding="utf-8")


# ---- helpers --------------------------------------------------------------

def _balanced_block(text: str, start: int, open_ch: str, close_ch: str) -> str:
    """Return the slice between `start` and the matching close bracket."""
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
        i += 1
    return text[start : i - 1]


def _const_array(text: str, var_name: str) -> set[str]:
    m = re.search(rf"const\s+{re.escape(var_name)}\s*=\s*\[", text)
    if not m:
        return set()
    body = _balanced_block(text, m.end(), "[", "]")
    return set(re.findall(r'"([^"]+)"', body))


def _detector_keys(text: str, detector: str) -> set[str]:
    """Resolve a DETECTOR_CONFIGS entry. Handles inline arrays, alias
    references (e.g. `median_frame: _MEDIAN_FRAME_KEYS`), and spread
    expressions (`..._MEDIAN_FRAME_KEYS`)."""
    m = re.search(rf"\b{re.escape(detector)}\s*:\s*", text)
    if not m:
        return set()
    after = text[m.end():]
    if after.startswith("["):
        body = _balanced_block(text, m.end() + 1, "[", "]")
        keys = set(re.findall(r'"([^"]+)"', body))
        spread_consts = re.findall(r"\.\.\.(\w+)", body)
        for name in spread_consts:
            keys |= _const_array(text, name)
        return keys
    alias = re.match(r"(\w+)", after)
    if alias:
        return _const_array(text, alias.group(1))
    return set()


def _help_text_keys(text: str) -> set[str]:
    m = re.search(r"const\s+HELP_TEXT\s*=\s*\{", text)
    if not m:
        return set()
    body = _balanced_block(text, m.end(), "{", "}")
    # top-level identifier keys: `foo: "..."` at the start of a line (after spaces).
    return set(re.findall(r'^\s*(\w+)\s*:\s*"', body, re.MULTILINE))


# ---- tests ----------------------------------------------------------------

class DetectorConfigSchemaTests(unittest.TestCase):
    """DETECTOR_CONFIGS is the source of truth for which knobs/badges
    belong to which detector. These tests guard its shape."""

    def test_three_known_detectors_present(self):
        for det in ("median_frame", "median_court_roi", "pose_skeleton_yolo"):
            with self.subTest(detector=det):
                keys = _detector_keys(APP_JS, det)
                self.assertGreater(
                    len(keys), 0, f"No keys parsed for {det}; check the literal"
                )

    def test_median_frame_includes_core_post_processing_knobs(self):
        keys = _detector_keys(APP_JS, "median_frame")
        for required in (
            "sample_fps",
            "median_bg_samples",
            "diff_threshold",
            "motion_threshold",
            "merge_gap_s",
            "min_segment_s",
            "segment_padding_s",
            "range_start_s",
            "range_end_s",
        ):
            self.assertIn(required, keys, f"median_frame missing {required}")

    def test_court_roi_extends_median_frame_with_weights(self):
        median = _detector_keys(APP_JS, "median_frame")
        roi = _detector_keys(APP_JS, "median_court_roi")
        self.assertTrue(
            median.issubset(roi),
            "median_court_roi should be a superset of median_frame keys",
        )
        for required in ("court_weight", "outside_weight", "near_camera_weight"):
            self.assertIn(required, roi)

    def test_pose_detector_includes_imgsz_and_excludes_motion_knobs(self):
        keys = _detector_keys(APP_JS, "pose_skeleton_yolo")
        for required in (
            "pose_model",
            "pose_conf",
            "pose_imgsz",
            "sample_fps",
            "range_start_s",
            "range_end_s",
        ):
            self.assertIn(required, keys, f"pose detector missing {required}")
        # YOLO detector doesn't run the median-frame post-processing pipeline,
        # so these knobs would be misleading on a pose analysis.
        for forbidden in (
            "diff_threshold",
            "motion_threshold",
            "court_weight",
            "outside_weight",
            "near_camera_weight",
            "median_bg_samples",
        ):
            self.assertNotIn(
                forbidden,
                keys,
                f"pose detector should not list {forbidden}",
            )


class HelpTextCoverageTests(unittest.TestCase):
    """Every config that can be displayed in either the start form or the
    Analysis Config card needs help text."""

    def test_every_detector_key_has_help_text(self):
        help_keys = _help_text_keys(APP_JS)
        all_keys = set()
        for det in ("median_frame", "median_court_roi", "pose_skeleton_yolo"):
            all_keys |= _detector_keys(APP_JS, det)
        missing = all_keys - help_keys
        self.assertEqual(
            missing,
            set(),
            f"DETECTOR_CONFIGS keys without HELP_TEXT entries: {sorted(missing)}",
        )


class HtmlContractTests(unittest.TestCase):
    """Help dots and algorithm filtering reference real keys."""

    def test_data_help_key_attributes_resolve_to_help_text(self):
        help_keys = _help_text_keys(APP_JS)
        used = set(re.findall(r'data-help-key="([^"]+)"', INDEX_HTML))
        self.assertGreater(len(used), 0, "no data-help-key attributes found")
        unknown = used - help_keys
        self.assertEqual(
            unknown,
            set(),
            f"data-help-key values without HELP_TEXT entries: {sorted(unknown)}",
        )

    def test_data_algorithms_reference_known_detectors(self):
        valid = {"median_frame", "median_court_roi", "pose_skeleton_yolo"}
        for match in re.finditer(r'data-algorithms="([^"]+)"', INDEX_HTML):
            for algo in match.group(1).split():
                with self.subTest(algorithm=algo):
                    self.assertIn(
                        algo,
                        valid,
                        f"data-algorithms references unknown detector '{algo}'",
                    )

    def test_help_popover_div_present(self):
        self.assertIn(
            'id="help-popover"',
            INDEX_HTML,
            "missing #help-popover element required by the click-help handler",
        )

    def test_help_dot_select_dropdown_options_match_detectors(self):
        # The <select id="start-algorithm"> options need to match
        # DETECTOR_CONFIGS keys, otherwise switching the dropdown can put the
        # form into a state where every label is hidden.
        match = re.search(
            r'<select[^>]*id="start-algorithm"[^>]*>(.*?)</select>',
            INDEX_HTML,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        options = set(re.findall(r'value="([^"]+)"', match.group(1)))
        valid = {"median_frame", "median_court_roi", "pose_skeleton_yolo"}
        self.assertEqual(
            options,
            valid,
            "start-algorithm options drifted from DETECTOR_CONFIGS keys",
        )

    def test_no_stray_title_attributes_left_on_help_dots(self):
        # We replaced title="..." with data-help-key on every .help-dot. If a
        # new one is added with title="..." it bypasses the centralized HELP_TEXT.
        for match in re.finditer(r'<button[^>]*class="help-dot"[^>]*>', INDEX_HTML):
            tag = match.group(0)
            self.assertNotIn(
                "title=",
                tag,
                f"help-dot button still has a title= attribute: {tag}",
            )
            self.assertIn(
                "data-help-key=",
                tag,
                f"help-dot button missing data-help-key: {tag}",
            )


class CssOverrideTests(unittest.TestCase):
    """The label hide rule must beat .analysis-start-controls label specificity."""

    def test_start_config_hidden_uses_important(self):
        # Matches: .start-config[hidden] { display: none !important; }
        # (or similar block) — the !important is required because the
        # .analysis-start-controls label rule has higher specificity than the
        # UA stylesheet's [hidden] rule.
        match = re.search(
            r"\.start-config\[hidden\][^}]*\{[^}]*\}",
            STYLE_CSS,
        )
        self.assertIsNotNone(
            match,
            "missing CSS rule .start-config[hidden] — algorithm-based hiding will silently fail",
        )
        block = match.group(0)
        self.assertIn(
            "display:",
            block.replace(" ", ""),
            "rule should set display",
        )
        self.assertIn(
            "!important",
            block,
            "the rule must use !important to beat .analysis-start-controls label specificity",
        )

    def test_help_popover_styles_present(self):
        self.assertIn(".help-popover", STYLE_CSS)


class HelpPopoverWiringTests(unittest.TestCase):
    """The popover handler should be defined and bound to .help-dot clicks."""

    def test_popover_show_function_defined(self):
        self.assertIn("function showHelpPopover(", APP_JS)
        self.assertIn("function hideHelpPopover(", APP_JS)

    def test_popover_handler_listens_for_help_dot_clicks(self):
        self.assertRegex(
            APP_JS,
            r"document\.addEventListener\(\s*[\"']click[\"']",
            "expected a delegated document click handler",
        )
        # And handles Escape to close.
        self.assertRegex(
            APP_JS,
            r"document\.addEventListener\(\s*[\"']keydown[\"']",
            "expected a keydown handler for Escape close",
        )


class RenderAnalysisConfigsTests(unittest.TestCase):
    """The pill renderer must filter by detector and apply hover help."""

    def test_filter_uses_detector_configs(self):
        m = re.search(
            r"function renderAnalysisConfigs\([^)]*\)\s*\{",
            APP_JS,
        )
        self.assertIsNotNone(m)
        body = _balanced_block(APP_JS, m.end(), "{", "}")
        self.assertIn(
            "DETECTOR_CONFIGS",
            body,
            "renderAnalysisConfigs should filter pills using DETECTOR_CONFIGS",
        )
        self.assertIn(
            "pill.title",
            body,
            "renderAnalysisConfigs should set pill.title for hover help",
        )
        self.assertIn(
            "HELP_TEXT",
            body,
            "renderAnalysisConfigs should source tooltips from HELP_TEXT",
        )


if __name__ == "__main__":
    unittest.main()
