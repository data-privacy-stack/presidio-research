"""Rendering clients for CanonicalMapper.

This module is intentionally separate from the mapper core so that additional
rendering targets (Markdown, JSON, CLI tables, …) can be added without touching
the mapping logic.
"""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

from presidio_evaluator.entity_mapping.data_objects import IssueSeverity

if TYPE_CHECKING:
    from presidio_evaluator.entity_mapping.mapper import CanonicalMapper


class MapperRenderer:
    """Renders a CanonicalMapper audit table to HTML or plain text.

    Example::

        from presidio_evaluator.entity_mapping import CanonicalMapper, MapperRenderer

        mapper = CanonicalMapper().analyze(results_df)
        MapperRenderer(mapper).render()          # Jupyter-aware
        html = MapperRenderer(mapper).build_html()  # raw HTML string
    """

    # Badge colours per projection_type: (background, text)
    _PROJ_COLORS: dict[str, tuple[str, str]] = {
        "EXACT": ("#d1f4e0", "#2da44e"),
        "TRIVIAL": ("#ddf4ff", "#0969da"),
        "MANUAL": ("#ddf4ff", "#0969da"),
        "UNRESOLVED": ("#ffebe9", "#cf222e"),
        "AMBIGUOUS": ("#fff8c5", "#d4a72c"),
        "CROSS_BRANCH": ("#fff1f0", "#d4492a"),
        "NONE": ("#f6f8fa", "#57606a"),
    }

    def __init__(self, mapper: CanonicalMapper) -> None:
        self._mapper = mapper

    def render(self) -> None:
        """Display audit table. Uses IPython.display in Jupyter; plain-text fallback."""
        try:
            from IPython.display import HTML, display  # noqa: PLC0415

            display(HTML(self.build_html()))
        except ImportError:
            self.print_text()

    def build_html(self) -> str:
        """Return a self-contained HTML audit table string."""
        m = self._mapper
        records = m._records
        ann_counts = m._label_annotation_counts
        pred_counts = m._label_prediction_counts

        type_counts: dict[str, int] = {}
        for rec in records.values():
            key = rec.projection_type or "NONE"
            type_counts[key] = type_counts.get(key, 0) + 1

        n_auto = type_counts.get("TRIVIAL", 0)
        depth_info = f"Eval depth: {m._eval_depth}" if m._eval_depth else "Not analyzed"
        summary_parts = [
            f'<span style="color:#57606a;font-size:12px">{depth_info}</span>',
        ]
        for key, label in [
            ("EXACT", "exact"),
            ("TRIVIAL", f"{n_auto} auto-fixed"),
            ("UNRESOLVED", "unresolved"),
            ("AMBIGUOUS", "ambiguous"),
            ("CROSS_BRANCH", "cross-branch"),
        ]:
            n = type_counts.get(key, 0)
            if n:
                _, color = self._PROJ_COLORS[key]
                summary_parts.append(
                    f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:10px;font-size:11px;margin:0 3px">'
                    f"{n} {label}</span>"
                )

        issue_style = {
            IssueSeverity.ERROR: {
                "bg": "#ffebe9",
                "border": "#cf222e",
                "title_color": "#cf222e",
                "icon": "ERROR",
            },
            IssueSeverity.WARNING: {
                "bg": "#fff8c5",
                "border": "#d4a72c",
                "title_color": "#b76e00",
                "icon": "WARNING",
            },
            IssueSeverity.INFO: {
                "bg": "#ddf4ff",
                "border": "#54aeff",
                "title_color": "#0969da",
                "icon": "INFO",
            },
        }
        blocks: list[str] = []
        for severity, group in groupby(m.get_issues(), key=lambda i: i.severity):
            style = issue_style[severity]
            items = "".join(
                f'<li style="margin:4px 0">{issue.message}</li>' for issue in group
            )
            blocks.append(
                f'<div style="background:{style["bg"]};border:1px solid {style["border"]};'
                f'border-radius:6px;padding:10px 14px;margin:10px 0">'
                f'<strong style="color:{style["title_color"]}">'
                f"[{style['icon']}]</strong>"
                f'<ol style="margin:6px 0 0 0;padding-left:18px;color:#24292f">{items}</ol>'
                f"</div>"
            )
        issues_html = "".join(blocks)

        def _row_sort_key(item: tuple) -> tuple:
            lbl, rec = item
            pt = rec.projection_type or "NONE"
            order = {
                "UNRESOLVED": 0,
                "AMBIGUOUS": 1,
                "CROSS_BRANCH": 2,
                "TRIVIAL": 3,
                "EXACT": 4,
                "MANUAL": 4,
                "NONE": 5,
            }
            tokens = ann_counts.get(lbl, 0) + pred_counts.get(lbl, 0)
            return (order.get(pt, 9), -tokens)

        rows = []
        for lbl, rec in sorted(records.items(), key=_row_sort_key):
            pt = rec.projection_type or "NONE"
            bg, badge_color = self._PROJ_COLORS.get(pt, ("#ffffff", "#666"))
            badge = (
                f'<span style="background:{badge_color};color:white;padding:1px 6px;'
                f'border-radius:3px;font-size:11px;font-family:monospace">{pt}</span>'
            )
            canonical_display = (
                f"<code>{rec.canonical}</code>"
                if rec.canonical
                else '<em style="color:#57606a">None</em>'
            )
            projected_display = (
                f"<code>{rec.projected}</code>"
                if rec.projected
                else '<em style="color:#57606a">-</em>'
            )
            ann_str = str(ann_counts.get(lbl, "-"))
            pred_str = str(pred_counts.get(lbl, "-"))
            rows.append(
                f'<tr style="background:{bg}">'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de;'
                f'font-family:monospace;font-weight:600">{lbl}</td>'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de;'
                f'font-size:11px;color:#57606a">{rec.tier}</td>'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de">'
                f"{canonical_display}</td>"
                f'<td style="padding:7px 10px;border:1px solid #d0d7de">{badge}</td>'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de;'
                f'font-family:monospace">{projected_display}</td>'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de;'
                f'text-align:center">{ann_str}</td>'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de;'
                f'text-align:center">{pred_str}</td>'
                f"</tr>"
            )

        return (
            '<div style="font-family:-apple-system,BlinkMacSystemFont,'
            'Segoe UI,Helvetica,Arial,sans-serif">'
            '<h3 style="margin-top:0;color:#24292f">Entity Label Mapping Audit</h3>'
            f"{issues_html}"
            f'<div style="margin:10px 0">{"".join(summary_parts)}</div>'
            '<table style="width:100%;border-collapse:collapse;font-size:13px">'
            '<thead><tr style="background:#f6f8fa;border-bottom:2px solid #d0d7de">'
            '<th style="padding:7px 10px;border:1px solid #d0d7de;text-align:left">'
            "Raw Label</th>"
            '<th style="padding:7px 10px;border:1px solid #d0d7de;text-align:left">'
            "ID Tier</th>"
            '<th style="padding:7px 10px;border:1px solid #d0d7de;text-align:left">'
            "Canonical</th>"
            '<th style="padding:7px 10px;border:1px solid #d0d7de;text-align:left">'
            "Projection</th>"
            '<th style="padding:7px 10px;border:1px solid #d0d7de;text-align:left">'
            "Projected To</th>"
            '<th style="padding:7px 10px;border:1px solid #d0d7de;text-align:center">'
            "Annotations</th>"
            '<th style="padding:7px 10px;border:1px solid #d0d7de;text-align:center">'
            "Predictions</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        )

    def print_text(self) -> None:
        """Print a plain-text audit table to stdout."""
        m = self._mapper
        records = m._records
        ann_counts = m._label_annotation_counts
        pred_counts = m._label_prediction_counts

        depth_info = f"Eval depth: {m._eval_depth}" if m._eval_depth else "Not analyzed"
        n_blocking = sum(
            1
            for i in m._issues
            if i.severity in (IssueSeverity.ERROR, IssueSeverity.WARNING)
        )
        print(
            f"Entity Label Mapping Audit  "
            f"({depth_info}, {len(records)} labels, {n_blocking} blocking issues)\n"
        )
        for issue in m._issues:
            print(f"  [{issue.severity.value.upper()}] {issue.message}")
        print()
        print(
            f"{'Label':<30} {'ID Tier':<18} {'Canonical':<30} "
            f"{'Projection':<14} {'Projected To':<30} {'Ann':>6} {'Pred':>6}"
        )
        print("-" * 140)
        for lbl, rec in sorted(records.items()):
            canonical = rec.canonical or "-"
            projected = rec.projected or "-"
            pt = rec.projection_type or "-"
            ann = str(ann_counts.get(lbl, "-"))
            pred = str(pred_counts.get(lbl, "-"))
            print(
                f"{lbl:<30} {rec.tier:<18} {canonical:<30} "
                f"{pt:<14} {projected:<30} {ann:>6} {pred:>6}"
            )
