"""Rendering clients for CanonicalMapper.

This module is intentionally separate from the mapper core so that additional
rendering targets (Markdown, JSON, CLI tables, …) can be added without touching
the mapping logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from presidio_evaluator.entity_mapping.data_objects import IssueSeverity, IssueType

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

    # Badge colours per tier: (background, text)
    _PROJ_COLORS: dict[str, tuple[str, str]] = {
        "EXACT": ("#d1f4e0", "#2da44e"),
        "FUZZY": ("#ddf4ff", "#0969da"),
        "COUNTRY": ("#ddf4ff", "#0969da"),
        "COUNTRY_FALLBACK": ("#fff8c5", "#d4a72c"),
        "MANUAL": ("#ddf4ff", "#0969da"),
        "UNRESOLVED": ("#ffebe9", "#cf222e"),
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

    def render_summary(self) -> None:
        """Display a compact per-label summary table in Jupyter; plain-text fallback."""
        try:
            from IPython.display import HTML, display  # noqa: PLC0415

            display(HTML(self.build_summary_html()))
        except ImportError:
            self.print_text()

    def build_summary_html(self) -> str:
        """Return a compact HTML table: one row per label with counts and match confidence.

        Columns: Label | Canonical match | Projected to | Annotations | Predictions | Confidence
        """
        m = self._mapper
        records = m._records
        ann_counts = m._label_annotation_counts
        pred_counts = m._label_prediction_counts

        tier_label = {
            "EXACT": "exact",
            "FUZZY": "fuzzy",
            "COUNTRY": "country prefix",
            "COUNTRY_FALLBACK": "country fallback",
            "MANUAL": "manual",
            "NONE": "—",
            "UNRESOLVED": "not found",
        }

        def _conf_cell(rec) -> str:  # noqa: ANN001
            if rec.tier == "FUZZY" and rec.score is not None:
                pct = int(rec.score * 100)
                bar_color = (
                    "#2da44e" if pct >= 80 else "#d4a72c" if pct >= 60 else "#cf222e"
                )
                return (
                    f'<div style="display:flex;align-items:center;gap:6px">'
                    f'<div style="flex:0 0 80px;background:#eee;border-radius:4px;height:8px">'
                    f'<div style="width:{pct}%;background:{bar_color};'
                    f'border-radius:4px;height:8px"></div></div>'
                    f'<span style="font-size:11px;color:#57606a">{pct}%</span>'
                    f"</div>"
                )
            tl = tier_label.get(rec.tier, rec.tier)
            color = (
                "#2da44e"
                if rec.tier in ("EXACT", "MANUAL")
                else (
                    "#d4a72c"
                    if rec.tier in ("COUNTRY", "COUNTRY_FALLBACK")
                    else "#cf222e"
                )
            )
            return f'<span style="font-size:11px;color:{color}">{tl}</span>'

        # Sort: annotation-only labels first (by ann count desc), then pred-only
        all_labels = sorted(
            records.keys(),
            key=lambda lbl: (-ann_counts.get(lbl, 0), -pred_counts.get(lbl, 0)),
        )

        rows = []
        for lbl in all_labels:
            rec = records[lbl]
            ann = ann_counts.get(lbl, 0)
            pred = pred_counts.get(lbl, 0)
            if rec.resolved:
                projected_cell = f'<code style="font-size:12px">{rec.resolved}</code>'
            else:
                projected_cell = '<em style="color:#57606a;font-size:12px">—</em>'
            rows.append(
                f"<tr>"
                f'<td style="padding:6px 10px;border:1px solid #d0d7de;'
                f'font-family:monospace;font-weight:600">{lbl}</td>'
                f'<td style="padding:6px 10px;border:1px solid #d0d7de">{projected_cell}</td>'
                f'<td style="padding:6px 10px;border:1px solid #d0d7de;'
                f'text-align:right;color:#57606a;font-size:12px">{ann if ann else "—"}</td>'
                f'<td style="padding:6px 10px;border:1px solid #d0d7de;'
                f'text-align:right;color:#57606a;font-size:12px">{pred if pred else "—"}</td>'
                f'<td style="padding:6px 10px;border:1px solid #d0d7de">{_conf_cell(rec)}</td>'
                f"</tr>"
            )

        table = (
            '<table style="width:100%;border-collapse:collapse;font-size:13px">'
            '<thead><tr style="background:#f6f8fa">'
            '<th style="padding:6px 10px;border:1px solid #d0d7de;text-align:left">Label</th>'
            '<th style="padding:6px 10px;border:1px solid #d0d7de;text-align:left">Mapped to</th>'
            '<th style="padding:6px 10px;border:1px solid #d0d7de;text-align:right">Annotations</th>'
            '<th style="padding:6px 10px;border:1px solid #d0d7de;text-align:right">Predictions</th>'
            '<th style="padding:6px 10px;border:1px solid #d0d7de;text-align:left">Confidence</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
        return (
            '<div style="font-family:-apple-system,BlinkMacSystemFont,'
            'Segoe UI,Helvetica,Arial,sans-serif;max-width:900px">'
            '<h4 style="margin:0 0 6px;color:#24292f">Label mapping summary</h4>'
            '<p style="font-size:12px;color:#57606a;margin:0 0 8px">'
            "One row per label. <em>Mapped to</em> is the resolved hierarchy entity. "
            "Counts are pre-mapping token occurrences.</p>" + table + "</div>"
        )

    def build_html(self) -> str:
        """Return a self-contained HTML audit table string."""
        m = self._mapper
        records = m._records
        ann_counts = m._label_annotation_counts
        pred_counts = m._label_prediction_counts

        tier_priority = {
            "UNRESOLVED": 0,
            "EXACT": 1,
            "FUZZY": 2,
            "COUNTRY": 3,
            "COUNTRY_FALLBACK": 4,
            "MANUAL": 5,
            "NONE": 6,
        }

        # ── Shared helpers ────────────────────────────────────────────────

        def _render_resolution_options(issue) -> str:  # noqa: ANN001
            opts = issue.resolution_options
            if not opts:
                return ""
            top = opts[0]
            top_html = (
                f'<div style="font-size:12px">'
                f'<code style="background:#f6f8fa;padding:2px 5px;'
                f'border-radius:3px;font-size:11px">'
                f"mapper.map({top.mapper_call!r})"
                f"</code>"
                f'&nbsp;<span style="color:#57606a">— {top.description}</span>'
                f"</div>"
            )
            if len(opts) > 1:
                other_items = "".join(
                    f"<li>"
                    f'<code style="font-size:11px">mapper.map({o.mapper_call!r})</code>'
                    f'&nbsp;<span style="color:#57606a">— {o.description}</span>'
                    f"</li>"
                    for o in opts[1:]
                )
                top_html += (
                    f'<details style="margin-top:4px;font-size:12px">'
                    f'<summary style="color:#57606a;cursor:pointer">'
                    f"{len(opts) - 1} more option(s)</summary>"
                    f'<ul style="margin:4px 0;padding-left:18px">{other_items}</ul>'
                    f"</details>"
                )
            return top_html

        def _section_heading(title: str, subtitle: str = "") -> str:
            sub = (
                f'<span style="font-weight:normal;font-size:12px;'
                f'color:#57606a;margin-left:8px">{subtitle}</span>'
                if subtitle
                else ""
            )
            return (
                f'<h4 style="margin:22px 0 4px;color:#24292f;'
                f'border-bottom:2px solid #d0d7de;padding-bottom:4px">'
                f"{title}{sub}</h4>"
            )

        # ── Issue descriptions for gap cards (blocking issues) ────────────
        gap_why = {
            IssueType.UNRESOLVED: (
                "This label <strong>doesn't appear in the entity hierarchy</strong>, "
                "so the mapper has no basis for matching it to a canonical "
                "representation. It may be a typo, a model-specific tag, or an "
                "entirely new entity type."
            ),
            IssueType.COLLISION_CROSS_BRANCH: (
                "This label maps to a hierarchy entity on a "
                "<strong>different branch</strong> from all entities in your dataset. "
                "Since there's no natural correspondence, the mapper can't resolve it "
                "automatically."
            ),
            IssueType.PREDICTION_ONLY: (
                "The model predicts this entity type, but the dataset "
                "<strong>never annotates it</strong>. Every prediction would count as "
                "a false positive with no possible true positives which would "
                "make precision zero and recall undefined. Consider ignoring this label "
                "if you don't want it to affect your metrics."
            ),
        }

        # ── Short descriptions for INFO issues merged into label tables ───
        table_issue_desc = {
            IssueType.COLLISION_SAME_BRANCH: (
                "Same-branch depth mismatch — handled automatically by hierarchical evaluation "
                "(branch/detailed projection)."
            ),
            IssueType.DATASET_ONLY: (
                "The dataset annotates this entity type, but the model never predicts it "
                "(model coverage gap)."
            ),
        }

        # ── Build label → INFO-issue map for merging into label tables ────
        info_issue_map: dict = {}
        for issue in m.get_issues():
            if issue.severity == IssueSeverity.INFO:
                for lbl in issue.labels:
                    info_issue_map[lbl] = issue

        # ── Section pairs ─────────────────────────────────────────────────
        ann_pairs = sorted(
            [(lbl, rec) for lbl, rec in records.items() if ann_counts.get(lbl, 0) > 0],
            key=lambda x: (
                tier_priority.get(x[1].tier or "NONE", 9),
                -ann_counts.get(x[0], 0),
            ),
        )

        pred_pairs = sorted(
            [(lbl, rec) for lbl, rec in records.items() if pred_counts.get(lbl, 0) > 0],
            key=lambda x: (
                tier_priority.get(x[1].tier or "NONE", 9),
                -pred_counts.get(x[0], 0),
            ),
        )

        # ── Label table (with grouped INFO-issue rows) ────────────────────
        def _label_table(pairs: list, token_col: dict) -> str:  # noqa: ANN001
            if not pairs:
                return (
                    '<p style="color:#57606a;font-size:13px;font-style:italic;'
                    'margin:4px 0">None</p>'
                )

            has_issues = any(lbl in info_issue_map for lbl, _ in pairs)
            _td = 'style="padding:6px 10px;border:1px solid #d0d7de'
            rows = []

            for lbl, rec in pairs:
                pt = rec.tier or "NONE"
                row_bg, _ = self._PROJ_COLORS.get(pt, ("#ffffff", "#666"))
                display_entity = rec.resolved
                projected_cell = (
                    f'<code style="font-size:12px">{display_entity}</code>'
                    if display_entity
                    else '<em style="color:#cf222e;font-size:12px">unresolved</em>'
                )
                tokens = token_col.get(lbl, 0)
                row = (
                    f'<tr style="background:{row_bg}">'
                    f'<td {_td};font-family:monospace;font-weight:600">{lbl}</td>'
                    f'<td {_td}">{projected_cell}</td>'
                    f'<td {_td};text-align:right;color:#57606a;font-size:12px">'
                    f"{tokens if tokens else '—'}</td>"
                )
                if has_issues:
                    issue = info_issue_map.get(lbl)
                    issue_desc = table_issue_desc.get(issue.type, "") if issue else ""
                    fix_html = (
                        _render_resolution_options(issue)
                        if issue and issue.resolution_options
                        else ("—" if issue else "")
                    )
                    row += (
                        f'<td {_td};font-size:12px;color:#24292f;line-height:1.5">'
                        f"{issue_desc}</td>"
                        f'<td {_td};font-size:12px">{fix_html}</td>'
                    )
                row += "</tr>"
                rows.append(row)

            _th = 'style="padding:6px 10px;border:1px solid #d0d7de'
            header_cells = (
                f'<th {_th};text-align:left">Label</th>'
                f'<th {_th};text-align:left">Mapped to</th>'
                f'<th {_th};text-align:right">Tokens</th>'
            )
            if has_issues:
                header_cells += (
                    f'<th {_th};text-align:left">Issue</th>'
                    f'<th {_th};text-align:left">Suggested fix</th>'
                )
            return (
                '<table style="width:100%;border-collapse:collapse;'
                'font-size:13px;margin-top:8px">'
                f'<thead><tr style="background:#f6f8fa">{header_cells}</tr></thead>'
                "<tbody>" + "".join(rows) + "</tbody></table>"
            )

        # ── Gap cards (ERROR / WARNING only) ─────────────────────────────
        sev_style = {
            IssueSeverity.ERROR: {
                "bg": "#ffebe9",
                "border": "#cf222e",
                "badge_bg": "#cf222e",
                "label": "ERROR",
            },
            IssueSeverity.WARNING: {
                "bg": "#fff8c5",
                "border": "#d4a72c",
                "badge_bg": "#b76e00",
                "label": "WARNING",
            },
        }

        gap_cards = []
        for issue in m.get_issues():
            if issue.severity == IssueSeverity.INFO:
                continue
            sty = sev_style[issue.severity]
            why = gap_why.get(issue.type, "")
            lbl_tags = " ".join(
                f'<code style="background:#f6f8fa;padding:1px 5px;'
                f'border-radius:3px;font-size:12px">{lbl}</code>'
                for lbl in issue.labels
            )
            sev_badge = (
                f'<span style="background:{sty["badge_bg"]};color:white;'
                f'padding:1px 8px;border-radius:10px;font-size:11px">'
                f"{sty['label']}</span>"
            )
            fix_html = _render_resolution_options(issue)
            gap_cards.append(
                f'<div style="background:{sty["bg"]};border:1px solid {sty["border"]};'
                f'border-radius:6px;padding:12px 16px;margin:10px 0">'
                f'<div style="margin-bottom:6px">'
                f"{sev_badge}&nbsp;&nbsp;{lbl_tags}"
                f"</div>"
                f'<div style="font-size:13px;color:#24292f;line-height:1.6">'
                f"{why}</div>"
                f"{fix_html}"
                f"</div>"
            )

        gaps_section = (
            "".join(gap_cards)
            if gap_cards
            else (
                '<p style="color:#2da44e;font-size:13px;margin:6px 0">'
                "&#10003; No blocking issues. All labels are fully resolved.</p>"
            )
        )

        # ── Summary header ────────────────────────────────────────────────
        n_labels = len(records)
        depth_info = f"{n_labels} labels identified" if n_labels else "Not analyzed"
        n_blocking = sum(1 for i in m.get_issues() if i.severity == IssueSeverity.ERROR)
        status_msg = (
            f'<span style="color:#cf222e;font-weight:600">'
            f"{n_blocking} issue(s) need your attention</span>"
            if n_blocking
            else '<span style="color:#2da44e;font-weight:600">'
            "&#10003; Ready for evaluation</span>"
        )

        # ── Assemble ──────────────────────────────────────────────────────
        return (
            '<div style="font-family:-apple-system,BlinkMacSystemFont,'
            'Segoe UI,Helvetica,Arial,sans-serif;max-width:900px">'
            '<h3 style="margin-top:0;color:#24292f">'
            "Entity Mapping Audit"
            f'&nbsp;<span style="font-size:13px;font-weight:normal;color:#57606a">'
            f"{depth_info}</span></h3>"
            f'<div style="margin-bottom:4px">{status_msg}</div>'
            + _section_heading(
                "1. Blocking issues",
                f"{n_blocking} requiring action" if n_blocking else "none",
            )
            + '<p style="font-size:12px;color:#57606a;margin:2px 0 4px">'
            "Labels that could not be mapped automatically (ERROR / WARNING). "
            "Resolve these before calling <code>mapper.get_mapped_results_dataframe()</code>.</p>"
            + gaps_section
            + _section_heading(
                "2. Annotation labels",
                f"{len(ann_pairs)} label(s) from your dataset",
            )
            + '<p style="font-size:12px;color:#57606a;margin:2px 0 4px">'
            "The entity labels in your ground-truth annotations. "
            "Non-blocking issues (INFO) are shown inline.</p>"
            + _label_table(ann_pairs, ann_counts)
            + _section_heading(
                "3. Prediction labels",
                f"{len(pred_pairs)} label(s) from the model",
            )
            + '<p style="font-size:12px;color:#57606a;margin:2px 0 4px">'
            "The entity labels your model outputs. "
            "Non-blocking issues (INFO) are shown inline.</p>"
            + _label_table(pred_pairs, pred_counts)
            + "</div>"
        )

    def print_text(self) -> None:
        """Print a plain-text audit table to stdout."""
        m = self._mapper
        records = m._records
        ann_counts = m._label_annotation_counts
        pred_counts = m._label_prediction_counts

        depth_info = (
            f"{len(m._records)} labels identified" if m._records else "Not analyzed"
        )
        n_blocking = sum(1 for i in m._issues if i.severity == IssueSeverity.ERROR)
        print(
            f"Entity Label Mapping Audit  "
            f"({depth_info}, {len(records)} labels, {n_blocking} unresolved)\n"
        )
        for issue in m._issues:
            print(f"  [{issue.severity.value.upper()}] {issue.message}")
        print()
        print(
            f"{'Label':<30} {'ID Tier':<18} {'Resolved':<30} {'Score':<14} {'Ann':>6} {'Pred':>6}"
        )
        print("-" * 110)
        for lbl, rec in sorted(records.items()):
            resolved = rec.resolved or "-"
            tier = rec.tier or "-"
            score = f"{rec.score:.0%}" if rec.score is not None else "-"
            ann = str(ann_counts.get(lbl, "-"))
            pred = str(pred_counts.get(lbl, "-"))
            print(f"{lbl:<30} {tier:<18} {resolved:<30} {score:<14} {ann:>6} {pred:>6}")
