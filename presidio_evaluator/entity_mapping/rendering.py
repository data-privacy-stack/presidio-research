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
            canonical_differs = (
                rec.canonical and rec.projected and rec.canonical != rec.projected
            )
            if rec.projected:
                projected_cell = f'<code style="font-size:12px">{rec.projected}</code>'
                if canonical_differs:
                    projected_cell += (
                        f' <span style="color:#57606a;font-size:11px">'
                        f"(via {rec.canonical})</span>"
                    )
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
            "One row per label. <em>Mapped to</em> is the final canonical entity used for "
            "evaluation. When an intermediate match differs (e.g. depth-4 label projected "
            "up to a depth-3 surface node), it is shown in parentheses. "
            "Counts are pre-mapping token occurrences.</p>" + table + "</div>"
        )

    def build_html(self) -> str:
        """Return a self-contained HTML audit table string."""
        m = self._mapper
        records = m._records
        ann_counts = m._label_annotation_counts
        pred_counts = m._label_prediction_counts

        proj_priority = {
            "UNRESOLVED": 0,
            "AMBIGUOUS": 1,
            "CROSS_BRANCH": 2,
            "TRIVIAL": 3,
            "EXACT": 4,
            "MANUAL": 4,
            "NONE": 5,
        }

        # ── Shared helpers ────────────────────────────────────────────────

        def _status_badge(pt: str) -> str:
            _, color = self._PROJ_COLORS.get(pt, ("#f6f8fa", "#57606a"))
            label = {
                "EXACT": "mapped",
                "TRIVIAL": "auto-mapped",
                "MANUAL": "manual",
                "AMBIGUOUS": "&#9888; needs decision",
                "CROSS_BRANCH": "&#9888; wrong branch",
                "UNRESOLVED": "&#10007; not found",
                "NONE": "",
            }.get(pt, pt)
            return (
                f'<span style="background:{color};color:white;padding:1px 7px;'
                f"border-radius:10px;font-size:11px;white-space:nowrap"
                f'">{label}</span>'
            )

        def _why_note(rec) -> str:  # noqa: ANN001, PLR0911
            """One-liner explaining why this label was mapped the way it was."""
            tier = rec.tier
            pt = rec.projection_type or "NONE"
            canonical = rec.canonical
            projected = rec.projected

            if tier == "MANUAL":
                return f"Manually mapped to <code>{projected}</code>."
            if pt == "EXACT":
                if tier in ("EXACT",):
                    return "Exact alias match in the entity hierarchy."
                if tier == "FUZZY":
                    return f"Fuzzy-matched to <code>{canonical}</code> in the entity hierarchy."
                if tier in ("COUNTRY", "COUNTRY_FALLBACK"):
                    return f"Matched via country-prefix to <code>{canonical}</code>."
            if pt == "TRIVIAL":
                if canonical != projected:
                    return (
                        f"<code>{canonical}</code> is a depth-4+ node; auto-projected "
                        f"up to its depth-3 canonical representative <code>{projected}</code>."
                    )
                return f"Descendant of <code>{projected}</code>; auto-projected to the canonical surface."
            if pt == "AMBIGUOUS":
                return "Broad parent label — maps to multiple sub-types on the canonical surface."
            if pt == "CROSS_BRANCH":
                return "Mapped to a different hierarchy branch; cannot auto-resolve."
            if pt == "UNRESOLVED":
                return "Not found in the entity hierarchy."
            return ""

        def _label_table(pairs: list, token_col: dict) -> str:  # noqa: ANN001
            if not pairs:
                return (
                    '<p style="color:#57606a;font-size:13px;font-style:italic;'
                    'margin:4px 0">None</p>'
                )
            rows = []
            for lbl, rec in pairs:
                pt = rec.projection_type or "NONE"
                row_bg, _ = self._PROJ_COLORS.get(pt, ("#ffffff", "#666"))
                if rec.projected:
                    proj_cell = f'<code style="font-size:12px">{rec.projected}</code>'
                    if pt == "TRIVIAL" and rec.canonical != rec.projected:
                        proj_cell += (
                            f' <span style="color:#57606a;font-size:11px">'
                            f"(via {rec.canonical})</span>"
                        )
                    note = _why_note(rec)
                    if note:
                        proj_cell += (
                            f'<br><span style="color:#57606a;font-size:11px;'
                            f'font-style:italic">{note}</span>'
                        )
                else:
                    proj_cell = (
                        '<em style="color:#cf222e;font-size:12px">unresolved</em>'
                    )
                rows.append(
                    f'<tr style="background:{row_bg}">'
                    f'<td style="padding:6px 10px;border:1px solid #d0d7de;'
                    f'font-family:monospace;font-weight:600">{lbl}</td>'
                    f'<td style="padding:6px 10px;border:1px solid #d0d7de">'
                    f"{proj_cell}</td>"
                    f'<td style="padding:6px 10px;border:1px solid #d0d7de">'
                    f"{_status_badge(pt)}</td>"
                    f'<td style="padding:6px 10px;border:1px solid #d0d7de;'
                    f'text-align:right;color:#57606a;font-size:12px">'
                    f"{token_col.get(lbl, 0)}</td>"
                    f"</tr>"
                )
            return (
                '<table style="width:100%;border-collapse:collapse;'
                'font-size:13px;margin-top:8px">'
                '<thead><tr style="background:#f6f8fa">'
                '<th style="padding:6px 10px;border:1px solid #d0d7de;'
                'text-align:left">Label</th>'
                '<th style="padding:6px 10px;border:1px solid #d0d7de;'
                'text-align:left">Canonical representation</th>'
                '<th style="padding:6px 10px;border:1px solid #d0d7de;'
                'text-align:left">Status</th>'
                '<th style="padding:6px 10px;border:1px solid #d0d7de;'
                'text-align:right">Tokens</th>'
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
            )

        def _render_resolution_options(issue) -> str:  # noqa: ANN001
            opts = issue.resolution_options
            if not opts:
                return ""
            top = opts[0]
            top_html = (
                f'<div style="margin-top:10px;font-size:12px">'
                f"<strong>Suggested fix:</strong>&nbsp;"
                f'<code style="background:#f6f8fa;padding:2px 5px;'
                f'border-radius:3px;font-size:11px">'
                f"mapper.map({top.mapper_call!r})"
                f"</code>"
                f'&nbsp;<span style="color:#57606a">- {top.description}</span>'
                f"</div>"
            )
            if len(opts) > 1:
                other_items = "".join(
                    f"<li>"
                    f'<code style="font-size:11px">mapper.map({o.mapper_call!r})</code>'
                    f'&nbsp;<span style="color:#57606a">- {o.description}</span>'
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

        # ── Section 1: Annotation labels ──────────────────────────────────
        ann_pairs = sorted(
            [(lbl, rec) for lbl, rec in records.items() if ann_counts.get(lbl, 0) > 0],
            key=lambda x: (
                proj_priority.get(x[1].projection_type or "NONE", 9),
                -ann_counts.get(x[0], 0),
            ),
        )

        # ── Section 2: Prediction labels ──────────────────────────────────
        pred_pairs = sorted(
            [(lbl, rec) for lbl, rec in records.items() if pred_counts.get(lbl, 0) > 0],
            key=lambda x: (
                proj_priority.get(x[1].projection_type or "NONE", 9),
                -pred_counts.get(x[0], 0),
            ),
        )

        # ── Section 3: Gap cards ──────────────────────────────────────────
        gap_why = {
            IssueType.COLLISION_AMBIGUOUS: (
                "This label maps to a hierarchy entity that is a "
                "<strong>coarser-grained ancestor</strong> of multiple canonical "
                "surface entities. This is <strong>non-blocking</strong> — use "
                "<code>evaluator.calculate_hierarchical_scores(mapped_df)</code> to "
                "score at each granularity level automatically. Alternatively, use "
                "<code>mapper.map()</code> to manually remap to a specific sub-type."
            ),
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
            IssueType.DATASET_ONLY: (
                "The dataset annotates this entity type, but the model "
                "<strong>never predicts it</strong>. All annotations will count as "
                "false negatives. This is a model coverage gap, not a mapping "
                "problem unless you think some other model entity should "
                "be considered here."
            ),
            IssueType.COLLISION_TRIVIAL: (
                "This label was <strong>automatically projected</strong> to the "
                "nearest canonical representative on the same branch. "
                "Evaluation will use the projected canonical entity."
                "No action is required unless you wish to manually change this"
            ),
        }

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
            IssueSeverity.INFO: {
                "bg": "#ddf4ff",
                "border": "#54aeff",
                "badge_bg": "#0969da",
                "label": "INFO",
            },
        }
        # Per-type style overrides (applied after severity defaults)
        type_style_override = {
            IssueType.COLLISION_AMBIGUOUS: {
                "bg": "#e6f6f6",
                "border": "#2aa198",
                "badge_bg": "#2aa198",
                "label": "INFO",
            },
        }

        gap_cards = []
        for issue in m.get_issues():
            sty = type_style_override.get(issue.type, sev_style[issue.severity])
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
            # Token co-occurrence detail for AMBIGUOUS
            extra = ""
            if issue.type == IssueType.COLLISION_AMBIGUOUS and issue.overlap_counts:
                sorted_ovl = sorted(
                    ((e, cnt) for e, cnt in issue.overlap_counts.items() if cnt > 0),
                    key=lambda x: -x[1],
                )
                if sorted_ovl:
                    ovl_items = " &nbsp;&middot;&nbsp; ".join(
                        f"<code>{e}</code>: {cnt}" for e, cnt in sorted_ovl
                    )
                    extra = (
                        f'<div style="margin-top:8px;font-size:12px;color:#57606a">'
                        f"<strong>Token co-occurrences (annotation &harr; prediction):"
                        f"</strong> {ovl_items}"
                        f"</div>"
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
                f"{extra}"
                f"{fix_html}"
                f"</div>"
            )

        gaps_section = (
            "".join(gap_cards)
            if gap_cards
            else (
                '<p style="color:#2da44e;font-size:13px;margin:6px 0">'
                "&#10003; No gaps. All labels are fully resolved.</p>"
            )
        )

        # ── Summary header ────────────────────────────────────────────────
        surface = m._canonical_surface
        depth_info = (
            f"{len(surface)} canonical entities (per-branch)"
            if surface
            else "Not analyzed"
        )
        n_blocking = sum(
            1
            for i in m.get_issues()
            if i.severity in (IssueSeverity.ERROR, IssueSeverity.WARNING)
        )
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
                "1. Gaps",
                f"{n_blocking} requiring action" if n_blocking else "none",
            )
            + '<p style="font-size:12px;color:#57606a;margin:2px 0 4px">'
            "Labels that could not be mapped automatically. Resolve these before "
            "calling <code>mapper.get_mapped_results_dataframe()</code>.</p>"
            + gaps_section
            + _section_heading(
                "2. Annotation labels",
                f"{len(ann_pairs)} label(s) from your dataset",
            )
            + '<p style="font-size:12px;color:#57606a;margin:2px 0 4px">'
            "The entity labels in your ground-truth annotations. "
            "Each must map to exactly one canonical representation to be scored.</p>"
            + _label_table(ann_pairs, ann_counts)
            + _section_heading(
                "3. Prediction labels",
                f"{len(pred_pairs)} label(s) from the model",
            )
            + '<p style="font-size:12px;color:#57606a;margin:2px 0 4px">'
            "The entity labels your model outputs. "
            "Each must map to a canonical representation to be scored.</p>"
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
            f"{len(m._canonical_surface)} canonical entities (per-branch)"
            if m._canonical_surface
            else "Not analyzed"
        )
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
