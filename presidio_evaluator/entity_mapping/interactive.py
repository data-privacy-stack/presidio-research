"""
Interactive entity mapping utilities for experiment setup.

This module provides tools to automatically identify entities from models and datasets,
suggest mappings, and allow users to review/modify mappings before running experiments.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from collections import Counter
import json
import logging
from pathlib import Path
import html

try:
    from IPython.display import HTML, display

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

from presidio_evaluator.data_objects import InputSample
from .mapper import (
    create_presidio_mapper,
    EntityMapper,
)

logger = logging.getLogger(__name__)


class EntityMappingHelper:
    """
    Interactive helper for entity mapping in experiments.

    This class provides an intuitive workflow for:
    1. Auto-detecting entities from models and datasets
    2. Suggesting intelligent mappings
    3. Allowing user review and modifications
    4. Handling unmapped entities

    Example:
        >>> helper = EntityMappingHelper(
        ...     dataset=dataset,
        ...     model=analyzer_engine
        ... )
        >>> helper.review_mapping()
        >>>
        >>> # Make adjustments
        >>> helper.set_mapping("CUSTOMER_NAME", "PERSON")
        >>> helper.exclude_dataset_entities(["INTERNAL_ID"])
        >>>
        >>> # Get final configuration
        >>> mapping = helper.get_mapping()
        >>> entities = helper.get_model_entities_to_use()
    """

    def __init__(
        self,
        dataset: Union[List[InputSample], str, Path],
        model: Any,
        language: str = "en",
        mapper: Optional[Union[EntityMapper, Callable]] = None,
        threshold: float = 0.40,
    ):
        """
        Initialize the entity mapping helper.

        :param dataset: List of InputSample objects or path to JSON file
        :param model: Model object (Presidio Analyzer, Transformers, SpaCy, etc.)
        :param language: Language code for models that require it
        :param mapper: Optional custom mapper (uses Presidio mapper by default)
        :param threshold: Threshold for semantic similarity mapping
        """
        # Load dataset if needed
        if isinstance(dataset, (str, Path)):
            logger.info(f"Loading dataset from {dataset}")
            self.dataset = InputSample.read_dataset_json(dataset)
            self.dataset_path = str(dataset)
        else:
            self.dataset = dataset
            self.dataset_path = None

        self.model = model
        self.language = language
        self.threshold = threshold

        # Set up mapper
        if mapper is None:
            self.mapper = create_presidio_mapper()
        else:
            self.mapper = mapper

        # Extract entities
        self._extract_all_entities()

        # User modifications
        self._manual_mappings: Dict[str, Optional[str]] = {}
        self._excluded_dataset_entities: set = set()
        self._excluded_model_entities: set = set()

        # Generate initial mapping
        self._regenerate_mapping()

    def _extract_all_entities(self):
        """Extract entities from dataset and model."""
        # Dataset entities with counts
        self._dataset_entity_counts = get_dataset_entities(
            self.dataset, include_counts=True
        )
        self._all_dataset_entities = set(self._dataset_entity_counts.keys())

        # Model entities
        self._all_model_entities = get_model_entities(self.model, self.language)

        # Handle zero-shot models (e.g., GLiNER, LLMs)
        # For zero-shot, use dataset entities as the model entities
        self._is_zero_shot = False
        if not self._all_model_entities:
            logger.warning(
                "Could not auto-detect model entities. Assuming zero-shot model. "
                "Using dataset entities as the entities to pass to the model."
            )
            self._all_model_entities = list(self._all_dataset_entities)
            self._is_zero_shot = True

        logger.debug(
            f"Found {len(self._all_dataset_entities)} dataset entities, "
            f"{len(self._all_model_entities)} model entities{' (zero-shot)' if self._is_zero_shot else ''}"
        )

    def _regenerate_mapping(self):
        """Generate or regenerate the entity mapping."""
        # Get active entities (after exclusions)
        active_dataset_entities = [
            e
            for e in self._all_dataset_entities
            if e not in self._excluded_dataset_entities
        ]
        active_model_entities = [
            e
            for e in self._all_model_entities
            if e not in self._excluded_model_entities
        ]

        # For zero-shot models, create identity mapping
        if self._is_zero_shot:
            # Identity mapping: dataset entity → same entity for model
            self._suggested_mapping = {e: e for e in active_dataset_entities}
            self._mapping_scores = {e: 1.0 for e in active_dataset_entities}
        else:
            # Generate suggested mapping using mapper
            self._suggested_mapping, self._mapping_scores = suggest_mapping(
                dataset_entities=active_dataset_entities,
                model_entities=active_model_entities,
                mapper=self.mapper,
                return_scores=True,
            )

        # Apply manual overrides
        for dataset_entity, model_entity in self._manual_mappings.items():
            if dataset_entity in self._suggested_mapping:
                self._suggested_mapping[dataset_entity] = model_entity

        # Identify unmapped entities (requires user action)
        # Entities are only unmapped if they have no mapping AND weren't manually set to None
        self._unmapped_entities = [
            e
            for e, m in self._suggested_mapping.items()
            if m is None and e not in self._manual_mappings
        ]

    def review_mapping(
        self,
        show_scores: bool = False,
        show_excluded: bool = False,
        format: str = "html",
    ):
        """
        Display current mapping status.

        :param show_scores: Whether to show similarity scores
        :param show_excluded: Whether to show excluded entities
        :param format: Display format - "html" (default) or "compact"
        """
        if format == "html":
            self._review_mapping_html(show_scores, show_excluded)
        elif format == "compact":
            self._review_mapping_compact(show_scores, show_excluded)
        else:
            raise ValueError(f"Unknown format '{format}'. Use 'html' or 'compact'.")

    def _review_mapping_compact(
        self, show_scores: bool = False, show_excluded: bool = False
    ):
        """Compact text-based mapping review."""
        active_dataset_entities = [
            e
            for e in self._all_dataset_entities
            if e not in self._excluded_dataset_entities
        ]
        active_model_entities = [
            e
            for e in self._all_model_entities
            if e not in self._excluded_model_entities
        ]

        mapped_count = sum(1 for v in self._suggested_mapping.values() if v is not None)
        unmapped_count = len(self._unmapped_entities)

        print(
            f"\nMAPPING: {mapped_count} mapped, {unmapped_count} unmapped | "
            f"Dataset: {len(active_dataset_entities)} active, {len(self._excluded_dataset_entities)} excluded | "
            f"Model: {len(active_model_entities)} entities"
        )

        if self._is_zero_shot:
            print("🔄 ZERO-SHOT mode enabled")

        # Only show unmapped entities (the critical ones)
        if self._unmapped_entities:
            print(f"\n⚠️  UNMAPPED ({len(self._unmapped_entities)}):")
            for entity in sorted(self._unmapped_entities):
                count = self._dataset_entity_counts.get(entity, 0)
                print(f"  • {entity} ({count} samples)")
            print(
                "  💡 You can: 1) map to a model entity, 2) map to None (FN penalty), or 3) exclude them."
            )
        else:
            print("\n✅ All entities mapped! Ready to evaluate.")
            print("  💡 Entities mapped to None will be counted as False Negatives.")

        # Show manual mappings if any
        if self._manual_mappings:
            print(f"\n🔧 Manual ({len(self._manual_mappings)}): ", end="")
            manual_items = [
                f"{k}→{v}" for k, v in sorted(self._manual_mappings.items())
            ]
            print(", ".join(manual_items))

        # Show excluded if requested
        if show_excluded:
            if self._excluded_dataset_entities:
                print(
                    f"\n🚫 Excluded dataset: {', '.join(sorted(self._excluded_dataset_entities))}"
                )
            if self._excluded_model_entities:
                print(
                    f"🚫 Excluded model: {', '.join(sorted(self._excluded_model_entities))}"
                )

    def _review_mapping_html(
        self, show_scores: bool = False, show_excluded: bool = False
    ):
        """HTML-based mapping review with styled tables."""
        active_dataset_entities = [
            e
            for e in self._all_dataset_entities
            if e not in self._excluded_dataset_entities
        ]
        active_model_entities = [
            e
            for e in self._all_model_entities
            if e not in self._excluded_model_entities
        ]

        mapped_count = sum(1 for v in self._suggested_mapping.values() if v is not None)
        unmapped_count = len(self._unmapped_entities)

        # Build HTML
        html_parts = []
        html_parts.append(
            '<div style="font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif;">'
        )

        # Header
        html_parts.append(
            '<h3 style="margin-top: 0; color: #24292f;">🗺️ Entity Mapping Review</h3>'
        )

        # Zero-shot indicator
        if self._is_zero_shot:
            html_parts.append(
                '<div style="background: #ddf4ff; border-left: 4px solid #0969da; padding: 12px; margin: 10px 0; border-radius: 6px;">'
            )
            html_parts.append(
                "<strong>🔄 ZERO-SHOT mode:</strong> Dataset entities will be passed to the model during evaluation."
            )
            html_parts.append("</div>")

        # Summary stats
        html_parts.append(
            '<div style="display: flex; gap: 20px; margin: 15px 0; flex-wrap: wrap;">'
        )
        html_parts.append(
            '<div style="background: #f6f8fa; padding: 10px 15px; border-radius: 6px; border: 1px solid #d0d7de;">'
        )
        html_parts.append(
            f"<strong>Dataset:</strong> {len(active_dataset_entities)} active, {len(self._excluded_dataset_entities)} excluded</div>"
        )
        html_parts.append(
            '<div style="background: #f6f8fa; padding: 10px 15px; border-radius: 6px; border: 1px solid #d0d7de;">'
        )
        html_parts.append(
            f"<strong>Model:</strong> {len(active_model_entities)} entities</div>"
        )
        html_parts.append(
            f'<div style="background: {"#d1f4e0" if unmapped_count == 0 else "#fff8c5"}; padding: 10px 15px; border-radius: 6px; border: 1px solid {"#a2ddb8" if unmapped_count == 0 else "#d4a72c"};">'
        )
        html_parts.append(
            f"<strong>Status:</strong> {mapped_count} mapped, {unmapped_count} unmapped</div>"
        )
        html_parts.append("</div>")

        # Calculate total samples for percentage calculation
        total_samples = len(self.dataset)

        # Dataset entities (collapsible)
        html_parts.append(
            '<details style="margin: 15px 0; background: #f6f8fa; padding: 10px; border-radius: 6px; border: 1px solid #d0d7de;">'
        )
        html_parts.append(
            f'<summary style="cursor: pointer; font-weight: 600; padding: 5px;">📊 Dataset Entities ({len(active_dataset_entities)})</summary>'
        )
        html_parts.append('<div style="margin-top: 10px; padding-left: 10px;">')
        # Sort by count (most common first)
        sorted_dataset_entities = sorted(
            active_dataset_entities,
            key=lambda e: self._dataset_entity_counts.get(e, 0),
            reverse=True,
        )
        for entity in sorted_dataset_entities:
            count = self._dataset_entity_counts.get(entity, 0)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            entity_escaped = html.escape(entity)
            html_parts.append(
                f'<span style="display: inline-block; background: #ffffff; border: 1px solid #d0d7de; padding: 4px 8px; margin: 3px; border-radius: 4px; font-family: monospace; font-size: 12px;">'
                f'{entity_escaped} <span style="color: #57606a; font-size: 11px;">({count} samples, {percentage:.1f}%)</span></span>'
            )
        html_parts.append("</div></details>")

        # Model entities (collapsible)
        html_parts.append(
            '<details style="margin: 15px 0; background: #f6f8fa; padding: 10px; border-radius: 6px; border: 1px solid #d0d7de;">'
        )
        html_parts.append(
            f'<summary style="cursor: pointer; font-weight: 600; padding: 5px;">📦 Model Entities ({len(active_model_entities)})</summary>'
        )
        html_parts.append('<div style="margin-top: 10px; padding-left: 10px;">')
        for entity in sorted(active_model_entities):
            entity_escaped = html.escape(entity)
            html_parts.append(
                f'<span style="display: inline-block; background: #ffffff; border: 1px solid #d0d7de; padding: 4px 8px; margin: 3px; border-radius: 4px; font-family: monospace; font-size: 12px;">{entity_escaped}</span>'
            )
        html_parts.append("</div></details>")

        # Mapping table
        html_parts.append(
            '<h4 style="margin-top: 20px; color: #24292f;">📋 Entity Mapping</h4>'
        )
        html_parts.append(
            '<table style="width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px;">'
        )
        html_parts.append(
            '<thead><tr style="background: #f6f8fa; border-bottom: 2px solid #d0d7de;">'
        )
        html_parts.append(
            '<th style="padding: 8px; text-align: left; border: 1px solid #d0d7de;">Dataset Entity</th>'
        )
        html_parts.append(
            '<th style="padding: 8px; text-align: left; border: 1px solid #d0d7de;">→ Model Entity</th>'
        )
        html_parts.append(
            '<th style="padding: 8px; text-align: center; border: 1px solid #d0d7de;">Samples</th>'
        )
        html_parts.append(
            '<th style="padding: 8px; text-align: center; border: 1px solid #d0d7de;">Confidence</th>'
        )
        html_parts.append("</tr></thead><tbody>")

        # Sort by confidence (low to high): unmapped (0.0) first, then mapped to None, then by score ascending, then manual (1.0)
        def get_sort_key(entity):
            model_entity = self._suggested_mapping.get(entity)
            is_manual = entity in self._manual_mappings
            is_manually_none = is_manual and self._manual_mappings[entity] is None

            if not model_entity and not is_manually_none:
                # Unmapped (no action taken) - show first
                return (0, 0.0, entity)
            elif is_manually_none:
                # Manually mapped to None - show after unmapped
                return (1, 0.0, entity)
            elif is_manual:
                # Manual mapping to actual entity - highest confidence
                return (3, 1.0, entity)
            else:
                # Auto-mapped - use actual score
                score = self._mapping_scores.get(entity, 0.5)
                return (2, score, entity)

        sorted_entities = sorted(active_dataset_entities, key=get_sort_key)

        for dataset_entity in sorted_entities:
            model_entity = self._suggested_mapping.get(dataset_entity)
            count = self._dataset_entity_counts.get(dataset_entity, 0)
            is_manual = dataset_entity in self._manual_mappings
            is_manually_none = (
                is_manual and self._manual_mappings[dataset_entity] is None
            )

            # Calculate confidence
            if not model_entity and not is_manually_none:
                # Unmapped (no action taken)
                confidence = 0.0
            elif is_manually_none:
                # Explicitly mapped to None
                confidence = 1.0
            elif is_manual:
                confidence = 1.0
            else:
                confidence = self._mapping_scores.get(dataset_entity, 0.5)

            # Row styling based on confidence
            if not model_entity and not is_manually_none:
                # Unmapped (no action taken)
                bg_color = "#fff8c5"  # Yellow for unmapped
                border_color = "#d4a72c"
                status = "⚠️"
                mapping_text = "<em style='color: #cf222e;'>NOT MAPPED</em>"
            elif is_manually_none:
                # Explicitly mapped to None
                bg_color = "#f3f4f6"  # Gray for None
                border_color = "#d0d7de"
                status = "○"
                mapping_text = "<em style='color: #57606a;'>None</em>"
            elif confidence >= 0.7:
                bg_color = "#d1f4e0"  # Green for high confidence
                border_color = "#a2ddb8"
                status = "✓"
                mapping_text = html.escape(model_entity)
            else:
                bg_color = "#fff3cd"  # Light yellow for low confidence
                border_color = "#e5c365"
                status = "⚠"
                mapping_text = html.escape(model_entity)

            dataset_entity_escaped = html.escape(dataset_entity)
            html_parts.append(
                f'<tr style="background: {bg_color}; border: 1px solid {border_color};">'
            )
            html_parts.append(
                f'<td style="padding: 8px; border: 1px solid {border_color}; font-family: monospace; font-weight: 600;">{status} {dataset_entity_escaped}</td>'
            )
            html_parts.append(
                f'<td style="padding: 8px; border: 1px solid {border_color}; font-family: monospace;">{mapping_text}</td>'
            )
            html_parts.append(
                f'<td style="padding: 8px; border: 1px solid {border_color}; text-align: center;">{count}</td>'
            )

            # Confidence column (always shown now)
            if is_manually_none:
                conf_badge = '<span style="background: #57606a; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-family: monospace;">None ✎</span>'
            elif not model_entity:
                conf_color = "#cf222e"
                conf_badge = f'<span style="background: {conf_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-family: monospace;">{confidence:.2f}</span>'
            else:
                conf_color = (
                    "#2da44e"
                    if confidence >= 0.7
                    else ("#d4a72c" if confidence > 0.0 else "#cf222e")
                )
                conf_badge = f'<span style="background: {conf_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-family: monospace;">{confidence:.2f}</span>'
                if is_manual and confidence == 1.0:
                    conf_badge = '<span style="background: #0969da; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-family: monospace;">1.00 ✎</span>'
            html_parts.append(
                f'<td style="padding: 8px; border: 1px solid {border_color}; text-align: center;">{conf_badge}</td>'
            )
            html_parts.append("</tr>")

        html_parts.append("</tbody></table>")

        # Add legend for confidence scores
        html_parts.append(
            '<div style="margin: 10px 0; padding: 10px; background: #f6f8fa; border-radius: 6px; border: 1px solid #d0d7de; font-size: 12px;">'
        )
        html_parts.append("<strong>Confidence legend:</strong> ")
        html_parts.append(
            '<span style="background: #0969da; color: white; padding: 2px 6px; margin: 0 5px; border-radius: 3px;">1.00 ✎ Manual</span>'
        )
        html_parts.append(
            '<span style="background: #57606a; color: white; padding: 2px 6px; margin: 0 5px; border-radius: 3px;">None ✎ Mapped to None</span>'
        )
        html_parts.append(
            '<span style="background: #2da44e; color: white; padding: 2px 6px; margin: 0 5px; border-radius: 3px;">≥0.70 High</span>'
        )
        html_parts.append(
            '<span style="background: #d4a72c; color: white; padding: 2px 6px; margin: 0 5px; border-radius: 3px;">0.01-0.69 Low</span>'
        )
        html_parts.append(
            '<span style="background: #cf222e; color: white; padding: 2px 6px; margin: 0 5px; border-radius: 3px;">0.00 Unmapped</span>'
        )
        html_parts.append("</div>")

        # Excluded entities (collapsible)
        if show_excluded and (
            self._excluded_dataset_entities or self._excluded_model_entities
        ):
            html_parts.append(
                '<details style="margin: 15px 0; background: #fff1f0; padding: 10px; border-radius: 6px; border: 1px solid #ffccc7;">'
            )
            html_parts.append(
                '<summary style="cursor: pointer; font-weight: 600; padding: 5px;">🚫 Excluded Entities</summary>'
            )
            html_parts.append('<div style="margin-top: 10px;">')

            if self._excluded_dataset_entities:
                html_parts.append(
                    '<p style="margin: 5px 0; font-weight: 600;">Dataset entities (excluded from evaluation):</p>'
                )
                for entity in sorted(self._excluded_dataset_entities):
                    count = self._dataset_entity_counts.get(entity, 0)
                    entity_escaped = html.escape(entity)
                    html_parts.append(
                        f'<div style="padding: 4px 8px; margin: 3px 0; background: white; border: 1px solid #ffccc7; border-radius: 4px;">✗ {entity_escaped} ({count} samples)</div>'
                    )

            if self._excluded_model_entities:
                html_parts.append(
                    '<p style="margin: 15px 0 5px 0; font-weight: 600;">Model entities (predictions will be ignored):</p>'
                )
                for entity in sorted(self._excluded_model_entities):
                    entity_escaped = html.escape(entity)
                    html_parts.append(
                        f'<div style="padding: 4px 8px; margin: 3px 0; background: white; border: 1px solid #ffccc7; border-radius: 4px;">✗ {entity_escaped}</div>'
                    )

            html_parts.append("</div></details>")

        # Warning for unmapped entities
        if self._unmapped_entities:
            html_parts.append(
                '<div style="background: #fff8c5; border: 2px solid #d4a72c; padding: 15px; margin: 20px 0; border-radius: 6px;">'
            )
            html_parts.append(
                f'<h4 style="margin: 0 0 10px 0; color: #9a6700;">⚠️  Action Required: {len(self._unmapped_entities)} Unmapped Entities</h4>'
            )
            html_parts.append(
                '<p style="margin: 5px 0;">The following entities exist in your dataset but have no mapping:</p>'
            )
            html_parts.append('<ul style="margin: 10px 0; padding-left: 25px;">')
            for entity in sorted(self._unmapped_entities):
                count = self._dataset_entity_counts.get(entity, 0)
                entity_escaped = html.escape(entity)
                html_parts.append(
                    f'<li style="font-family: monospace; font-weight: 600;">{entity_escaped} <span style="color: #57606a;">({count} samples)</span></li>'
                )
            html_parts.append(
                '</ul><p style="margin: 10px 0; font-weight: 600;">You must take action:</p>'
            )
            html_parts.append('<ol style="margin: 5px 0; line-height: 1.6;">')
            html_parts.append(
                "<li><strong>Map to a model entity:</strong> <code>helper.set_mapping('ENTITY', 'TARGET')</code><br><span style=\"color: #57606a; font-size: 12px;\">The model will evaluate these entities using the mapped target entity.</span></li>"
            )
            html_parts.append(
                "<li><strong>Map to None:</strong> <code>helper.set_mapping('ENTITY', None)</code><br><span style=\"color: #57606a; font-size: 12px;\">Keep in dataset but penalize model for not detecting (counted as False Negatives).</span></li>"
            )
            html_parts.append(
                "<li><strong>Exclude them:</strong> <code>helper.exclude_dataset_entities(['ENTITY'])</code><br><span style=\"color: #57606a; font-size: 12px;\">Remove these entities from evaluation entirely (samples filtered out).</span></li>"
            )
            html_parts.append("</ol></div>")
        else:
            html_parts.append(
                '<div style="background: #d1f4e0; border: 2px solid #2da44e; padding: 15px; margin: 20px 0; border-radius: 6px;">'
            )
            html_parts.append(
                '<h4 style="margin: 0; color: #116329;">✅ All entities mapped! Ready to run experiment.</h4>'
            )
            html_parts.append(
                '<p style="margin: 10px 0 0 0; color: #116329; font-size: 13px;">💡 Note: Entities mapped to <code>None</code> will be counted as False Negatives since the model doesn\'t support them.</p>'
            )
            html_parts.append("</div>")

        html_parts.append("</div>")

        # Display HTML
        if IPYTHON_AVAILABLE:
            display(HTML("".join(html_parts)))
        else:
            # Fallback to compact format if IPython not available
            print("⚠️  IPython not available. Falling back to compact format.")
            self._review_mapping_compact(show_scores, show_excluded)

    def set_mapping(self, dataset_entity: str, model_entity: Optional[str]):
        """
        Manually set a mapping for a dataset entity.

        :param dataset_entity: Entity type from the dataset
        :param model_entity: Target entity type from the model, or None if the model doesn't support this entity

        Example:
            >>> helper.set_mapping("CUSTOMER_NAME", "PERSON")
            >>> helper.set_mapping("INTERNAL_ID", None)  # Model doesn't support this - will be FN in evaluation
        """
        if dataset_entity not in self._all_dataset_entities:
            print(f"⚠️  Warning: '{dataset_entity}' not found in dataset entities")
            return

        if model_entity is not None and model_entity not in self._all_model_entities:
            print(f"⚠️  Warning: '{model_entity}' is not in model's supported entities")
            print(
                f"   Available model entities: {', '.join(sorted(self._all_model_entities))}"
            )
            print(
                "   Mapping will still be set, but the model may not recognize this entity."
            )

        self._manual_mappings[dataset_entity] = model_entity
        self._regenerate_mapping()

        print(f"✓ Mapping set: {dataset_entity} → {model_entity or 'None'}")

        # Show quick summary
        unmapped_count = len(self._unmapped_entities)
        if unmapped_count > 0:
            print(f"   ({unmapped_count} entities still unmapped)")

    def exclude_dataset_entities(self, entities: Union[str, List[str]]):
        """
        Exclude dataset entities from evaluation.

        These entities will be ignored during evaluation (not counted in metrics).

        :param entities: Single entity or list of entities to exclude

        Example:
            >>> helper.exclude_dataset_entities("INTERNAL_CODE")
            >>> helper.exclude_dataset_entities(["SYSTEM_ID", "DEBUG_INFO"])
        """
        if isinstance(entities, str):
            entities = [entities]

        # Show warning about consequences
        valid_entities = [e for e in entities if e in self._all_dataset_entities]
        if valid_entities:
            print(f"\n⚠️  Excluding {len(valid_entities)} dataset entity(ies):")
            for entity in valid_entities:
                count = self._dataset_entity_counts.get(entity, 0)
                print(f"   • {entity} ({count} samples)")
            print("\n📌 Impact:")
            print(
                "   - Samples containing these entities will be filtered out from evaluation"
            )
            print(
                "   - Model predictions for these entities will be ignored in all metrics"
            )
            print()

        for entity in entities:
            if entity not in self._all_dataset_entities:
                print(f"⚠️  Warning: '{entity}' not found in dataset entities")
                continue
            self._excluded_dataset_entities.add(entity)

        self._regenerate_mapping()

        print(f"✓ Excluded {len(valid_entities)} dataset entity(ies)")
        unmapped_count = len(self._unmapped_entities)
        if unmapped_count > 0:
            print(f"   ({unmapped_count} entities still unmapped)")

    def exclude_model_entities(self, entities: Union[str, List[str]]):
        """
        Exclude model entities from evaluation.

        Predictions for these entities will be ignored.

        :param entities: Single entity or list of entities to exclude

        Example:
            >>> helper.exclude_model_entities("DATE_TIME")
            >>> helper.exclude_model_entities(["NRP", "ORGANIZATION"])
        """
        if isinstance(entities, str):
            entities = [entities]

        for entity in entities:
            if entity not in self._all_model_entities:
                print(f"⚠️  Warning: '{entity}' not found in model entities")
                continue
            self._excluded_model_entities.add(entity)

        self._regenerate_mapping()

        print(f"✓ Excluded {len(entities)} model entity(ies)")

    def get_mapping(self) -> Dict[str, str]:
        """
        Get the final entity mapping (excluding None values).

        Entities mapped to None are excluded as they represent dataset entities
        that the model doesn't support.

        :return: Dict mapping dataset entities to model entities
        :raises ValueError: If there are unmapped entities
        """
        if self._unmapped_entities:
            raise ValueError(
                f"Cannot get mapping: {len(self._unmapped_entities)} entities are still unmapped. "
                f"Unmapped: {', '.join(sorted(self._unmapped_entities))}. "
                f"Use set_mapping() to map them (including to None) or exclude_dataset_entities() to exclude them."
            )

        return {k: v for k, v in self._suggested_mapping.items() if v is not None}

    def get_model_entities_to_use(self) -> List[str]:
        """
        Get the list of model entities to use in evaluation.

        Entities mapped to None are included using their original dataset names
        (None means identity mapping - use entity as-is).

        :return: List of model entities (after exclusions)
        :raises ValueError: If there are unmapped entities
        """
        if self._unmapped_entities:
            raise ValueError(
                f"Cannot get model entities: {len(self._unmapped_entities)} dataset entities "
                f"are still unmapped. Resolve unmapped entities first."
            )

        # Get unique model entities from the mapping
        # For entities mapped to None, include the original dataset entity name
        mapped_entities = set()
        for dataset_entity, model_entity in self._suggested_mapping.items():
            if model_entity is None:
                # None means identity mapping - use original dataset entity
                mapped_entities.add(dataset_entity)
            else:
                mapped_entities.add(model_entity)

        # Remove excluded model entities
        active_entities = [
            e for e in mapped_entities if e not in self._excluded_model_entities
        ]

        return sorted(active_entities)

    def get_filtered_dataset(self) -> List[InputSample]:
        """
        Returns the dataset with excluded entities removed.

        Entities mapped to None are kept in the dataset with their original names
        (None means identity mapping - use entity as-is, not filtered out).

        :return: List of InputSamples with only non-excluded entities
        :raises ValueError: If there are unmapped entities in the dataset
        """
        # Check for unmapped entities first
        if self._unmapped_entities:
            raise ValueError(
                f"Cannot filter dataset: {len(self._unmapped_entities)} entities are unmapped. "
                f"Unmapped entities: {', '.join(sorted(self._unmapped_entities))}. "
                f"You must either map them using set_mapping() (including to None) or exclude them using exclude_dataset_entities()."
            )

        # Filter out samples that only contain excluded entities
        filtered = []
        for sample in self.dataset:
            # Check if sample has any non-excluded entities
            has_valid_entity = False
            for span in sample.spans:
                if span.entity_type not in self._excluded_dataset_entities:
                    has_valid_entity = True
                    break

            if has_valid_entity:
                filtered.append(sample)

        logger.info(
            f"Filtered dataset: {len(filtered)}/{len(self.dataset)} samples retained"
        )
        return filtered


def get_model_entities(model: Any, language: str = "en") -> List[str]:
    """
    Automatically detect supported entities from a model.

    Supports:
    - Presidio Analyzer (via analyzer.get_supported_entities())
    - Presidio AnalyzerEngine wrapper
    - TransformersNlpEngine (via NerModelConfiguration mapping or model config)
    - Transformers models (via config.label2id or config.id2label)
    - SpaCy models (via pipeline component labels)
    - Any model with .entities attribute

    :param model: The model object (Presidio, Transformers, SpaCy, etc.)
    :param language: Language code for models that require it
    :return: List of entity types supported by the model

    Examples:
        >>> # Presidio Analyzer
        >>> from presidio_analyzer import AnalyzerEngine
        >>> analyzer = AnalyzerEngine()
        >>> entities = get_model_entities(analyzer)

        >>> # TransformersNlpEngine with user mapping
        >>> from presidio_analyzer.nlp_engine import TransformersNlpEngine
        >>> nlp_engine = TransformersNlpEngine(
        ...     models=model_config,
        ...     ner_model_configuration=ner_config
        ... )
        >>> entities = get_model_entities(nlp_engine)

        >>> # Transformers model
        >>> from transformers import AutoModelForTokenClassification
        >>> model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        >>> entities = get_model_entities(model)

        >>> # Model wrapper
        >>> from presidio_evaluator.models import PresidioAnalyzerWrapper
        >>> wrapper = PresidioAnalyzerWrapper()
        >>> entities = get_model_entities(wrapper)
    """
    entities = []

    # Try Presidio AnalyzerEngine (direct or wrapped)
    if hasattr(model, "get_supported_entities"):
        try:
            entities = model.get_supported_entities(language)
            logger.info(f"Detected {len(entities)} entities from Presidio Analyzer")
            return entities
        except Exception as e:
            logger.debug(f"Failed to get entities via get_supported_entities: {e}")

    # Try wrapped Presidio model
    if hasattr(model, "analyzer_engine") and hasattr(
        model.analyzer_engine, "get_supported_entities"
    ):
        try:
            entities = model.analyzer_engine.get_supported_entities(language)
            logger.info(
                f"Detected {len(entities)} entities from wrapped Presidio Analyzer"
            )
            return entities
        except Exception as e:
            logger.debug(f"Failed to get entities from analyzer_engine: {e}")

    # Try entities attribute (common in model wrappers)
    if hasattr(model, "entities") and model.entities:
        entities = model.entities
        logger.info(f"Detected {len(entities)} entities from model.entities attribute")
        return entities

    # Try Transformers model config
    if hasattr(model, "config"):
        config = model.config

        # Try id2label
        if hasattr(config, "id2label") and config.id2label is not None:
            try:
                labels = list(config.id2label.values())
                # Filter out special tokens and strip BIO prefixes
                entities = set()
                for label in labels:
                    if label in ["O", "PAD", "[PAD]", "[CLS]", "[SEP]"]:
                        continue
                    # Strip BIO/BILUO prefixes
                    if label.startswith(("B-", "I-", "O-")):
                        entity = label[2:]
                    elif label.startswith(("U-", "L-")):
                        entity = label[2:]
                    else:
                        entity = label
                    if entity and entity != "O":
                        entities.add(entity)
                entities = sorted(entities)
                logger.info(
                    f"Detected {len(entities)} entities from Transformers config.id2label"
                )
                return entities
            except (TypeError, AttributeError) as e:
                logger.debug(f"Failed to extract entities from id2label: {e}")

        # Try label2id
        if hasattr(config, "label2id") and config.label2id is not None:
            try:
                labels = list(config.label2id.keys())
                entities = set()
                for label in labels:
                    if label in ["O", "PAD", "[PAD]", "[CLS]", "[SEP]"]:
                        continue
                    if label.startswith(("B-", "I-", "O-")):
                        entity = label[2:]
                    elif label.startswith(("U-", "L-")):
                        entity = label[2:]
                    else:
                        entity = label
                    if entity and entity != "O":
                        entities.add(entity)
                entities = sorted(entities)
                logger.info(
                    f"Detected {len(entities)} entities from Transformers config.label2id"
                )
                return entities
            except (TypeError, AttributeError) as e:
                logger.debug(f"Failed to extract entities from label2id: {e}")

    # Try SpaCy pipeline
    if hasattr(model, "pipe_names") and "ner" in model.pipe_names:
        try:
            ner_component = model.get_pipe("ner")
            if hasattr(ner_component, "labels"):
                entities = list(ner_component.labels)
                logger.info(
                    f"Detected {len(entities)} entities from SpaCy NER component"
                )
                return entities
        except Exception as e:
            logger.debug(f"Failed to get entities from SpaCy pipeline: {e}")

    # Try TransformersNlpEngine
    if hasattr(model, "ner_model_configuration") or hasattr(model, "_engines"):
        try:
            entities = _get_transformers_nlp_engine_entities(model)
            if entities:
                logger.info(
                    f"Detected {len(entities)} entities from TransformersNlpEngine"
                )
                return entities
        except Exception as e:
            logger.debug(f"Failed to get entities from TransformersNlpEngine: {e}")

    # If nothing worked, return empty list
    if not entities:
        logger.warning(
            "Could not automatically detect model entities. Please provide them manually."
        )
        return []

    return entities


def _get_transformers_nlp_engine_entities(nlp_engine: Any) -> List[str]:
    """
    Extract entity types from a TransformersNlpEngine.

    Checks in this order:
    1. User-provided mapping in NerModelConfiguration.model_to_presidio_entity_mapping
    2. Underlying transformer model's config (id2label/label2id)

    :param nlp_engine: TransformersNlpEngine object
    :return: List of entity types
    """
    entities = set()

    # Strategy 1: Check if user provided a mapping in NerModelConfiguration
    if hasattr(nlp_engine, "ner_model_configuration"):
        config = nlp_engine.ner_model_configuration
        if config and hasattr(config, "model_to_presidio_entity_mapping"):
            mapping = config.model_to_presidio_entity_mapping
            if mapping:
                # Get unique entity values from the mapping
                # These are the Presidio entity names the user wants to use
                entities.update(mapping.values())
                logger.info(
                    f"Found entities from NerModelConfiguration mapping: {sorted(entities)}"
                )
                if entities:
                    return sorted(entities)

    # Strategy 2: Extract from underlying transformer models
    if hasattr(nlp_engine, "_engines"):
        for lang_code, engine in nlp_engine._engines.items():
            # Try to access the transformers model
            if hasattr(engine, "model"):
                transformer_model = engine.model
                if hasattr(transformer_model, "config"):
                    config = transformer_model.config

                    # Try id2label
                    if hasattr(config, "id2label") and config.id2label:
                        try:
                            for label in config.id2label.values():
                                # Skip special tokens
                                if label in ["O", "PAD", "[PAD]", "[CLS]", "[SEP]"]:
                                    continue
                                # Strip BIO/BILUO prefixes
                                if label.startswith(("B-", "I-", "O-")):
                                    entity = label[2:]
                                elif label.startswith(("U-", "L-")):
                                    entity = label[2:]
                                else:
                                    entity = label
                                if entity and entity != "O":
                                    entities.add(entity)
                        except (TypeError, AttributeError) as e:
                            logger.debug(f"Failed to extract from id2label: {e}")

                    # Try label2id
                    if not entities and hasattr(config, "label2id") and config.label2id:
                        try:
                            for label in config.label2id.keys():
                                if label in ["O", "PAD", "[PAD]", "[CLS]", "[SEP]"]:
                                    continue
                                if label.startswith(("B-", "I-", "O-")):
                                    entity = label[2:]
                                elif label.startswith(("U-", "L-")):
                                    entity = label[2:]
                                else:
                                    entity = label
                                if entity and entity != "O":
                                    entities.add(entity)
                        except (TypeError, AttributeError) as e:
                            logger.debug(f"Failed to extract from label2id: {e}")

            if entities:
                logger.info(
                    f"Found entities from transformer model config: {sorted(entities)}"
                )
                break

    return sorted(entities)


def get_dataset_entities(
    dataset: List[InputSample],
    include_counts: bool = False,
) -> Union[List[str], Dict[str, int]]:
    """
    Extract entity types from a dataset.

    Extracts entities from both spans and tags to handle all cases.
    Uses existing InputSample methods for entity extraction.

    :param dataset: A list of InputSample objects
    :param include_counts: If True, return dict with entity counts; if False, return list of entities
    :return: List of entity types or dict mapping entity types to sample counts

    Examples:
        >>> # From loaded dataset
        >>> dataset = InputSample.read_dataset_json("data.json")
        >>> entities = get_dataset_entities(dataset)
        >>> # Returns: ['PERSON', 'EMAIL', 'PHONE', ...]

        >>> # With counts
        >>> entity_counts = get_dataset_entities(dataset, include_counts=True)
        >>> # Returns: {'PERSON': 45, 'EMAIL': 30, 'PHONE': 25, ...}
    """
    entity_types = InputSample.extract_entity_types(dataset)
    logger.info(f"Found {len(entity_types)} unique entity types in dataset")

    if not include_counts:
        return sorted(entity_types)

    # Count how many samples contain each entity type
    entity_sample_counts: Dict[str, int] = Counter()
    for sample in dataset:
        sample_entities = set()

        # Extract from spans
        if sample.spans:
            for span in sample.spans:
                sample_entities.add(span.entity_type)

        # Extract from tags (tokens)
        if sample.tags:
            for tag in sample.tags:
                # Strip BIO/BILUO prefixes
                if tag.startswith(("B-", "I-", "O-")):
                    entity_type = tag[2:]
                elif tag.startswith(("U-", "L-")):
                    entity_type = tag[2:]
                else:
                    entity_type = tag

                if entity_type and entity_type != "O":
                    sample_entities.add(entity_type)

        # Increment count for each unique entity in this sample
        for entity in sample_entities:
            entity_sample_counts[entity] += 1

    return dict(entity_sample_counts)


def suggest_mapping(
    dataset_entities: List[str],
    model_entities: List[str],
    mapper: Optional[EntityMapper] = None,
    return_scores: bool = False,
) -> Union[Dict[str, Optional[str]], Tuple[Dict[str, Optional[str]], Dict[str, float]]]:
    """
    Suggest automatic mapping from dataset entities to model entities.

    Uses intelligent mapping strategies (exact match, pattern matching, semantic similarity).

    :param dataset_entities: List of entity types from the dataset
    :param model_entities: List of entity types supported by the model
    :param mapper: Optional custom EntityMapper. If None, uses create_presidio_mapper()
    :param return_scores: If True, also return similarity scores for each mapping
    :return: Dict mapping dataset entities to model entities (and optionally scores)

    Examples:
        >>> dataset_entities = ['FIRST_NAME', 'EMAIL', 'SSN']
        >>> model_entities = ['PERSON', 'CONTACT', 'ID', 'LOCATION']
        >>> mapping = suggest_mapping(dataset_entities, model_entities)
        >>> # Returns: {'FIRST_NAME': 'PERSON', 'EMAIL': 'CONTACT', 'SSN': 'ID'}
    """
    if mapper is None:
        # Use Presidio mapper by default
        mapper_func = create_presidio_mapper()
    else:
        mapper_func = mapper

    mappings = {}
    scores = {}

    for dataset_entity in dataset_entities:
        # Try to map - the mapper now returns (entity, confidence) tuples
        if hasattr(mapper_func, "map"):
            result = mapper_func.map(dataset_entity, model_entities)
        else:
            # Assume it's a callable
            result = mapper_func(dataset_entity, model_entities)

        # Extract entity and confidence from result
        if result is None:
            # No mapping found
            mappings[dataset_entity] = None
            if return_scores:
                scores[dataset_entity] = 0.0
        elif isinstance(result, tuple):
            # New format: (entity, confidence)
            mapped_entity, confidence = result
            mappings[dataset_entity] = mapped_entity
            if return_scores:
                scores[dataset_entity] = confidence
        else:
            # Legacy format: just the entity (for backward compatibility with old mappers)
            mappings[dataset_entity] = result
            if return_scores:
                # Assume exact match if no confidence provided
                scores[dataset_entity] = 1.0

    if return_scores:
        return mappings, scores
    return mappings
