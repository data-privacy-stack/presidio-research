#!/usr/bin/env python3
"""Build (or serve) the Presidio-Research documentation with Zensical.

Zensical (https://zensical.org) is the successor to Material for MkDocs. It can
read an existing ``mkdocs.yml`` natively, but it does **not** run MkDocs plugins
and it does not (yet) render Jupyter notebooks. Presidio-Research keeps its
source of truth outside ``docs/``: the landing page is the top-level
``README.md`` and the tutorials are notebooks under ``notebooks/``.

So, before invoking Zensical, this script *collects* those sources into a
staging copy of ``docs/``:

  * ``README.md``            -> ``<staging>/docs/index.md``  (the site Home)
  * ``notebooks/*.ipynb``    -> ``<staging>/docs/notebooks/*.ipynb``

and then pre-converts every collected notebook to Markdown with ``nbconvert``
so Zensical renders it natively (full theme, nav, search and TOC) instead of
serving a raw ``.ipynb`` download.

Notebook support is accepted on the Zensical backlog but unscheduled:
  - request : https://github.com/zensical/zensical/issues/52
  - backlog : https://github.com/zensical/backlog/issues/9

Steps
-----
1. Load ``mkdocs.yml`` and collect every ``*.ipynb`` referenced in the nav.
2. Mirror ``docs/`` into ``.zensical-build/docs/`` so the real source tree
   stays untouched (Material-compatible, minimal diff, easy to rebase).
3. Collect ``README.md`` -> ``index.md`` and the nav notebooks into the staging
   tree, absolutise the README's repo-relative links, and convert each notebook
   to Markdown (images land in a sibling ``<name>_files/`` directory).
4. Rewrite in-repo ``*.ipynb`` links to the converted ``*.md`` pages.
5. Emit a generated ``zensical.yml`` whose ``docs_dir`` is the staging tree,
   whose nav points at the ``.md`` files, and which drops MkDocs-only plugins.
6. Run ``zensical build`` (default) or ``zensical serve`` against that config.

Executables are taken from ``PATH`` by default. Override with:

  JUPYTER_BIN          path to the ``jupyter`` entry point (default: ``jupyter``)
  ZENSICAL_BIN         path to the ``zensical`` entry point (default: ``zensical``)

Usage
-----
  python scripts/zensical_build.py                # build to ./site
  python scripts/zensical_build.py serve -a :8001 # serve (extra args passed on)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MKDOCS_CONFIG = REPO_ROOT / "mkdocs.yml"
GENERATED_CONFIG = REPO_ROOT / "zensical.yml"

# Where the README's repo-relative links (to source files that are not part of
# the docs site) are re-pointed. Branch can be overridden in CI via DOCS_REF.
REPO_SLUG = "data-privacy-stack/presidio-research"
DOCS_REF = os.environ.get("DOCS_REF", "main")
GH_BLOB = f"https://github.com/{REPO_SLUG}/blob/{DOCS_REF}"
GH_RAW = f"https://raw.githubusercontent.com/{REPO_SLUG}/{DOCS_REF}"

# The site Home page is collected from the top-level README.
INDEX_SOURCE = REPO_ROOT / "README.md"

# Everything Zensical-specific is generated into this staging tree so the real
# ``docs/`` stays pristine (Material-compatible, minimal diff, easy rebase).
STAGING_DIR = REPO_ROOT / ".zensical-build"
STAGING_DOCS = STAGING_DIR / "docs"
STAGING_DOCS_REL = STAGING_DOCS.relative_to(REPO_ROOT).as_posix()


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #
class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that tolerates MkDocs' ``!!python/name:`` tags.

    We only need to *read* the config to discover notebooks; the generated
    config is produced by string transformation so these tags round-trip
    untouched.
    """


def _ignore_python_name(loader, suffix, node):  # noqa: ANN001, ARG001
    return None


_TolerantLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:", _ignore_python_name
)


def _collect_notebook_refs(nav) -> list[str]:
    """Return every ``*.ipynb`` path referenced in a MkDocs nav structure."""
    found: list[str] = []

    def walk(node) -> None:
        if isinstance(node, str):
            if node.endswith(".ipynb"):
                found.append(node)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)

    walk(nav)
    # De-duplicate while preserving order.
    return list(dict.fromkeys(found))


# --------------------------------------------------------------------------- #
# Staging
# --------------------------------------------------------------------------- #
def _stage_docs(src: Path, dst: Path) -> None:
    """Mirror the real docs tree into the staging dir (source stays untouched)."""
    if shutil.which("rsync"):
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", f"{src}/", f"{dst}/"], check=True)
    else:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


# --------------------------------------------------------------------------- #
# Collection (README -> index.md, notebooks/ -> docs)
# --------------------------------------------------------------------------- #
# Markdown link/image or HTML href/src whose URL is captured in group 2.
_URL_RE = re.compile(r"""(!?\]\(|(?:href|src)=["'])([^)"'\s]+)""")


def _absolutise_readme_links(text: str, notebook_rels: set[str]) -> str:
    """Re-point the README's repo-relative links so they work on the site.

    Links to collected notebooks are left relative (the notebook-link pass later
    rewrites them to the rendered ``.md`` page). Any other repo-relative link is
    re-pointed at GitHub so it does not 404 on the docs site; images use the raw
    host, everything else the blob host. Absolute URLs and pure anchors are left
    untouched.
    """

    def repl(match: re.Match) -> str:
        prefix, url = match.group(1), match.group(2)
        if url.startswith(("http://", "https://", "//", "mailto:", "#")):
            return match.group(0)
        path, _, frag = url.partition("#")
        clean_path = path.lstrip("./")
        # Notebook links become on-site pages; leave them for the .ipynb pass.
        if clean_path in notebook_rels:
            return match.group(0)
        # Links into docs/ resolve to on-site pages: the docs tree is the site
        # root, so drop the leading ``docs/`` and keep the link relative.
        if clean_path.startswith("docs/") and clean_path.endswith(".md"):
            rel = clean_path[len("docs/") :]
            return f"{prefix}{rel}" + (f"#{frag}" if frag else "")
        is_image = prefix.startswith("!") or prefix.startswith(("src=",))
        base = GH_RAW if is_image else GH_BLOB
        clean = path.lstrip("./")
        return f"{prefix}{base}/{clean}" + (f"#{frag}" if frag else "")

    return _URL_RE.sub(repl, text)


def _collect_sources(notebook_rels: list[str], docs_dir: Path) -> None:
    """Collect README (-> index.md) and nav notebooks into the staging docs."""
    # 1) README.md -> index.md, with repo-relative links absolutised.
    if not INDEX_SOURCE.exists():
        raise SystemExit(f"error: index source not found: {INDEX_SOURCE}")
    nb_names = {rel.lstrip("./") for rel in notebook_rels}
    index_text = INDEX_SOURCE.read_text(encoding="utf-8")
    index_text = _absolutise_readme_links(index_text, nb_names)
    (docs_dir / "index.md").write_text(index_text, encoding="utf-8")
    print(f"  collected README.md -> {docs_dir.name}/index.md")

    # 2) notebooks referenced in nav (paths are docs-relative and mirror the
    #    repo layout, e.g. ``notebooks/1_Generate_data.ipynb``).
    for rel in notebook_rels:
        src = REPO_ROOT / rel
        if not src.exists():
            print(f"  ! skip (missing notebook): {rel}", file=sys.stderr)
            continue
        dst = docs_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  collected {rel}")


# --------------------------------------------------------------------------- #
# Notebook conversion
# --------------------------------------------------------------------------- #
def _convert_notebooks(notebooks: list[str], docs_dir: Path) -> list[Path]:
    """Convert notebooks to Markdown in place; return generated paths."""
    jupyter = os.environ.get("JUPYTER_BIN", "jupyter")
    generated: list[Path] = []

    for rel in notebooks:
        src = docs_dir / rel
        if not src.exists():
            print(f"  ! skip (missing): {rel}", file=sys.stderr)
            continue

        out_md = src.with_suffix(".md")
        print(f"  - {rel} -> {out_md.relative_to(docs_dir)}")
        subprocess.run(
            [
                jupyter,
                "nbconvert",
                "--to",
                "markdown",
                "--output",
                src.stem,
                "--output-dir",
                str(src.parent),
                str(src),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        # Drop the raw .ipynb from the staging tree so Zensical does not also
        # copy it into the build output as a downloadable asset.
        src.unlink()
        generated.append(out_md)

    return generated


# --------------------------------------------------------------------------- #
# Link rewriting
# --------------------------------------------------------------------------- #
# Matches the URL of a Markdown link ``](...ipynb`` or an HTML ``href="...ipynb``.
# The match stops at ``.ipynb`` so any ``#fragment`` is preserved untouched.
_LINK_RE = re.compile(r"""(\]\(|href=["'])([^)"'#?\s]+\.ipynb)""")


def _rewrite_links(docs_dir: Path, notebooks: list[str]) -> None:
    """Point in-repo ``*.ipynb`` links at the converted ``*.md`` pages.

    Only links that resolve to a converted notebook are touched — external
    (GitHub) links and links to non-converted notebooks are left alone.
    """
    converted = {(docs_dir / rel).resolve() for rel in notebooks}
    total = 0

    for md in docs_dir.rglob("*.md"):
        base = md.parent
        text = md.read_text(encoding="utf-8")
        hits = 0

        def repl(match: re.Match) -> str:
            nonlocal hits
            prefix, url = match.group(1), match.group(2)
            if url.startswith(("http://", "https://", "//", "mailto:")):
                return match.group(0)
            target = (base / urllib.parse.unquote(url)).resolve()
            if target in converted:
                hits += 1
                return prefix + url[: -len(".ipynb")] + ".md"
            return match.group(0)

        new = _LINK_RE.sub(repl, text)
        if hits:
            md.write_text(new, encoding="utf-8")
            total += hits

    print(f"  rewrote {total} notebook link(s) to .md")


# --------------------------------------------------------------------------- #
# Generated config
# --------------------------------------------------------------------------- #
def _write_generated_config(raw: str, notebooks: list[str]) -> None:
    """Rewrite nav .ipynb -> .md, drop MkDocs-only plugins, point at staging."""
    text = raw
    for rel in notebooks:
        text = text.replace(rel, rel[: -len(".ipynb")] + ".md")

    # Remove any ``- mkdocs-*:`` plugin entry and its indented children (Zensical
    # does not run MkDocs plugins; ``search`` is built in).
    text = re.sub(
        r"[ \t]*-[ \t]*mkdocs-[^\n]*\n(?:[ \t]+[^\n]*\n)*",
        "",
        text,
    )

    # Build from the staging copy (converted notebooks + rewritten links).
    if re.search(r"(?m)^docs_dir:.*$", text):
        text = re.sub(r"(?m)^docs_dir:.*$", f"docs_dir: {STAGING_DOCS_REL}", text)
    else:
        text = f"docs_dir: {STAGING_DOCS_REL}\n" + text

    banner = (
        "# AUTOGENERATED by scripts/zensical_build.py - DO NOT EDIT.\n"
        "# Source of truth is mkdocs.yml. The Home page and notebook tutorials\n"
        "# are collected into a staging docs tree and notebook nav entries have\n"
        "# been converted to Markdown, because Zensical does not run MkDocs\n"
        "# plugins or render notebooks (backlog: zensical/backlog#9).\n"
        f"# docs_dir points at the generated staging tree ({STAGING_DOCS_REL}).\n\n"
    )
    GENERATED_CONFIG.write_text(banner + text, encoding="utf-8")
    print(f"  wrote {GENERATED_CONFIG.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    """Stage docs, collect+convert sources, write zensical.yml, build or serve."""
    command = argv[0] if argv else "build"
    passthrough = argv[1:]
    if command not in {"build", "serve"}:
        # Treat unknown first arg as a passthrough flag for `build`.
        command, passthrough = "build", argv

    raw = MKDOCS_CONFIG.read_text(encoding="utf-8")
    config = yaml.load(raw, Loader=_TolerantLoader)
    docs_dir = REPO_ROOT / config.get("docs_dir", "docs")

    notebooks = _collect_notebook_refs(config.get("nav", []))

    print(f"Staging docs -> {STAGING_DOCS_REL}")
    _stage_docs(docs_dir, STAGING_DOCS)

    print("Collecting sources (README + notebooks)...")
    _collect_sources(notebooks, STAGING_DOCS)

    print(f"Converting {len(notebooks)} notebook(s)...")
    _convert_notebooks(notebooks, STAGING_DOCS)

    print("Rewriting notebook links...")
    _rewrite_links(STAGING_DOCS, notebooks)

    print("Generating Zensical config...")
    _write_generated_config(raw, notebooks)

    zensical = os.environ.get("ZENSICAL_BIN", "zensical")
    cmd = [zensical, command, "-f", str(GENERATED_CONFIG), *passthrough]
    print(f"Running: {' '.join(cmd)}")

    # ``serve`` is interactive/long-running, so run it once. ``build`` can be
    # killed by the OS on memory spikes; that crash is transient, so retry a
    # couple of times before giving up.
    if command != "build":
        return subprocess.run(cmd, check=False).returncode

    attempts = max(1, int(os.environ.get("ZENSICAL_BUILD_RETRIES", "3")))
    rc = 0
    for attempt in range(1, attempts + 1):
        rc = subprocess.run(cmd, check=False).returncode
        if rc == 0:
            return 0
        print(f"  zensical build failed (exit {rc}), attempt {attempt}/{attempts}")
    print(f"  zensical build still failing after {attempts} attempt(s)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
