"""Generate clean, short recipe docs from minecraft-data (PrismarineJS).

Downloads the authoritative items.json + recipes.json for one Java version, then
writes one short recipe doc per craftable item to data/recipes/<Item>.md.

minecraft-data stores a SEPARATE recipe per ingredient variant (e.g. one Stick
recipe per plank type, one Furnace recipe per stone type). We MERGE all recipes
of an item so the doc shows the generic ingredient ("Planks", or
"Cobblestone or Blackstone or Cobbled Deepslate") instead of one arbitrary variant.

These focused recipe docs are the crafting corpus: precise recipe retrieval with
complete coverage (the craftable list = the keys of recipes.json). Broader
questions (usage, lore, mobs) are answered by live minecraft.wiki search instead.
"""
import os
import re
import requests

VERSION = "1.21.11"
BASE = f"https://raw.githubusercontent.com/PrismarineJS/minecraft-data/master/data/pc/{VERSION}"
OUT = os.path.join("data", "recipes")
WIKI = "https://minecraft.wiki/w/"


def load(name):
    r = requests.get(f"{BASE}/{name}", timeout=60)
    r.raise_for_status()
    return r.json()


def normalize(name):
    """Collapse wood-variant families so '* Planks' -> 'Planks'."""
    if name and name.endswith(" Planks"):
        return "Planks"
    return name


def cell_names(entry, id2name):
    """All ingredient names for one grid cell (int id / {id} / [ids] / None)."""
    if entry is None:
        return []
    if isinstance(entry, list):
        out = []
        for e in entry:
            out += cell_names(e, id2name)
        return out
    if isinstance(entry, dict):
        entry = entry.get("id")
    nm = id2name.get(entry)
    return [nm] if nm else []


def _order(name_set):
    # single-word names first (e.g. Cobblestone before Cobbled Deepslate)
    return sorted(name_set, key=lambda n: (len(n.split()), len(n), n))


def fmt_cell(name_set):
    if not name_set:
        return None
    return " or ".join(_order(name_set)[:3])


def short_name(name_set):
    """One short representative for a grid cell (full alternatives go in Ingredients)."""
    return _order(name_set)[0] if name_set else "(empty)"


def merged_grid(recipes, id2name):
    """Merge all shaped recipes of an item into one grid; each cell = the set of
    distinct (normalized) ingredient names used there across variants."""
    shaped = [r for r in recipes if r.get("inShape")]
    if not shaped:
        return None
    rows = max(len(r["inShape"]) for r in shaped)
    cols = max((len(row) for r in shaped for row in r["inShape"]), default=0)
    cells = [[set() for _ in range(cols)] for _ in range(rows)]
    for r in shaped:
        sh = r["inShape"]
        for i in range(rows):
            row = sh[i] if i < len(sh) else []
            for j in range(cols):
                for nm in cell_names(row[j] if j < len(row) else None, id2name):
                    cells[i][j].add(normalize(nm))
    return cells  # list of list of set (raw)


def shapeless_counts(recipes, id2name):
    for r in recipes:
        if r.get("ingredients"):
            counts = {}
            for cell in r["ingredients"]:
                names = cell_names(cell, id2name)
                if names:
                    nm = normalize(names[0])
                    counts[nm] = counts.get(nm, 0) + 1
            if counts:
                return counts
    return None


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def render(name, recipes, id2name):
    counts, shape_desc = {}, ""
    grid = merged_grid(recipes, id2name)
    if grid:
        lines = []
        for row in grid:
            lines.append(" | ".join(short_name(c) for c in row))
            for c in row:
                if c:
                    full = fmt_cell(c)
                    counts[full] = counts.get(full, 0) + 1
        shape_desc = "Grid (rows of the crafting table):\n" + "\n".join(lines)
    else:
        counts = shapeless_counts(recipes, id2name) or {}
        shape_desc = "Shapeless recipe (no fixed positions)."

    if not counts:
        return None

    out_count = 1
    for r in recipes:
        res = r.get("result")
        if isinstance(res, dict) and res.get("count"):
            out_count = res["count"]
            break

    ing_summary = ", ".join(f"{k} x{v}" for k, v in counts.items())
    yield_txt = f" (makes {out_count})" if out_count and out_count > 1 else ""
    return (
        f"# {name}\n"
        f"source: {WIKI}{safe_filename(name)}\n\n"
        f"To craft {name}{yield_txt}, you need: {ing_summary}.\n"
        f"Ingredients: {ing_summary}\n"
        f"{shape_desc}\n"
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Downloading minecraft-data {VERSION} ...")
    items = load("items.json")
    recipes = load("recipes.json")
    id2name = {it["id"]: it.get("displayName") or it.get("name", str(it["id"])) for it in items}
    print(f"  items: {len(id2name)} | recipe groups: {len(recipes)}")

    written = 0
    for result_id, recipe_list in recipes.items():
        try:
            rid = int(result_id)
        except ValueError:
            rid = result_id
        name = id2name.get(rid)
        if not name:
            continue
        doc = render(name, recipe_list, id2name)
        if doc:
            with open(os.path.join(OUT, f"{safe_filename(name)}.md"), "w", encoding="utf-8") as f:
                f.write(doc)
            written += 1
    print(f"Done. {written} recipe docs -> {OUT}/  (next: python ingest.py --folder data/recipes)")


if __name__ == "__main__":
    main()
