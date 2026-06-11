"""Generate eval/golden.jsonl: 50 normal cases (auto from recipe docs) + 20 edge cases.

Normal cases pull the first ingredient from each recipe doc as the expected keyword
(handling "A or B" variants as any-of), and vary the question phrasing (Thai+English).
Edge cases are hand-written: refusals, out-of-domain, greetings, counts, typos, reverse.

Run:  python eval/generate_golden.py
"""
import os
import re
import json
import random

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPES = os.path.join(os.path.dirname(HERE), "data", "recipes")

# recognizable items to prefer (skipped automatically if the file doesn't exist)
PREFERRED = [
    "Crafting_Table", "Furnace", "Chest", "Stick", "Torch", "Ladder", "Bowl",
    "Bread", "Cookie", "Cake", "Bucket", "Paper", "Book", "Bookshelf", "Clock",
    "Compass", "Shears", "Flint_and_Steel", "Fishing_Rod", "Bow", "Arrow", "Shield",
    "Anvil", "Hopper", "Dropper", "Dispenser", "Piston", "Observer", "Lever",
    "Redstone_Torch", "Redstone_Lamp", "Note_Block", "Jukebox", "Beacon", "Cauldron",
    "Brewing_Stand", "Iron_Bars", "Glass_Pane", "Rail", "Minecart", "Lantern",
    "Campfire", "Smithing_Table", "Stonecutter", "Grindstone", "Loom", "Barrel",
    "Blast_Furnace", "Smoker", "Target", "TNT", "Bone_Meal", "Iron_Block",
    "Gold_Block", "Diamond_Block", "Enchanting_Table",
]

TEMPLATES = [
    "{n} คราฟจากอะไร", "{n} คราฟยังไง", "{n} ทำจากอะไร", "วิธีคราฟ {n}",
    "how to craft {n}", "what do you need to make a {n}", "{n} ใช้อะไรทำ",
    "อยากได้ {n} ต้องใช้อะไร",
]


def first_ingredient(doc_text):
    m = re.search(r"^Ingredients:\s*(.+)$", doc_text, re.M)
    if not m:
        return None
    first = m.group(1).split(",")[0].strip()          # "Blackstone or Cobblestone x8"
    first = re.sub(r"\s*x\d+\s*$", "", first).strip()  # drop the "x8"
    parts = [p.strip() for p in first.split(" or ")]   # variants -> any-of
    return parts if len(parts) > 1 else parts[0]


def normal_cases():
    files = {os.path.splitext(f)[0] for f in os.listdir(RECIPES) if f.endswith(".md")}
    order = [p for p in PREFERRED if p in files] + sorted(files - set(PREFERRED))
    cases = []
    for i, key in enumerate(order):
        if len(cases) >= 50:
            break
        with open(os.path.join(RECIPES, key + ".md"), encoding="utf-8") as fh:
            text = fh.read()
        ing = first_ingredient(text)
        if not ing:
            continue
        name = key.replace("_", " ")
        q = TEMPLATES[i % len(TEMPLATES)].format(n=name)
        cases.append({"question": q, "must_contain": [ing], "kind": "normal"})
    return cases


REFUSAL = ["เพียงพอ", "enough", "ไม่มีข้อมูล", "don't have", "not enough", "ขออภัย", "ไม่พบ"]

EDGE = [
    # non-craftable -> should refuse
    # non-craftable: pivoted to broad answers (mined/found), no longer refusals
    {"question": "Diamond Ore คราฟจากอะไร", "must_contain": [["ขุด", "mine", "mined", "mining", "ไม่สามารถคราฟ", "cannot be crafted", "ไม่มีข้อมูล", "don't have"]], "kind": "edge", "note": "non-craftable: answered broadly (mined)"},
    {"question": "Bedrock คราฟยังไง", "must_contain": [["ไม่มีข้อมูล", "ไม่สามารถคราฟ", "cannot be crafted", "don't have", "not enough", "ขุด", "mine", "ไม่พบ"]], "kind": "edge", "note": "non-craftable: answered broadly"},
    {"question": "Dirt คราฟจากอะไร", "must_contain": [["ขุด", "mine", "mined", "ไม่ต้องคราฟ", "ไม่สามารถคราฟ", "cannot be crafted", "ไม่มีข้อมูล"]], "kind": "edge", "note": "non-craftable: answered broadly (mined/found)"},
    {"question": "how do you craft Obsidian", "must_contain": [["lava", "water", "ลาวา", "ขุด", "mine", "mined", "cannot be crafted", "ไม่สามารถคราฟ"]], "kind": "edge", "note": "non-craftable: answered broadly (water+lava)"},
    {"question": "Grass Block คราฟจากอะไร", "must_contain": [["ไม่สามารถคราฟ", "cannot be crafted", "ขุด", "mine", "mined", "ธรรมชาติ", "natural", "Silk Touch", "ไม่มีข้อมูล", "ไม่พบ"]], "kind": "edge", "note": "non-craftable: answered broadly"},
    # smelting (now covered)
    {"question": "Glass ได้จากการเผาอะไร", "must_contain": [["Sand"]], "kind": "edge", "note": "smelting"},
    # out of domain
    {"question": "ราคาบิทคอยน์วันนี้เท่าไหร่", "must_contain": [REFUSAL], "kind": "edge", "note": "out of domain"},
    {"question": "เมืองหลวงของประเทศไทยคือ", "must_contain": [REFUSAL], "kind": "edge", "note": "out of domain"},
    {"question": "2 บวก 2 เท่ากับเท่าไหร่", "must_contain": [REFUSAL], "kind": "edge", "note": "out of domain"},
    # greetings (no intent routing yet -> exposes limitation)
    {"question": "สวัสดีครับ", "must_contain": [REFUSAL + ["สวัสดี", "hello", "ช่วย"]], "kind": "edge", "note": "greeting"},
    {"question": "hello", "must_contain": [REFUSAL + ["hello", "hi", "help"]], "kind": "edge", "note": "greeting; known limitation: no greeting intent routing"},
    # counts / yields
    {"question": "Stick คราฟได้กี่อันต่อครั้ง", "must_contain": ["4"], "kind": "edge", "note": "yield count"},
    {"question": "คราฟ Torch ครั้งนึงได้กี่อัน", "must_contain": ["4"], "kind": "edge", "note": "yield count"},
    # variants
    {"question": "Furnace ใช้หินอะไรได้บ้าง", "must_contain": [["Cobblestone", "Blackstone", "Deepslate"]], "kind": "edge", "note": "variants"},
    {"question": "Stick ทำจากไม้ชนิดไหนได้บ้าง", "must_contain": [["Planks", "Bamboo"]], "kind": "edge", "note": "variants"},
    # english phrasing
    {"question": "how to craft a chest", "must_contain": ["Planks"], "kind": "edge", "note": "english"},
    {"question": "what do I need for a crafting table", "must_contain": ["Planks"], "kind": "edge", "note": "english"},
    # typo
    {"question": "Furnce คราฟจากอะไร", "must_contain": [["Cobblestone", "Blackstone", "Deepslate"]], "kind": "edge", "note": "typo"},
    # reverse lookup (hard -> may expose limitation)
    {"question": "อะไรที่คราฟจาก cobblestone 8 ก้อน", "must_contain": [["Furnace", "เตา"]], "kind": "edge", "note": "reverse lookup"},
    # ingredient question
    {"question": "Planks คราฟจากอะไร", "must_contain": [["Log", "Wood", "ไม้"]], "kind": "edge", "note": "intermediate item"},
    # multi-ingredient "what can I make with X + Y" (works via semantic retrieval; not exhaustive)
    {"question": "stick กับ ถ่าน ผสมกันแล้วได้เป็นอะไร", "must_contain": [["Torch", "คบ"]], "kind": "edge", "note": "multi-ingredient combine"},
    {"question": "เหล็ก 3 อัน และ แท่งไม้สอง สามารถทำเครื่องมืออะไรได้บ้าง วัตถุดิบสามารถเหลือได้", "must_contain": [["Pickaxe", "Axe", "เสียม", "ขวาน"]], "kind": "edge", "note": "multi-ingredient: tools from X+Y"},
]


SMELT_CASES = [
    {"question": "Iron Ingot ได้จากการเผาอะไร", "must_contain": [["Iron Ore", "Raw Iron"]], "kind": "normal"},
    {"question": "Raw Gold เผาแล้วได้อะไร", "must_contain": ["Gold"], "kind": "normal"},
    {"question": "how do I smelt Sand", "must_contain": ["Glass"], "kind": "normal"},
    {"question": "Cobblestone เผาได้อะไร", "must_contain": ["Stone"], "kind": "normal"},
    {"question": "เนื้อวัวดิบเผาได้อะไร", "must_contain": [["Cooked", "Steak", "สุก"]], "kind": "normal"},
    {"question": "Charcoal ได้ยังไง", "must_contain": [["Log", "Wood"]], "kind": "normal"},
]


def main():
    random.seed(7)
    cases = normal_cases() + SMELT_CASES + EDGE
    with open(os.path.join(HERE, "golden.jsonl"), "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_normal = sum(1 for c in cases if c["kind"] == "normal")
    print(f"Wrote {len(cases)} cases ({n_normal} normal + {len(EDGE)} edge) -> eval/golden.jsonl")


if __name__ == "__main__":
    main()
