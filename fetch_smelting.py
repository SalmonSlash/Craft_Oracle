"""Generate short smelting (furnace) recipe docs -> data/smelting/.

minecraft-data's recipes.json covers crafting-grid recipes only, so the common
smelting recipes (ore -> ingot, raw food -> cooked, etc.) are curated here. Same
short-doc format as the crafting recipe docs, ingested into the same collection.
"""
import os
import re

OUT = os.path.join("data", "smelting")
WIKI = "https://minecraft.wiki/w/"

# output -> list of accepted inputs (smelt any of them in a Furnace)
SMELT = {
    "Iron Ingot": ["Iron Ore", "Deepslate Iron Ore", "Raw Iron"],
    "Gold Ingot": ["Gold Ore", "Deepslate Gold Ore", "Nether Gold Ore", "Raw Gold"],
    "Copper Ingot": ["Copper Ore", "Deepslate Copper Ore", "Raw Copper"],
    "Netherite Scrap": ["Ancient Debris"],
    "Glass": ["Sand", "Red Sand"],
    "Stone": ["Cobblestone"],
    "Smooth Stone": ["Stone"],
    "Brick": ["Clay Ball"],
    "Terracotta": ["Clay"],
    "Nether Brick": ["Netherrack"],
    "Charcoal": ["Log", "Wood"],
    "Green Dye": ["Cactus"],
    "Sponge": ["Wet Sponge"],
    "Popped Chorus Fruit": ["Chorus Fruit"],
    "Dried Kelp": ["Kelp"],
    "Deepslate": ["Cobbled Deepslate"],
    "Cracked Stone Bricks": ["Stone Bricks"],
    "Smooth Sandstone": ["Sandstone"],
    "Smooth Red Sandstone": ["Red Sandstone"],
    "Smooth Quartz": ["Quartz Block"],
    "Cooked Beef": ["Raw Beef"],
    "Cooked Porkchop": ["Raw Porkchop"],
    "Cooked Chicken": ["Raw Chicken"],
    "Cooked Mutton": ["Raw Mutton"],
    "Cooked Rabbit": ["Raw Rabbit"],
    "Cooked Cod": ["Raw Cod"],
    "Cooked Salmon": ["Raw Salmon"],
    "Baked Potato": ["Potato"],
}


def safe(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def main():
    os.makedirs(OUT, exist_ok=True)
    for output, inputs in SMELT.items():
        ins = " or ".join(inputs)
        doc = (
            f"# {output} (smelting)\n"
            f"source: {WIKI}{safe(output)}\n\n"
            f"To obtain {output} by smelting, put {ins} in a Furnace.\n"
            f"Smelting: {ins} -> {output} (in a Furnace)\n"
        )
        with open(os.path.join(OUT, f"{safe(output)}_smelting.md"), "w", encoding="utf-8") as f:
            f.write(doc)
    print(f"Done. {len(SMELT)} smelting docs -> {OUT}/")


if __name__ == "__main__":
    main()
