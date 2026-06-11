"""One-command setup for anyone cloning this repo.

    python bootstrap.py

What it does:
  1. creates .env from .env.example if missing (then asks you to paste your key)
  2. checks the required key is filled in
  3. ingests the corpus into the vector store (local file OR Qdrant Cloud,
     depending on your .env) -- this is the one-time embedding step

After it finishes:  streamlit run app.py
"""
import os
import shutil
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    env = os.path.join(HERE, ".env")
    example = os.path.join(HERE, ".env.example")

    # 1) scaffold .env
    if not os.path.exists(env):
        shutil.copy(example, env)
        print("Created .env from .env.example.")
        print("  -> Open .env and paste your OpenRouter key into BOTH")
        print("     LLM_API_KEY and EMBED_API_KEY (same key).")
        print("  -> Optional: set QDRANT_URL + QDRANT_API_KEY to use Qdrant Cloud")
        print("     instead of a local file index.")
        print("Then run `python bootstrap.py` again.")
        return

    # 2) check the key is present
    try:
        from dotenv import load_dotenv
        load_dotenv(env)
    except ImportError:
        print("Run `pip install -r requirements.txt` first.")
        return
    if not os.getenv("LLM_API_KEY") or not os.getenv("EMBED_API_KEY"):
        print("LLM_API_KEY / EMBED_API_KEY are empty in .env.")
        print("Paste your OpenRouter key into both, then re-run `python bootstrap.py`.")
        return

    # 3) ingest (one-time embedding into the vector store): crafting recipes
    #    (fresh collection) + smelting recipes (appended).
    print("Ingesting the Minecraft corpus into the vector store (one-time)...")
    subprocess.run(
        [sys.executable, "ingest.py", "--folder", "data/recipes", "--recreate"],
        cwd=HERE, check=True,
    )
    subprocess.run(
        [sys.executable, "ingest.py", "--folder", "data/smelting"],
        cwd=HERE, check=True,
    )
    print("\nSetup complete. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
