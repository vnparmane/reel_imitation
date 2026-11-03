from pathlib import Path
import sqlite3

db_file = Path("data/trajectories.db")
if db_file.exists():
    with sqlite3.connect(db_file) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(reels)").fetchall()]
    if "trajectory_id" not in cols:
        backup = db_file.with_suffix(".db.bak")
        db_file.replace(backup)
        print(f"Moved legacy database to {backup}")


pyproject = Path("pyproject.toml")
if pyproject.exists():
    text = pyproject.read_text(encoding="utf-8")
    if "[tool.setuptools.package-data]" not in text:
        text += '\n[tool.setuptools.package-data]\nreel_clone = ["memory/*.sql"]\n'
        pyproject.write_text(text, encoding="utf-8", newline="\n")
        print("Added [tool.setuptools.package-data] to pyproject.toml")