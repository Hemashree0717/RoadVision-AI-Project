
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "road_results.db"

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                input_path TEXT NOT NULL,
                output_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_result(job_id, original_name, input_path, output_path):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO results(id, original_name, input_path, output_path) VALUES (?, ?, ?, ?)",
            (job_id, original_name, input_path, output_path)
        )

def list_results():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, original_name, input_path, output_path, created_at "
            "FROM results ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
