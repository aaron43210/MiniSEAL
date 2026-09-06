# src/hitl/db.py

import sqlite3
from typing import List, Dict, Any
import os

class QuarantineDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.initialize_schema()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_schema(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Proposals Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quarantine_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    passage TEXT NOT NULL,
                    question TEXT NOT NULL,
                    proposed_answer TEXT NOT NULL,
                    status TEXT CHECK(status IN ('PENDING', 'APPROVED', 'REJECTED', 'TRAINED')) NOT NULL DEFAULT 'PENDING',
                    risk_level TEXT CHECK(risk_level IN ('LOW', 'MEDIUM', 'HIGH')) NOT NULL DEFAULT 'LOW',
                    risk_flags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # LoRA Adapt Jobs Tracking Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS adapter_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_status TEXT CHECK(job_status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED')) NOT NULL,
                    run_loss REAL,
                    adapter_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_proposal(self, passage: str, question: str, answer: str, risk_level: str, flags: List[str]) -> int:
        flags_str = ",".join(flags) if flags else None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO quarantine_proposals (passage, question, proposed_answer, status, risk_level, risk_flags)
                VALUES (?, ?, ?, 'PENDING', ?, ?)
            """, (passage, question, answer, risk_level, flags_str))
            conn.commit()
            return cursor.lastrowid

    def list_proposals(self, status: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM quarantine_proposals WHERE status = ?", (status,))
            return [dict(row) for row in cursor.fetchall()]

    def update_status(self, proposal_id: int, new_status: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE quarantine_proposals SET status = ? WHERE id = ?", (new_status, proposal_id))
            conn.commit()
