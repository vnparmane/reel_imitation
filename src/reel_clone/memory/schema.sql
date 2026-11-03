CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    start_time TEXT NOT NULL,
    end_time TEXT,
    total_reels INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reels (
    reel_id TEXT PRIMARY KEY,
    session_id TEXT,
    duration_sec REAL,
    watch_time_sec REAL,
    final_action INTEGER, -- 0: CONTINUE, 1: SKIP, 2: LIKE
    loop_count INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);