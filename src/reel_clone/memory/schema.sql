CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    start_time TEXT NOT NULL,
    end_time TEXT,
    total_reels INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reels (
    trajectory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    reel_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    duration_sec REAL,
    watch_time_sec REAL,
    final_action INTEGER,
    liked INTEGER DEFAULT 0,
    censored INTEGER DEFAULT 0,
    loop_count INTEGER DEFAULT 0,
    npz_filename TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_reels_session ON reels(session_id);
CREATE INDEX IF NOT EXISTS idx_reels_reel_id ON reels(reel_id);