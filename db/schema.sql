-- db/schema.sql
CREATE TABLE IF NOT EXISTS image_metadata (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    width INT,
    height INT,
    format VARCHAR(50),
    file_size BIGINT,
    processed_at TIMESTAMP DEFAULT NOW()
);
