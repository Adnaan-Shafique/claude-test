-- Migration for databases created with the ORIGINAL init.sql (the eight-table
-- version with no is_active/is_admin, no per-user example scoping, and no
-- conversation-scoped short-term memory).
--
-- Fresh installs should run ../init.sql instead; this file is only for
-- upgrading an existing deployment in place.
--
--   psql -h <host> -U <user> -d <database> -f migrations/001_auth_and_scoping.sql
--
-- Run it inside a transaction so a partial upgrade cannot be left behind.
BEGIN;

-- 1. Users: approval workflow + optional email.
ALTER TABLE coding_agent_schema.users
    ADD COLUMN IF NOT EXISTS email       TEXT,
    ADD COLUMN IF NOT EXISTS is_active   BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS is_admin    BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS approved_by INTEGER REFERENCES coding_agent_schema.users(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ;

-- Accounts that already existed were created by an admin via curl, so they
-- keep working: activate them rather than locking everyone out on upgrade.
UPDATE coding_agent_schema.users SET is_active = TRUE, approved_at = NOW() WHERE approved_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS users_username_lower_idx
    ON coding_agent_schema.users (LOWER(username));

-- 2. Short-term memory becomes conversation-scoped.
ALTER TABLE coding_agent_schema.short_term_memory
    ADD COLUMN IF NOT EXISTS conversation_id INTEGER
        REFERENCES coding_agent_schema.conversations(id) ON DELETE CASCADE;
CREATE INDEX IF NOT EXISTS short_term_user_conv_idx
    ON coding_agent_schema.short_term_memory (user_id, conversation_id, created_at ASC);

-- 3. Messages remember which file the question was asked against.
ALTER TABLE coding_agent_schema.messages
    ADD COLUMN IF NOT EXISTS file_name TEXT;

-- 4. Examples become per-user. Rows already in these tables have no owner and
--    would otherwise leak across accounts, so they are dropped rather than
--    assigned to an arbitrary user.
DELETE FROM coding_agent_schema.golden_examples;
DELETE FROM coding_agent_schema.flagged_answers;

ALTER TABLE coding_agent_schema.golden_examples
    ADD COLUMN IF NOT EXISTS user_id INTEGER
        REFERENCES coding_agent_schema.users(id) ON DELETE CASCADE;
ALTER TABLE coding_agent_schema.flagged_answers
    ADD COLUMN IF NOT EXISTS user_id INTEGER
        REFERENCES coding_agent_schema.users(id) ON DELETE CASCADE;

ALTER TABLE coding_agent_schema.golden_examples  ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE coding_agent_schema.flagged_answers  ALTER COLUMN user_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS golden_examples_user_idx  ON coding_agent_schema.golden_examples (user_id);
CREATE INDEX IF NOT EXISTS flagged_answers_user_idx  ON coding_agent_schema.flagged_answers (user_id);

-- 5. Indexes the original schema was missing.
CREATE INDEX IF NOT EXISTS conversations_user_created_idx
    ON coding_agent_schema.conversations (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS messages_conversation_created_idx
    ON coding_agent_schema.messages (conversation_id, created_at ASC);
CREATE INDEX IF NOT EXISTS feedback_user_idx
    ON coding_agent_schema.feedback (user_id);

COMMIT;
