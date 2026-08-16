# Runbook: course_id migration against production (Supabase)

**Status: not yet run.** This document is delivered alongside a series of
local commits that have not been pushed. Nothing here has touched Supabase.

## What changed since the previous version of this runbook

The migration is no longer reachable from the learner-facing request path
at all. Previously, `build_controller()` called `initialize_schema()`,
which would silently perform the legacy course_id migration on whichever
learner request happened to trigger it first. That is no longer possible:

- `SQLiteBKTRepository.initialize_schema()` (called by `build_controller()`
  on every course selection/login) is now **runtime-safe only**: it creates
  the schema when the database is genuinely empty, verifies the schema
  version when one already exists, and **raises**
  `SchemaMigrationRequiredError` -- without changing anything -- when it
  finds a legacy pre-course_id database. `app/bootstrap.py` translates that
  into a `BootstrapError` with a maintenance-style message; `app/main.py`
  and `app/ui/course_selector.py` catch it and show that message instead of
  crashing.
- The legacy-data rebuild itself now lives in exactly one place:
  `bkt.sqlite_repository.run_course_ownership_migration()`. It is never
  called by application runtime code -- only by the protected CLI below.
- A `schema_migrations` table records the applied schema version, so
  readiness is determined by reading that value back, never by inferring
  from table shape or catching a generic database error.
- `QUIZ_MAINTENANCE_MODE` (default `false`) is a new setting: when enabled,
  `app/main.py` shows a maintenance message and returns before any learner
  login/course-selection/submission code runs, while still allowing the
  settings/catalogue-load step (basic startup diagnostics) to execute.

**Entry point (protected, explicit, CLI-only):**

```bash
python -m scripts.migrate_course_ownership --database "$QUIZ_DATABASE_URL"
python -m scripts.migrate_course_ownership --database "$QUIZ_DATABASE_URL" --confirm
```

Without `--confirm` the command only reports the current schema status
(read-only) and exits -- it never mutates anything. `--confirm` is required
to actually run the migration. This is the *only* supported way to run
`run_course_ownership_migration()` against a real database.

## What the migration does

`run_course_ownership_migration()` backfills every existing row in
`attempt_events`, `mastery_snapshots`, `recommendation_events`,
`content_gap_events`, `bkt_model_metadata`, `learner_sessions`, and
`question_presentations` with `course_id="intro-ai"`, rebuilds each
table's constraints (composite PKs on `learner_sessions`/`bkt_model_metadata`,
composite FKs restored on `mastery_snapshots`→`attempt_events` and
`question_presentations`→`learner_sessions`), and records
`course_id_v1` in `schema_migrations`. It is idempotent -- a second run
against an already-current database is a no-op (`{}` returned, nothing
touched).

## Failure safety

- **Postgres and SQLite alike**: the whole migration runs inside one
  transaction. If anything fails partway -- including an FK violation
  caught during `INSERT ... SELECT` on a rebuilt child table, or any other
  error -- the transaction rolls back automatically. A failed migration
  attempt leaves the database exactly as it was; there is no partial-
  migration state to clean up. Verified directly: `tests/
  test_schema_migration_safety.py::test_migration_failure_rolls_back_completely`
  forces a failure on the last statement inside the migration transaction
  and asserts every row is byte-for-byte unchanged afterward.
- **Rollback if you discover a problem *after* the migration commits
  successfully** (e.g., an application-level issue found later): there is
  no built-in "undo." The table rebuild drops the original table and
  renames a new one into place, so reversing it means restoring from the
  backup taken in Step 1 below, not re-running application code.

---

## Deployment order

### Step 1 — Backup

Take a fresh Supabase backup immediately before starting. Options, in order
of preference:

1. **Supabase dashboard**: Database → Backups → trigger an on-demand backup,
   or confirm point-in-time recovery (PITR) is enabled and note the current
   timestamp as your recovery point.
2. **Manual `pg_dump`** (belt-and-suspenders, gives you a portable local
   copy):
   ```bash
   pg_dump "$QUIZ_DATABASE_URL" \
     --format=custom \
     --file="backup-pre-course-id-migration-$(date +%Y%m%dT%H%M%S).dump" \
     --table=attempt_events --table=mastery_snapshots \
     --table=recommendation_events --table=content_gap_events \
     --table=bkt_model_metadata --table=learner_sessions \
     --table=question_presentations
   ```
   Store this somewhere outside the deploy environment (it contains learner
   data).

Do not proceed until you have a backup or a confirmed PITR recovery point.

### Step 2 — Record pre-migration row counts and schema version

```bash
psql "$QUIZ_DATABASE_URL" -c "
SELECT 'attempt_events' AS table_name, COUNT(*) FROM attempt_events
UNION ALL SELECT 'mastery_snapshots', COUNT(*) FROM mastery_snapshots
UNION ALL SELECT 'recommendation_events', COUNT(*) FROM recommendation_events
UNION ALL SELECT 'content_gap_events', COUNT(*) FROM content_gap_events
UNION ALL SELECT 'bkt_model_metadata', COUNT(*) FROM bkt_model_metadata
UNION ALL SELECT 'learner_sessions', COUNT(*) FROM learner_sessions
UNION ALL SELECT 'question_presentations', COUNT(*) FROM question_presentations
ORDER BY table_name;
"
python -m scripts.migrate_course_ownership --database "$QUIZ_DATABASE_URL"
```

Save both outputs -- the row counts are compared again in Step 6, and the
schema-status line (`empty` / `current` / `migration_required`) confirms
you are about to migrate a genuinely legacy database, not something
already current.

### Step 3 — Enable maintenance mode

Set `QUIZ_MAINTENANCE_MODE=true` in the deployment environment (alongside
existing settings) and restart/redeploy the process. `app/main.py` will
show a maintenance message and refuse all learner logins, course
selections, and submissions, while the settings/catalogue-load step still
runs so you can confirm the process starts cleanly.

### Step 4 — Deploy the migration-aware runtime

Deploy the code containing this change (the redesigned
`initialize_schema()`, `run_course_ownership_migration()`, the
`SchemaMigrationRequiredError` handling in `app/bootstrap.py`, and the
`QUIZ_MAINTENANCE_MODE` gate in `app/main.py`) if it is not already live.
With maintenance mode on, this is safe: the new runtime-safe
`initialize_schema()` will detect the legacy schema and raise
`SchemaMigrationRequiredError` rather than migrate anything, and
maintenance mode independently blocks the learner path regardless.

### Step 5 — Run the explicit migration

From a machine with the deployed code and network access to Supabase
(**not** inside a learner's request path):

```bash
python -m scripts.migrate_course_ownership --database "$QUIZ_DATABASE_URL" --confirm
```

Expected output: `schema status: migration_required`, then `migration
succeeded: {...}` with one entry per table that had rows to migrate (e.g.
`{'attempt_events': 1842, 'mastery_snapshots': 1842, ...}`).

If this fails, **stop** -- the transaction rolled back automatically (see
Failure safety above), so production is unchanged. Read the error, fix the
cause, and re-run this step. Do not proceed to Step 8 (disabling
maintenance mode) until this step completes cleanly.

### Step 6 — Validate: foreign keys, schema version, counts, course_id backfill

```bash
python -m scripts.migrate_course_ownership --database "$QUIZ_DATABASE_URL"
```
Expect `schema status: current`.

```bash
psql "$QUIZ_DATABASE_URL" -c "
SELECT conname, conrelid::regclass AS table_name
FROM pg_constraint
WHERE contype = 'f'
  AND conrelid::regclass::text IN ('mastery_snapshots', 'question_presentations')
ORDER BY table_name;
"
```
Expect two rows (one composite, course-aware FK constraint per table).

```bash
psql "$QUIZ_DATABASE_URL" -c "
SELECT COUNT(*) AS orphan_mastery_snapshots FROM mastery_snapshots ms
WHERE NOT EXISTS (
  SELECT 1 FROM attempt_events ae
  WHERE ae.course_id = ms.course_id AND ae.attempt_id = ms.source_attempt_id
);
SELECT COUNT(*) AS orphan_question_presentations FROM question_presentations qp
WHERE NOT EXISTS (
  SELECT 1 FROM learner_sessions ls
  WHERE ls.learner_id = qp.learner_id AND ls.course_id = qp.course_id
);
"
```
Both counts must be `0`.

Re-run the exact row-count query from Step 2. Every count must match
exactly (the migration only adds a `course_id` column and rebuilds
constraints -- it never adds, removes, or duplicates rows). Then confirm
the backfill:

```bash
psql "$QUIZ_DATABASE_URL" -c "
SELECT DISTINCT course_id FROM attempt_events
UNION SELECT DISTINCT course_id FROM mastery_snapshots
UNION SELECT DISTINCT course_id FROM learner_sessions;
"
```
Expect exactly one value: `intro-ai`.

### Step 7 — Smoke test intro-ai

Still with `QUIZ_MAINTENANCE_MODE=true` set, run the smoke test against a
staging pointer at the now-migrated database (or a temporary override in a
non-traffic-serving process), or briefly flip maintenance mode off for a
single canary check before the full Step 8 rollout:

1. Load the app. Confirm it starts without error.
2. Log in with a **real pre-existing learner ID** (one that has history).
   Select "AI" at the course selector.
3. Confirm prior progress is visible and correct (`get_progress` reflects
   the learner's actual pre-migration mastery -- this is the strongest
   end-to-end proof that Step 5 preserved their data).
4. Answer a new question. Confirm it saves (no FK rejection) and mastery
   updates.
5. Confirm the admin sidebar stays hidden (`QUIZ_ADMIN_STATUS_ENABLED` is
   unset/false in production config) -- not migration-related, but worth
   reconfirming as part of the same deploy.

### Step 8 — Disable maintenance mode

Set `QUIZ_MAINTENANCE_MODE=false` (or unset it) and restart/redeploy. The
now-migrated database means every learner's next course selection resolves
`SchemaStatus.CURRENT` immediately -- no further schema work happens on
their request.

### Step 9 — Monitor logs and latency

Watch application logs for `SchemaMigrationRequiredError` or `BootstrapError`
(would indicate a process talking to a database that didn't get migrated --
e.g. a stale connection string) and watch warm login/submit/mastery-update/
next-question latency for the first period of real traffic after reopening.

---

## Rollback

If a problem surfaces **before** Step 5 completes: nothing to roll back --
maintenance mode has blocked all learner writes, and the transaction never
committed.

If a problem surfaces **after** Step 5 commits successfully but something
is still wrong (an application bug unrelated to the migration itself,
discovered later):

1. Re-enable `QUIZ_MAINTENANCE_MODE=true` to stop learner writes.
2. Restore from the Step 1 backup:
   - Supabase PITR: restore to the timestamp noted in Step 1.
   - Manual dump: `pg_restore --clean --if-exists -d "$QUIZ_DATABASE_URL" backup-....dump`
     (only restores the 7 tables backed up in Step 1 -- nothing else in the
     database is touched; note this also reverts `schema_migrations` to
     whatever state it was in before Step 5, if it existed at all).
3. Re-deploy the previous application version (pre this feature) against
   the restored database, which is back in its pre-migration shape.
4. Investigate offline against a copy before attempting the migration again.

There is no partial/incremental rollback path within the new schema itself --
restoring from backup is the only supported way back to the pre-migration
state once Step 5 has committed.
