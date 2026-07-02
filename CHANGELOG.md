# Changelog

## [0.1.1] - 2026-07-02

### Fixed
- **CLI:** `openeval report` now exits with a non-zero status code (exit code 1) when encountering corrupted or malformed evaluation result JSON files. Previously, it silently logged a warning to standard error and exited with 0, which could mask data loss and cause false-positive passes in CI/CD environments.
