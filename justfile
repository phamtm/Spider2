set dotenv-load := true

default:
    @just --list

# Run one Spider 2.0 Snow question by name, using the matching gold SQL file.
run question:
    @set -eu; \
    src="spider2-snow/evaluation_suite/gold/sql/{{question}}.sql"; \
    if [ ! -f "$src" ]; then \
        echo "Unknown question: {{question}}"; \
        exit 1; \
    fi; \
    tmp_dir="$(mktemp -d)"; \
    trap 'rm -rf "$tmp_dir"' EXIT INT TERM; \
    cp -f "$src" "$tmp_dir/{{question}}.sql"; \
    cd spider2-snow/evaluation_suite && \
    ../../.venv/bin/python evaluate.py --mode sql --result_dir "$tmp_dir" --gold_dir gold --max_workers 1 --timeout 60
