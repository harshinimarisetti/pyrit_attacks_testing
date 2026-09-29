```python
"""
garak_all_attacks.py

Standalone NVIDIA Garak vulnerability scanner for a local Ollama model.

What this script does
---------------------
1. Checks that Ollama is reachable.
2. Checks that the requested model exists.
3. Runs ALL Garak probes against the model.
4. Uses low concurrency to reduce connection pressure on local Ollama.
5. Captures the raw Garak JSONL report.
6. Parses the Garak report into a structured JSON summary.
7. Creates a CSV summary suitable for client reporting.
8. Does not require PyRIT.

Requirements
------------
pip install garak ollama

Ollama
------
Ollama must be running locally.

Example:
    ollama list

Run
---
python garak_all_attacks.py
"""

import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

OLLAMA_TAGS_URL = (
    f"{OLLAMA_BASE_URL}/api/tags"
)

# Change this to your client model.
MODEL_NAME = "llama3:8b-instruct-fp16"

# Output directory.
OUTPUT_DIR = Path("garak_results")

# Garak raw report prefix.
GARAK_REPORT_PREFIX = "garak_full_scan"

# Keep concurrency low for a local Ollama server.
PARALLEL_REQUESTS = 1
PARALLEL_ATTEMPTS = 1

# Number of model generations per probe prompt.
GENERATIONS = 1

# Retry settings for Ollama preflight.
OLLAMA_RETRIES = 5
OLLAMA_RETRY_DELAY = 2

# Garak command timeout is intentionally not imposed here.
# Large Garak runs can legitimately take a long time.


# =============================================================================
# UTILITY
# =============================================================================

def timestamp():
    """Return a filesystem-safe timestamp."""

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def safe_filename(value):
    """Convert a string into a filesystem-safe filename."""

    value = str(value)

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )

    return value.strip(
        "._"
    ) or "unknown"


# =============================================================================
# OLLAMA CONNECTIVITY
# =============================================================================

def check_ollama():
    """
    Check whether Ollama is reachable and whether MODEL_NAME exists.
    """

    print(
        "\n"
        + "=" * 90
    )

    print(
        "OLLAMA PRE-FLIGHT CHECK"
    )

    print(
        "=" * 90
    )

    print(
        f"Ollama URL : {OLLAMA_BASE_URL}"
    )

    print(
        f"Model      : {MODEL_NAME}"
    )

    last_error = None

    for attempt in range(
        1,
        OLLAMA_RETRIES + 1,
    ):

        try:
            request = Request(
                OLLAMA_TAGS_URL,
                headers={
                    "Accept":
                        "application/json",
                },
                method="GET",
            )

            with urlopen(
                request,
                timeout=10,
            ) as response:

                raw = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

                if response.status != 200:

                    raise RuntimeError(
                        f"HTTP {response.status}"
                    )

            payload = json.loads(
                raw
            )

            models = payload.get(
                "models",
                [],
            )

            installed_names = {
                item.get("name")
                for item in models
                if item.get("name")
            }

            print(
                f"[OK] Ollama reachable "
                f"(attempt {attempt}/{OLLAMA_RETRIES})"
            )

            print(
                f"[OK] {len(installed_names)} "
                "model(s) reported by Ollama."
            )

            if MODEL_NAME not in installed_names:

                print(
                    "\n[ERROR] Requested model was "
                    "not found in Ollama."
                )

                print(
                    f"        Requested: {MODEL_NAME}"
                )

                print(
                    "\nInstalled models:"
                )

                for model in sorted(
                    installed_names
                ):
                    print(
                        f"  - {model}"
                    )

                print(
                    "\nRun:"
                )

                print(
                    f"  ollama pull {MODEL_NAME}"
                )

                return False

            print(
                f"[OK] Target model found: "
                f"{MODEL_NAME}"
            )

            return True

        except (
            HTTPError,
            URLError,
            TimeoutError,
            ConnectionError,
            OSError,
            RuntimeError,
            json.JSONDecodeError,
        ) as exc:

            last_error = str(
                exc
            )

            print(
                f"[WARNING] Ollama check "
                f"{attempt}/{OLLAMA_RETRIES} failed: "
                f"{exc}"
            )

            if attempt < OLLAMA_RETRIES:
                time.sleep(
                    OLLAMA_RETRY_DELAY
                )

    print(
        "\n[ERROR] Ollama is not available."
    )

    print(
        f"[ERROR] Last error: {last_error}"
    )

    print(
        "\nStart Ollama and verify with:"
    )

    print(
        "  ollama list"
    )

    return False


# =============================================================================
# GARAK COMMAND
# =============================================================================

def build_garak_command():
    """
    Build the Garak command.

    IMPORTANT:
    No --spec / --probes argument is supplied because Garak's default
    behavior is to run all known probes.
    """

    command = [
        sys.executable,

        "-m",
        "garak",

        "--target_type",
        "ollama",

        "--target_name",
        MODEL_NAME,

        "--generations",
        str(
            GENERATIONS
        ),

        "--parallel_requests",
        str(
            PARALLEL_REQUESTS
        ),

        "--parallel_attempts",
        str(
            PARALLEL_ATTEMPTS
        ),

        "--report_prefix",
        GARAK_REPORT_PREFIX,
    ]

    return command


# =============================================================================
# FIND THE GARAK REPORT
# =============================================================================

def find_latest_garak_report(
    search_dir=Path("."),
):
    """
    Find the most recently modified Garak report JSONL file.
    """

    candidates = list(
        search_dir.glob(
            "*.report.jsonl"
        )
    )

    if not candidates:
        return None

    candidates.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return candidates[0]


# =============================================================================
# READ GARAK JSONL
# =============================================================================

def read_jsonl(
    report_path,
):
    """
    Read a JSONL file safely.

    Returns a list of JSON objects.
    Malformed lines are ignored and returned separately.
    """

    entries = []
    malformed_lines = []

    with open(
        report_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                entries.append(
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError:

                malformed_lines.append(
                    line_number
                )

    return (
        entries,
        malformed_lines,
    )


# =============================================================================
# EXTRACT STRUCTURED RESULTS
# =============================================================================

def extract_results(
    entries,
):
    """
    Extract probe summaries and evaluated attempts from Garak's JSONL.

    The exact report structure can evolve between Garak releases, so this
    parser intentionally preserves unknown fields and extracts what is
    available.
    """

    probe_summaries = []

    evaluated_attempts = []

    hits = []

    run_metadata = {}

    digest = {}

    for entry in entries:

        entry_type = entry.get(
            "entry_type"
        )

        # ---------------------------------------------------------------------
        # Run metadata
        # ---------------------------------------------------------------------
        if entry_type == "start_run setup":

            run_metadata = {
                key: value
                for key, value in entry.items()
                if key not in {
                    "entry_type",
                }
            }

        elif entry_type == "init":

            run_metadata.update(
                {
                    key: value
                    for key, value in entry.items()
                    if key not in {
                        "entry_type",
                    }
                }
            )

        # ---------------------------------------------------------------------
        # Probe summaries
        # ---------------------------------------------------------------------
        elif entry_type == "probe_summary":

            probe_summaries.append(
                entry
            )

        # ---------------------------------------------------------------------
        # Evaluations
        # ---------------------------------------------------------------------
        elif entry_type == "eval":

            evaluated_attempts.append(
                entry
            )

            # Try multiple common representations of detector hits.
            if (
                entry.get("passed") is False
                or entry.get("hit") is True
                or entry.get("status") == "fail"
            ):
                hits.append(
                    entry
                )

        # ---------------------------------------------------------------------
        # Digest
        # ---------------------------------------------------------------------
        elif entry_type == "digest":

            digest = entry

    return {
        "run_metadata":
            run_metadata,

        "probe_summaries":
            probe_summaries,

        "evaluations":
            evaluated_attempts,

        "hits":
            hits,

        "digest":
            digest,
    }


# =============================================================================
# PROBE SUMMARY NORMALIZATION
# =============================================================================

def normalize_probe_summaries(
    probe_summaries,
):
    """
    Convert Garak probe-summary entries into a consistent schema.

    We preserve the complete original object under 'raw' so that no
    Garak information is thrown away.
    """

    normalized = []

    for item in probe_summaries:

        row = {
            "probe":
                item.get(
                    "probe",
                    item.get(
                        "name",
                        "unknown",
                    ),
                ),

            "detector":
                item.get(
                    "detector",
                    "",
                ),

            "passed":
                item.get(
                    "passed",
                ),

            "failed":
                item.get(
                    "failed",
                ),

            "total":
                item.get(
                    "total",
                ),

            "pass_rate":
                item.get(
                    "pass_rate",
                ),

            "failure_rate":
                item.get(
                    "failure_rate",
                ),

            "raw":
                item,
        }

        normalized.append(
            row
        )

    return normalized


# =============================================================================
# HIT EXTRACTION
# =============================================================================

def normalize_hits(
    hit_entries,
):
    """
    Create a compact finding list.

    The original Garak evaluation entry is retained under 'raw'.
    """

    findings = []

    for index, item in enumerate(
        hit_entries,
        start=1,
    ):

        findings.append(
            {
                "finding_id":
                    f"GARAK-{index:04d}",

                "probe":
                    item.get(
                        "probe",
                        item.get(
                            "probe_name",
                            "unknown",
                        ),
                    ),

                "detector":
                    item.get(
                        "detector",
                        item.get(
                            "detector_name",
                            "unknown",
                        ),
                    ),

                "status":
                    item.get(
                        "status",
                    ),

                "passed":
                    item.get(
                        "passed",
                    ),

                "hit":
                    item.get(
                        "hit",
                    ),

                "prompt":
                    item.get(
                        "prompt",
                        item.get(
                            "prompt_text",
                            "",
                        ),
                    ),

                "response":
                    item.get(
                        "response",
                        item.get(
                            "output",
                            "",
                        ),
                    ),

                "raw":
                    item,
            }
        )

    return findings


# =============================================================================
# CREATE STRUCTURED JSON
# =============================================================================

def write_structured_json(
    source_report,
    parsed_data,
    output_path,
):
    """
    Write a client-friendly structured JSON report.
    """

    probe_summary = normalize_probe_summaries(
        parsed_data[
            "probe_summaries"
        ]
    )

    findings = normalize_hits(
        parsed_data[
            "hits"
        ]
    )

    structured = {
        "report_metadata": {
            "scanner":
                "NVIDIA Garak",

            "target_type":
                "ollama",

            "model":
                MODEL_NAME,

            "ollama_endpoint":
                OLLAMA_BASE_URL,

            "scan_started":
                parsed_data[
                    "run_metadata"
                ].get(
                    "start_time",
                ),

            "generated_at":
                datetime.now().isoformat(),

            "source_report":
                str(
                    source_report
                ),
        },

        "configuration": {
            "all_probes":
                True,

            "generations":
                GENERATIONS,

            "parallel_requests":
                PARALLEL_REQUESTS,

            "parallel_attempts":
                PARALLEL_ATTEMPTS,
        },

        "results": {
            "probe_count":
                len(
                    probe_summary
                ),

            "finding_count":
                len(
                    findings
                ),

            "probe_summaries":
                probe_summary,

            "findings":
                findings,

            "digest":
                parsed_data[
                    "digest"
                ],
        },

        "raw_report_statistics": {
            "jsonl_entries":
                len(
                    parsed_data.get(
                        "evaluations",
                        [],
                    )
                ),

            "evaluated_attempts":
                len(
                    parsed_data[
                        "evaluations"
                    ]
                ),

            "detected_hits":
                len(
                    findings
                ),
        },
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            structured,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return structured


# =============================================================================
# CREATE CSV
# =============================================================================

def write_csv(
    structured,
    output_path,
):
    """
    Create a flat probe-level CSV.

    This is useful for importing into Excel or a client dashboard.
    """

    rows = (
        structured[
            "results"
        ][
            "probe_summaries"
        ]
    )

    fieldnames = [
        "probe",
        "detector",
        "passed",
        "failed",
        "total",
        "pass_rate",
        "failure_rate",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    key:
                        row.get(
                            key,
                            "",
                        )
                    for key in fieldnames
                }
            )


# =============================================================================
# CREATE FINDINGS CSV
# =============================================================================

def write_findings_csv(
    structured,
    output_path,
):
    """
    Create a finding-level CSV with prompt and response evidence.
    """

    rows = (
        structured[
            "results"
        ][
            "findings"
        ]
    )

    fieldnames = [
        "finding_id",
        "probe",
        "detector",
        "status",
        "passed",
        "hit",
        "prompt",
        "response",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    key:
                        row.get(
                            key,
                            "",
                        )
                    for key in fieldnames
                }
            )


# =============================================================================
# PRINT CONSOLE SUMMARY
# =============================================================================

def print_summary(
    structured,
):
    """
    Print a human-readable summary.
    """

    results = structured[
        "results"
    ]

    print(
        "\n"
        + "=" * 90
    )

    print(
        "GARAK SCAN SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        f"Target model    : "
        f"{structured['report_metadata']['model']}"
    )

    print(
        f"Target type     : "
        f"{structured['report_metadata']['target_type']}"
    )

    print(
        f"Probes detected : "
        f"{results['probe_count']}"
    )

    print(
        f"Findings        : "
        f"{results['finding_count']}"
    )

    print(
        "=" * 90
    )

    print(
        "\nPROBE RESULTS"
    )

    print(
        "-" * 90
    )

    if not results[
        "probe_summaries"
    ]:

        print(
            "No probe_summary entries "
            "were found in the report."
        )

    else:

        for row in results[
            "probe_summaries"
        ]:

            print(
                f"Probe: {row['probe']}"
            )

            if row["detector"]:

                print(
                    f"  Detector     : "
                    f"{row['detector']}"
                )

            print(
                f"  Passed       : "
                f"{row['passed']}"
            )

            print(
                f"  Failed       : "
                f"{row['failed']}"
            )

            print(
                f"  Total        : "
                f"{row['total']}"
            )

            print(
                f"  Pass rate    : "
                f"{row['pass_rate']}"
            )

            print(
                f"  Failure rate : "
                f"{row['failure_rate']}"
            )

            print()

    print(
        "\nFINDINGS"
    )

    print(
        "-" * 90
    )

    if not results[
        "findings"
    ]:

        print(
            "No detected findings were extracted."
        )

    else:

        for finding in results[
            "findings"
        ]:

            print(
                f"{finding['finding_id']} | "
                f"Probe={finding['probe']} | "
                f"Detector={finding['detector']}"
            )

            prompt = finding.get(
                "prompt",
                "",
            )

            response = finding.get(
                "response",
                "",
            )

            if prompt:

                print(
                    "  Prompt: "
                    + prompt[:300]
                )

            if response:

                print(
                    "  Response: "
                    + response[:500]
                )

            print()


# =============================================================================
# RUN GARAK
# =============================================================================

def run_garak():
    """
    Run Garak and return the generated report path.
    """

    command = (
        build_garak_command()
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "GARAK ALL-PROBE SCAN"
    )

    print(
        "=" * 90
    )

    print(
        f"Target model       : {MODEL_NAME}"
    )

    print(
        "Probe selection     : ALL"
    )

    print(
        f"Generations/prompt : {GENERATIONS}"
    )

    print(
        f"Parallel requests  : {PARALLEL_REQUESTS}"
    )

    print(
        f"Parallel attempts  : {PARALLEL_ATTEMPTS}"
    )

    print(
        "\nCommand:"
    )

    print(
        " ".join(command)
    )

    print(
        "\nStarting Garak..."
    )

    start_time = time.time()

    try:

        process = subprocess.Popen(
            command,

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            bufsize=1,
        )

    except FileNotFoundError as exc:

        print(
            "\n[ERROR] Could not start Garak."
        )

        print(
            "Install with:"
        )

        print(
            "  pip install garak ollama"
        )

        raise exc

    if process.stdout is not None:

        for line in process.stdout:

            print(
                f"[GARAK] "
                f"{line.rstrip()}"
            )

    return_code = (
        process.wait()
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "GARAK PROCESS FINISHED"
    )

    print(
        "=" * 90
    )

    print(
        f"Exit code : {return_code}"
    )

    print(
        f"Duration  : {elapsed:.2f} seconds"
    )

    if return_code != 0:

        print(
            "\n[WARNING] Garak exited "
            "with a non-zero code."
        )

        print(
            "Check garak.log and the generated "
            "JSONL report for details."
        )

    return return_code


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        + "=" * 90
    )

    print(
        "NVIDIA GARAK - STANDALONE LLM SECURITY SCANNER"
    )

    print(
        "=" * 90
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        "Mode : ALL GARAK PROBES"
    )

    # -------------------------------------------------------------------------
    # Create output directory.
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Pre-flight.
    # -------------------------------------------------------------------------

    if not check_ollama():

        print(
            "\n[STOP] Ollama pre-flight failed."
        )

        print(
            "[STOP] Garak scan will not start."
        )

        sys.exit(
            1
        )

    # -------------------------------------------------------------------------
    # Run Garak.
    # -------------------------------------------------------------------------

    exit_code = run_garak()

    # -------------------------------------------------------------------------
    # Locate generated report.
    # -------------------------------------------------------------------------

    report_path = (
        find_latest_garak_report(
            Path(".")
        )
    )

    if report_path is None:

        print(
            "\n[ERROR] Garak completed but "
            "no *.report.jsonl file was found."
        )

        print(
            "Look at garak.log for details."
        )

        sys.exit(
            exit_code
            if exit_code != 0
            else 2
        )

    print(
        "\n[OK] Raw Garak report:"
    )

    print(
        f"     {report_path.resolve()}"
    )

    # -------------------------------------------------------------------------
    # Parse report.
    # -------------------------------------------------------------------------

    entries, malformed_lines = (
        read_jsonl(
            report_path
        )
    )

    print(
        f"\n[OK] Parsed {len(entries)} "
        "JSONL entries."
    )

    if malformed_lines:

        print(
            "[WARNING] Malformed JSONL "
            f"lines: {malformed_lines}"
        )

    parsed_data = extract_results(
        entries
    )

    # -------------------------------------------------------------------------
    # Create client-oriented filenames.
    # -------------------------------------------------------------------------

    run_id = (
        f"{timestamp()}_"
        f"{safe_filename(MODEL_NAME)}"
    )

    structured_json_path = (
        OUTPUT_DIR
        / f"{run_id}_structured.json"
    )

    probe_csv_path = (
        OUTPUT_DIR
        / f"{run_id}_probe_summary.csv"
    )

    findings_csv_path = (
        OUTPUT_DIR
        / f"{run_id}_findings.csv"
    )

    # -------------------------------------------------------------------------
    # Structured JSON.
    # -------------------------------------------------------------------------

    structured = (
        write_structured_json(
            source_report=
                report_path,

            parsed_data=
                parsed_data,

            output_path=
                structured_json_path,
        )
    )

    # -------------------------------------------------------------------------
    # CSV outputs.
    # -------------------------------------------------------------------------

    write_csv(
        structured,
        probe_csv_path,
    )

    write_findings_csv(
        structured,
        findings_csv_path,
    )

    # -------------------------------------------------------------------------
    # Print summary.
    # -------------------------------------------------------------------------

    print_summary(
        structured
    )

    # -------------------------------------------------------------------------
    # Final output paths.
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 90
    )

    print(
        f"Raw Garak JSONL : "
        f"{report_path.resolve()}"
    )

    print(
        f"Structured JSON : "
        f"{structured_json_path.resolve()}"
    )

    print(
        f"Probe CSV       : "
        f"{probe_csv_path.resolve()}"
    )

    print(
        f"Findings CSV    : "
        f"{findings_csv_path.resolve()}"
    )

    print(
        "\nGarak log:"
    )

    print(
        f"{Path('garak.log').resolve()}"
    )

    print(
        "\n"
        + "=" * 90
    )

    if exit_code == 0:

        print(
            "GARAK SCAN COMPLETED"
        )

    else:

        print(
            "GARAK SCAN COMPLETED WITH ERRORS"
        )

    print(
        "=" * 90
    )

    # Return Garak's exit code so CI/CD can consume it.
    sys.exit(
        exit_code
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
```
