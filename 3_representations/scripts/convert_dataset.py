"""
Convert existing analysis_results JSON to the new little_steer JSONL format.

Usage:
    cd Reasoning_behaviours/
    uv run python little_steer/scripts/convert_dataset.py \\
        --input data/analysis_results_detailed_harmbench_and_strong_reject_gemini2.0.json \\
        --output data/little_steer_dataset.jsonl \\
        --verbose
"""

import argparse
from pathlib import Path

import sys
from little_steer.data.converter import convert_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert analysis_results JSON to little_steer JSONL format."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input JSON file (analysis_results_*.json)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to the output JSONL file"
    )
    parser.add_argument(
        "--include-bad", action="store_true",
        help="Include entries where correctly_extracted=False (default: skip them)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    entries = convert_file(
        input_path=args.input,
        output_path=args.output,
        skip_incorrectly_extracted=not args.include_bad,
        verbose=args.verbose or True,
    )

    print(f"\n📊 Summary:")
    print(f"   Total entries converted: {len(entries)}")

    # Count unique labels
    all_labels: dict[str, int] = {}
    for entry in entries:
        for ann in entry.annotations:
            for lbl in ann.labels:
                all_labels[lbl] = all_labels.get(lbl, 0) + 1

    print(f"   Unique labels: {len(all_labels)}")
    print(f"   Top labels by frequency:")
    for lbl, count in sorted(all_labels.items(), key=lambda x: -x[1])[:10]:
        print(f"     {lbl}: {count}")


if __name__ == "__main__":
    main()
