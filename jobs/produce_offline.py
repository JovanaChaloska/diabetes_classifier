"""Kafka producer for the online / streaming phase.

Streams data/offline.csv row by row as JSON messages into the Kafka
topic health_data. The diabetes label column is removed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from typing import Dict, Iterator

# Make ``jobs`` importable when this file is executed directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kafka import KafkaProducer  # noqa: E402

from jobs.config import (  # noqa: E402
    ALL_FEATURES,
    INPUT_TOPIC,
    KAFKA_BOOTSTRAP_HOST,
    LABEL_COL,
    OFFLINE_DATA_PATH,
    PRODUCER_DELAY_SECONDS,
)


def iter_csv_rows(path: str) -> Iterator[Dict[str, float]]:
    """Yield one dict per CSV row, keeping only the feature columns.

    Values are coerced to ``float`` so the downstream JSON is numeric (the
    Spark side casts to ``DoubleType`` either way, but sending numbers
    instead of strings avoids surprises when inspecting Kafka messages).
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in ALL_FEATURES if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV at {path} is missing expected feature columns: {missing}"
            )
        for row in reader:
            payload = {}
            for col in ALL_FEATURES:
                raw = row[col]
                if raw is None or raw == "":
                    payload[col] = None
                else:
                    payload[col] = float(raw)
            # Explicitly drop the label, it must never be sent.
            row.pop(LABEL_COL, None)
            yield payload


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--bootstrap-servers",
        default=KAFKA_BOOTSTRAP_HOST,
        help=f"Kafka bootstrap servers (default: {KAFKA_BOOTSTRAP_HOST!r}).",
    )
    parser.add_argument(
        "--topic",
        default=INPUT_TOPIC,
        help=f"Destination topic (default: {INPUT_TOPIC!r}).",
    )
    parser.add_argument(
        "--csv",
        default=OFFLINE_DATA_PATH,
        help=f"Path to the CSV to stream (default: {OFFLINE_DATA_PATH!r}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=PRODUCER_DELAY_SECONDS,
        help=(
            "Seconds to wait between messages "
            f"(default: {PRODUCER_DELAY_SECONDS})."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of messages to send (for smoke tests).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print(
        f"Producing rows from {args.csv} -> topic {args.topic!r} "
        f"on {args.bootstrap_servers} (delay={args.delay}s)",
        flush=True,
    )

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        security_protocol="PLAINTEXT",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=20,
        acks="all",
    )

    # Ctrl-C handling
    stopping = {"flag": False}

    def _handle_sigint(signum, frame):
        stopping["flag"] = True
        print("\nInterrupt received, flushing and closing producer...", flush=True)

    signal.signal(signal.SIGINT, _handle_sigint)

    sent = 0
    start = time.time()
    try:
        for payload in iter_csv_rows(args.csv):
            if stopping["flag"]:
                break
            producer.send(args.topic, value=payload)
            sent += 1
            if sent % 1000 == 0:
                rate = sent / (time.time() - start)
                print(f"  sent {sent:,} messages ({rate:.0f}/s)", flush=True)
            if args.limit is not None and sent >= args.limit:
                break
            if args.delay > 0:
                time.sleep(args.delay)
    finally:
        producer.flush()
        producer.close()

    elapsed = time.time() - start
    print(
        f"Done. {sent:,} messages sent in {elapsed:.1f}s "
        f"({sent / elapsed:.0f} msg/s).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
