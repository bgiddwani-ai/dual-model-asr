"""Check that the published evidence supports its summaries and corpus claims."""
import csv
import hashlib
import json
import statistics
import unittest
from collections import defaultdict
from pathlib import Path

from scripts.project import ROOT


class RecordedResults(unittest.TestCase):
    def test_all_225_trial_ratios_and_45_medians(self):
        directory = ROOT / "benchmarks/results"
        with (directory / "trials.csv").open() as handle:
            trials = list(csv.DictReader(handle))
        with (directory / "summary.csv").open() as handle:
            summaries = list(csv.DictReader(handle))
        self.assertEqual((len(trials), len(summaries)), (225, 45))
        grouped = defaultdict(list)
        for row in trials:
            self.assertAlmostEqual(float(row["rtfx"]), float(row["audio_seconds"]) / float(row["inference_seconds"]), places=7)
            grouped[row["configuration"]].append(row)
        self.assertEqual(set(grouped), {row["configuration"] for row in summaries})
        rejected = 0
        for summary in summaries:
            group = grouped[summary["configuration"]]
            self.assertEqual({row["trial"] for row in group}, {"1", "2", "3", "4", "5"})
            for field in ["rtfx", "inference_seconds", "wer_pct", "cer_pct", "empty_outputs", "audio_seconds"]:
                self.assertAlmostEqual(float(summary[field]), statistics.median(float(row[field]) for row in group), places=7)
            if summary["quality_status"] == "rejected_empty_outputs":
                rejected += 1
                self.assertEqual(float(summary["wer_pct"]), 100)
                self.assertEqual(float(summary["empty_outputs"]), 256)
        self.assertEqual(rejected, 7)

    def test_frozen_corpora_have_portable_unique_recordings_and_expected_duration(self):
        for name, count, seconds in [("librispeech.jsonl", 2528, 17252.786), ("fleurs-hi.jsonl", 256, 2782.5)]:
            rows = [json.loads(line) for line in (ROOT / "benchmarks" / name).read_text().splitlines()]
            paths = [Path(row["audio_filepath"]) for row in rows]
            self.assertEqual(len(rows), count)
            self.assertEqual(len(set(paths)), count)
            self.assertTrue(all(not path.is_absolute() and ".." not in path.parts for path in paths))
            self.assertEqual([r["duration"] for r in rows], sorted(r["duration"] for r in rows))
            self.assertAlmostEqual(sum(r["duration"] for r in rows), seconds, places=5)
            self.assertTrue(all(row["text"].strip() for row in rows))
        expected = json.loads((ROOT / "benchmarks/manifest-sha256.json").read_text())
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / "benchmarks" / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__": unittest.main()
