import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image


class DatasetDownloadTests(unittest.TestCase):
    def setUp(self):
        import main_down

        self.downloader = main_down
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.summary = self.root / "image_summary.json"
        self.dataset = self.root / "dataset.json"
        image_buffer = io.BytesIO()
        Image.new("RGB", (24, 24), "blue").save(image_buffer, format="PNG")
        self.image_bytes = image_buffer.getvalue()
        self.requests = []
        requests = self.requests
        image_bytes = self.image_bytes

        class ImageHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"invalid image" if self.path == "/bad" else image_bytes)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ImageHandler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        settings = patch.multiple(
            self.downloader, DOWNLOAD_IMAGES_DIR=self.images,
            IMAGE_SUMMARY_PATH=self.summary, DOWNLOAD_RETRIES=1, REQUEST_TIMEOUT=1,
        )
        settings.start()
        self.addCleanup(settings.stop)

    def write_dataset(self, records):
        self.dataset.write_text(json.dumps(records), encoding="utf-8")

    def record(self, path, label="flood"):
        return {"url": self.url + path, "incidents": {label: 1}}

    def image_path(self, key):
        return self.images / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".png")

    def run_download(self, limit):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.downloader.main([
                "--json-path", str(self.dataset), "--limit", str(limit),
            ])

    def read_summary(self):
        return json.loads(self.summary.read_text(encoding="utf-8"))

    def test_limit_counts_readable_disaster_images_and_skips_bad_downloads(self):
        self.write_dataset({
            "traffic": self.record("/traffic", "traffic jam"),
            "broken": self.record("/bad"),
            "one": self.record("/one"),
            "two": self.record("/two"),
            "spare": self.record("/spare"),
        })
        self.assertEqual(self.run_download(2), 0)
        self.assertEqual(self.requests, ["/bad", "/one", "/two"])
        self.assertEqual(set(self.read_summary()), {"one", "two"})
        for key in ("one", "two"):
            with Image.open(self.image_path(key)) as image:
                self.assertEqual(image.mode, "RGB")
                image.verify()

    def test_downloads_original_dataset_disaster_labels(self):
        self.write_dataset({
            "one": self.record("/one", "flooded"),
            "two": self.record("/two", "on fire"),
        })
        self.assertEqual(self.run_download(2), 0)
        self.assertEqual(set(self.read_summary()), {"one", "two"})

    def test_rerun_reuses_images_and_extending_target_downloads_only_missing_images(self):
        self.write_dataset({str(i): self.record(f"/{i}") for i in range(3)})
        self.assertEqual(self.run_download(1), 0)
        self.assertEqual(self.run_download(1), 0)
        self.assertEqual(self.run_download(3), 0)
        self.assertEqual(self.requests, ["/0", "/1", "/2"])
        self.assertEqual(len(self.read_summary()), 3)

    def test_smaller_target_preserves_previously_downloaded_records(self):
        self.write_dataset({str(i): self.record(f"/{i}") for i in range(3)})
        self.run_download(3)
        self.assertEqual(self.run_download(1), 0)
        self.assertEqual(len(self.read_summary()), 3)
        self.assertEqual(len(self.requests), 3)

    def test_recovers_saved_image_without_summary_after_interruption(self):
        self.write_dataset({"one": self.record("/one")})
        self.image_path("one").write_bytes(self.image_bytes)
        self.assertEqual(self.run_download(1), 0)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.read_summary()["one"]["downloaded_file"], self.image_path("one").name)

    def test_redownloads_corrupt_cached_image(self):
        self.write_dataset({"one": self.record("/one")})
        self.image_path("one").write_bytes(b"truncated")
        self.assertEqual(self.run_download(1), 0)
        self.assertEqual(self.requests, ["/one"])

    def test_interruption_saves_completed_record_for_resume(self):
        self.write_dataset({"one": self.record("/one"), "two": self.record("/two")})
        with patch.object(self.downloader, "download_image", side_effect=[self.image_bytes, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                self.run_download(2)
        self.assertEqual(set(self.read_summary()), {"one"})
        self.assertEqual(self.run_download(2), 0)
        self.assertEqual(self.requests, ["/two"])

    def test_returns_incomplete_status_when_valid_data_cannot_reach_target(self):
        self.write_dataset({"broken": self.record("/bad"), "one": self.record("/one")})
        self.assertEqual(self.run_download(2), 2)
        self.assertEqual(set(self.read_summary()), {"one"})

    def test_corrupt_summary_is_preserved_instead_of_silently_replaced(self):
        self.write_dataset({"one": self.record("/one")})
        self.summary.write_text("broken summary", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_download(1)
        self.assertEqual(self.summary.read_text(encoding="utf-8"), "broken summary")
        self.assertEqual(self.requests, [])

    def test_rechecks_cached_metadata_and_excludes_non_disaster_records(self):
        records = {"one": self.record("/one"), "traffic": self.record("/traffic", "traffic jam")}
        self.write_dataset(records)
        self.image_path("traffic").write_bytes(self.image_bytes)
        self.summary.write_text(json.dumps({"traffic": records["traffic"]}), encoding="utf-8")
        self.assertEqual(self.run_download(1), 0)
        self.assertEqual(set(self.read_summary()), {"one"})
        self.assertTrue(self.image_path("traffic").is_file())

    def test_negative_target_is_rejected_before_downloading(self):
        self.write_dataset({"one": self.record("/one")})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            self.run_download(-1)
        self.assertEqual(failure.exception.code, 2)
        self.assertEqual(self.requests, [])


class DownloadJobGuardTests(unittest.TestCase):
    def run_job(self, allocation):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        executable = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not executable:
            self.skipTest("Bash is unavailable")
        environment = os.environ.copy()
        for key in ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_SUBMIT_DIR"):
            environment.pop(key, None)
        environment.update(allocation)
        script = Path(__file__).resolve().parents[1] / "scripts/download_incidents.slurm"
        return subprocess.run(
            [executable, str(script)], env=environment,
            capture_output=True, text=True, timeout=15,
        )

    def test_refuses_execution_without_a_slurm_allocation(self):
        result = self.run_job({})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Submit this script with sbatch", result.stderr)

    def test_refuses_a_login_node_even_when_slurm_variables_are_set(self):
        result = self.run_job({"SLURM_JOB_ID": "1", "SLURMD_NODENAME": "login01"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not the login node", result.stderr)


if __name__ == "__main__":
    unittest.main()
