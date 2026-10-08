import contextlib
import io
import unittest
from unittest.mock import patch

import requests

from src.helper.image import download_image


def response(status):
    result = requests.Response()
    result.status_code = status
    result.url = "https://example.test/image.png"
    result._content = b"image bytes"
    return result


class ImageDownloadRetryTests(unittest.TestCase):
    def test_does_not_retry_unavailable_or_rate_limited_urls_immediately(self):
        for status in (403, 404, 410, 429):
            with self.subTest(status=status), patch("src.helper.image.requests.get", return_value=response(status)) as get:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertIsNone(download_image("https://example.test/image.png"))
                self.assertEqual(get.call_count, 1)

    def test_can_recover_from_a_transient_server_error(self):
        with patch("src.helper.image.requests.get", side_effect=[response(503), response(200)]) as get:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(download_image("https://example.test/image.png"), b"image bytes")
            self.assertEqual(get.call_count, 2)

    def test_can_recover_from_a_connection_timeout(self):
        with patch("src.helper.image.requests.get", side_effect=[requests.Timeout("timeout"), response(200)]) as get:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(download_image("https://example.test/image.png"), b"image bytes")
            self.assertEqual(get.call_count, 2)

    def test_stops_when_the_retry_budget_is_exhausted(self):
        with patch("src.helper.image.requests.get", side_effect=requests.Timeout("timeout")) as get:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(download_image("https://example.test/image.png", retries=2))
            self.assertEqual(get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
