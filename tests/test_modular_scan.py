import unittest
import json
from app.main import app

class TestModularScan(unittest.TestCase):
    def test_modular_scan_router_registration(self):
        # Verify routes are registered in the FastAPI app
        routes = [r.path for r in app.routes]
        self.assertIn("/hit-study/{analysis_id}/pose-scan", routes)
        self.assertIn("/hit-study/{analysis_id}/audio-scan", routes)
        self.assertIn("/hit-study/{analysis_id}/pose-scan/save", routes)
        self.assertIn("/hit-study/{analysis_id}/audio-scan/save", routes)

if __name__ == "__main__":
    unittest.main()
