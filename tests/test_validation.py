import os
import unittest

class TestLangChainRAGIntegrity(unittest.TestCase):
    def test_readme_exists(self):
        self.assertTrue(os.path.exists("README.md"), "README.md must be present")

    def test_makefile_exists(self):
        self.assertTrue(os.path.exists("Makefile"), "Makefile must be present")

    def test_ci_workflow_exists(self):
        self.assertTrue(os.path.exists(".github/workflows/ci.yml"), "CI workflow must be present")

    def test_healthcheck_script_exists(self):
        self.assertTrue(os.path.exists("scripts/healthcheck.sh"), "Healthcheck script must be present")

    def test_environment_template_exists(self):
        self.assertTrue(os.path.exists("config/production.env.example"), "Env template must be present")

    def test_gitattributes_exists(self):
        self.assertTrue(os.path.exists(".gitattributes"), ".gitattributes must be present")

    def test_security_policy_exists(self):
        self.assertTrue(os.path.exists("SECURITY.md"), "SECURITY.md must be present")

    def test_architecture_diagram_exists(self):
        self.assertTrue(os.path.exists("docs/architecture.svg"), "docs/architecture.svg must be present")

    def test_pipeline_implements_chroma_and_mmr(self):
        self.assertTrue(os.path.exists("pipeline.py"), "pipeline.py must be present")
        with open("pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("RecursiveCharacterTextSplitter", content)
        self.assertIn("search_type=\"mmr\"", content)
        self.assertIn("DocumentRAGPipeline", content)

    def test_fastapi_app_configured(self):
        self.assertTrue(os.path.exists("app.py"), "app.py must be present")
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("FastAPI", content)
        self.assertIn("/query", content)
        self.assertIn("/ingest", content)
        self.assertIn("/health", content)

    def test_chain_implements_grounding(self):
        self.assertTrue(os.path.exists("chain.py"), "chain.py must be present")
        with open("chain.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("ChatPromptTemplate", content)
        self.assertIn("format_docs", content)

if __name__ == "__main__":
    unittest.main()
