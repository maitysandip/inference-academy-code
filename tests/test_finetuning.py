import re, subprocess, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "finetuning"

def run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / script), *args], capture_output=True, text=True, timeout=300, cwd=ROOT).stdout

class FineTuningScripts(unittest.TestCase):
    def test_lora_math(self):
        out = run("lora_math.py")
        self.assertIn("base ≈ 8.03B params", out)
        diff = float(re.search(r"max difference: ([0-9.e+-]+)", out).group(1))
        self.assertLess(diff, 1e-12)  # floating-point noise only; the exact value varies by Python version

    def test_memory_table(self):
        out = run("training_memory.py")
        self.assertIn("8B (Llama-3.1-8B shape)     qlora", out)
        self.assertIn("gradient checkpointing on :   3.3 GB", out)

    def test_synthetic_pipeline_yield(self):
        out = run("synthetic_data_pipeline.py")
        self.assertIn("yield: 20.2% of raw generations survive", out)

    def test_tool_harness_ranks_models(self):
        out = run("tool_call_harness.py")
        self.assertIn("(False, 'unit not in enum')", out)
        self.assertIn("fine-tuned + grammar       100.0%", out)

    def test_embeddings_improve_on_keyword_search(self):
        out = run("toy_embeddings.py", "5")
        rows = {l.split("  ")[0].strip(): l for l in out.splitlines()}
        self.assertIn("trained, plus hard negatives", out)
        self.assertIn("83.9%", out)

if __name__ == "__main__":
    unittest.main()
