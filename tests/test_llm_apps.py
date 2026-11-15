import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_apps"))

import budgets  # noqa: E402
import context_packing  # noqa: E402
import doc_chunking  # noqa: E402
import mcp_minimal  # noqa: E402
import rag_eval  # noqa: E402


class TestLlmApps(unittest.TestCase):
    def test_mcp_lifecycle(self):
        c = mcp_minimal.Client([sys.executable, mcp_minimal.__file__, "--server"])
        init = c.request("initialize", {"protocolVersion": mcp_minimal.PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
        self.assertEqual(init["result"]["protocolVersion"], mcp_minimal.PROTOCOL_VERSION)
        c.notify("notifications/initialized")
        names = [t["name"] for t in c.request("tools/list")["result"]["tools"]]
        self.assertEqual(names, ["search_notes", "add"])
        self.assertTrue(c.request("tools/call", {"name": "add", "arguments": {"a": 1, "b": "x"}})["result"]["isError"])
        self.assertEqual(c.request("tools/call", {"name": "nope", "arguments": {}})["error"]["code"], -32602)
        self.assertEqual(c.request("bogus")["error"]["code"], -32601)
        c.close()

    def test_packing_respects_budget_and_pins(self):
        for budget in (250, 400):
            ordered, used, _ = context_packing.pack(context_packing.ITEMS, budget)
            self.assertLessEqual(used, budget)
            self.assertEqual(ordered[0]["id"], "system")
            self.assertEqual(ordered[-1]["id"], "question")

    def test_cached_prefix_is_cheaper(self):
        self.assertLess(budgets.call("large", 3000, 400, cached=2500)["cost"], budgets.call("large", 3000, 400)["cost"])

    def test_stemming_improves_recall(self):
        self.assertGreater(rag_eval.evaluate(3, True)[0], rag_eval.evaluate(3, False)[0])

    def test_table_rows_become_sentences(self):
        rows = [c for c in doc_chunking.structured_chunks(doc_chunking.DOC) if "Plan Team:" in c]
        self.assertEqual(len(rows), 1)
        self.assertIn("150 per seat", rows[0])


if __name__ == "__main__":
    unittest.main()
