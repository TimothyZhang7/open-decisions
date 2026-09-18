"""Exercise the same public SDK and primitives on three real model runtimes."""
import argparse
import json
from pathlib import Path

from open_decisions import LocalClient, Choice, Noul, Score, Gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-vl", required=True)
    parser.add_argument("--qwen-text", required=True)
    parser.add_argument("--hf-model", required=True)
    args = parser.parse_args()
    questions = {
        "team": Choice(instructions="Which team handles this request?", criteria={
            "billing": "Payment or refund", "technical": "Software error", "other": "Something else"}),
        "human": Noul(instructions="Does the customer request a human agent?"),
        "urgency": Score(instructions="How severe is this software issue?", criteria=[
            "Cosmetic only", "A feature is impaired with a workaround", "Production is down for everyone"]),
        "gates": Gates(instructions="Determine whether the request requires sustained work and whether essential context is missing.",
                       criteria={"delegate": "Sustained investigation will be needed", "ask": "Essential context is missing"}),
    }
    rows = []
    for backend, path, options in [
        ("mlx-vlm", args.qwen_vl, {}), ("mlx-lm", args.qwen_text, {}),
        ("transformers", args.hf_model, {"device": "cpu"}),
    ]:
        with LocalClient(path, backend=backend, backend_options=options) as client:
            r = client.evaluate(state="Production is down for all users after our latest deployment. Investigate the available logs and repository. Please get a human agent involved.", questions=questions)
            assert r.backend == backend and set(r.answers) == set(questions)
            assert r.usage.model_evaluations == 4 and r.usage.output_tokens == 0
            a = client.evaluate(state="Please refund a duplicate charge.", questions={"team": questions["team"]})
            b = client.evaluate(state="The software crashes on launch.", questions={"team": questions["team"]})
            assert a.usage.output_tokens == b.usage.output_tokens == 0
            rows.append({"info": client.info, "all_primitives": r.model_dump(),
                         "changed_state": [a.model_dump(), b.model_dump()]})
            print(backend, client.info, "all four primitives returned; output_tokens=0", flush=True)
            print("changed states:", a.answers["team"].choice, b.answers["team"].choice, flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/backend-matrix.json").write_text(json.dumps(rows, indent=2)+"\n")


if __name__ == "__main__":
    main()
