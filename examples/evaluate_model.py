"""Small, reproducible smoke evaluation; synthetic fixtures, not a quality benchmark.

Run: python examples/evaluate_model.py --model .models/qwen3-vl-4b-instruct-4bit
Writes individual decisions, expected labels, timing and cache differential checks.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

from PIL import Image, ImageDraw, ImageFont

from open_decisions import Choice, Noul, Score, Gates, ImageInput, EvaluationRequest
from open_decisions.engine import Engine, compile_question, probabilities
from open_decisions.backends.mlx_vlm import MLXVLMBackend


def fixtures(root):
    root.mkdir(parents=True, exist_ok=True)
    for color, shape in [("red", "circle"), ("blue", "square"), ("green", "triangle")]:
        image = Image.new("RGB", (336, 336), "white")
        draw = ImageDraw.Draw(image)
        if shape == "circle":
            draw.ellipse((60, 60, 276, 276), fill=color)
        elif shape == "square":
            draw.rectangle((60, 60, 276, 276), fill=color)
        else:
            draw.polygon([(168, 40), (40, 290), (296, 290)], fill=color)
        image.save(root / f"{color}.png")
    image = Image.new("RGB", (600, 320), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    draw.multiline_text((35, 30), "DEMO RECEIPT\nCoffee       $4.00\nSandwich     $8.00\nTOTAL       $12.00", fill="black", font=font, spacing=20)
    image.save(root / "receipt.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results/smoke.json")
    args = parser.parse_args()
    assets = Path(__file__).parent / "fixtures"
    fixtures(assets)
    started = time.perf_counter()
    backend = MLXVLMBackend(args.model)
    load_ms = (time.perf_counter()-started)*1000
    engine = Engine(backend)
    cases = []

    def add(name, state, question, expected, image=None):
        cases.append((name, EvaluationRequest(state=state, questions={"decision": question},
                     images=[ImageInput.from_file(path) for path in (image if isinstance(image, list) else [image])]
                     if image else []), expected))

    team = Choice(instructions="Which team should handle this request?", criteria={
        "billing": "Payments, invoices, refunds", "technical": "Software errors, integrations",
        "shipping": "Deliveries, tracking", "other": "None of these"})
    for i, (state, answer) in enumerate([
        ("I was charged twice. Please refund the duplicate charge.", "billing"),
        ("My parcel never arrived. Where is it?", "shipping"),
        ("The API returns HTTP 500 when I upload a file.", "technical"),
        ("Do you sponsor local community events?", "other"),
        ("Please email a VAT invoice for last month's payment.", "billing"),
        ("The package tracking number says delivered to the wrong address.", "shipping"),
        ("The app crashes whenever I open Settings.", "technical"),
    ]):
        add(f"team_{i}", state, team, answer)
    human = Noul(instructions="Does the customer explicitly request to speak with a human support agent?")
    for i, (state, answer) in enumerate([
        ("Can I speak to a real person?", True), ("Thanks, that solved it!", False),
        ("Please connect me to a human support agent.", True),
        ("Do not transfer me to a person. Just show me the help page.", False),
    ]):
        add(f"human_{i}", state, human, answer)
    severity = Score(instructions="How severe is the software issue?", criteria=[
        "Cosmetic issue with no effect on functionality", "A feature is impaired but a workaround exists",
        "A core function is unavailable and there is no workaround"])
    for i, state in enumerate([
        "The icon is slightly misaligned. Everything works normally.",
        "Export fails in Safari. Users can export successfully in Chrome as a workaround.",
        "All users are unable to sign in. The production service is unavailable and there is no workaround.",
    ]):
        add(f"severity_{i}", state, severity, i)
    router = json.loads((Path(__file__).parent / "router.json").read_text())["questions"]["routing"]
    for i, (state, expected) in enumerate([
        ({"request": "Thanks!"}, {"delegate": False, "ask": False}),
        ({"request": "What is 7 times 8?"}, {"delegate": False, "ask": False}),
        ({"request": "Inspect the codebase, implement SSO, and run the tests.",
          "context": "Repository, requirements and credentials are available."}, {"delegate": True, "ask": False}),
        ({"request": "Book it for me.", "context": "No earlier messages or referent for 'it'."}, {"delegate": False, "ask": True}),
        ({"request": "Compare six CRM products using public pricing and reviews, then write a report."},
         {"delegate": True, "ask": False}),
    ]):
        add(f"router_{i}", state, Gates.model_validate(router), expected)
    color = Choice(instructions="What is the color of the single large shape in the image?", criteria={
        "red": "Red", "blue": "Blue", "green": "Green", "other": "Another color"})
    shape = Choice(instructions="What shape is shown in the image?", criteria={
        "circle": "Circle", "square": "Square", "triangle": "Triangle"})
    for name, geometry in [("red", "circle"), ("blue", "square"), ("green", "triangle")]:
        add(f"color_{name}", "Judge the attached image.", color, name, assets / f"{name}.png")
        add(f"shape_{name}", "Judge the attached image.", shape, geometry, assets / f"{name}.png")
    add("receipt_total", "Read the attached receipt.", Choice(instructions="What is the TOTAL amount on this receipt?",
        criteria={"4": "$4.00", "8": "$8.00", "12": "$12.00", "20": "$20.00"}), "12", assets / "receipt.png")
    pair = Choice(instructions="What are the colors of the shapes in the first and second images, in that order?",
                  criteria={"red_blue": "First red, second blue", "blue_red": "First blue, second red", "same": "Both the same color"})
    for colors in [("red", "blue"), ("blue", "red")]:
        add("pair_" + "_".join(colors), "The first image is followed by the second. Compare them.", pair,
            "_".join(colors), [assets / f"{c}.png" for c in colors])
    by_name = {case[0]: case for case in cases}
    records = []
    for name, request, expected in cases:
        result = engine.evaluate(request)
        answer = result.answers["decision"]
        if isinstance(expected, dict):
            actual = {key: value >= .5 for key, value in answer.gates.items()}
        elif isinstance(expected, bool):
            actual = answer.noul >= .5
        elif isinstance(expected, int):
            actual = round(answer.score)
        else:
            actual = answer.choice
        record = dict(name=name, request=request.model_dump(exclude={"images"}), image=bool(request.images),
                      expected=expected, actual=actual, passed=actual == expected, response=result.model_dump())
        records.append(record)
        print(f"{name:20} {str(actual == expected):5} {result.latency_ms:7.1f} ms  {actual}", flush=True)

    # Same policy, changing image/state; cache reuse must agree with full prefill.
    differentials = []
    for case in [by_name[name] for name in ["team_0", "team_1", "color_red", "color_blue", "color_green", "team_3", "pair_red_blue"]]:
        name, request, _ = case
        backend.cache_enabled = True
        cached = engine.evaluate(request)
        backend.cache_enabled = False
        fresh = engine.evaluate(request)
        backend.cache_enabled = True
        p = cached.answers["decision"].probabilities
        q = fresh.answers["decision"].probabilities
        delta = max(abs(p[key] - q[key]) for key in p)
        same = cached.answers["decision"].choice == fresh.answers["decision"].choice
        differentials.append(dict(name=name, max_probability_delta=delta, same_choice=same,
                                  cached_tokens=cached.usage.cached_tokens, fresh_tokens=fresh.usage.cached_tokens))
        assert same and delta < .02, differentials[-1]

    # Check the optimized final-position head against MLX-VLM's ordinary forward.
    from open_decisions.images import decode_images
    reference_checks = []
    for name, request, _ in [by_name[key] for key in ["team_0", "color_red", "receipt_total", "pair_red_blue"]]:
        question = request.questions["decision"]
        system, keys, _ = compile_question(question, backend.labels)
        images = decode_images(request.images)
        state = json.dumps(request.state, ensure_ascii=False, separators=(",", ":"))
        prompt = backend._format([
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": "State: " + state}]},
        ], True)
        inputs = backend.prepare_inputs(backend.processor, images=images or None, prompts=prompt,
                                        image_token_index=backend.network.config.image_token_index,
                                        max_pixels=1024*1024, min_pixels=32*32*4)
        ids, mask = inputs.pop("input_ids"), inputs.pop("attention_mask", None)
        raw = backend.network(ids, mask=mask, **inputs).logits[0, -1].astype(backend.mx.float32)
        reference = raw[backend.mx.array(backend.label_ids[:len(keys)])]
        backend.mx.eval(reference)
        observed = backend.score(system, state, images, len(keys))
        rp, op = probabilities(reference.tolist(), 1), probabilities(observed.logits, 1)
        delta = max(abs(a-b) for a, b in zip(rp, op))
        reference_checks.append(dict(name=name, max_probability_delta=delta,
                                     max_logit_delta=max(abs(a-b) for a, b in zip(reference.tolist(), observed.logits))))
        assert delta < .02, reference_checks[-1]

    text_latencies = [r["response"]["latency_ms"] for r in records if not r["image"] and r["response"]["usage"]["cached_tokens"]]
    image_latencies = [r["response"]["latency_ms"] for r in records if r["image"]]
    result = {
        "kind": "synthetic_smoke_evaluation", "hardware": platform.platform(), "load_ms": load_ms,
        "versions": {name: importlib.metadata.version(name) for name in ["mlx-vlm", "mlx", "transformers", "pillow"]},
        "model": args.model, "total": len(records), "passed": sum(r["passed"] for r in records),
        "warm_text_median_ms": statistics.median(text_latencies), "image_median_ms": statistics.median(image_latencies),
        "output_tokens": sum(r["response"]["usage"]["output_tokens"] for r in records),
        "cases": records, "cache_checks": differentials, "reference_forward_checks": reference_checks,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"cases", "cache_checks", "reference_forward_checks"}}, indent=2))


if __name__ == "__main__":
    main()
