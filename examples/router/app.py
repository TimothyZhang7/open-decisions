"""Router demo: all inference and decision semantics come from Open Decisions."""
import argparse
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles
from open_decisions.server import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="mlx-vlm", choices=["mlx-vlm", "mlx-lm", "transformers"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    options = {"device": args.device, "vision": args.vision} if args.backend == "transformers" else {}
    app = create_app(args.model, backend=args.backend, backend_options=options)
    app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True), name="router-example")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
