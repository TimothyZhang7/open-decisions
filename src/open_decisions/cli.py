import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="open-decisions")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Start the decision API")
    serve.add_argument("--model", required=True)
    serve.add_argument("--backend", choices=["mlx-vlm", "mlx-lm", "transformers"], default="mlx-vlm")
    serve.add_argument("--device", default="auto")
    serve.add_argument("--vision", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8772)
    evaluate = sub.add_parser("evaluate", help="Evaluate a JSON request locally")
    evaluate.add_argument("request", type=Path)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--backend", choices=["mlx-vlm", "mlx-lm", "transformers"], default="mlx-vlm")
    download = sub.add_parser("download", help="Download an open model from Hugging Face")
    download.add_argument("repo")
    download.add_argument("--revision")
    download.add_argument("--directory", required=True)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .server import create_app
        options = {"device": args.device, "vision": args.vision} if args.backend == "transformers" else {}
        uvicorn.run(create_app(args.model, backend=args.backend, backend_options=options), host=args.host, port=args.port)
    elif args.command == "download":
        from huggingface_hub import snapshot_download
        print(snapshot_download(args.repo, revision=args.revision, local_dir=args.directory,
                                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.md", "LICENSE*"]))
    else:
        from .client import LocalClient
        from .schema import EvaluationRequest
        request = EvaluationRequest.model_validate_json(args.request.read_text())
        with LocalClient(args.model, backend=args.backend) as client:
            print(client.evaluate_request(request).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
