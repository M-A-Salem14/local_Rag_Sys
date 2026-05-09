import os
import torch

# Disable broken CUDA (RTX 5070 sm_120 needs PyTorch 2.7+)
if torch.cuda.is_available():
    try:
        _ = torch.zeros(1, device="cuda")
    except RuntimeError:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Force re-evaluation by patching cuda availability
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0

import argparse
from indexer  import index_directory
from pipeline import run_rag

def main():
    parser = argparse.ArgumentParser(description="Local RAG System")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("index", help="Index documents in ./data")

    ask = sub.add_parser("ask", help="Query the RAG system")
    ask.add_argument("question", type=str)
    ask.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.cmd == "index":
        index_directory()

    elif args.cmd == "ask":
        result = run_rag(args.question, verbose=args.verbose)
        print(f"\n{'='*60}\n{result['answer']}\n")
        print("Sources:")
        for s in set(result["sources"]):
            print(f"  • {s}")

if __name__ == "__main__":
    main()
