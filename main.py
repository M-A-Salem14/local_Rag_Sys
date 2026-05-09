import argparse
from config   import DEVICE
from indexer  import index_directory
from pipeline import run_rag


def main():
    print(f"Device: {DEVICE}")

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
