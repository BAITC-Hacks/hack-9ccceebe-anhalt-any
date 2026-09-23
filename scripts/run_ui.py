"""Start the WindAgent dashboard and its existing API on localhost."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="WindAgent AI — веб-интерфейс")
    parser.add_argument("--port", type=int, default=8000, help="Local port (default: 8000)")
    args = parser.parse_args()
    import uvicorn
    print(f"WindAgent AI: http://127.0.0.1:{args.port}/ui/", flush=True)
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
