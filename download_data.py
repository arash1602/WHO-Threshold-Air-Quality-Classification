"""Download and unpack the SensEURCity v03 dataset from Zenodo.

    python download_data.py                   # -> ./senseurcity_data_v03/{dataset,metadata}
    python download_data.py --dest D:/data    # anywhere else; then set AQM_DATA_DIR

Source: Van Poppel et al., "SensEURCity: A multi-city air quality dataset collected
for 2020/2021 using open low-cost sensor systems", Scientific Data 10, 322 (2023).
Record: https://doi.org/10.5281/zenodo.17858205 (version 3.0.0, CC-BY-4.0).

The archive is 5.7 GB and unpacks to about 21 GB. The download is verified against
the MD5 published on the record before anything is extracted, and an interrupted
download resumes where it stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

URL = "https://zenodo.org/api/records/17858205/files/senseurcity_data_v03.zip/content"
MD5 = "821fb5eb0124009e958d43e61220f4fd"
SIZE = 5_703_882_048
N_DEVICE_FILES = 84          # <City>_<device>.csv in version 3.0.0 (plus 2 reference-only files)


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: str) -> None:
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have >= SIZE:
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r, open(path, "ab" if have else "wb") as f:
        if have and r.status != 206:        # server ignored the range: start over
            f.seek(0)
            f.truncate()
            have = 0
        done = have
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e9:6.2f} / {SIZE / 1e9:.2f} GB", end="", flush=True)
    print()


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=os.path.join(here, "senseurcity_data_v03"),
                    help="directory to unpack into (default ./senseurcity_data_v03)")
    ap.add_argument("--keep-zip", action="store_true", help="keep the archive afterwards")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    zpath = os.path.join(args.dest, "senseurcity_data_v03.zip")
    print(f"downloading {URL}\n  -> {zpath}")
    download(URL, zpath)
    print("verifying MD5 ...")
    got = md5sum(zpath)
    if got != MD5:
        print(f"MD5 mismatch: got {got}, expected {MD5}. Delete {zpath} and rerun.",
              file=sys.stderr)
        return 1
    print(f"extracting to {args.dest} ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(args.dest)
    data_dir = os.path.join(args.dest, "dataset")
    n = len([f for f in os.listdir(data_dir) if f.endswith(".csv") and "_" in f
             and f.split("_", 1)[0] in ("Antwerp", "Oslo", "Zagreb")])
    if n != N_DEVICE_FILES:
        print(f"expected {N_DEVICE_FILES} device CSVs in {data_dir}, found {n}", file=sys.stderr)
        return 1
    if not args.keep_zip:
        os.remove(zpath)
    print(f"done: {n} device files in {data_dir}")
    if os.path.abspath(args.dest) != os.path.join(here, "senseurcity_data_v03"):
        print(f"set AQM_DATA_DIR={data_dir} (or pass --data-dir) so the scripts find it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
