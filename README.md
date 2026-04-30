# Voucher Extractor

Pulls the account number, amount, and signatures out of a voucher photo.
Uses PaddleOCR for the text and OpenCV for the rest (finding the paper, flattening the photo, cutting out the signatures).

## Files

```
voucher_extractor.py
requirements.txt
README.md
```

## Setup

Make a venv and activate it:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the packages:
```bash
pip install -r requirements.txt
```

This takes a while (around 1 GB of stuff to download). On the first run, PaddleOCR will also fetch its model files (~120 MB). After that it works offline.

## Running it

```bash
python voucher_extractor.py voucher.jpg
```
Add `--debug` if you want to see how the paper detection and dewarping went:
```bash
python voucher_extractor.py voucher.jpg --out results --debug
```

## What you get back

```
results/
├── extraction_summary.json
├── aligned.png
├── debug/                    (only with --debug)
│   ├── 01_corners_detected.png
│   └── 02_dewarped.png
└── signatures/
    ├── approved_by_sig_color.png
    └── main_signature_color.png
```

The JSON file has the extracted values:

```json
{
  "account_number": "",
  "amount": "",
  "signatures": {
    "approved_by_sig": {  },
    "main_signature":  {  }
  },
  "diagnostics": { }
}
```

The `diagnostics` section shows what PaddleOCR considered before picking a winner. Useful when something looks off.


## How it works

1. Find the paper in the photo (it's brighter and less colorful than most surfaces).
2. Warp it flat — that way the fields stay in the same place even if the photo was tilted.
3. Crop each field and run PaddleOCR on it. The amount fields are upscaled extra because the digits are small.
4. The amount also appears in the Rs. box on the right, so we use that as a sanity check. If both reads agree, the score gets a big boost.
5. For signatures, find the blue ink with a color filter and save the cropped region.
