# Running the seed-sensitivity experiment (Windows)

This reruns the paper's selected models with four new random seeds. It does **not** repeat the hyperparameter search. When it finishes, send back the `seed_results` folder.

You need about 30 GB of free disk space (the dataset unpacks to 21 GB). An NVIDIA graphics card makes it much faster but is not required.

## 1. Install Python

Install Python 3.13 from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"** during installation.

## 2. Set up the code

Unzip the code folder, open **Command Prompt** in it (type `cmd` in the folder's address bar and press Enter), then run:

```
python -m venv .venv
.venv\Scripts\activate
```

**If the PC has an NVIDIA graphics card**, install the GPU build of PyTorch: go to https://pytorch.org/get-started/locally/, choose Stable / Windows / Pip / Python / the newest CUDA version, and run the command it shows.

**If it does not**, install the standard build:

```
pip install torch
```

Then, in both cases:

```
pip install numpy pandas scikit-learn scipy matplotlib
```

(Do not use `pip install -r requirements.txt` here: it pins an exact PyTorch version and can replace the GPU build with a CPU-only one.)

## 3. Download the dataset (once, about 5.7 GB)

```
python download_data.py
```

## 4. Run, in this order

Each step must succeed before the next.

```
python seed_refit.py --smoke
```
A two-minute test that the setup works. It ends with "Smoke test passed".

```
python seed_refit.py --check
```
Reruns the paper's own seed on a few folds and compares the scores. It ends with "Reproduces the paper." If it instead reports a difference, still continue, but include that line when you send the results.

```
python seed_refit.py --workers 2
```
The full run. With a GPU, expect a few hours; without one, possibly overnight. It saves after every model, so if the PC restarts, run the same command again and it continues where it stopped.

The first run of any of these spends about 8 minutes reading the dataset; later runs start in seconds.

## 4b. Optional: recheck a sample (about 10 minutes)

```
python seed_refit.py --recheck 30
```
Reruns 30 randomly chosen refits and compares them with the recorded results. On the same machine they should be identical; it ends with "reproduce the recorded results exactly".

## 5. Send back

Zip the `seed_results` folder and send it. That's everything needed; `seed_analysis.py` turns it into the paper's table.
