# Generating Rescue Images from Natural Disaster Images

This project explores image generation for scientific research, with a focus on transforming natural disaster scenes into rescue-oriented images.

## Technology Badges

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Conda](https://img.shields.io/badge/Conda-Environment-44A833)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Enabled-EE4C2C)
![CUDA](https://img.shields.io/badge/CUDA-Required-76B900)

## Requirements

- Python `>= 3.11`
- NVIDIA GPU with `40 GB VRAM` or more
- Conda installed on your machine

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ImageGeneration
```

### 2. Create a Conda environment

```bash
conda create -n rescue-image python=3.11 -y
conda activate rescue-image
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
### 4. Create the data folder 
- Create a data folder in the root. Including 2 subfolders: data/input and data/output 
- We are using `incidents1M` for dataset. You can get here: https://github.com/ethanweber/IncidentsDataset. Thank you for authors. 
### 5. Verify the installation

```bash
python main.py
```

### 6. Run on VPS with Slurm

```bash
sbatch sbatch.sh
```

Run the batch command from the repository root so the relative log paths resolve correctly. If your virtual environment lives elsewhere, set `VENV_PATH` before submitting the job.

## Notes

- The project is designed for GPU execution.
- For best performance, use an NVIDIA GPU that meets or exceeds the 40 GB VRAM requirement.
- If your CUDA driver or toolkit differs from the environment used to build the dependencies, you may need to adjust the PyTorch and CUDA packages accordingly.

Build with Cloudian 💙 Cloud
