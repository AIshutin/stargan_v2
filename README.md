# StarGAN v2

Changes:
- adding noise to images
- label smoothing

Things to try:
- spectral norm
- adaptive residuals

# Setup

```bash
pip3 install -r requirements.txt
```

# Run

```bash
python3 train.py
```

Then you can compute LPIPS by running:
```bash
python3 test.py --checkpoint checkpoints/my/checkpoint.pt
```

Open visualizer.ipynb to visualize results.

# Pretrained

Undertrained (10k steps) checkpoint can be downloaded via:
```bash
gdown --fuzzy https://drive.google.com/file/d/1CJwZOoVV8aCa2Jvn2cBoDpJLZXnI3KjN/view?usp=sharing
```