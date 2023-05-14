# controlnet-demosaicing

# Setup
`prepare.py`: Data preprocessing. Here we resize all Flickr-8k images to (256, 256) and pixelate them.
`train_controlnet.py`: Train ControlNet on pixelated Flickr-8k with *denoising* condition.
`inference.py`: Evaluate ControlNet.

# Demonstration
See `6_8301_final_project_eval.ipynb`
