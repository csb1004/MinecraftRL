"""Independent E2 PPO training; explicit calibration and train-only retention."""
from minecraftrl import execution_learning as profile
from minecraftrl.training.adapter import main

if __name__=='__main__':main(profile)
