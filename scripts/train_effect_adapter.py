"""Train an E1 candidate from verified new train trajectories; never promotes it."""
from minecraftrl import effect_learning
from minecraftrl.training.adapter import main

if __name__ == '__main__':
    main(profile=effect_learning)
