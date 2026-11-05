"""Compatibility entrypoint for independent R1 adapter training.

Implementation: minecraftrl.training.adapter; validation remains importable here.
"""
from minecraftrl.training.adapter import main
from minecraftrl.training.validation import validate


if __name__ == '__main__':
    main()
