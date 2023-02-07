VAE Playland :joystick:
=======================

A VAE playland

## Installation

Change directory to root of the package, and then type on the command line:

```
pip install -e .
```

## How to Run

Change entity and group of W&B logger (or set them in a config file).

```bash
vae-train data=iris model=simplebi trainer.logger.entity=x trainer.logger.group=y
```
