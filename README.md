# Pi-HiQRL
Original implementation of the Physics-informed Hierarchical Quasimetric Reinforcement Learning (Pi-HiQRL) algorithm.

### Installation

Our implementations require Python 3.9+ and additional dependencies, including `jax >= 0.4.26`.
To install these dependencies, run:

```shell
pip install -r requirements.txt
```

### Running Pi-HiQRL

To train an agent, you can run the `main.py` script.
Training metrics, evaluation metrics, and videos are logged via `wandb` by default.
Here are some example commands (see [hyperparameters.sh](hyperparameters.sh) for the full list of commands):

```shell
# pointmaze-large-navigate-v0 (Pi-HiQRL)
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/pi_hiqrl.py 
```

### Acknowledgements

We want to thank the authors of the [OGBench](https://seohong.me/projects/ogbench/) benchmark for their amazing work.
Most of this code is based on their implementations. Please refer also to their [GitHub Repo](https://github.com/seohongpark/ogbench).
