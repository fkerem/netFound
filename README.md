# netFound: Foundation Model for Network Security

This is a source code for the netFound model by the Systems & Networking Lab, UC Santa Barbara

## Description

## Checkpoint
TODO: here there will be a link to a pretrained checkpoint

## How to use

### Start here
The easiest way to check that the preprocessing code and model work is to use the provided Dockerfile and Makefile.

1. Build a docker container: `docker build -t netfound:test .`. The docker container will contain the source code and the small test dataset, located in the folder data/test.
2. Run the container: `docker run -it netfound:test`. You should have a shell inside the container in the folder /workspace.
3. Run `make all` to start the preprocessing, pretraining, and finetuning of the netFound model on the test data.

You can explore how Makefile and Dockerfile are constructed:
1. Dockerfile creates a container that has all the dependencies installed
2. data/test folder contains raw pcap files in a certain format to be preprocessed
3. Makefile:preprocess contains instructions on how to run the preprocessing code which will filter, split, and tokenize the data.
4. Makefile:pretrain contains instructions on how to run the pretraining on the preprocessed data.
5. Makefile:finetune contains instructions on how to run the finetuning on the preprocessed data.

### Bring Your Own Data


## How to cite
```
@misc{guthula2024netfoundfoundationmodelnetwork,
      title={netFound: Foundation Model for Network Security}, 
      author={Satyandra Guthula and Roman Beltiukov and Navya Battula and Wenbo Guo and Arpit Gupta},
      year={2024},
      eprint={2310.17025},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2310.17025}, 
}
```
