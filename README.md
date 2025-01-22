# netFound: Foundation Model for Network Security

This is a source code for the netFound model by the Systems & Networking Lab, UC Santa Barbara

## Description

netFound is a network traffic foundation model that uses transformer architecture and includes a pretraining phase on unlabeled data to achieve high results.  

Key features:
- netFound takes raw PCAP data as input
- netFound can (and need) be pretrained on the unlabeled dataset
- netFound uses Hierarchical Transformer architecture to take into account packet burst and flow behavior
- netFound uses burst metadata (inter arrival time, number of bytes per burst, etc)

Corresponding paper: https://arxiv.org/abs/2310.17025

## Checkpoint

https://huggingface.co/snlucsb/netFound-640M-base

The checkpoint is pretrained on ~450mln flows of the real-world network traffic of the University of California, Santa Barbara.  

Pretrained model metrics:  
```
  eval_loss                         =     1.8847
  eval_macro_mlm_f1                 =     0.4038
  eval_macro_mlm_prec               =     0.7205
  eval_macro_mlm_recall             =     0.3005
  eval_mlm_acc                      =     0.8514
  eval_swapped_macro_pred_f1        =     0.9605
  eval_swapped_macro_pred_prec      =      0.963
  eval_swapped_macro_pred_recall    =     0.9603
  eval_swapped_pred_acc             =     0.9605
  eval_swapped_weighted_pred_f1     =     0.9605
  eval_swapped_weighted_pred_prec   =     0.9628
  eval_swapped_weighted_pred_recall =     0.9605
  eval_weighted_mlm_f1              =     0.8451
  eval_weighted_mlm_prec            =     0.8816
  eval_weighted_mlm_recall          =     0.8514
  perplexity                        =     6.5842
  
  Total params:  643,825,672
```

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

To use your own data, the easy way is to run the scripts/preprocess_data.py on your dataset.  
For pretraining, create the next folder structure:
  - folder_name
    - raw
      - *.pcap

Then run the scripts/preprocess_data.py on the folder_name:  
`python3 scripts/preprocess_data.py --input_folder folder_name --action pretrain --tokenizer_config configs/TestPretrainingConfig.json --combined`
- The script will create intermediate folders (extracted, split, etc). The resulting tokens would be in the "tokens" folder.
- You can use different tokenizer config (mostly to change internal and external IPs to define directions)
- You can remove --combined flags to create multiple arrow files (one per original pcap). This is usually better for parallelization between nodes when using multiple data loaders.
- You can use --tcp_options to include TCPOptions in the data, but for this your data need to be preprocessed with additional flag "1" (as a last argument) when using 3_extract_fields (to include tcpoptions) and config file with TCPOptions should be provided

For fine-tuning, use the same structure as for pretraining, but separate different classes to different folders.
- folder_name1
  - raw
    - *.pcap
- folder_name2
  - raw
    - *.pcap

Then run the scripts/preprocess_data.py on the folder_name:  
`python3 scripts/preprocess_data.py --input_folder folder_name --action finetune --tokenizer_config configs/TestPretrainingConfig.json --combined`

- **Attention**: folder names should be integers (1, 2, 3, etc.) and will be used as class labels. Only integer numbers are supported as class labels.
- The resulting arrow files (preprocessed data) have the "labels" column which can be modified manually if needed to change the corresponding labels (incl. for regression labels)

  
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
