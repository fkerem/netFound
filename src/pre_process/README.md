Following are the instructions to create the tokens from Nprint vectors.

Add the input and output directory in a Config.json file following the example of UCSBPretrainingConfig.json for pretraining or UCSBFinetuningConfig.json for Finetuning.

Add the reference of the Config file in the `TokenizeNpt.py` file and run to get the tokens and metadata for each flow

`Tokenize.py` is the counterpart where Nprint is not used and csvs with relevant fields are used instead.

Next to collect the tokens and meta data into a single file use the CollectTokensInFiles.py add the output folder used in the TokenizeNpt.py as `dirpath` and output file as `outputpath`. Also set the flag for finetuning depending on the use case.

If the dataset is for finetuning, we need to split the test and train datasets. `SplitToTestTrain.py` is used to split the test and train datasets. Update the paths to the entire dataset, test dataset and train dataset

## Field extraction

input: merged pcap with flows

1. Process all pcaps and leave only tcp/udp/icmp packets
`./1_tshark_extraction.sh input_folder output_folder`
2. split pcap by flows  
`./2_pcap_splitting.sh input_folder output_folder`
3. extract packet features from each flow
`./3_extract_fields.sh input_folder output_folder`

### File structure
Resulting folder structure:
- <pcap_filename>(folder)
  - <pcap_filename>.pcap.<protocol_number(1,6,17)>
  - <pcap_filename>.pcap.<protocol_number(1,6,17)>
  - ...

### File structure and fields

Each file is a binary stream of packets in a custom format.
First byte is always a procotol number: 1 for ICMP, 6 for TCP, 17 for UDP.
Then, each packet is represented by a sequence of fields without separators.

So, the packet with TCP protocol will have the following structure:
- uint8_t: protocol number
- packet0 representation
- packet1 representation
- ...

#### ICMP packet structure
- uint64_t: unix timestamp with nanoseconds
- uint8_t: IP header length (in bytes)
- uint8_t: Type of Service
- uint16_t: Total Length
- uint8_t: IP Flags
- uint8_t: TTL
- uint32_t: Source IP (as an integer)
- uint32_t: Destination IP (as an integer)
- uint8_t: ICMP type
- uint8_t: ICMP code
- 12 bytes of data padded with zeros

#### TCP packet structure
- uint64_t: unix timestamp with nanoseconds
- uint8_t: IP header length (in bytes)
- uint8_t: Type of Service
- uint16_t: Total Length
- uint8_t: IP Flags
- uint8_t: TTL
- uint32_t: Source IP (as an integer)
- uint32_t: Destination IP (as an integer)
- uint16_t: Source Port
- uint16_t: Destination Port
- uint8_t: TCP flags
- uint16_t: TCP Window Size
- uint32_t: Relative Sequence Number
- uint32_t: Relative Acknowledgement Number
- uint16_t: TCP urgent pointer
- 12 bytes of data padded with zeros

#### UDP packet structure
- uint64_t: unix timestamp with nanoseconds
- uint8_t: IP header length (in bytes)
- uint8_t: Type of Service
- uint16_t: Total Length
- uint8_t: IP Flags
- uint8_t: TTL
- uint32_t: Source IP (as an integer)
- uint32_t: Destination IP (as an integer)
- uint16_t: Source Port
- uint16_t: Destination Port
- uint16_t: Length
- 12 bytes of data padded with zeros