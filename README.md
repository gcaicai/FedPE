# FedPC
This work is about a federated KG embedding method that ensures privacy preservation and the dynamic participation of clients.
It introduces an encryption protocol based on bilinear maps to process client data, achieving ciphertext alignment to enhance the quality of global embeddings while
preserving data privacy.

## Data
We put our experimential datasets in `.\fed_data`, and more datasets can be found at [OpenKE](https://github.com/thunlp/OpenKE).

Static and dynamic splitting of federated datasets can be found in `.\process_datasets.py` and `.\process_dynamic_datasets.py`, separately.

## Privacy
We provide codes about the quantitative analyses of reconstruction attacks, member inference attacks and property inference attacks.

The codes of three types of attacks are in `.\recon_member_attacks.py` and `.\proper_infer_attacks.py`.

Encryption codes are in `.\data_encryption.py`.

## Run
We give the example script `.\exp.sh` for reproducing our experimental results.

## Acknowledgement
* [FedE](https://github.com/zjukg/FedE)
* [FedR](https://github.com/taokz/FedR)
* [pypbc](https://github.com/debatem1/pypbc)
* [WLGK](https://github.com/adriancaruana/py_wlgk)
* [OpenKE](https://github.com/thunlp/OpenKE)
