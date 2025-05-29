# FedPE

This work is about a federated KG embedding method that supports privacy preservation and secure participants extension.

It utilizes bilinear map-based cryptographic techniques to enable encrypted representation and embedding training of entities and relations, and further introduces a ciphertext-level consistency mechanism to support secure collaborative embedding training between existing and newly joined participants without revealing any original data.

## Data

We put our experimential datasets in `.\fed_data`, and detailed information can be found at [OpenKE](https://github.com/thunlp/OpenKE).

The process of federated data can be found in `.\process_datasets.py`.

## Privacy

We provide codes about the quantitative analyses of privacy attacks, which can be found in `.\recon_member_attacks.py` and `.\proper_infer_attacks.py`.

Data encryption codes are in `.\data_encryption.py`.

## Run

We give the example script `.\exp.sh` for reproducing the experimental results.

## Acknowledgement

* [FedE](https://github.com/zjukg/FedE)
* [FedR](https://github.com/taokz/FedR)
* [pypbc](https://github.com/debatem1/pypbc)
* [WLGK](https://github.com/adriancaruana/py_wlgk)
* [OpenKE](https://github.com/thunlp/OpenKE)
