# Public release boundary

This repository contains an allowlisted subset of scientific code and compact, public-source aggregate results. It does not include patient records or private deployment inventories.

- Excluded: `.secrets`, API credentials, account/session files, SSH configuration, `.agents`, `.codex`, environments, caches, model weights, activation tensors, raw FASTA/VCF, logs and administrative monitor/launcher scripts.
- `.gitignore` prevents common future additions. It does not sanitize tracked files. Selected text and Git candidates are separately scanned before publishing; the original source history is not copied.
- Aggregate figures are generated from the checked-in CSV/JSON, avoiding screenshots of server terminals, internal paths or account information.
- Models, external APIs and public datasets retain their own usage terms. The selected public workflow does not need an API key for the CPU checks. Do not put tokens in source, notebooks or command-line examples.
- GPU scripts can download public resources or models when explicitly run. Review their dependencies and intended inputs first. No automatic inference service, scheduled job or paid API request is enabled by cloning this repository.
- Input/result hashes identify artifacts; they do not anonymize sensitive source data or prove who created a file. Do not publish new patient or internal data merely because it has a hash.
- Research results are not validated clinical decisions. Clinical labels, measured experimental effects, model predictions and numerical checks must remain distinct.

The original research workspace was preserved. Future disclosure of a credential requires removing it from publication and rotating it; ignore rules cannot revoke an exposed key.
