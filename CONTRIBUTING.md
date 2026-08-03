# Contributing to rimeX

## Development setup

See the [Install](README.md#install) section of the README for how to set up
a development environment (editable pip install, optional CDO dependency,
conda alternative).

## Reporting issues and opening pull requests

- Report bugs or request features via [GitHub Issues](https://github.com/iiasa/rimeX/issues).
- For pull requests: fork the repo, create a branch off `main`, and open a PR
  describing what changed and why. Keep PRs focused — separate unrelated
  changes into their own PRs where practical.
- Please check that existing scripts/tests still run before opening a PR.

## Licensing of contributions

rimeX is licensed under AGPL-3.0-or-later and is also made available under a
separate commercial licence.

By submitting a contribution, you confirm that you have the right to do so, and
you agree that your contribution is licensed under AGPL-3.0-or-later and may
also be included in versions of rimeX distributed under IIASA's commercial
licence terms.

If you are contributing on behalf of an employer, please ensure you have the
necessary permission.

## SPDX headers

All source files in this repository carry an
`# SPDX-License-Identifier: AGPL-3.0-or-later` header. New files should too —
see `scripts/add_spdx_headers.py` for the tool used to add these.
