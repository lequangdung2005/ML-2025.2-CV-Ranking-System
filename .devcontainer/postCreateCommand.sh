#!/usr/bin/env bash

# Install Dependencies
poetry lock
poetry install

# # Install pre-commit hooks
poetry run pre-commit install --install-hooks

