#!/usr/bin/env bash

source ~/.bashrc

export VIRTUALENVWRAPPER_PYTHON=$(which python3.8)
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
source /home/ohad/.local/bin/virtualenvwrapper.sh

workon ohad

git fetch
git reset --hard origin/main

python ../src/attribute_filler.py