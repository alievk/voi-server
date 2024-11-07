#!/bin/bash

# this is a bad idea to pull deps from packages like TTS
# because it causes conflicts with other packages
# so we manually install the deps in requirements.txt
# and then install the rest of the packages in requirements_nodep.txt

pip install -U pip
pip install -r requirements.txt
pip install --no-deps -r requirements_nodep.txt