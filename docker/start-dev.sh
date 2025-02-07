#!/bin/bash

sudo /usr/sbin/sshd -D &

if [ -z "${JUPYTER_TOKEN}" ]; then
    echo "JUPYTER_TOKEN environment variable is not set"
    tail -f /dev/null
else
    jupyter-lab --ip=0.0.0.0 --port=8500 --no-browser --notebook-dir=/home/user --ServerApp.token=${JUPYTER_TOKEN}
fi
