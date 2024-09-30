#!/bin/bash

/usr/sbin/sshd -D &

runuser -l user -c "jupyter-lab --ip=0.0.0.0 --port=8500 --no-browser --notebook-dir=/home/user/sources"
