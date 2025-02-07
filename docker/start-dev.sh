#!/bin/bash

sudo /usr/sbin/sshd -D &

jupyter-lab --ip=0.0.0.0 --port=8500 --no-browser --notebook-dir=/home/user
