#!/bin/bash

sudo /usr/sbin/sshd -D &

cd /home/user/sources/voice-agent-core && python3 ws_server.py