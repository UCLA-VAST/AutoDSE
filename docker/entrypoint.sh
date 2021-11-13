#!/bin/bash

# prevent SIGINT from interrupting redis
setsid redis-server >/dev/null 2>&1 &

export HOME=/home

exec /opt/merlin/entrypoint.sh "$@"
