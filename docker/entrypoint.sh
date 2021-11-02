#!/bin/bash

redis-server &> /dev/null &

export HOME=/home

exec /opt/merlin/entrypoint.sh "$@"
