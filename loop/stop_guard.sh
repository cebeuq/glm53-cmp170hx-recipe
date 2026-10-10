#!/usr/bin/env bash
# Stop the guard without matching ssh/bash -c command lines (anchored to argv[0..1] == "bash loop/guard.sh").
pids=$(pgrep -f '^bash loop/guard\.sh'); [ -n "$pids" ] && kill $pids
pgrep -f '^journalctl -kf -n0 -o short-iso' | xargs -r kill
