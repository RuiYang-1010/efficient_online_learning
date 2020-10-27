#!/bin/bash

while true; do
    ps -C $1 -o pid=,%mem=,vsz= >> mem.log
    gnuplot mem_plot.plt
    sleep 1
done
