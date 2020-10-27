set term png small size 800,600
set output "mem_log.png"

set ylabel "VSZ"
set y2label "%MEM"

set ytics nomirror
set y2tics nomirror in

set yrange [0:*]
set y2range [0:*]

plot "mem.log" using 3 with lines axes x1y1 title "VSZ", \
     "mem.log" using 2 with lines axes x1y2 title "%MEM"