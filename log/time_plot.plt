set terminal pdfcairo enhanced dashed size 6,4
#set grid nopolar

set output 'time_log.pdf'
set key width -6 inside right top vertical Right noreverse enhanced autotitles box linetype -1 linewidth 1.000

#set title ""
set xlabel "frame number"
set ylabel "runtime [ms]"
#set xrange [0:1]
set yrange [0:1900]
#set xtics 0.1
#set ytics 0.1
#set ytics ("120" 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000)

plot 'orf_time_log' using 1:($2*1000.0) title "online random forests (Ours): 25.0 ms +/- 8.4 ms" with lines lt 1 lc rgb "orange" lw 2,\
     'svm_time_log' using 1:($2*1000.0) title "online SVM (Yan'18): 35.2 ms +/- 7.4 ms" with lines lt 1 lc rgb "green" lw 2

