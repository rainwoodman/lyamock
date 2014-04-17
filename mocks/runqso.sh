for i in $*; do
qsub -q bigmem -l nodes=1:ppn=64 -j eo -N run$i <<EOF
    cd \$PBS_O_WORKDIR
    cd $i
    rm datadir/QSOCatelog.raw
    rm datadir/bootstrap.npz
    python /home/yfeng1/physics/lyamock/main/sightlines.py paramfile
#    python /home/yfeng1/physics/lyamock/main/gaussian.py paramfile
#    python /home/yfeng1/physics/lyamock/main/matchmeanF.py paramfile
#    python /home/yfeng1/physics/lyamock/main/spectra.py paramfile
#    python /home/yfeng1/physics/lyamock/main/measuremeanF.py paramfile
#    python /home/yfeng1/physics/lyamock/fit/pixelcorr.py paramfile
    python /home/yfeng1/physics/lyamock/fit/bootstrap.py paramfile
EOF
done
