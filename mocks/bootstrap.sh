for i in $*; do
qsub -q physics -l nodes=1:ppn=16 -j eo -N BS$i <<EOF
    cd \$PBS_O_WORKDIR
    cd $i
#    python /home/yfeng1/physics/lyamock/fit/pixelcorr.py paramfile
#    python /home/yfeng1/physics/lyamock/fit/pixelcorr2d.py paramfile
    python /home/yfeng1/physics/lyamock/fit/bootstrap.py paramfile
EOF
done
