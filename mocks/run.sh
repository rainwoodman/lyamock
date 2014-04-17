for i in $*; do
mkdir $i
rm -rf $i/datadir
mkdir $i/datadir
sed -e "s;@SEED@;$RANDOM$RANDOM;" paramfile.temp > $i/paramfile
qsub -q physics -l nodes=1:ppn=16 -j eo -N run$i <<EOF
    cd \$PBS_O_WORKDIR
    cd $i
    python /home/yfeng1/physics/lyamock/main/sightlines.py paramfile
    python /home/yfeng1/physics/lyamock/main/gaussian.py paramfile
    python /home/yfeng1/physics/lyamock/main/matchmeanF.py paramfile
    python /home/yfeng1/physics/lyamock/main/spectra.py paramfile
    python /home/yfeng1/physics/lyamock/main/measuremeanF.py paramfile
#    python /home/yfeng1/physics/lyamock/fit/pixelcorr.py paramfile
    python /home/yfeng1/physics/lyamock/fit/bootstrap.py paramfile
EOF
done
