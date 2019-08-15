#! /bin/bash -e
CWD=$(dirname $(readlink -f "$0"))
rm -rf "${CWD}/test_1_PRUSVI"
OUTPUT_DIR="${CWD}/test_1_PRUSVI/jigsaw"
mkdir -p $OUTPUT_DIR
FORT14=PRUSVI_Jigsaw_fort.14
STATIONS_FILE=/ddnas/jreniel/ADCIRC/HSOFS/Sandy2012/postSandyUpdate/control/hotstart/fort.15
export OUTPUT_FORT14="${OUTPUT_DIR}/${FORT14}"
mkdir -p $OUTPUT_DIR
source ${OUTPUT_DIR}/../../../.geomesh_env/bin/activate
nosetests test_Jigsaw.py
cd "$OUTPUT_DIR"
rm -rf coldstart
rm -rf hotstart
GenerateTidalRun "${OUTPUT_FORT14}" \
				  $(date +"%Y-%m-%dT%H:%M" -d "30 days ago") \
				  7 30 "${OUTPUT_DIR}" \
				  --stations-file=${STATIONS_FILE} \
				  --eso-sampling-frequency=6 \
				  --eso-harmonic-analysis \
				  --eso-coldstart \
				  --ego-sampling-frequency=0.02 \
				  --ego-coldstart 
# ssh -T teslak80 'rm -rf ~/geomesh_tests/jigsaw/PRUSVI'
ssh -T teslak80 'mkdir -p ~/geomesh_tests/jigsaw/PRUSVI'
rsync -zah --partial --progress "${OUTPUT_DIR}/" "teslak80:~/geomesh_tests/jigsaw/PRUSVI/"
ssh -T teslak80 "ln -f ~/geomesh_tests/jigsaw/PRUSVI/${FORT14} ~/geomesh_tests/jigsaw/PRUSVI/fort.14"
ssh -T teslak80 << EOT
source ~/.local/opt/environment.sh
cd ~/geomesh_tests/jigsaw/PRUSVI
mkdir coldstart
cd coldstart
ln -sf ../fort.14 ./fort.14
ln -sf ../fort.13 ./fort.13
ln -sf ../fort.15.coldstart ./fort.15
adcprep --np 47 --partmesh
adcprep --np 47 --prepall
mpiexec -np 48 padcirc -W 1
# mkdir ../hotstart
# cd ../hotstart
# ln -sf ../fort.14
# ln -sf ../fort.13
# ln -sf ../fort.15.hotstart ./fort.15
# ln -sf ../coldstart/fort.67.nc ./
# adcprep --np 47 --partmesh 
# adcprep --np 47 --prepall
# mpiexec -np 48 padcirc -W 1
EOT
ssh -T teslak80 "rm -rf ~/geomesh_tests/jigsaw/PRUSVI/fort.14"
ssh -T teslak80 "rm -rf ~/geomesh_tests/jigsaw/PRUSVI/coldstart/PE*"
rsync -zah --no-links --partial --progress "teslak80:~/geomesh_tests/jigsaw/PRUSVI/" "${OUTPUT_DIR}/"
PlotMaxele "${OUTPUT_DIR}/coldstart/maxele.63.nc"
