d
#$ -j y
#$ -S /bin/bash
#$ -q amd2.q
#$ -p 0
#
export GUROBI_HOME=/share/apps/gurobi/current/linux64
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
export GRB_LICENSE_FILE=$GUROBI_HOME/../gurobi.lic
echo $ARG1
date
/share/apps/python/2.7/bin/python qtm_solve.py $ARG1
date
