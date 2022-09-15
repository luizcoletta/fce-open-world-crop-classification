import subprocess
import shlex
import os
#import pip


processes = ([
    'dp_ceratocystis1',
    'dp_ceratocystis2',
    'dp_ceratocystis5',
    'dp_ceratocystis10',
    'dp_ceratocystis20'

])

#https://stackoverflow.com/questions/16044612/converting-a-1-2-3-4-to-a-float-or-int-in-python
#https://stackoverflow.com/questions/24528464/converting-a-string-of-the-form-a-b-c-to-a-list-in-python-without-going-th

#Parametros
sel_list = " -selection EDS silhueta0 silhueta1 entropia"
classifiers = " -classifiers IC_EDS SVM SVM SVM"
insert_nc= " -insert_nc 2 3" # [iter nc]

procs = []
for pname in processes:

    command = shlex.split("python3 main.py -dataset " + pname + classifiers + insert_nc + sel_list)
    logfile = 'logs/' + "exp_"+pname+ '.log'
    with open(logfile, 'w') as f:

        proc = subprocess.Popen(command, stdout=f)
        procs.append(proc)

for proc in procs:
    proc.wait()


command = shlex.split("python3 plot_final_results.py -datasets dp_ceratocystis1 dp_ceratocystis2 dp_ceratocystis5 dp_ceratocystis10 dp_ceratocystis20"
                      + sel_list +
                      " -language pt")
pro = subprocess.Popen(command)
pro.wait()