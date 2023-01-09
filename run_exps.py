import subprocess
import shlex
import os
#import pip
#https://stackoverflow.com/questions/16044612/converting-a-1-2-3-4-to-a-float-or-int-in-python
#https://stackoverflow.com/questions/24528464/converting-a-string-of-the-form-a-b-c-to-a-list-in-python-without-going-th

datasets = []
#-----------------------------------------------
# deep features
dp = (['dp_ceratocystis1','dp_ceratocystis2','dp_ceratocystis5','dp_ceratocystis10','dp_ceratocystis20'])
#dp = (['dp_ceratocystis1'])



#Parametros
dp_parameters = " -selection prob_class_desc silh_inc " \
                "-classifiers NNO incremental " \
                "-insert_nc 2 3" # [iter nc]


#-----------------------------------------------
# hand craft features
hc = (['hc_ceratocystis1','hc_ceratocystis2','hc_ceratocystis5','hc_ceratocystis10','hc_ceratocystis20'])

#Parametros
hc_parameters = " -selection prob_class_desc silh_inc " \
           "-classifiers NNO incremental " \
           "-insert_nc 2 3" # [iter nc]
#-----------------------------------------------

# VAE features
vae = (['vae_ceratocystis1','vae_ceratocystis2','vae_ceratocystis5','vae_ceratocystis10','vae_ceratocystis20'])

#Parametros
vae_parameters = " -selection prob_class_desc silh_inc  "\
                "-classifiers NNO incremental " \
                 "-use_vae True " \
            "-latent_dim 8 "\
                 "-insert_nc 2 3" # [iter nc]
#-----------------------------------------------

mnist = (['mnist2D'])

#Parametros
mnist_parameters = " -selection entropia "\
                "-classifiers SVM " \
                 "-insert_nc 2 3" # [iter nc]
#-----------------------------------------------

datasets.append([dp, dp_parameters])
datasets.append([hc, hc_parameters])
datasets.append([vae, vae_parameters])


for data,params in datasets:

    procs = []

    for pname in data:

        command = shlex.split("python3 main.py -dataset " + pname + params)
        logfile = 'logs/' + "exp_"+pname+ '.log'
        with open(logfile, 'w') as f:

            proc = subprocess.Popen(command, stdout=f)
            procs.append(proc)

    for proc in procs:
        proc.wait()



    command = shlex.split("python3 plot_final_results.py -datasets "
                          + str(data).strip('[]').replace(',',' ')
                          + " -" + params.split('-')[1]
                          + "-" + params.split('-')[2] +
                          "-language pt")
    pro = subprocess.Popen(command)
    pro.wait()

    command = shlex.split("python3 summary_table.py -datasets "
                          + str(data).strip('[]').replace(',', ' ')
                          + " -" + params.split('-')[1]
                          + "-" + params.split('-')[2] +
                          "-language pt")
    pro = subprocess.Popen(command)
    pro.wait()


