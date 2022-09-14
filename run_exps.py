import subprocess
import shlex
import os
#import pip


'''
def install(package):
    pip.main(['install', package])


### https://stackoverflow.com/questions/48097428/how-to-check-and-install-missing-modules-in-python-at-time-of-execution
def install_all_packages(modules_to_try):
    for module in modules_to_try:
        try:
            __import__(module[0])
            print(">> " + module[0] + " already installed!")
        except ImportError as e:
            #install(e.name)
            install(module[1])


install_pkg = False
'''

processes = ([
    'dp_ceratocystis1',
    'dp_ceratocystis2'


    #"plot_final_results.py"



])

sel_list = " -selection silhueta0 silhueta1 entropia"
classifiers = " -classifiers svm svm svm"
insert_nc= " -insert_nc 2 3"

'''
if (install_pkg):
    required_pkg_list = [('numpy', 'numpy'),
                     ('scipy', 'scipy'),
                     ('pandas', 'pandas'),
                     ('matplotlib', 'matplotlib'),
                     ('seaborn', 'seaborn'),
                     ('plotly', 'plotly'),
                     ('sklearn', 'sklearn'),
                     ('cv2', 'opencv-python'),
                     ('PIL', 'Pillow'),
                     ('tensorflow', 'tensorflow>=2.4.1'),
                     ('keras', 'keras'),
                     ('glob', 'glob'),
                     ('random', 'random'),
                     ('json', 'json'),
                     ('os', 'os'),
                     ('six', 'six'),
                     ('sys', 'sys'),
                     ('argparse', 'argparse'),
                     ('subprocess', 'subprocess'),
                     ('skbuild', 'scikit-build'),
                     ('cmake', 'cmake'),
                     ('pylab', 'pylab'),
                     ('keras_segmentation', 'git+https://github.com/luizfsc/ext-semantic-segmentation')] # https://github.com/divamgupta/image-segmentation-keras
    install_all_packages(required_pkg_list)
'''


procs = []
for pname in processes:

    command = shlex.split("python3 main.py -dataset " + pname + classifiers + insert_nc + sel_list)
    logfile = 'logs/' + "exp_"+pname+ '.log'
    with open(logfile, 'w') as f:

        proc = subprocess.Popen(command, stdout=f)
        procs.append(proc)

for proc in procs:
    proc.wait()


command = shlex.split("python3 plot_final_results.py -datasets dp_ceratocystis1 dp_ceratocystis2"
                      + sel_list+
                      " -language pt")
pro = subprocess.Popen(command)
pro.wait()