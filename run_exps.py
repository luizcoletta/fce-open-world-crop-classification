import subprocess
import os
import pip

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
processes = ([
    "main.py",
    #"exp2_training_unet_mini.py",
    #"exp3_training_unet.py",
    #"exp4_training_resnet50_unet.py",
    #"exp5_training_mobilenet_unet.py",
    #"exp6_training_segnet.py",
    #"exp7_training_vgg_segnet.py",
    #"exp8_training_resnet50_segnet.py",
    #"exp9_training_mobilenet_segnet.py"
])

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

procs = []
for pname in processes:
    logfile = 'logs/' + os.path.splitext(pname)[0] + '.log'
    with open(logfile, 'w') as f:
        proc = subprocess.Popen(['python3', pname], stdout=f)
        procs.append(proc)

for proc in procs:
    proc.wait()
