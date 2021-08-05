import pip # installing required packages requires pip installation ;-)


def install(package):
    pip.main(['install', package])


### https://stackoverflow.com/questions/48097428/how-to-check-and-install-missing-modules-in-python-at-time-of-execution
def install_required_packages(modules_to_try):
    for module in modules_to_try:
        try:
            __import__(module[0])
            print(">> " + module[0] + " already installed!")
        except ImportError as e:
            #install(e.name)
            install(module[1])


def main():
    install_pkg = True
    
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
                     ('tensorflow', 'tensorflow'),
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
    install_required_packages(required_pkg_list)


if __name__ == "__main__":
    main()
