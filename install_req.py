import sys
import subprocess
import pkg_resources # usado para verificar pacotes já instalados no python

class install_missing_pkg:
    ### https://stackoverflow.com/questions/48097428/how-to-check-and-install-missing-modules-in-python-at-time-of-execution

    #https://stackoverflow.com/questions/12332975/installing-python-module-within-code
    '''
        if __name__ == "__main__":
            main()
    '''

    def __init__(self):

        self.main()

    def install_required_packages(self, missing_pkg):

        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_pkg]) # o * desempacota a lista missing_pkg

    def main(self):

        install_pkg = True

        if install_pkg:
            required_pkg_list = {'numpy',
                                 'scipy',
                                 'pandas',
                                 'matplotlib',
                                 'seaborn',
                                 'plotly',
                                 'scikit-learn',
                                 'tensorflow',
                                 'kaleido'}

        installed = {pkg.key for pkg in pkg_resources.working_set}

        missing_pkg = required_pkg_list-installed

        if missing_pkg:

            print('\nSome packages are not installed yet!!')
            print('------------------------------------------\n')
            print('Installing...\n')

            self.install_required_packages(missing_pkg)

            print('\nMissing packages are now installed and ready to use')
            print('--------------------------------------------------------\n')




