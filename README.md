# fce-open-world-crop-classification

Python code for detecting and learning new classes of threats present in crops

In order to run the experiments carried out over the Ceratocystis wilt dataset,
first install the repo:


```
cd ~
git clone https://github.com/luizcoletta/fce-open-world-crop-classification.git
```

Then execute the file run_exps.py:

```
cd ~/fce-open-world-crop-classification
python3 run_exps.py
```

The outcomes produced are stored in 'results' folder, where each subfolder 
refers to a dataset:

```
~/fce-open-world-crop-classification/
      |----results/
          |----dp_ceratocystis1/
          |----dp_ceratocystis2/
          |----dp_ceratocystis5/
          |----dp_ceratocystis10/
          |----dp_ceratocystis20/
          |----hc_ceratocystis1/
          |----hc_ceratocystis2/
          |----hc_ceratocystis5/
          |----hc_ceratocystis10/
          |----hc_ceratocystis20/
          |----vae_ceratocystis1/
          |----vae_ceratocystis2/
          |----vae_ceratocystis5/
          |----vae_ceratocystis10/
          |----vae_ceratocystis20/
```

Log files are yielded as well, showing the experiment's behavior through the iterations.
These files are stored in 'logs' folder according to the nomenclature pattern exp_{dataset}.log:

```
~/fce-open-world-crop-classification/
      |----logs/
          |----exp_dp_ceratocystis1.log
          |----exp_dp_ceratocystis2.log
          |----exp_dp_ceratocystis5.log
          |----exp_dp_ceratocystis10.log
          |----exp_dp_ceratocystis20.log
          |----exp_hc_ceratocystis1.log
          |----exp_hc_ceratocystis2.log
          |----exp_hc_ceratocystis5.log
          |----exp_hc_ceratocystis10.log
          |----exp_hc_ceratocystis20.log
          |----exp_vae_ceratocystis1.log
          |----exp_vae_ceratocystis2.log
          |----exp_vae_ceratocystis5.log
          |----exp_vae_ceratocystis10.log
          |----exp_vae_ceratocystis20.log

```

## Experiments Settings

When performing experiments in this code, some parameters can be defined by the user in order to define the conditions desired.


```
    Parameters: 
    
    n_classes -> Number of classes in test set
    dataset -> dataset name
    use_vae (True/False) -> enables the use of VAE to extract features if True    
    vae_epochs -> amount of epochs for VAE
    latent_dim -> amount of latent variables used to represent the intances (lenght of the feature vector)
    classifiers -> name of the classifier that it will be used
    selection -> criterion's name applied to find new classes
    insert_nc ([iter, nc])-> defines the iteraction (iter) where a new class (nc) appear in the test set 
    iteractions -> number of iteractions executed by the framework
    language -> defines the language used in graphics: 'pt' for portuguese or 'en' for english
```



