# fce-open-world-crop-classification

Python code for detecting and learning new classes of threats present in crops

In order to run the experiments carried out over the Ceratocystis wilt dataset,
first install the repo:


```
cd ~
git clone https://github.com/luizcoletta/fce-open-world-crop-classification.git
```

Then execute the file `run_exps.py`:

```
cd ~/fce-open-world-crop-classification
python3 run_exps.py
```

The outcomes produced are stored in `results` folder, where each subfolder 
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
These files are stored in `logs` folder according to the nomenclature pattern `exp_{dataset}.log`:

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
    latent_dim -> amount of latent variables used to represent the intances (length of the feature vector)
    classifiers -> name of the classifier that it will be used
    selection -> criterion's name applied to find new classes
    insert_nc ([iter, nc])-> defines the iteraction (iter) where a new class with label 'nc' appears in the test set 
    iteractions -> number of iteractions executed by the framework
    language -> defines the language used in graphics: 'pt' for portuguese or 'en' for english
```

In the file `run_exps.py` these parameters are specified as arguments from the ArgParse Package, so they are defined through
 command lines. Example:

```
dp = (['dp_ceratocystis1','dp_ceratocystis2','dp_ceratocystis5','dp_ceratocystis10','dp_ceratocystis20'])

dp_parameters = " -selection  silhouette silh_mod entropy random "\
                "-classifiers iCaRL iCaRL iCaRL iCaRL "\
                "-insert_nc 2 3"
datasets.append([dp, dp_parameters])    

```

The variable `dp` refers to the feature space where one or more datasets lies to, while at same time, its content defines the 
parameter `dataset`. The variable `dp_parameters` can be used to define others parameters seen previously.

So, the experiments settings are stablished following the pattern:

```
{feature space} = (['{dataset-1}','{dataset-2}','{dataset-n}'])

{feature space}_parameters = " -selection  {criterion-1} {criterion-2} {criterion-n} "\
                "-classifiers {classifier-1} {classifier-2} {classifier-n} "\
                "-insert_nc {iter} {nc} "\
                "-{other_parameters}"
datasets.append([{feature space}, {feature space}_parameters])    

```

When the framework is executed, the `{classifier-1}` is used along with `{criterion-1}`, `{classifier-n}` is applied with 
`{criterion-n}` and so on.

In this implementation the following classifiers and criteria are available:

```
Classifiers: 
   -NNO
   -iCaRL 
   -SVM
   -IC-EDS

Criteria: 
   -prob_class_desc (for NNO)
   -silhouette (or silhueta) 
   -silh_mod 
   -entropy (or entropia) 
   -random (or aleatória)
   -EDS (for IC-EDS)
```








