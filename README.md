# astropy_LMC
Please install the required dependencies by running
```
pip install -r requirements.txt
``` 
in the terminal.

Please copy the 'LMC' directory into the folder and ensure it has the following structure
```
astropy_LMC
|--- LMC
|    |--- lmc_askap.fits
|    |--- lmc_ha_csub.fits
|    |--- HII_boundaries
|    |    |--- mcels-l001.reg
|    |    |--- mcels-l002.reg
|    |    |--- mcels-l003.reg       
|    |--- SNR_boundaries
|    |    |--- 0449-6903.reg
|    |    |--- 0449-6920.reg
|    |    |--- 0450-7050.reg
```

When creating a new dataset called `dataset_name`, it will be saved in the DATASET directory with the following structure
```
astropy_LMC
|--- DATASET
|    |--- dataset_name
|    |    |--- data
│    |    |    |--- my_dataset
│    |    |    |    |--- img
│    |    |    |    |    |--- train
│    |    |    |    |    |    |--- xxx{img_suffix}
│    |    |    |    |    |    |--- yyy{img_suffix}
│    |    |    |    |    |    |--- zzz{img_suffix}
│    |    |    |    |    |--- val
│    |    |    |    |--- ann
│    |    |    |    |    |--- train
│    |    |    |    |    |    |--- xxx{ann_suffix}
│    |    |    |    |    |    |--- yyy{ann_suffix}
│    |    |    |    |    |    |--- zzz{ann_suffix}
│    |    |    |    |    |--- val
```

## First run...
The first time you run the jupyter notebook to create a dataset, the program will create the annotation array (~10 min) and will save it in the directory for future use. The next time you run the program it will load the saved array and will generate the dataset much faster. It will also save a png image of this array for viewing.

Additionally, this initial run will create the DATASET directory. 
