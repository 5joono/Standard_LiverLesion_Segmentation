# 2D Liver/Lesion Segmentation Pipeline
### ___Authors___: [_Karsten Roth_](karsten.rh1@gmail.com), [_Tomasz Konopczynski_](Tomasz.Konopczynski@medma.uni-heidelberg.de)

---
## Introduction
This repository contains the full pipeline to train a liver- and lesionsegmentation network on the basis of a standard [UNet](https://arxiv.org/abs/1505.04597) on the [LiTS-dataset](https://competitions.codalab.org/competitions/17094).
The network setup is done in a modular fashion to allow for, within a defined range, arbitrary network architectures.

Using the UNet as a baseline, one can vary architectural parameters s.a. depth and filtersizes, as well as inclusions of extending elements
like [skip-connections](https://arxiv.org/abs/1512.03385), a dense setup inspired by [DenseNet](https://arxiv.org/abs/1608.06993)/[Tiramisu](https://arxiv.org/abs/1611.09326), although using summation instead of concatenation, layer stage output concatenation similar to https://arxiv.org/pdf/1802.10508.pdf,
the inclusion of auxiliary inputs (_i.e. per UNet-stage, add an auxiliary loss based on a downscaled version of the target image that needs to be recreated using that stages features_), [Squeeze-and-Excitation Modules](https://arxiv.org/abs/1709.01507) and a convolutional replacement for maxpooling is provided.

Finally, one can fill the channel axis with volume frames
surrounding the input slice to incorporate semi-3D information, as suggested by [Han](https://arxiv.org/abs/1704.07239).
For big network architectures batch normalization may be set to `False` to use [GroupNorm](https://arxiv.org/abs/1803.08494) with smaller batchsizes instead, while getting similar results.

In addition to that, training uses either dice or crossentropy loss, where for the latter precomputed weightmaps emphasising lesion/liver(boundary)pixels are used.  
Both loss functions can also be extended to add a focal loss term taken and adapted from [Ecoffet](https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c).

To start and as a baseline, training setups for a basic UNet structure as well as an advanced setup are given, where for the latter all extension were included.  

_An exemplary architecture sketch_:

![Exemplary Architecture Styles](/Images/ExUNet.png)

To run training on these from scratch, starting at the data download, follow the instructions listed below.

Running the provided pipeline yiels (without ensembling, single 2D-based networks):

Architecture | Test Dice/Case Lesion | Test Dice Global Lesion | Test Dice/Case Liver | Test Dice Global Liver
-------------|-----------------------|-------------------------|----------------------|-----------------------
2D Easy UNet |     0.6240            |        0.7930           |        0.9500        |      0.9540
2D Hard UNet |     0.6540            |        0.7930           |        0.9390        |      0.9510

---

## Folder Structure
```
Repository Standard_LiverLesion_Segmentation
│   README.md
│   requirements.txt    
│   installer.sh
|
|
|
└───Train_Networks (scripts for network training)
│   │   Train_Networks.py
│   │   Function_Library.py
│   │
│   └───Training_Setup_Files (training and network parameters)
│       │   Baseline_Parameters.txt
│       │   Small_UNet_Liver.txt
│       │   Small_UNet_Lesion.txt
│       │   Big_UNet_Liver.txt
│       │   Big_UNet_Lesion.txt
│   
└───Data_Preparation (scripts to convert volumetric format to slice format)
|   │   Convert_NibVolumes_to_Slices.py
│   
└───Network_Zoo (network library)
|   │   network_zoo.py
│   
└───Network_Graph_Visualization (function to visualize computational graph)
|   │   visualize.py
│   
└───Make_Test_Submission (generate test liver and lesion masks)
|    │   generate_liver_masks.py
|    │   make_test_submission.py
│   
└───Utilities (utility functions)
|    │   General_Utilities.py
|    │   Network_Utilities.py
|    │   PyTorch_Datasets.py
│   
└───Images (Readme images)
|
│   
|
└───[will be generated by Conversion Script] LOADDATA (Savefolder for 2D Slice Data)
│   
└───[will be generated by Training Scripts]  SAVEDATA (Savefolder for network weights)
│   
└───[generated by hand] OriginalData (Savefolder for downloaded .nii-files)

```
---


## QUICK USAGE
### [1.] Requirements
This repository is build around `Python3` and `PyTorch 0.4.1` (tested with `cuda 8`).   
For ease of use, simply set up a conda environment (requiring at least miniconda installation from https://conda.io/miniconda.html) via   
`conda create -n <Env_Name> python=3.6` and, after doing  
`source activate <Env_Name>`, running     
`bash installer.sh`.

### [2.] Download Data and Setting Correct Folders
If a CodaLab-Account is available, LiTS Training and TestData may be [downloaded](https://competitions.codalab.org/competitions/17094#participate) to a folder of choice.   
___Note: The data requires roughly 60-70GB of storage.___

Once this is done, place the data in a folder structure with `OriginalData` as parent:  
Test files to `<path>/OriginalData/Test_Data` and training files to `<path>/OriginalData/Training_Data`.   
The former contains all test-volumes in `.nii`-format, the latter houses both all `segmentation-X.nii`-files as well as all the `volume-X.nii`-files.   

In the case of other datasets, please set it up in the same way, i.e. mask data as `segmentation-X.nii` and input volumes as `volume-X.nii` in the same folder structure.

Overall, it should look something like:

```
Repository Standard_LiverLesion_Segmentation (or any other save folder)
|
└───OriginalData
│   │
│   └───Training_Data
│   |   │   segmentation-0.nii
│   |   │   segmentation-1.nii
│   |   │   ...
│   |   │   volume-0.nii
│   |   │   volume-1.nii
│   |   │   ...
│   |
│   └───Test_Data
│       │   test-volume-0.nii
│       │   test-volume-1.nii
│       │   ...
```

__IMPORTANT__:  
If you don't want to declare custom paths throughout, save the `OriginalData`-folder to this repository under `/Standard_LiverLesion_Segmentation/OriginalData` (_see above_). If so, resort to the __(Default)__-segments in the detailed explanation or equivalently __step 3__ in this Quick Guide.   
This will also save everything else to this repository. Overall, this can require up to `150GB` of disk space.  
Otherwise, execute the __(Custom)__-segments (see detailed explanation).


### [3.] Preparation, Training and Testing

To use provided network architectures (`Small_UNet_Liver/Lesion.txt` and `Big_UNet_Liver/Lesion.txt` for small and big UNet-Variant respectively) for training(with __preset hyperparameters__) and letting the script set save paths within this repository, do:

#### (I.) __Prepare Dataset__:  

___WARNING___: _THIS GENERATES DATA IN .npy-FORMAT AND REQUIRES UP TO AN ADDITIONAL 80GB OF STORAGE!_
```
python Data_Preparation/Convert_NibVolumes_to_Slices3D.py
python Data_Preparation/Convert_NibVolumes_to_Slices3D.py --is_test_data
```
This adds another two folder in the repository: `LOADDATA/Training_Data_2D` and `LOADDATA/Test_Data_2D`. Each folder has layout `LOADDATA/Training_Data_2D/volume-X/slice-Y.npy`.  
Do note that within the `Data_Preparation`-folder, there is a `Convert_NibVolumes_to_Slices.py`-file if one wishes to generate weightmaps on a 2D slice-by-slice basis. The _3D_ version generates weightmaps with regards to the full 3D volume.

#### (II.) __Train Liver Segmentation Network__:

___NOTE___:  
_For Training, the data in_ `LOADDATA/Training_Data_2D` _is split into a training and validation set in_ `90% to 10%` _relation. The exact assignment is dependent on the pipeline seed.  
If one wishes to adjust this (division/seed) or the_ `gpu-id`_, add in_ `/Train_Networks/Training_Setup_Files/<Small/Big_UNet_Liver/Lesion.txt>` the lines (under `##### Training`):  

```
train_val_split: [percentage of choice, e.g. 0.9 for default]
seed: [int of choice, 1 for default]
gpu:  [int of choice, 1 for default]
```

__MOVING ON:__

_For the small UNet:_
```
python Train_Networks/Train_Networks.py --search_setup Small_UNet_Liver.txt --no_date
```
_For the Big UNet:_
```
python Train_Networks/Train_Networks.py --search_setup Big_UNet_Liver.txt --no_date
```

#### (III.) __Train Lesion Segmentation Network__:
___NOTE:___ _The standard setup uses the network weights used for liver segmentation as __initialization__ to speed up training! If you want to run without initialization, please go to the respective text files and comment out the `initialization: [...]`-line with `%`._

_For the small UNet:_
```
python Train_Networks/Train_Networks.py --search_setup Small_UNet_Lesion.txt --no_date
```
_For the big UNet:_
```
python Train_Networks/Train_Networks.py --search_setup Big_UNet_Lesion.txt --no_date
```

#### (IV.) __Generate Liver Masks__:
___NOTE:___ _The setup shown uses a single network for Test-Set Segmentation. If one wishes to use multiple networks, simply set the flag `--use_all`. This will make use of all the networks stored in `SAVEDATA/Standard_Liver_Networks`.

_For the small UNet:_
```
python Make_Test_Submission/generate_liver_masks.py --network_choice vUnet2D_liver_small
```
_For the big UNet:_
```
python Make_Test_Submission/generate_liver_masks.py --network_choice vUnet2D_liver_big
```

#### (V.) __Generate Lesion Masks and Submission Files__:
___NOTE:___ _The setup shown uses a single network for Test-Set Segmentation. If one wishes to use multiple networks, simply set the flag `--use_all`. This will make use of all the networks stored in `SAVEDATA/Standard_Lesion_Networks`. In addition, running lesion segmentation requires liver segmentation to be completed beforehand.

_For the small UNet:_
```
python Make_Test_Submission/make_test_submission.py --network_choice vUnet2D_lesion_small
```
_For the big UNet:_
```
python Make_Test_Submission/make_test_submission.py --network_choice vUnet2D_lesion_big
```


__Zip the resulting files that were saved in `Test_Segmentations/Test_Submissions` and upload.__



---


## Detailed Explanation
For every following script execution, we distinguish between __(Default)__-setup, i.e. letting the script set all paths, and __(Custom)__-setup, allowing to set all paths and parameters manually.

### [1.] Preparation of the dataset
We wish to convert the volumetric data to a slice-wise format to save RAM during training as well as making data shuffling easier. From the folder `Data_Preparation`, execute

* __(Custom)__

```
python Data_Preparation/Convert_NibVolumes_to_Slices.py --path_2_training_volumes <path>/OriginalData/Training_Data --save_path_4_training_slices <where_to_save_2D_training_data>
```

* __(Default)__

```
python Data_Preparation/Convert_NibVolumes_to_Slices.py
```

Note that on an average SSD this takes roughly one hour.

To generate the test data in correct format, run the same lines but setting the flag `--is_test_data`.  
In the __(Custom)__-case, replace `<path>/OriginalData/Training_Data` with `<path>/OriginalData/Test_Data` and
`--save_path_4_training_slices <where_to_save_2D_test_data>`.

Note that during this run, weightmaps such as in the image below (_volume slice, liver mask, liver weightmask, lesion mask, lesion weightmask_)

![Example of liver/lesion weightmaps](/Images/weightmap_example.png)

will be generated both for liver and lesion masks. These will later be combined with crossentropy-based loss to place more weights on liver/lesion boundaries. Using this instead of standard cross entropy and even plain dice loss boosts segmentation performance.




### [2.] Training Liver and Lesion Segmentation Network

To set up the network architecture of choice, take a look at `Train_Networks/Training_Setup_Files/Baseline_Parameters.txt`, which contains all pipeline parameters that can be tweaked.  

These parameters are divided into three sections: __Training__ to contain all parameters relevant to train the network (e.g. opimization parameters), __Network__ containing the Architectural Parameters and __Paths__ containing all relevant paths. Each parameter is explained in more depth in this file, however some minor notes:
Use `#` to denote the generation of any kind of section (e.g. __Training__). Use `%` to make comments. Otherwise, the setup follows a general python dict layout.

In addition to the `Baseline_Parameters.txt`-file, an additional text file option is provided, namely `Train_Networks/Training_Setup_Files/Variation_Parameters.txt`. This allows to
easily run grid searches and to concat multiple runs in a clear way. By using the `=`-operator, one can separate different runs. If setting multiple elements in a list following a parameter, variations in this parameter will be used in the respective number of runs.


* ___Example___: Testing for various learning rates and batchsizes:  
In `Variation_Parameters.txt`, make a run section via `==== (optional description)`.   
Then, add the section and follow up with the parameters to change, i.e. in this case:
`lr: [0.001, 0.0001, 0.00001]` and `bs: [4,8,32]`.   
This will concat six different runs, each time varying one of the parameters.


If leaving these files as they are now, they either belong to the standard UNet training run (`Small_UNet_Liver/Lesion.txt`) or the advanced training run (`Big_UNet_Liver/Lesion.txt`).
The default training setup uses a `85% to 15%` split; where the network weights are saved whenever Validation Dice increases.

When training a custom Liver Segmentation network, remember to, within the __Training__-section, set `data: 'liver'` (`data: 'lesion'` for Lesion Segmentation respectively).


Finally, the complete approach would look something like (remember to adjust `Training_Path` and `Save_Path` for Liver/Lesion Segmentation accordingly in `Small/Big_UNet_Liver/Lesion.txt` if in __Custom__-mode):

* (__Liver Segmentation__) [executed within `Train_Networks`]

  ```
  python Train_Networks --search_setup Small/Big_UNet_Liver.txt
  ```

  Note that if the Liver Segmentation Architecture coincides with the Lesion Segmentation Architecture, one can add this as an initialization accordingly by setting the __Network__-section flag `initialization: 'path_to_saved_network_folder'`.

* (__Lesion Segmentation__) [executed within `Train_Networks`]

  ```
  python Train_Networks --search_setup Small/Big_UNet_Lesion.txt
  ```

_One obviously can create new setup files with different names. They just need to be passed to the python script._



### [3.] Generating Test Predictions & Uploading
Finally, to generate the submission masks to be uploaded, make use of the files in `/Make_Test_Submission`. First run

* __(Custom)__

  ```
  python Make_Test_Submission/generate_liver_masks.py --test_data <path_to_test_files> --network_choice <name_of_network_folder> --network_folder <path_to_network_to_use_for_liver_segmentation> --save_folder <where_to_save_liver_segmentation_masks>
  ```  

*  __(Default)__

  ```
  python Make_Test_Submission/generate_liver_masks.py --network_choice vUnet2D_liver_2d_unet_<easy/hard>_baseline
  ```

to generate the liver segmentation masks. This is done separately to the Lesion segmentation to allow for bigger networks requiring nearly full GPU Memory, as well as the fact that once a sufficient liver segmentation has been achieved, these masks can be used for other lesion segmentation setups.  


Following the Liver Segmentation, run Lesion Segmentation similarly by running

* __(Custom)__

  ```
  python Make_Test_Submission/make_test_submission.py --test_data <path_to_test_files> --network_choice <name_of_network_to_use> --network_folder <path_to_network_to_use> --path_2_liv_seg <path_to_precomputed_liver_masks> --save_folder <where to save predictions>
  ```


* __(Default)__

  ```
  python Make_Test_Submission/make_test_submission.py --network_choice vUnet2D_lesion_2d_unet_<easy/hard>baseline
  ```


This will output a folder with submission-ready `.nii`-files under `Test_Segmentations/Test_Submissions`, that only need to be zipped and uploaded to the [LiTS challenge website](https://competitions.codalab.org/competitions/17094#participate-submit_results).



### [4.] Results
Using either network, the following results can be achieved using an end-to-end, 2D approach:

Architecture | Test Dice/Case Lesion | Test Dice Global Lesion | Test Dice/Case Liver | Test Dice Global Liver
-------------|-----------------------|-------------------------|----------------------|-----------------------
2D Easy UNet |     0.6240            |        0.7930           |        0.9500        |      0.9540
2D Hard UNet |     0.6540            |        0.7930           |        0.9390        |      0.9510


---


## Additional Information


### Graph Visualisation
If designing a network architecture, it might prove useful to check the computational graph to see if it resembles the target structure. To do so, simply declare your network architecture in `Network_Graph_Visualization` and run
```
python visualize.py --search_setup <Network_Setup_of_Choice>
```   
Note that for this to work, `graphviz` needs to be installed, e.g. via `pip install graphviz`.

For the easy UNet setup, we get the computational graph

![Easy_UNet_Graph](/Network_Graph_Visualization/Network_Graphs/Easy_UNet.svg)

and for the more complicated setup

![Hard_UNet_Graph](/Network_Graph_Visualization/Network_Graphs/Hard_UNet_without_aux_visualization.svg)

(_Load images directly to zoom in and see more details._)

### Regarding the network save folder
During training, following elements are saved:
  * Network weights, Optimizer Weights, Scheduler weights and other parameters to recreate training state (saved in a `checkpoint.pth.tar`-dict after each epoch and `checkpoint_best_val.pth.tar` every time validation dice increases).

  * Pipeline parameters in easy-to-read text-form `parameters.txt`.
  * Python-dict-form `hypa.pkl` of pipeline parameters.
  * During training, a csv-logger will be updated in each epoch, saving time, loss and dice score.   
  * To qualitatively judge the training run, sample segmentations will be generated as well, both for the current set of network weights as well as the current best validation scored weights, for both training and validation set.   
  * Finally, training performances is visualized in a `training_results.svg`-plot.
