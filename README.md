# AdaptMVSNet

# NanHuLab Affiliation

# installion 
* These project ned to be installed: torch 1.12.1, python 3.8, cuda >= 10.1
* Please install Tensorrt, if you want to test in fast inference.
```
pip install -r requirements.txt
```


# Download 
* Preprocessed training/validation data: [DTU's evaluation set](https://drive.google.com/file/d/1jN8yEQX0a-S22XwUjISM8xSJD39pFLL_/view?usp=sharing) ,[BlendedMVS](https://github.com/YoYo000/BlendedMVS) ,[Tanks&Temples](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view?usp=sharing). Each dataset is already organized as follows:
```
root_directory
├──scan1 (scene_name1)
├──scan2 (scene_name2) 
      ├── images                 
      │   ├── 00000000.jpg       
      │   ├── 00000001.jpg       
      │   └── ...                
      ├── cams                   
      │   ├── 00000000_cam.txt   
      │   ├── 00000001_cam.txt   
      │   └── ...                
      └── pair.txt  
``` 

# Testing 
* running eval.sh, using dtu dataset
* In eval.sh, set DTU_TESTING, ETH3D_TESTING or TANK_TESTING as the root directory of corresponding dataset and uncomment the evaluation command for corresponding dataset (default is to evaluate on DTU's evaluation set). If you want to change the output location (default is same as input one), modify the --output_folder parameter.
* CKPT_FILE is the checkpoint file (our pretrained model is ./checkpoints/params_000007.ckpt), change it if you want to use your own model.
* Test on GPU by running sh eval.sh. The code includes depth map estimation and depth fusion. The outputs are the point clouds in ply format.
