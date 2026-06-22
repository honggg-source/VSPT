
# Visual and Semantic Prior for Thermal Infrared Tracking(TMM)

## Abstract
We propose a new FFT-based TIR tracking framework based on visual and semantic priors, termed VSPT. Specifically, this framework consists of three main components: a pre-trained ViT-based tracker, a visual prior branch, and a semantic prior branch. Given that visual features such as multi-scale and shape are crucial for TIR tracking tasks, we first propose a visual prior module employing a dynamic multi-scale CNN architecture to model this prior information. Second, considering the challenges faced by TIR tracking, like target deformation, occlusion, and background distractors, we propose a semantic prior module based on a pre-trained MAE-based infrared large model.  This branch captures semantic information in TIR images through a dynamic masking mechanism. Third, we design a feature injector and a dynamic feature extractor based on the cross-attention mechanism. The feature injector integrates these priors into each Transformer block of ViT via dynamic residual connections, while the feature extractor simulates multiple levels of prior features injected into the prior branches.

<p align="center">
  <img width="85%" src="https://github.com/hongg-source/VSPT/assets/adapter+mae.png" alt="Framework"/>
</p>

## Download
- You can download several our trained model from [Baidu Pan](https://pan.baidu.com/s/1czaAeie5iD8hvXjJV401Pw).
- You can download the tracking raw results of three benchmarks from [Baidu Pan](https://pan.baidu.com/s/1knmuUTv72cLwhy40eUOMlA).



## Install the environment
**Option1**: Use the Anaconda
```
conda create -n ostrack python=3.8
conda activate ostrack
bash install.sh
```

**Option2**: Use the Anaconda
```
conda env create -f ostrack_cuda113_env.yaml
```

**Option3**: Use the docker file

We provide the full docker file here.


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lsotb
            |-- test
            |-- train
        -- ptbtir
            |-- test
        -- lashertir
            |-- test
   ```


## Training
- Download pre-trained [OSTrack weights](https://github.com/botaoye/OSTrack) and [InfMAE weights](https://github.com/liufangcen/InfMAE) and put it under `$PROJECT_ROOT$/pretrained_models` .
- VSPT can be trained using a single NVIDIA A100 GPU with 80GB of memory.
- Please run
```
python lib/train/run_training.py
```


## Evaluation
Download the model weights from [Baidu Pan](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd?usp=sharing) 

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/ostrack`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- LSOTB or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py # need to modify tracker configs and names
python tracking/analysis_results.py # need to modify tracker configs and names
```


## Visualization or Debug 
[Visdom](https://github.com/fossasia/visdom) is used for visualization. 
1. Alive visdom in the server by running `visdom`.

2. Simply set `--debug 1` during inference for visualization.
3. Open `http://localhost:8097` in your browser (remember to change the IP address and port according to the actual situation).

4. Then you can visualize the candidate elimination process.



## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single A4000 GPU.

```
# Profiling vitb_256_mae_ce_32x4_ep300
python tracking/profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300
# Profiling vitb_384_mae_ce_32x4_ep300
python tracking/profile_model.py --script ostrack --config vitb_384_mae_ce_32x4_ep300
```


## Citation

If you have any questions, feel free to contact us ([liuqiao.hit@gmail.com](mailto:liuqiao.hit@gmail.com)).
