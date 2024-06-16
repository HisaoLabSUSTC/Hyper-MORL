# Hyper-MORL

This repository contains the implementation for the paper **Learning Pareto Set for Multi-Objective Continuous Robot Control**  **(IJCAI 2024)** 

In this paper, we propose an MORL algorithm called Hyper-MORL which learns a continuous representation of the Pareto set in a high-dimensional policy parameter space using a single hypernet.

![illustration](\\172.18.36.77\shutianye\Github\Hyper-MORL\illustration.png)

## Installation Dependencies

If you use the conda virtual environment, then simply create a virtual environment named **hypermorl** by:

```shell
conda env create -f environment.yml
```

If you want to install packages manually,  then you can check the **environment.yml**  file and install necessary packages.

Next, we will briefly show how to run the code. For more details, please refer to **visualization/demonstration.ipynb**.

## Training 

The **script** folder contains all training scripts. For example, you can run our algorithm on Walker2d-v2 for nine runs:

```python
python scripts/walker.py --num-seeds 9
```

For your convenience, we have provided pretrained models for all problems in the **pretrained** folder.

## Test

After a hypernet is trained, you can input an arbitrary preference to obtain the learned policy parameters corresponding to the given preference. For example,

```python
python/visualization/test_hypermorl.py --env-name MO-Walker2d-v2
```

The input preferences, output policy parameters and evaluated objective values are saved in **/results/sample** by default.

## Visualization

We provide some scripts to visualize the relation between the input preference and the objective values of the corresponding output policy.

```python
python visualization/pref_obj_visualization_2d.py
python visualization/pref_obj_visualization_3d.py
```

After execute the above command, you will see an interactive figure. In the figure, you can adjust the input preference by drag the point in the preference space. Moreover, you can also visualize the behavior of the output policy by clicking the button.

To visualize the parameters of different policy, we use t-SNE to embed high-dimensional parameter space into two-dimensional space. These codes can be found in the **visualization/demonstration.ipynb**.

## Acknowledgement

Our codes are implemented based on the PGMORL repository.

## Citation

If you find our paper or code is useful, please consider citing:

```bib
@inproceedings{shu2024Learning,
title = {Learning Pareto Set for Multi-Objective Continuous Robot Control},
author = {Tianye Shu and Ke Shang and Cheng Gong and Yang Nan and Hisao Ishibuchi},
booktitle={International Joint Conference on Artificial Intelligence},
year = {2024}
}
```

  