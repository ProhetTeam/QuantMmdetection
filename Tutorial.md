
## Environment Configuration
- python3.6+
- [pytorch 1.6.0](https://pytorch.org)
- [mmcv](https://mmcv.readthedocs.io/en/latest/#installation)

## DownLoad Code
- clone from tanfeiyang's gitcore 
    ```python
    git clone git@git-core.megvii-inc.com:tanfeiyang/mmdetection.git
    ```
- pull the lastest version and checkout the branch
    ```
    git pull origin tanfeiyang/dev
    git checkout -b tanfeiyang/dev
    ```
- update the third party of the transformer library
    ```python
    git submodule update --init --recursive
    ```

- compile
    ```
    python setup.py develop --user
    ```
## HOW TO RUN THE CODE:
- first of all the compute resource you need to ralunch

    ```shell
    rlaunch --cpu=32 --gpu=8 --memory=100000 --preemptible=no  --group=research_facerec_multi_machine_2 -- zsh
    ```
- secondly, if you only have one machine with 8 gpu
    ```shell
    python -m torch.distributed.launch --nproc_per_node=8 tools/train.py configs/atss/atss_r50_nuscenes_multiscale.py --launcher pytorch
    ```
-  multi machine,(2 node both with 8 gpu)

    **node1 ：** run on first machine
    ```
    bash ./tools/dist_train.sh 2 0 10.124.161.208（ip） 8 ./configs/atss/atss_r50_nuscenes_v2.py pytorch
    ```
    **node2:** run on the other machine
    ```
    bash ./tools/dist_train.sh 2 1 10.124.161.208（ip） 8 ./configs/atss/atss_r50_nuscenes_v2.py pytorch
    ```
`Tips:` you can query the ip address  by the command  `ifconfig`


