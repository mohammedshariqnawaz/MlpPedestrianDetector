[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/f2dnet-fast-focal-detection-network-for/pedestrian-detection-on-caltech)](https://paperswithcode.com/sota/pedestrian-detection-on-caltech?p=f2dnet-fast-focal-detection-network-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/f2dnet-fast-focal-detection-network-for/pedestrian-detection-on-citypersons)](https://paperswithcode.com/sota/pedestrian-detection-on-citypersons?p=f2dnet-fast-focal-detection-network-for)

# F2DNet

<img title="Frankfurt" src="gifs/gm.png" width="800" />

F2DNet is a [Pedestron](https://github.com/hasanirtiza/Pedestron) based repository which implements a novel, two-staged detector i.e. Fast Focal Detection Network for pedestrian detection.

[Pedestron](https://128.84.21.199/pdf/2003.08799.pdf) is a [MMdetection](https://github.com/open-mmlab/mmdetection) based repository, that focuses on the advancement of research on pedestrian detection. We provide a list of detectors, both general purpose and pedestrian specific to train and test. Moreover, we provide pre-trained models and benchmarking of several detectors on different pedestrian detection datasets. Additionally, we provide processed annotations and scripts to process the annotation of different pedestrian detection benchmarks. If you use Pedestron, please cite us (see at the end) and other respective sources.

# :fire: **Updates** :fire:
* 🧨 **We have realeased [PedesFormer - Transformer Based Pedestrian Detection](https://github.com/hasanirtiza/PedesFormer-SwinTransformer-For-Pedestrian-Detection) repo (particularly Swin Transformer) along with pre-trained models. Stay tune for the updates.** 🧨





### YouTube demo
* [Caltech](https://www.youtube.com/watch?v=cemN7JbgxWE&feature=youtu.be) and [EuroCity Persons](https://www.youtube.com/watch?v=ZFObRPa_yMM). Pre-Trained model available.

### Leaderboards 
* [CityPersons](https://github.com/cvgroup-njust/CityPersons)
* [EuroCity Persons](https://eurocity-dataset.tudelft.nl/eval/benchmarks/detection)

### Installation
Please refer to [base repository](https://github.com/hasanirtiza/Pedestron) for step-by-step installation. 

### List of detectors

In addition to configuration for different detectors provided in [base repository](https://github.com/hasanirtiza/Pedestron) we provide configuration for F2DNet.


### Following datasets are currently supported 
* [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)
* [CityPersons](https://github.com/cvgroup-njust/CityPersons)
* [EuroCity Persons](https://eurocity-dataset.tudelft.nl/)

### Datasets Preparation
Please refer to base repository for dataset preparation.


# Benchmarking 
### Benchmarking of F2DNet on pedestrian detection datasets
| Dataset            | &#8595;Reasonable |  &#8595;Small   |  &#8595;Heavy   | 
|--------------------|:----------:|:--------:|:--------:|
| CityPersons        |  **8.7**   | **11.3** | **32.6** | 
| EuroCityPersons    |    6.1     |   10.7   |   28.2   | 
| Caltech Pedestrian |  **2.2**   | **2.5**  | **38.7** |

### Benchmarking of F2DNet when trained using extra data on pedestrian detection datasets
| Dataset            | Config                                                                                                       | Model                                                                                        | &#8595;Reasonable | &#8595;Small |  &#8595;Heavy   | 
|--------------------|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|:----------:|:------------:|:--------:|
| CityPersons        | cascade_hrnet | Cascade Mask R-CNN                                                                           |  **7.5**   |   **8.0**    |   28.0   |
| CityPersons        | [ecp_cp](https://github.com/AbdulHannanKhan/F2DNet/blob/master/configs/f2dnet/cp/ecp_sup.py)                 | [F2DNet](https://drive.google.com/file/d/1IrwvdLtpOjUpmz2_IXWENbVNAQtEZKn-/view?usp=sharing) |    7.8     |     9.4      | **26.2** |
| Caltech Pedestrian | cascade_hrnet | Cascade Mask R-CNN                                                                           |  **1.7**   |              |   25.7   |
| Caltech Pedestrian | [ecp_cp_caltech](https://github.com/AbdulHannanKhan/F2DNet/blob/master/configs/f2dnet/caltech/ecp_cp_sup.py) | [F2DNet](https://drive.google.com/file/d/1DzcKR-tKy-Oa6uVoiYUt_q_7h5iwwCeh/view?usp=sharing)                                                                                   |  **1.7**   |   **2.1**    | **20.4** |


# References

* [Pedestron](https://openaccess.thecvf.com/content/CVPR2021/papers/Hasan_Generalizable_Pedestrian_Detection_The_Elephant_in_the_Room_CVPR_2021_paper.pdf)

### Please cite the following work
[AxXiv2022](https://arxiv.org/pdf/2203.02331.pdf)
```
@Article{khan:2022,
  author      = {Khan, Abdul Hannan AND Munir, Mohsin AND Elst, Ludger van AND Dengel, Andreas},
  title       = {F2DNet: Fast Focal Detection Network for Pedestrian Detection},
  version     = {1},
  date        = {2022-03-04},
  eprinttype  = {arxiv},
  eprintclass = {cs.CV, I.2.10; I.4.8; I.5.4},
  eprint      = {http://arxiv.org/abs/2203.02331v1},
  url         = {http://arxiv.org/abs/2203.02331v1}
}

```

[ArXiv2022](https://arxiv.org/pdf/2201.03176.pdf)
```
@article{hasan2022pedestrian,
  title={Pedestrian Detection: Domain Generalization, CNNs, Transformers and Beyond},
  author={Hasan, Irtiza and Liao, Shengcai and Li, Jinpeng and Akram, Saad Ullah and Shao, Ling},
  journal={arXiv preprint arXiv:2201.03176},
  year={2022}
}
```
