# Benchmarking and Boosting Transformers for Medical Image Classification

We benchmark how well existing transformer variants that use various (supervised and self-supervised) pre-training methods perform against CNNs on a variety of medical classification tasks. Furthermore, given the data-hungry nature of transformers and the annotation-deficiency challenge of medical imaging, we present a practical approach for bridging the domain gap between photographic and medical images by utilizing unlabeled large-scale in-domain data. 



<p align="center"><img width=60% alt="FrontCover" src="https://github.com/Mda233/BenchmarkTransformers/blob/main/media/FrontCover.png"></p>

## Publication
<b>Benchmarking and Boosting Transformers for Medical Image Classification </b> <br/>
[DongAo Ma](https://github.com/Mda233)<sup>1</sup>,[Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher)<sup>1</sup>, [Jiaxuan Pang](https://github.com/MRJasonP)<sup>1</sup>, [Nahid Ul Islam](https://github.com/Nahid1992)<sup>1</sup>, [Fatemeh Haghighi](https://github.com/fhaghighi)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Mayo Clinic <br/>

International Conference on Medical Image Computing and Computer Assisted Intervention ([MICCAI 2022](https://conferences.miccai.org/2022/en/)); Domain Adaptation and Representation Transfer (DART) 

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-16852-9_2) ([PDF](https://github.com/Mda233/BenchmarkTransformers/blob/main/media/Benchmarking%20and%20Boosting%20Transformers%20for%20Medical%20Image%20Classification%20.pdf), [Supplementary material](https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-031-16852-9_2/MediaObjects/538914_1_En_2_MOESM1_ESM.pdf))  | [Code](https://github.com/jlianglab/BenchmarkTransformers) | [Poster](https://github.com/Mda233/BenchmarkTransformers/blob/main/media/Poster_Benchmarking&Boosting_Final(updated).pdf) | [Slides](https://github.com/Mda233/BenchmarkTransformers/blob/main/media/BenchmarkTransformers_DART22.pdf) | Presentation ([YouTube](https://youtu.be/VMyU8UWLPFg))

## Major results from our work

1. **Pre-training is more vital for transformer-based models than for CNNs in medical imaging.**

<p align="center"><img width=90% alt="Result1" src="https://github.com/Mda233/BenchmarkTransformers/blob/main/media/Result1.png"></p>

In medical imaging, good initialization is more vital for transformer-based models than for CNNs.  When training from scratch, transformers perform significantly worse than CNNs on all target tasks. However, with supervised or self-supervised pre-training on ImageNet, transformers can offer the same results as CNNs, highlighting the importance of pre-training when using transformers for medical imaging tasks. We conduct statistical analysis between the best of six pre-trained transformer models and the best of three pre-trained CNN models.

</br>

2. **Self-supervised learning based on masked image modeling is a preferable option to supervised baselines for medical imaging.**

<p align="center"><img width=90% alt="Result2" src="https://github.com/Mda233/BenchmarkTransformers/blob/main/media/Result2.png"></p>

Self-supervised SimMIM model with the Swin-B backbone outperforms fully- supervised baselines. The best methods are bolded while the second best are underlined. For every target task, we conduct statistical analysis between the best (bolded) vs. others. Green-highlighted boxes indicate no statistically significant difference at the p = 0.05 level.

</br>

3. **Self-supervised domain-adaptive pre-training on a larger-scale domain-specific dataset better bridges the domain gap between photographic and medical imaging.**

<p align="center"><img width=90% alt="Result3" src="https://github.com/Mda233/BenchmarkTransformers/blob/main/media/Result3.png"></p>

The domain-adapted pre-trained model which utilized a large number of in-domain data (X-rays(926K)) in an SSL manner achieves the best performance across all five target tasks. The best methods are bolded while the second best are underlined. For each target task, we conducted the independent two sample t-test between the best (bolded) vs. others. The absence of a statistically significant difference at the p = 0.05 level is indicated by green-highlighted boxes. 

*X-rays(926K): To check what datasets are used for the domain-adaptive pre-training, please see the [Supplementary material](https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-031-16852-9_2/MediaObjects/538914_1_En_2_MOESM1_ESM.pdf).

</br>

### Requirements
+ Python
+ Install PyTorch ([pytorch.org](http://pytorch.org))


## Pre-trained models

You can download the pretrained models used/developed in our paper as follows:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Category</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Training Dataset</th>
<th valign="bottom">Training Objective</th>
<th valign="bottom">model</th>

<tr>
<td rowspan = "2">Domain-adapted models</td>
<td>Swin-Base</td>
<td>ImageNet &#8594; X-rays(926K)</td>
<td>SimMIM &#8594; SimMIM</td>   
<td><a href="https://zenodo.org/record/7101953/files/simmim_swinb_ImageNet_Xray926k.pth?download=1">download</a></td>

</tr>
<tr> 
 <td>Swin-Base</td>
<td>ImageNet &#8594; ChestX-ray14</td>
<td>SimMIM &#8594; SimMIM</td>   
<td><a href="https://zenodo.org/record/7101953/files/simmim_swinb_ImageNet_ChestXray14.pth?download=1">download</a></td>
 </tr>

<tr >
<td rowspan = "2">In-domain models</td>
<td>Swin-Base</td>
<td>X-rays(926K)</td>
<td>SimMIM</td>   
<td><a href="https://zenodo.org/record/7101953/files/simmim_swinb_Scratch_Xray926k.pth?download=1">download</a></td>
<tr >
<td>Swin-Base</td>
<td>ChestX-ray14</td>
<td>SimMIM</td>   
<td><a href="https://zenodo.org/record/7101953/files/simmim_swinb_Scratch_ChestXray14.pth?download=1">download</a></td>
</tr>

 
</tbody></table>

## Fine-tuing of pre-trained models on target task
1. Download the desired pre-trained model.
2. Download the desired dataset; you can simply add any other dataset that you wish.
3. Run the following command by the desired parameters. For example, to finetune our pre-trained ImageNet &#8594; X-rays(926K) model on ChestX-ray14, run:
```bash
python main_classification.py --data_set ChestXray14  \
--model swin_base \
--init simmim \
--pretrained_weights [PATH_TO_MODEL]/simmim_swinb_ImageNet_Xray926k.pth \
--data_dir [PATH_TO_DATASET] \
--train_list dataset/Xray14_train_official.txt \
--val_list dataset/Xray14_val_official.txt \
--test_list dataset/Xray14_test_official.txt \
--lr 0.01 --opt sgd --epochs 200 --warmup-epochs 0 --batch_size 64
```

Or, to evaluate the official released ImageNet models from timm on ChestX-ray14, run:
```bash
python main_classification.py --data_set ChestXray14  \
--model vit_base \
--init imagenet_21k \
--data_dir [PATH_TO_DATASET] \
--train_list dataset/Xray14_train_official.txt \
--val_list dataset/Xray14_val_official.txt \
--test_list dataset/Xray14_test_official.txt \
--lr 0.1 --opt sgd --epochs 200 --warmup-epochs 20 --batch_size 64
```

## Citation
If you use this code or use our pre-trained weights for your research, please cite our paper:
```
@inproceedings{Ma2022Benchmarking,
    title="Benchmarking and Boosting Transformers for Medical Image Classification",
    author="Ma, DongAo and Hosseinzadeh Taher, Mohammad Reza and Pang, Jiaxuan and Islam, Nahid UI and Haghighi, Fatemeh and Gotway, Michael B and Liang, Jianming",
    booktitle="Domain Adaptation and Representation Transfer",
    year="2022",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="12--22",
    isbn="978-3-031-16852-9"
}
```

## Acknowledgement
This research has been supported in part by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and in part by the NIH under Award Number R01HL128785. The content is solely the responsi- bility of the authors and does not necessarily represent the official views of the NIH. This work has utilized the GPUs provided in part by the ASU Research Computing and in part by the Extreme Science and Engineering Discovery En- vironment (XSEDE) funded by the National Science Foundation (NSF) under grant numbers: ACI-1548562, ACI-1928147, and ACI-2005632. The content of this paper is covered by patents pending.




## License

Released under the [ASU GitHub Project License](./LICENSE).
