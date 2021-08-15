# Blackbird or Robin? - Implementation of a Fully Automated Machine Learning Pipeline for Bird Vocalization Recognition

#### Josafat-Mattias Burmeister, Maximilian Götz

## Contact

josafat-mattias.burmeister@student.hpi.de

maximilian.goetz@student.hpi.de

## Abstract

<div style="text-align: justify">
Audio recorders that capture bird vocalizations are increasingly used in conservation biology to monitor bird
populations. However, labeling the collected audio data requires trained ornithologists and is a very time-consuming task. To facilitate training of deep learning models that automate the labeling process, this work implements an end-to-end machine learning pipeline for the automatic recognition of bird vocalizations in audio files. The presented pipeline can be run both as a notebook and as a Kubeflow pipeline. The pipeline includes steps for collecting and downloading suitable training data, preprocessing and filtering the training data, and training, tuning, and evaluating deep learning models.
The pipeline is evaluated using an example sample dataset with ten different bird vocalization classes from the Xeno-Canto database. On this dataset an average F<sub>1</sub>-score of ... is achieved.
</div>

## Motivation

<div style="text-align: justify">

Estimates of bird populations are an essential element in conservation biology for determining the conservation status of bird species and for planning conservation measures [1, 2]. Therefore, monitoring programs exist for many bird species in which the abundance of the target species‚ is regularly surveyed in selected study areas. Conventionally, monitoring is conducted by trained ornithologists who survey the observed species using standardized methods [2]. In recent years, monitoring by human observers has been increasingly complemented by passive acoustic monitoring with audio recorders. The use of audio recorders reduces bias caused by human disturbances and allows for data collection on a larger spatial and temporal scale [1]. However, labeling and interpreting the collected audio files requires trained ornithologists and is a very time-consuming task. Using machine learning, significant progress has been made in recent years to automate the labeling process. In particular, deep convolutional neural networks that treat audio classification as an image classification problem have emerged as a promising approach. As the classification of bird vocalizations is associated with various challenges, the problem is not yet completely solved. One challenge is that audio recordings often contain background noise and overlapping vocalizations of multiple bird species. In addition, the songs of some bird species differ between individuals and regions. Moreover, most publicly available training data are only weakly labeled at the file level, but do not include temporal annotations.

Since different monitoring projects focus on different bird species and research questions, individual models are usually required for each monitoring project. To reduce the effort of training and fine-tuning custom models, this work aims to implement a flexible machine-learning pipeline for recognition of bird vocalizations in audio files. In previous work, convolutional neural networks trained on spectrogram images have yielded promising results. Therefore, we adopt this approach and focus our pipeline design on training such models. To support a wide range of applications, we aim for a flexible design in terms of the training dataset and model architecture used. To optimize models for custom datasets, our pipeline should also support the tuning of model hyperparameters.

</div>
