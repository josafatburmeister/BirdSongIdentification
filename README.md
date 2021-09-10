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

Estimates of bird populations are an essential element in conservation biology for determining the conservation status of bird species and for planning conservation measures [\cite{monitorung-overview}, \cite{audio-monitorung}]. Therefore, monitoring programs exist for many bird species in which the abundance of the target species‚ is regularly surveyed in selected study areas. Conventionally, monitoring is conducted by trained ornithologists who survey the observed species using standardized methods [\cite{monitorung-overview}]. In recent years, monitoring by human observers has been increasingly complemented by passive acoustic monitoring with audio recorders. The use of audio recorders reduces bias caused by human disturbances and allows for data collection on a larger spatial and temporal scale [\cite{audio-monitorung}]. However, labeling and interpreting the collected audio files requires trained ornithologists and is a very time-consuming task. Using machine learning, significant progress has been made in recent years to automate the labeling process. In particular, deep convolutional neural networks that treat audio classification as an image classification problem have emerged as a promising approach [\cite{sprengel-2016}, \cite{sevilla-2017}, \cite{koh-2018}]. As the classification of bird vocalizations is associated with various challenges, the problem is not yet completely solved. One challenge is that audio recordings often contain background noise and overlapping vocalizations of multiple bird species. In addition, the songs of some bird species differ between individuals and regions. Moreover, most publicly available training data are only weakly labeled at the file level, but do not include temporal annotations [\cite{nips4bplus}].

Since different monitoring projects focus on different bird species and research questions, individual models are usually required for each monitoring project. To reduce the effort of training and fine-tuning custom models, this work aims to implement a flexible machine-learning pipeline for recognition of bird vocalizations in audio files. In previous work, convolutional neural networks trained on spectrogram images have yielded promising results. Therefore, we adopt this approach and focus our pipeline design on training such models. To support a wide range of applications, we aim for a flexible design in terms of the training dataset and model architecture used. To optimize models for custom datasets, our pipeline should also support the tuning of model hyperparameters.

</div>

## Related Work

<div style="text-align: justify">

A major driver of research in automatic bird sound recognition is the _BirdCLEF_ challenge, which has been held annually since 2014 [\cite{bird-clef-2014}, \cite{bird-clef-2015}, \cite{bird-clef-2016}, \cite{bird-clef-2017}, \cite{bird-clef-2018}, \cite{bird-clef-2019}, \cite{bird-clef-2020}]. The objective of this challenge is to recognize bird vocalizations in so-called soundscape files, which are longer, omnidirectional audio recordings that usually contain a variety of bird vocalizations [\cite{bird-clef-2026}]. The provided training data in the BirdCLEF challenge consists mainly of so-called focal recordings from the _Xeno-Canto_ database, which usually contain vocalizations of only one species [\cite{bird-clef-2026}]. The Xeno-Canto database is a public database that collects audio recordings of bird vocalizations worldwide. The recordings included in the Xeno-Canto collection are usually a few minutes long and typically contain only one focal species. The Xeno-Canto database also provides various metadata, such as the recording location, recording quality, and the age and sex of the recorded bird [\cite{xeno-canto}].

Besides the Xeno-Canto database, several other datasets that include recordings of bird vocalizations are publicly available. These include the Chernobyl dataset, the Warblr dataset, the freefield1010 dataset, the PolandNFC dataset, the Birdvox-Full-Night dataset, and the NIPS4B dataset [\cite{warblr}, \cite{birdvox}, \cite{nips4bplus}]. While the other datasets only include presence-absence tags for bird vocalizations, the NIPS4B dataset also provides species tags [\cite{nips4bplus}]. The NIPS4BPlus dataset provides time-accurate annotations of bird vocalizations for a subset of the recordings in the NIPS4B dataset [\cite{nips4bplus}].

Over the last years, various approaches to automatic bird sound recognition have been investigated. Lassek approached the problem with random forests trained on low-level features of audio files and statistical features of spectrograms [\cite{lasseck2013}, \cite{lasseck2014}, \cite{lasseck2015}]. Müller and Marti employed recurrent neural networks (RNN) for bird sound recognition, namely an LSTM architecture [\cite{muller2018}]. However, deep convoplutional neural networks have emerged as the most promising approach. In order to use convolutional neural networks (CNN) for audio classification, spectrogram images are generated from the audio data, turning the audio classification task into an image classification task. In recent years, this approach has produced very good results in the BirdCLEF challenge.

For example, Sprengel et al. won the BirdCLEF challenge in 2016 by training a CNN with five convolutional layers on fixed-size spectrograms. In order to identify relevant sections of the audio files when generating the spectrograms, Sprengel et al. implemented noise filtering. Using a sequence of image filters such as median threshholding, erosion filtering and dilation filtering, noise pixels are separated from signal pixels. To enlarge the training set, data augmentation techniques such as time shifts, pitch shifts, and background
noise were applied [\cite{sprengel-2016}].

In their winning submission to the 2017 BirdCLEF challenge, Sevilla employed an Inception-v4 model for classification of bird vocalizations. The model was trained using transfer learning and standard augmentation techniques, such as random hue, contrast, brightness, and saturation modifications. To guide the model's focus to relevant spectrogram regions, attention layers were added to Inception-v4 architecture [\cite{sevilla-2017}].

Following a similar approach, Koh et al. achieved second place in the BirdCLEF challenge in 2019. They trained Resnet18 and Inception models on Mel-scaled spectrograms. For noise filtering, an image filter-based algorithm was used as in Sprengel et al. To address data imbalanced, data augmentation techniques were also used, e.g. brightness adjustments, blurring, slight rotations, crops and background noise [\cite{koh2019}].

</div>

## Our Approach

### Use Case Specification

<div style="text-align: justify">

Our work aims to implement an end-to-end machine learning pipeline that automates the training bird sound recognition models. Based on promising results of previous work, we focus on training deep convolutional neural networks (DCNN) trained as image classification models on spectrograms. For data preprocessing and spectrogram creation, we largely follow the approach described by Koh et al. [\cite{koh-2018}]. With respect to the dataset and model architecture used, we aim for a flexible and extensible pipeline design.

To demonstrate and evaluate the capability of our pipeline, we consider the following use case: As the primary data source, we use the Xeno-Canto database, which is the largest publicly available collection of bird sound recordings. To train DCNN models, we convert the audio files from Xeno-Canto into spectrograms. The audio recordings from Xeno-Canto are usually dominated by the vocalizations of one focus species, but may include other bird vocalizations in the background. Since Xeno-Canto only includes file-level annotations, but no time-accurate annotations, we use only the focal species for spectrogram labeling and ignore the background species. In contrast to recordings in Xeno-Canto, recordings from monitoring projects usually contain multiple overlapping bird vocalizations. To generalize our models to such use cases, we train multi-label classification models, even though our training data is single-label data. To evaluate the performance of our models in such a scenario, we use the NIPS4BPlus dataset as an additional test dataset [\cite{nips4bplus}]. In order to obtain time-accurate predictions, we split the audio files into fixed-length chunks (1 second) and create separate spectrograms and thus separate predictions for each chunk.

</div>

### Pipeline Architecture

<div style="text-align: justify">

Conceptually, our machine learning pipeline consists of the following four stages:

(1) Download of audio data and labels

(2) Conversion of audio files into spectrograms and filtering of spectrograms containing only noise

(3) Training of DCNN image classification models on the spectrograms and tuning of model hyperparameters

(4) Model evaluation on test datasets

All pipeline steps are implemented by Python classes, which are described in more detail in the following sections. To support a wide range of applications, our pipeline can be run as both a Jupyter notebook and a Kubeflow pipeline. Both variants use the same Python implementation, with the definition of our Kubeflow pipeline providing a wrapper for the interface of our Python classes.

</div>

### Stage 1: Data Download

<div style="text-align: justify">

The downloader stage is responsible for downloading the audio files and labels needed for model training and evalution, and converting them into a consistent format.

To demonstrate the capability of our pipeline, we use both audio data from the Xeno-Canto database (for model training, validation and testing) and the NIPS4BPlus dataset (for model testing). The download of both datasets is implemented by separate downloader classes that inherit from a common base class. For downloading audio files from Xeno-Canto, we use the public Xeno-Canto API. The Xeno-Canto API allows searching for audio files based on a set of filter criteria (e.g., bird species, recording location, recording quality, and recording duration). The search returns the metadata of the matching audio files in JSON format, including download links for the audio files. Our Xeno-Canto downloader implementation supports most of the filter criteria of the Xeno-Canto API. Based on the criteria defined by the pipeline user, the downloader compiles training, validation and test sets. Our NIPS4BPlus downloader, on the other hand, only supports filtering by bird species and sound category, since no other metadata is available for the NIPS4Bplus dataset.

To speed up the download phase, our downloader classes use multithreading where possible. In addition, we implement local caching of files such that subsequent pipeline runs do not need to download them again. When the pipeline is run as a Jupyter notebook, an ordinary directory on the local disk is used for caching. When the pipeline is run as a Kubeflow pipeline, a Google Cloud Storage bucket is used for file caching.

### Stage 2: Spectrogram Creation

For spectrogram creation, we largely follow the approach described by Kot et al. [\cite{koh-2018}]. As in the work of Koh et al, we divide the audio files into non-overlapping 1-second chunks and create a mel-scale log-amplitude spectrogram for each chunk. The spectrogram creation is based on a short-time Fourier transform (STFT) of the amplitude signal, for which we use the Python sound processing library _Librosa_<sup>1</sup>. We choose the parameters of the STFT so that the resulting spectrograms have a size of approximately 224 x 112 pixels. Table 1 provides an overview of our STFT parameter settings, which are largely consistent with those of Koh et al. [\cite{koh-2018}]. The spectrogram images are stored as inverted grayscale images, so that high amplitudes are represented by dark pixels.

| Parameter         | Value     |
| ----------------- | --------- |
| Sampling rate     | 44100 Hz  |
| Window length     | 1024      |
| Hop length        | 196       |
| Minimum frequency | 500 Hz    |
| Maximum frequency | 15,000 Hz |

Table 1: Parameter settings of the short-time Fourier transform used for spectrogram creation.

Since the audio files from Xeno-Canto are only labeled at the file level, it is uncertain which parts of the recording contain bird vocalizations. To separate spectrograms that contain bird vocalizations from spectrograms that contain only noise, we implement noise filtering. For this purpose, we employ the noise filtering algorithm presented by Kahl et al. [\cite{kahl-2017}]. In this algorithm, multiple image filters are applied to each spectrogram to extract the signal pixels of the spectrogram, and then the number of signal rows is compared to a threshold value. First, the image is blurred with a median blur kernel of size 5. Next, a binary image is created by median filtering. In this process, all pixel values that are 1.5 times larger than the row and the column median are set to black and all other pixels are set to white. To remove isolated black pixels, spot removal and morphological closing operations are applied. Finally, the number of rows with black pixels (signal pixels) is compared to a predefined threshold, the signal threshold. If the number of signal rows is larger than the signal threshold, the spectrogram is assumed to contain bird vocalizations. If the number of signal rows is below a second threshold, the noise threshhold, the spectrogram is considered to contain only noise. To have a decision margin, we choose the noise threshold smaller than the signal threshold. To increase model robustness, our pipeline allows to include noise spectrograms for training as a separate class.

<sup>1</sup> https://librosa.org

### Stage 3: Model Training

The model training stage of our pipeline can be used either to train DCNN models with specified hyperparameters or to tune the model hyperparameters. It was implemented using the Pytorch framework<sup>2</sup> and the Torchvision library<sup>3</sup>. Building on the convetions of the Torchvision library, the training component is designed in such a way that the model architecture used can be easily exchanged. For our use case, we mainly use the ResNet18 architecture, as it has been successfully applied to bird sound classification in previous work [\cite{koh-2017}]. In addition, we also experiment with the ResNet50 and the DenseNet121 architectures. Our implementation supports training of both single-label and multi-label classification models. However, for our use case, we only use multi-label models since multiple bird calls may occur simultaneously in some recordings.

We train the models using a transfer learning approach. For this, we use models from the Torchvision Library that were pre-trained on the ImageNet dataset [\cite{image-net}]. We replace the fully-connected layers of the pre-trained models with classifiers suited to our classification tasks and then fine-tune some of the model layers on our data. Our implementation supports various degrees of transfer learning, which range from retraining only the fully connected layers to fine-tuning all model layers.

To select the best model from each training run, we use a macro-averaged F1-score as performance metric. The macro F1-score weights all classes equally and is therefore suitable for our use case, where model performance on rare classes is as important as performance on the frequent classes.

<sup>2</sup> https://pytorch.org

<sup>3</sup> https://pytorch.org/vision/stable/index.html

### Stage 4: Model Evaluation

In the model evaluation stage, the best model from the training stage is evaluated on test datasets that have not been used for model training or validation. In our use case, we use test data from Xeno-Canto and the NIPS4BPlus dataset to evaluate the models. As in the training stage, the macro-average F1 score is used as the primary evauation metric. Although model evaluation is conceptually a separate pipeline stage, in our Kubeflow pipeline we have implemented model training and evaluation as a joint pipeline component. Although model evaluation is conceptually a separate pipeline stage, in Kubeflow we have implemented model training and evaluation as a joint pipeline component for performance reasons.

### Data Exchange Between Pipeline Components

<div style="text-align: justify">

In Kubeflow pipelines, all outputs of the pipeline stages are stored as files and can be used as inputs for subsequent pipeline stages. Persisting the outputs increases the pipeline's robustness and facilitates failure recovery. Therefore, we follow this approach and use purely file-based interfaces to exchange data between the different components of our pipeline.

Listing 1 shows an example of the directory structure that is used to pass data between the data download and the spectrogram creation stage. As shown, the data download stage is required to create a file named "categories.txt" as well as a number of subdirectories, representing different datasets or different parts of a dataset (e.g., train, validation, and test set). The file "categories.txt" contains a line-by-line listing of all possible class names that may be used as labels for the audio files (Listing 2). Each of the subdirectories representing different datasets has to contain a subdirectory named "audio" and a label file named "audio.csv". The subdirectory "audio" contains the audio files of the respective dataset, which can be grouped in further subdirectories. The label file "audio.csv" contains one line per annotated sound event, i.e., per annotated bird vocalization. An example of such a label file is shown in Table 2. As shown, the label files must contain at least the following columns:

**id**: Identifier of the audio file that is unique across all datasets.

**file_path**: Path of the audio file relative to the subdirectory containing the dataset.

**start**: Start time of the annotated sound event, specified in milliseconds after the beginning of the audio file.

**end**: End time of the annotated sound event, specified in milliseconds after the beginning of the audio file.

**label**: Class label of the annotated sound event.

This label format can be used to support both single-label and multi-label classification tasks. In addition, both temporally annotated and file-level annotated datasets can be processed. In the latter case, only one label is created per file, with the start time set to 0 and the end time set to the length of the audio file in milliseconds.

```
├── categories.txt
├── train
│   ├── audio
│   │   ├── 368261.mp3
│   │   ├── 619980.mp3
│   │   └── ...
│   └── audio.csv
├── val
│   ├── audio
│   │   └── ...
│   └── audio.csv
└── test
    ├── audio
    │   └── ...
    └── audio.csv
```

**Listing 1**: Example of the directory structure that is used to pass data between the data download and the spectrogram creation stage.

```
Turdus_merula_song
Turdus_merula_call
Erithacus_rubecula_song
Erithacus_rubecula_call
```

**Listing 2**: Example of a "categories.txt" file that lists the labels that are used in a dataset.

**Table 2**: Example of a label file in CSV format used for audio file labeling in our pipeline.

| id     | file_path  | start | end    | label                   |
| ------ | ---------- | ----- | ------ | ----------------------- |
| 368261 | 368261.mp3 | 0     | 47000  | Turdus_merula_song      |
| 619980 | 619980.mp3 | 0     | 11000  | Erithacus_rubecula_call |
| 619980 | 619980.mp3 | 11000 | 174000 | Turdus_merula_call      |

Listing 3 shows an example of the directory structure that is used to pass data between the spectrogram creation and the model training stage. It is very similar to the directory structure that is used as input of the spectrogram creation stage (Listing 1). As shown, the output directory of the spectrogram creation stage also has to contain a "categories.txt" file matches the format shown in Listing 2. In addition, the spectrogram creation stage has to create a subdirectory named "spectrograms" and a label file "spectrograms.csv" for each dataset. The "spectrograms" subdirectory contains the spectrogram images of the respective dataset. The label file "spectrograms.csv" has to contain one label per spectrogram image. As shown in Table 3, it must contain at least the columns "id", "file_path" and one column per label class containing binary presence-absence labels.

```
├── categories.txt
├── train
│   ├── spectrograms
│   │   ├── 368261-0.png
│   │   ├── 368261-1_noise.png
│   │   ├── ...
│   │   ├── 619980-0.png
│   │   └── ...
│   └── spectrograms.csv
├── val
│   ├── spectrograms
│   │   └── ...
│   └── spectrograms.csv
└── test
    ├── spectrograms
    │   └── ...
    └── spectrograms.csv
```

**Listing 3**: Example of the directory structure that is used to pass data between the spectrogram creation and the model training stage.

**Table 3**: Example of a label file in CSV format used for spectrogram labeling in our pipeline.

| id     | file_path    | Turdus_merula_song | Turdus_merula_call | Erithacus_rubecula_song | Erithacus_rubecula_call |
| ------ | ------------ | ------------------ | ------------------ | ----------------------- | ----------------------- |
| 368261 | 368261-0.png | 1                  | 0                  | 0                       | 0                       |
| 368261 | 368261-1.png | 1                  | 0                  | 0                       | 0                       |
| 619980 | 619980-0.png | 0                  | 0                  | 0                       | 1                       |

</div>

## Experiments

To evaluate the performance of our pipeline, we use a sample dataset with ten classes of bird songs. The ten classes are those classes of the NIPS4BPlus dataset for which most recordings exist in Xeno-Canto. The dataset was compiled from recordings from Xeno-Canto. Only recordings that do not contain background species, have audio quality "A", and are not longer than 180 seconds were used. A maximum of 500 audio recordings were used per class, with 60% of the recordings used for model training and 20% each for model validation and testing. The class distribution of all data splits is shown in Table 2. The number of spectrograms per class depends on the number and length of audio recordings and ranges from 5,374 to 22,291 spectrograms per class in the training set. To avoid strong imbalances, the number of noise spectrograms included in the data splits was limited to the number of spectrograms of the most common bird vocalization class.

For the model testing, the NIPS4BPlus dataset was used in addition to the Xeno-Canto data. The NIPS4Bplus dataset was used in two different forms, which we call "NIPS4BPlus" and "filtered NIPS4BPlus". While the first form contains all audio recordings of the NIPS4BPlus dataset, the second form contains only recordings that contain at least one of the ten classes of our dataset. The class distribution of both variants is given in Table 3.

| Class name                    | No. recordings in training set | No. spectrograms in training set | No. recordings in validation set | No. spectrograms in validation set | No. recordings in test set | No. spectrograms in test set |
| ----------------------------- | ------------------------------ | -------------------------------- | -------------------------------- | ---------------------------------- | -------------------------- | ---------------------------- |
| Cyanistes caeruleus, song     | 163                            | 5,374                            | 55                               | 1,426                              | 55                         | 1,407                        |
| Erithacus rubecula, song      | 300                            | 13,269                           | 100                              | 4,668                              | 100                        | 5,027                        |
| Fringilla coelebs, song       | 300                            | 9,890                            | 100                              | 3,145                              | 100                        | 3,260                        |
| Luscinia megarhynchos, song   | 298                            | 18,156                           | 99                               | 5,817                              | 100                        | 6,472                        |
| Parus major, song             | 300                            | 11,026                           | 100                              | 3,721                              | 100                        | 3,552                        |
| Phylloscopus collybita, call  | 201                            | 4,990                            | 67                               | 1,352                              | 68                         | 1,599                        |
| Phylloscopus collybita, song  | 300                            | 12,474                           | 100                              | 4,219                              | 100                        | 4,006                        |
| Sylvia atricapilla, song      | 300                            | 14,786                           | 100                              | 5,304                              | 100                        | 4,968                        |
| Troglodytes troglodytes, song | 300                            | 11,001                           | 100                              | 2,990                              | 100                        | 3,225                        |
| Turdus philomelos, song       | 300                            | 22,291                           | 100                              | 6,901                              | 100                        | 6,995                        |
| Noise                         | -                              | 22,291                           | -                                | 6,901                              | -                          | 6,995                        |
| **Total**                     | **2,762**                      | **145,548**                      | **921**                          | **46,444**                         | **923**                    | **47,506**                   |

Table 2: Class distribution of the Xeno-Canto dataset used for training the baseline models and for tuning hyperparameters

| Class name                                   | No. recordings in NIPS4BPlus dataset | No. spectrograms in NIPS4BPlus dataset |
| -------------------------------------------- | ------------------------------------ | -------------------------------------- |
| Cyanistes caeruleus, song                    | 9                                    | 30                                     |
| Erithacus rubecula, song                     | 19                                   | 47                                     |
| Fringilla coelebs, song                      | 12                                   | 24                                     |
| Luscinia megarhynchos, song                  | 18                                   | 38                                     |
| Parus major, song                            | 15                                   | 40                                     |
| Phylloscopus collybita, call                 | 9                                    | 12                                     |
| Phylloscopus collybita, song                 | 14                                   | 31                                     |
| Sylvia atricapilla, song                     | 8                                    | 17                                     |
| Troglodytes troglodytes, song                | 9                                    | 31                                     |
| Turdus philomelos, song                      | 17                                   | 55                                     |
| Noise (whole NIPS4BPlus dataset)             | 549                                  | 2147                                   |
| Noise (filtered NIPS4BPlus dataset)          | 87                                   | 333                                    |
| **Total (filtered NIPS4BPlus / NIPS4BPlus)** | **107 / 569**                        | **486 / 2,300**                        |

Table 3: Class distribution of the NIPS4BPlus dataset used for model evaluation

### Baseline Setting

To establish a baseline for our experiments, we first train a model with a standard setting (Table 4). We train the model as a multi-label classification model with a confidence threshold of 0.5. To account for noise factors such as data shuffling and random weight initialization, we perform three training runs. From each run, we select the model with the highest macro-average F1-score and report the average of the F1-scores of these best models.

| Parameter               | Baseline Setting                                                  |
| ----------------------- | ----------------------------------------------------------------- |
| Model architecture      | ResNet18                                                          |
| Fine-tuned model layers | conv. layer 3-4, fc.                                              |
| Optimizer               | Adam                                                              |
| Learning rate           | 0.0001                                                            |
| Learning rate scheduler | cosine annealing learning rate scheduler with η<sub>min</sub> = 0 |
| Batch size              | 1024                                                              |
| Loss function           | Cross-entropy loss                                                |

Table 4: Training setting of our baseline model

### Hyperparameter-Tuning

With the goal of improving the performance of our baseline models, we tuned several model hyperparameters. The tuned hyperparameters are batch size, learning rate, regularization, probability of dropout, and the number of layers fine-tuned during transfer learning. Since related work has reported a linear dependence between batch Size and learning Rate, we have tuned these parameters in a paired fashion. All other hyperparameters were tuned independently, assuming that there are no dependencies between them.

### Additional Data

In addition to hyperparameter tuning, we also study how quality and size of the training dataset affect model performance. For this purpose, we compare the performance of our baseline model with the performance of models trained on two modified training datasets: In the first case, we used a training dataset with lower audio quality. For this, we set the minimum aud, and we used a maximum of 500 audio samples per sound class. In the second case, we used the same quality settings (minimum quality "E", up to ten background species) but increased the maximum number of audio files per class to 1000.

</div>

# References

<div style="text-align: justify">

<!-- image-net -->

Jia Deng et al. “ImageNet: A large-scale hierarchical image database”. In: Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (San Francisco, USA). IEEE, 2009, pp. 248–255. ISBN: 978-1-4244-3992-8. DOI: 10.1109/CVPR.2009.5206848.

<!-- bird-clef-2014 -->

[1] Hervé Goëau et al. “LifeCLEF Bird Identification Task 2014”. In: Working Notes of CLEF2014 - Conference and Labs of the Evaluation Forum (Sheffield, United Kingdom). Ed. by Linda Cappellato et al. Vol. 1180. CEUR Workshop Proceedings. CEUR, Sept. 2014, pp. 585–597. URL: https://hal.inria.fr/hal-01088829.

<!-- bird-clef-2015 -->

[2] Hervé Goëau et al. “LifeCLEF Bird Identification Task 2015”. In: Working Notes of CLEF 2015 - Conference and Labs of the Evaluation Forum (Toulouse, France). Ed. by Linda Cappellato et al. Vol. 1391. CEUR Workshop Proceedings. CEUR, Sept. 2015, pp. 1–11. URL: http://ceur-ws.org/Vol- 1391/156- CR.pdf.

<!-- bird-clef-2016 -->

[3] Hervé Goëau et al. “LifeCLEF Bird Identification Task 2016: The arrival of Deep learning”. In: Working NotesofCLEF2016-ConferenceandLabsoftheEvaluationForum(E ́vora,Portugal).Ed.byKrisztian Balog et al. Vol. 1609. CEUR Workshop Proceedings. CEUR, Sept. 2016, pp. 440–449. URL: http: //ceur-ws.org/Vol-1609/16090440.pdf.

<!-- bird-clef-2017 -->

[4] Hervé Goëau et al. “LifeCLEF Bird Identification Task 2017”. In: Working Notes of CLEF 2017 - Conference and Labs of the Evaluation Forum (Dublin, Ireland). Ed. by Linda Cappellato et al. Vol. 1866. CEUR Workshop Proceedings. CEUR, Sept. 2017, pp. 1–9. URL: http://ceur-ws.org/Vol- 1866/invited%5C_paper%5C_8.pdf.

<!-- bird-clef-2018 -->

[5] Hervé Goëau et al. “Overview of BirdCLEF 2018: Monospecies vs. Sundscape Bird Identification”. In: Working Notes of CLEF 2018 - Conference and Labs of the Evaluation Forum (Avignon, France). Ed. by Linda Cappellato et al. Vol. 2125. CEUR Workshop Proceedings. CEUR, Sept. 2018, pp. 1–12. URL: http://ceur-ws.org/Vol-2125/invited%5C_paper%5C_9.pdf.

<!-- kahl-2017 -->

Stefan Kahl et al. “Large-Scale Bird Sound Classification using Convolutional Neural Networks”. In: Working Notes of CLEF 2017 - Conference and Labs of the Evaluation Forum (Dublin, Ireland). Ed. by Linda Cappellato et al. Vol. 1866. CEUR Workshop Proceedings. CEUR, Sept. 2017, pp. 1–14. URL: http://ceur-ws.org/Vol-1866/paper_143.pdff.

<!-- bird-clef-2019 -->

[6] Stefan Kahl et al. “Overview of BirdCLEF 2019: Large-Scale Bird Recognition in Soundscapes”. In: Working Notes of CLEF 2019 - Conference and Labs of the Evaluation Forum (Lugano, Switzerland). Ed. by Linda Cappellato et al. Vol. 2380. CEUR Workshop Proceedings. CEUR, July 2019, pp. 1–9. URL: http://ceur-ws.org/Vol-2380/paper_256.pdf.

<!-- bird-clef-2020 -->

[7] Stefan Kahl et al. “Overview of BirdCLEF 2020: Bird Sound Recognition in Complex Acoustic Environments”. In: Working Notes of CLEF 2020 - Conference and Labs of the Evaluation Forum (Thessa- loniki, Greece). Ed. by Linda Cappellato et al. Vol. 2696. CEUR Workshop Proceedings. CEUR, Sept. 2020, pp. 1–14. URL: http://ceur-ws.org/Vol-2696/paper%5C_262.pdf.

<!-- koh2019 -->

[8] Chih-Yuan Koh et al. “Bird Sound Classification using Convolutional Neural Networks”. In: Working Notes of CLEF 2019 - Conference and Labs of the Evaluation Forum (Lugano, Switzerland). Ed. by Linda Cappellato et al. Vol. 2380. CEUR Workshop Proceedings. CEUR, July 2019, pp. 1–10. URL: http://ceur-ws.org/Vol-2380/paper_68.pdf.

<!-- lasseck2013 -->

[9] Mario Lasseck. “Bird Song Classification in Field Recordings: Winning Solution for NIPS4B 2013 Competition”. In: Proceedings of the Workshop on Neural Information Processing Scaled for Bioin- formatics (Lake Tahoe, USA). Ed. by Herve ́ Glotin et al. Jan. 2013, pp. 176–181. URL: http:// sabiod.lis- lab.fr/nips4b/.

<!-- lasseck2015 -->

[10] Mario Lasseck. “Improved Automatic Bird Identification through Decision Tree based Feature Selection and Bagging”. In: Working Notes of CLEF 2015 - Conference and Labs of the Evaluation Forum (Toulouse, France). Ed. by Linda Cappellato et al. Vol. 1391. CEUR Workshop Proceedings. CEUR, Sept. 2015, pp. 1–12. URL: http://ceur-ws.org/Vol-1391/160-CR.pdff.

<!-- lasseck2014 -->

[11] Mario Lasseck. “Large-Scale Identification of Birds in Audio Recordings”. In: Working Notes of CLEF 2014 - Conference and Labs of the Evaluation Forum (Sheffield, United Kingdom). Ed. by Linda Cap- pellato et al. Vol. 1180. CEUR Workshop Proceedings. CEUR, Sept. 2014, pp. 585–597. URL: http: //ceur-ws.org/Vol-1180/CLEF2014wn-Life-Lasseck2014.pdf.

<!-- birdvox -->

[12] Vincent Lostanlen et al. “Birdvox-Full-Night: A Dataset and Benchmark for Avian Flight Call Detection”. In: International Conference on Acoustics, Speech and Signal Processing - ICASSP 2018 (Calgary, Canada). IEEE Computer Society, 2018, pp. 266–270. ISBN: 978-1-5386-4658-8. DOI: 10. 1109/ICASSP.2018.8461410.

<!-- nips4bplus -->

[13] Veronica Morfi et al. “NIPS4Bplus: a richly annotated birdsong audio dataset.” In: PeerJ Computer Science 5.e223 (Oct. 7, 2019), pp. 1–12. ISSN: 2376-5992. DOI: 10.7717/peerj-cs.223.

<!-- muller2018 -->

[14] Lukas Müller and Mario Marti. “Bird sound classification using a bidirectional LSTM”. In: Working
Notes of CLEF 2018 - Conference and Labs of the Evaluation Forum (Avignon, France). Ed. by Linda Cappellato et al. Vol. 2125. CEUR Workshop Proceedings. CEUR, Sept. 2018, pp. 1–13. URL: http: //ceur-ws.org/Vol-2125/paper_134.pdf.

<!-- audio-monitorung -->

[15] Cristian Pérez-Granados and Juan Traba. “Estimating bird density using passive acoustic monitoring: a review of methods and suggestions for further research”. In: Ibis 163.3 (Feb. 2021), pp. 765–783. ISSN: 1474-919X. DOI: 10.1111/ibi.12944.

<!-- ross -->

[16] Jesse C. Ross and Paul E. Allen. “Random Forest for improved analysis efficiency in passive acoustic monitoring”. In: Ecological Acoustics 21 (2014), pp. 34–39. ISSN: 1574-9541. DOI: 10.1016/j.ecoinf.2013.12.002.

<!-- monitorung-overview -->

[17] Dirk S. Schmeller et al. “Bird-monitoring in Europe – a first overview of practices, motivations and aims”. In:
Nature Conservation 2 (Aug. 2012), pp. 41–57. ISSN: 1314-6947. DOI: 10.3897/ natureconservation.2.3644.

<!-- sevilla-2017 -->

[18] Antoine Sevilla and Herve ́ Glotin. “Audio Bird Classification with Inception-v4 extended with Time and Time-Frequency Attention Mechanisms”. In: Working Notes of CLEF 2017 - Conference and Labs of the Evaluation Forum (Dublin, Ireland). Ed. by Linda Cappellato et al. Vol. 1866. CEUR Workshop Proceedings. CEUR, Sept. 2017, pp. 1–8. URL: http://ceur-ws.org/Vol-1866/paper_ 177.pdf.

<!-- sprengel-2016 -->

[19] Elias Sprengel et al. “Audio Based Bird Species Identification using Deep Learning Techniques”. In: WorkingNotesofCLEF2016-ConferenceandLabsoftheEvaluationForum(E ́vora,Portugal).Ed.by Krisztian Balog et al. Vol. 1609. CEUR Workshop Proceedings. CEUR, Sept. 2016, pp. 547–559. URL: http://ceur-ws.org/Vol-1609/16090547.pdf.

<!-- xeno-canto -->

[20] Willem-Pier Vellinga and Robert Planque ́. “The Xeno-canto collection and its relation to sound recog- nition and classification”. In: Working Notes of CLEF 2015 - Conference and Labs of the Evaluation Forum (Toulouse, France). Ed. by Linda Cappellato et al. Vol. 1391. CEUR Workshop Proceedings. CEUR, Sept. 2015, pp. 1–10. URL: http://ceur-ws.org/Vol-1391/166-CR.pdf.

<!-- warblr -->

[21] Dan Stowell et al. “Automatic acoustic detection of birds through deep learning: The first Bird Audio Detection challenge”. In: Methods in Ecology and Evolution 10.3 (Mar. 2019), pp. 368–380. ISSN: 2041-210X. DOI: 10.1111/2041-210X.13103.

</div>

<div style="text-align: justify">

# Technical Documentation

## Running the pipeline in a Jupyter notebook

The individual steps of our pipeline are implemented by Python classes. To run the pipeline in a Jupyter notebook, instances of the corresponding classes must be created and certain methods must be called on them. In the following, we describe the setup of a typical notebook-based pipeline. The described pipeline is included in our repository as an Jupyter notebook named `demo.ipynb`.

### Installing the dependencies locally

Our pipeline requires Python 3.7 or higher and Jupyter, please make sure that both are installed. We recommend installing the dependencies of our pipeline in a virtual environment. To create a virtual environment, run the venv module inside the directory where you want to create the virtual environment::

```bash
python3 -m venv env
```

Once you have created a virtual environment, you may activate it. On Windows, run:

```bash
env\Scripts\activate.bat
```

On Unix or MacOS, run:

```bash
source ./env/bin/activate
```

To install the dependencies, run:

```bash
python3 -m pip install -r requirements-notebook.txt
```

Next, make sure that your Jupyter uses the Python installation of the virtual environment as kernel. If this is the case, you can start setting up the pipeline.

### Installing the dependencies in Google Colab

Besides local Jupyter notebooks, [Google Colab](https://colab.research.google.com/) can also be used to run our pipeline. To setup the pipeline in Google Colab, create an empty notebook in Google Colab and clone the repository:

```bash
!git clone github.com/josafatburmeister/BirdSongIdentification
```

To install the dependencies, run:

```bash
%cd /content/BirdSongIdentification/
!python3 -m pip install -r requirements-colab.txt
```

Afterwards, restart the Google Colab runtime so that the installed dependencies are loaded. To do this, go to the "Runtime" menu item and select "Restart runtime".

### Setting up the File Manager

As described in the section "Data Exchange Between Pipeline Components" our pipeline uses specific directory structures to exchange data between pipeline stages. The management of these directories is implemented by the `FileManager` class. The following code snippet creates an instance of this class:

```python
from general import FileManager

file_manager = FileManager("./data")
```

As you can see, the FileManager class is initialized with the path of the parent directory where the pipeline directory structure is to be created. In this example we use the directory `./data` to store the pipeline's data. As shown in Listing 1 and Listing 3, the output directories of the data download stage and the spectrogram creation stage have a very similar structure. As you may have noticed, the directory and file names are chosen so that both stages can write their output to the same directory without naming conflicts. Therefore, in our example, we can use the `./data` directory to store the output of both pipeline stages and only need to create a single FileManager object.

### Setting up the Logger

All our pipeline components use a shared logger to output status information. Per default, the logging level is set to `INFO`. For our example, we set the logging level to verbose:

```python
import logging

from general import logger

logger.setLevel(logging.VERBOSE)
```

### Pipeline Stage 1: Data Download

With this, we are ready to run the first pipeline stage that downloads the datasets. For this, we have to create instances of the respective downloader classes. In our example, we download audio data from Xeno-Canto and compile train, validation and test sets from it. To do this, we create an instance of the `XenoCantoDownloader` class and then call the `create_datasets` method on it. The `create_datasets` has a number of parameters that can be used to specify which data should be downloaded from Xeno-Canto. Among them is the `species_list` parameter, which specifies which bird species and sound categories should be included in the datasets, and the `maximum_samples_per_class` parameter, which specifies the maximum number of audio files per class that should be downloaded. A detailed documentation of all parameters can be found in the docstrings of the XenoCantoDownloader class.

```python
from downloader import XenoCantoDownloader

xc_downloader = XenoCantoDownloader(file_manager)

species_list=["Turdus merula, song, call", "Erithacus rubecula, song, call"]

xc_downloader.create_datasets(
    species_list=species_list,
    use_nips4b_species_list=False,
    maximum_samples_per_class=10,
    maximum_recording_length=180,
    test_size=0.4,
    min_quality="A",
    sound_types=["song", "call"],
    sexes=None,
    life_stages=None,
    exclude_special_cases=True,
    maximum_number_of_background_species=0,
    clear_audio_cache=False,
    clear_label_cache=False,
    )
```

Note that the FileManager object we created earlier is passed to the XenoCantoDownloader constructor. This way we make sure that the downloaded data will be placed in the `./data` directory.

In addition to the data from Xeno-Canto, we would like to use the NIPS4BPlus dataset for model evaluation in our example. To download this dataset, we create an instance of the `NIPS4BPlusDownloader` class and call the `download_nips4bplus_dataset` method on it:

```python
from downloader import NIPS4BPlusDownloader

nips4bplus_downloader = NIPS4BPlusDownloader(file_manager)

species_list=["Turdus merula, song, call", "Erithacus rubecula, song, call"]

nips4bplus_downloader.download_nips4bplus_dataset(species_list=species_list)
```

### Pipeline Stage 2:

After downloading the audio files, the next step is to convert them into spectrograms. To do this, we create an instance of the `SpectrogramCreator` class and call the `...` method on it:

```python
from downloader import NIPS4BPlusDownloader

nips4bplus_downloader = NIPS4BPlusDownloader(file_manager)

species_list=["Turdus merula, song, call", "Erithacus rubecula, song, call"]

nips4bplus_downloader.download_nips4bplus_dataset(species_list=species_list)
```

Since we want to use the `./data` directory as both input and output directory of the spectrogram creation stage, we pass the same FileManager object to the `audio_file_manager` parameter and the `spectorgram_file_manager` parameter of the SpectrogramCreator constructor.

As described in section "", our pipeline implements a prefiltering of the spectrograms into "signal" and "noise spectrograms". The signal_threshold and noise_threshold parameters of the create_spectrograms_for_datasets method control which spectrograms are classified as "signal" spectrograms and which are classified as "noise" filtering. Since the NIPS4BPlus dataset includes time-accurate annotations, we do not need noise filtering there and therefore set the parameters to zero.

### Pipeline Stage 3: Model Training and Hyperparameter Tuning

We can now train image classification models on the spectrograms that were produced in the previous step. Since we do not yet know which hyperparameter settings are most suitable, we start by tuning the hyperparameters batch size and learning rate. For this, we create an instance of the `HyperparameterTuner` class and call the `tune_model` method on it:

```python
# run hyperparameter tuning for batch size and learning rate

from training import hyperparameter_tuner

tuner = hyperparameter_tuner.HyperparameterTuner(
    file_manager,
    architecture="resnet18",
    experiment_name="Tuning of batch size and learning rate",
    batch_size=[32, 64, 128],
    early_stopping=True,
    include_noise_samples=True,
    layers_to_unfreeze=["layer3", "layer4", "avg_pool", "fc"],
    learning_rate=[0.01, 0.001, 0.0001],
    learning_rate_scheduler="cosine",
    monitor="f1-score",
    multi_label_classification=True,
    multi_label_classification_threshold=0.5,
    number_epochs=1,
    number_workers=0,
    optimizer="Adam",
    patience=3,
    p_dropout=0,
    track_metrics=False,
    wandb_entity_name="",
    wandb_key="",
    wandb_project_name="",
    weight_decay=0
)

tuner.tune_model()

```

As you can see, the HyperparameterTuner constructor takes a number of parameters that specify the model architecture, hyperparameters, and training settings. A detailed documentation of these hyperparameters can be found in the docstrings of the class. Note that for the hyperparameters to be tuned, a list of values to be tested is passed.

After tuning batch Size and learning rate, we now decide to train a model with fixed hyperparameters. For this, we create an instance of the `ModelTrainer` class and call the method `train_model` on it:

```python
from training import training

trainer = training.ModelTrainer(
    file_manager,
    architecture="resnet18",
    experiment_name="Test run",
    batch_size=64,
    early_stopping=False,
    is_hyperparameter_tuning=False,
    include_noise_samples=True,
    layers_to_unfreeze=["layer3", "layer4", "avg_pool", "fc"],
    learning_rate=0.0001,
    learning_rate_scheduler="cosine",
    multi_label_classification=True,
    multi_label_classification_threshold=0.5,
    number_epochs=10,
    number_workers=0,
    optimizer="Adam",
    p_dropout=0,
    track_metrics=False,
    wandb_entity_name="",
    wandb_key="",
    wandb_project_name="",
    weight_decay=0
)

best_average_model, best_minimum_model, best_models_per_class = trainer.train_model()
```

As you might have noticed, the constructor parameters of the ModelTrainer class are mainly the same as in the HyperparameterTuner class.

### Pipeline Stage 4: Model Evaluation

Finally, we would like to evaluate our classification model both on the Xeno-Canto test set and the NIPS4BPlus dataset. For this, we create an instance of the `ModelEvaluator class` and call the method `evaluate_model` on it. Since the confidence of our model on unseen data may be lower than on the training set, we run the model evaluation with different confidence thresholds:

```python
from training import model_evaluator

for confidence_threshold in [0.3, 0.4, 0.5]:
    evaluator = model_evaluator.ModelEvaluator(file_manager,
                                               architecture="resnet18",
                                               batch_size=32,
                                               include_noise_samples=True,
                                               multi_label_classification=True,
                                               multi_label_classification_threshold=confidence_threshold,
                                               track_metrics=False)

    evaluator.evaluate_model(model=best_average_model, model_name=f"test_model_{confidence_threshold}", dataset="test")
    evaluator.evaluate_model(model=best_average_model, model_name=f"test_model_{confidence_threshold}",
                             dataset="nips4bplus")
    evaluator.evaluate_model(model=best_average_model, model_name=f"test_model_{confidence_threshold}",
                             dataset="nips4bplus_all")
```

## Running the Pipeline in Kubeflow

To run our pipeline in Kubeflow, a Kubernetes cluster with a Kubeflow installation is required. Currently, our pipeline supports Kubeflow version 1.0.0.

First, the Kubeflow pipeline definition need to be compiled. To compile a pipeline defintion for a cluster with CPU nodes only, run inside the repository:

```bash
python3 kubeflow_pipeline/compile_pipeline.py compile_pipeline --use_gpu False
```

If there are GPU nodes available in your cluster, run to enable GPU-accelerated model training:

```bash
python3 kubeflow_pipeline/compile_pipeline.py compile_pipeline --use_gpu True
```

The compilation command will produce a pipeline defintion file named `birdsong_pipeline.yaml`.

Next, build and push the pipeline's Docker image to the Docker registry of your cluster:

```bash
python3 kubeflow_pipeline/compile_pipeline.py build_docker_image
```

Now, open the Kubeflow UI of your cluster, go to the "Pipelines" section and click "Upload Pipeline". Upload the `birdsong_pipeline.yaml` file that you compiled before:

![Kubeflow-1](https://lh6.googleusercontent.com/uyVJARH9d8HByY_g0LSA-BhSVB7KVE-mFo28nQOua6b1ZfnfBiFgrtTPrX-kgFs1scES3JQn7GaYMg=w2880-h766)

After creating the pipeline, click the "Create run" button, to start a pipeline run:

![Kubeflow-2](https://lh5.googleusercontent.com/iuUn-Pk9TtN6Af64cBNBMiseAgOz27XNGPsRqSVGOaqydNyYSaFY_z5W1vb9UPZWO_ZUcvGWEFvjl7IQjnLt=w2210-h1478-rw)

## Implementing Custom Pipeline Components

### Implementing Custom Data Downloaders

To apply our machine learning pipeline to another dataset than Xeno-Canto or NIPS4BPlus, it is necessary to implement a custom downloader for that dataset. This downloader must place the data in a directory structure as described in the section "Data Exchange Between Pipeline Components" (Listing 1). Let's suppose we want to implement a downloader for a dataset named "test" that will be stored under the path `/data`. To do this, we need to implement the following things:

(1) In the `/data` directory, the `categories.txt` file needs to be created. This file must list the class names of the data set in the format shown in Listing 2.

(2) The audio files of the dataset must be placed in the `/data/test/audio` directory.

(3) A label file in CSV format must be placed under the path `/data/test/audio.csv`. This label file must have the format shown in Table 2.

To facilitate the implementation of custom downloaders, we provide a `Downloader` class in the `downloader` module. This class implements several helper functions and can be used as base class for custom downloaders. The constructor of the Downloader class takes a `FileManager` object as an argument. The FileManager object has to be must be initialized with the path of the directory where the dataset is to be created. In the above example, the FileManager would be initialized with the `/data` directory.

By deriving custom downloader classes from the Downloader class, the following utility functions can be used:

(1) The Downloader class provides a method `save_categories_file(categories)`. This method takes a list of class names and creates a `categories.txt` file from it.

(2) Within the Downloader class, the FileManager object can be accessed using `self.path`. The FileManager provides several methods that facilitate the handling and manipulation of file paths. For example, `self.path.data_folder(<dataset name>, "audio")` can be used to obtain the absolute path of the directory where the audio files must be placed.

(3) The Downloader class implements a method `save_label_file(labels, dataset_name)`. This method takes a Pandas dataframe and creates a label file from it. The provided dataframe must contain at least the columns "id", "file_path", "label", "start", "end".

### Implementing Custom Spectrogram Creation Methods

To implement custom spectrogram creation methods, we recommend to create a custom class that is derived from the SpectrogramCreator class. The derived class should override the method `__get_spectrogram`. This method takes an Numpy ndarray as input that contains the amplitude values of the audio chunk for which the spectrogram is to be created. General parameters of spectogram creation, such as hop length, window length, minimum frequency, maximum frequency can be accessed as object attributes.

Um eigene Methoden der Spektrogram-Erzeugung zu implementieren, empfehlen wir eine von der SpectrogramCreator abgeleitete Klasse zu implementieren. Um eine andere Art von Spektorgrammen zu erzeugen, muss in der abgeleiteten Klasse die Methode `__get_spectrogram` überschrieben werden. Diese erhält als Eingabe ein Numpy ndarray, welches die Amplitudenwerte für den Chunk enthält, für den das Spektrogramm erzeugt werden soll. Die allgemeinen Parameter der Spektrogramm-Erzeugung (hop length, window length, minimum frequency, maximum frequency usw.) sind als Objekt-Attribute in der Methode zugreifbar.

### Implementing Custom Model Architectures

To integrate custom model architecture into the pipeline, a custom model class needs to be created in the `models` module and the model class needs to be registered in the `model_architectures` of the `__init__.py`. A custom model class needs to fullfil the following requirements:

(1) The model needs to be implemented in Pytorch and should be derived from the torch.nn.Module class.

(2) The model's constructor should have the following signature:

```python

**init**(self, architecture: str, num_classes: int,
layers_to_unfreeze: Optional[Union[str, List[str]]], logger: Optional[VerboseLogger],
p_dropout: float)

```

(3) The model's `forward` method receives an three-dimensional Pytorch tensor containing the spectrogram image as input. It has to return a one-dimensional tensor with `num_classes` being the number of entries. The result tensor should contain the log-odds of the classes.

Um eigene Modellarchitekturen in die Pipeline zu integrieren, muss im Modul models eine eigene Modellklasse angelegt werden und die Modellklasse im Dictionary `model_architectures` in der `__init__.py` des Moduls eingetragen werden. An eigene Modellklassen werden folgende Anforderungen gestellt:

(1) Das Modell muss in Pytorch implementiert und eine von torch.nn.Module abgeleitete Klasse sein.

(1) Der Konstruktor der Modellklasse sollte die folgende Signatur haben:

```python
__init__(self, architecture: str, num_classes: int,
                 layers_to_unfreeze: Optional[Union[str, List[str]]], logger: Optional[VerboseLogger],
                 p_dropout: float)
```

(2) Die forward-Methode des Modells muss einen Pytorch-Tensor mit den Spektrogramm-Bildern entegegen nehmen und einen Ergebnistensor mit `num_classes` Einträgen liefern.

</div>
