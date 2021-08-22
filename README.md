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

Following a similar approach, Koh et al. won the BirdCLEF challenge in 2018. They trained Resnet18 and Inception models on Mel-scaled spectrograms. For noise filtering, an image filter-based algorithm was used as in Sprengel et al. To address data imbalanced, data augmentation techniques were also used, e.g. brightness adjustments, blurring, slight rotations, crops and background noise [\cite{koh2018}].

</div>

## Methods

### Use Case Specification

<div style="text-align: justify">

Our work aims to implement an end-to-end machine learning pipeline that automates the training bird sound recognition models. Based on promising results of previous work, we focus on training deep convolutional neural networks (DCNN) trained as image classification models on spectrograms. For data preprocessing and spectrogram creation, we largely follow the approach described by Koh et al. [\cite{koh-2018}]. With respect to the dataset and model architecture used, we aim for a flexible and extensible pipeline design.

To demonstrate and evaluate the capability of our pipeline, we consider the following use case: As the primary data source, we use the Xeno-Canto database, which is the largest publicly available collection of bird sound recordings. To train DCNN models, we convert the audio files from Xeno-Canto into spectrograms. The audio recordings from Xeno-Canto are usually dominated by the vocalizations of one focus species, but may include other bird vocalizations in the background. Since Xeno-Canto only includes file-level annotations, but no time-accurate annotations, we use only the focal species for spectrogram labeling and ignore the background species. In contrast to recordings in Xeno-Canto, recordings from monitoring projects usually contain multiple overlapping bird vocalizations. To generalize our models to such use cases, we train multi-label classification models, even though our training data is single-label data. To evaluate the performance of our models in such a scenario, we use the NIPS4BPlus dataset as an additional test dataset [\cite{nips4bplus}]. In order to obtain time-accurate predictions, we split the audio files into fixed-length chunks (1 second) and create separate spectrograms and thus separate predictions for each chunk.

</div>

# References

<div style="text-align: justify">

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

<!-- bird-clef-2019 -->

[6] Stefan Kahl et al. “Overview of BirdCLEF 2019: Large-Scale Bird Recognition in Soundscapes”. In: Working Notes of CLEF 2019 - Conference and Labs of the Evaluation Forum (Lugano, Switzerland). Ed. by Linda Cappellato et al. Vol. 2380. CEUR Workshop Proceedings. CEUR, July 2019, pp. 1–9. URL: http://ceur-ws.org/Vol-2380/paper_256.pdf.

<!-- bird-clef-2020 -->

[7] Stefan Kahl et al. “Overview of BirdCLEF 2020: Bird Sound Recognition in Complex Acoustic Environments”. In: Working Notes of CLEF 2020 - Conference and Labs of the Evaluation Forum (Thessa- loniki, Greece). Ed. by Linda Cappellato et al. Vol. 2696. CEUR Workshop Proceedings. CEUR, Sept. 2020, pp. 1–14. URL: http://ceur-ws.org/Vol-2696/paper%5C_262.pdf.

<!-- koh2018 -->

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
