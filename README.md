# ORCA-CLEAN
ORCA-CLEAN: A Deep Denoising Toolkit for Killer Whale Communication

## General Description
ORCA-CLEAN, is a deep denoising network designed for denoising of killer whale (<em>Orcinus Orca</em>) underwater recordings, not requiring any clean ground-truth samples, in order to improve the interpretation and analysis of bioacoustic signals by biologists and various machine learning algorithms.<br>ORCA-CLEAN was trained exclusively on killer whale signals resulting in a significant signal enhancement.  To show and prove the transferability, robustness and generalization of ORCA-CLEAN even more, a deep denoising was also conducted for bird sounds (<em>Myiopsitta monachus</em>) and human speech.<br><br>

## Reference
If ORCA-CLEAN is used for your own research please cite the following publication: ORCA-CLEAN: A Deep Denoising Toolkit for Killer Whale Communication (https://www.isca-speech.org/archive/Interspeech_2020/abstracts/1316.html)

```
@inproceedings{Bergler-OC-2020,
  author={Christian Bergler and Manuel Schmitt and Andreas Maier and Simeon Smeele and Volker Barth and Elmar Nöth},
  title={ORCA-CLEAN: A Deep Denoising Toolkit for Killer Whale Communication},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={1136--1140},
  doi={10.21437/Interspeech.2020-1316},
  url={http://dx.doi.org/10.21437/Interspeech.2020-1316}
}
```

## License
GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007 (GNU GPLv3)

## General Information
Manuscript Title: <em>ORCA-CLEAN: A Deep Denoising Toolkit for Killer Whale Communication</em>. The following link will provide all original/denoised sound examples (killer whales, birds, and human speech), listed in the paper. All spectrograms can be viewed visually and/or listened to directly.

<br>[ORCA-CLEAN - Denoised Audio/Spectrogram Examples](https://christianbergler.github.io/ORCA-CLEAN/)<br><br>

## Python, Python Libraries, and Version
ORCA-CLEAN is a deep learning algorithm which was implemented in Python (Version=3.8) (Operating System: Linux) together with the deep learning framework PyTorch (Version=1.7.1, TorchVision=0.8.2, TorchAudio=0.7.2). Moreover it requires the following Python libraries: Pillow, MatplotLib, Librosa, TensorboardX, Matplotlib, Soundfile, Scikit-image, Six, Opencv-python (recent versions). ORCA-CLEAN is currently compatible with Python 3.8 and PyTorch (Version=1.9.0, TorchVision=0.10.0, TorchAudio=0.9.0)

## Required Filename Structure for Training
In order to properly load and preprocess your data to train the network you need to prepare the filenames of your audio data clips to fit the following template/format:

Filename Template: LABEL-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav

1st-Element: LABEL = a placeholder for any kind of string which describes the label of the respective sample, e.g. call-N9, orca, echolocation, etc.

2nd-Element: ID = unique ID (natural number) to identify the audio clip

3rd-Element: YEAR = year of the tape when it has been recorded

4th-Element: TAPENAME = name of the recorded tape (has to be unique in order to do a proper data split into train, devel, test set by putting one tape only in only one of the three sets

5th-Element: STARTTIME = start time of the audio clip in milliseconds with respect to the original recording (natural number)

6th-Element: ENDTIME = end time of the audio clip in milliseconds with respect to the original recording(natural number)

Due to the fact that the underscore (_) symbol was chosen as a delimiter between the single filename elements please do not use this symbol within your filename except for separation.

Examples of valid filenames:

call-Orca-A12_929_2019_Rec-031-2018-10-19-06-59-59-ASWMUX231648_2949326_2949919

Label Name=call-Orca-A12, ID=929, Year=2019, Tapename=Rec-031-2018-10-19-06-59-59-ASWMUX231648, Starttime in ms=2949326, Starttime in ms=2949919

orca-vocalization_2381_2010_101BC_149817_150055.wav

Label Name=orca-vocalization, ID=2381, Year=2010, Tapename=101BC, Starttime in ms=149817, Starttime in ms=150055

## Required Directory Structure for Training
ORCA-CLEAN does its own training, validation, and test split of the entire provided data archive. The entire data could be either stored in one single folder and ORCA-CLEAN will generate the datasplit by creating three CSV files (train.csv, val.csv, and test.csv) representing the partitions and containing the filenames. There is also the possibility to have a main data folder and subfolders containing special type of files e.g. N9_calls, N1_calls, unknown_orca_calls, etc.. ORCA-CLEAN will create for each subfolder a stand-alone train/validation/test split and merges all files of each subfolder partition together. ORCA-CLEAN ensures that no audio files of a single tape are spread over training/validation/testing. Therefore it moves all files of one tape into only one of the three partitions. If there is only data from one tape or if one of the three partitions do not contain any files the training will not be started. By default ORCA-CLEAN uses 70% of the files for training, 15% for validation, and 15% for testing. In order to guarantee such a distribution it is important to have a similar amount of labeled files per tape. This is just an example command in order to start the training:

## Network Training
For a detailed description about each possible training option we refer to the usage/code in main.py (usage: main.py -h). This is just an example command in order to start network training:

```main.py --debug --max_train_epochs 150 --noise_dir_train path_to_noise_dir_train --noise_dir_val path_to_noise_dir_val --noise_dir_test path_to_noise_dir_test --lr 10e-4 --batch_size 16 --num_workers 6 --data_dir path_to_data_dir --model_dir path_to_model_dir --log_dir path_to_log_dir --checkpoint_dir path_to_checkpoint_dir --summary_dir path_to_summary_dir --augmentation 1 --random_val --n_fft 4096 --hop_length 441 --sequence_len 1280 --freq_compression linear```

## Network Testing and Evaluation
During training ORCA-CLEAN will be verified on an independent validation set. In addition ORCA-CLEAN will be automatically evaluated on the test set. As evaluation criteria the validation loss is utilized. All documented results/images and the entire training process could be reviewed via tensorboard and the automatic generated summary folder:

```tensorboard --logdir /directory_to_model/summaries/```

There exist also the possibility to evaluate your model on either a folder including unseen audio recordings (.wav) of different length, or a stand-alone audio tape. The prediction script (predict.py) reads and denoises incoming audio input (.wav). ORCA-CLEAN removes noise of the respective input spectrogram and reconstructs the denoised audio. Moreover, there is an option to visualize noisy network input and denoised network output. For a detailed description about each possible option we refer to the usage/code in predict.py (usage: predict.py -h). This is just an example command in order to start the prediction:

Example Command:

```predict.py --debug --model_path path_to_orca_clean_model_pickel --log_dir path_to_log_dir --sequence_len 1 --num_workers 6 --output_dir path_to_output_dir --no_cuda --visualize --input_file path_to_folder_with_audio_files_OR_entire_recording```