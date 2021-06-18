from data_preparation import filepaths, downloader, spectrograms


class PipelineSteps:
    def download_xeno_canto_data(self, gcs_bucket, target_dir, species_list, **kwargs):
        path_manager = filepaths.PathManager(target_dir, gcs_bucket=gcs_bucket)
        xc_downloader = downloader.XenoCantoDownloader(path_manager)
        xc_downloader.create_datasets(species_list, **kwargs)

    def create_spectrograms(self, input_dir, gcs_bucket, target_dir, chunk_length, clear_spectrogram_cache=False):
        audio_path_manager = filepaths.PathManager(
            input_dir, gcs_bucket=gcs_bucket)
        spectrogram_path_manager = filepaths.PathManager(
            target_dir, gcs_bucket=gcs_bucket)
        spectrogram_creator = spectrograms.SpectrogramCreator(
            chunk_length, audio_path_manager, spectrogram_path_manager)

        spectrogram_creator.create_spectrograms_for_splits(
            splits=["train", "val", "test"], clear_spectrogram_cache=clear_spectrogram_cache)
