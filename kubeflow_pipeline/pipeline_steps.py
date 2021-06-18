from data_preparation import filepaths, downloader, spectrograms


class PipelineSteps:
    def download_xeno_canto_data(self, gcs_bucket: str, target_dir: str, species_list: list[str], **kwargs):
        path_manager = filepaths.PathManager(target_dir, gcs_bucket=gcs_bucket)
        xc_downloader = downloader.XenoCantoDownloader(path_manager)
        xc_downloader.create_datasets(species_list, **kwargs)

    def create_spectrograms(self, input_dir: str, gcs_bucket: str, target_dir: str, chunk_length: int,
                            clear_spectrogram_cache: bool = False):
        audio_path_manager = filepaths.PathManager(
            input_dir, gcs_bucket=gcs_bucket)
        spectrogram_path_manager = filepaths.PathManager(
            target_dir, gcs_bucket=gcs_bucket)
        spectrogram_creator = spectrograms.SpectrogramCreator(
            chunk_length, audio_path_manager, spectrogram_path_manager)

        spectrogram_creator.create_spectrograms_for_splits(
            splits=["train", "val", "test"], clear_spectrogram_cache=clear_spectrogram_cache)
