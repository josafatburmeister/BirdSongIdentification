from data_preparation import filepaths, downloader


class PipelineSteps:
    def download_xeno_canto_data(self, gcs_path, target_dir, species_list, maximum_samples_per_class=100, test_size=0.35,
                                 min_quality="E", sound_types=None, sexes=None,
                                 life_stages=None, exclude_special_cases=True,
                                 maximum_number_of_background_species=None, verbose=False):
        path_manager = filepaths.PathManager(target_dir, gcs_path=gcs_path)
        xc_downloader = downloader.XenoCantoDownloader(path_manager)
        xc_downloader.create_datasets(species_list, maximum_samples_per_class=100, test_size=0.35, min_quality="E",
                                      sound_types=None, sexes=None,
                                      life_stages=None, exclude_special_cases=True,
                                      maximum_number_of_background_species=None, verbose=False)
