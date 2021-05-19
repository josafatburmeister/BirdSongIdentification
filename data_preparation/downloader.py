import json
import os
import pandas as pd
import requests
import shutil
import tqdm


class XenoCantoDownloader():
    def __init__(self, target_dir):
        self.xeno_canto_url = "https://www.xeno-canto.org/api/2/recordings"
        self.audio_dir = os.path.join(target_dir, "audio")
        self.label_dir = os.path.join(target_dir, "labels")
        self.url_dir = os.path.join(target_dir, "url")

        if not os.path.exists(self.audio_dir):
            os.mkdir(self.audio_dir)

        if not os.path.exists(self.label_dir):
            os.mkdir(self.label_dir)

        if not os.path.exists(self.url_dir):
            os.mkdir(self.url_dir)

    def audio_dir_path(self, species_name):
        return os.path.join(self.audio_dir, species_name.replace(" ", "_"))

    def label_file_path(self, species_name):
        file_name = "{}.json".format(species_name.replace(" ", "_"))
        return os.path.join(self.label_dir, file_name)

    def url_file_path(self, species_name):
        file_name = "{}.txt".format(species_name.replace(" ", "_"))
        return os.path.join(self.url_dir, file_name)

    def download_file(self, url, target_file):
        r = requests.get(url, stream=True)

        if r.status_code == 200:
            r.raw.decode_content = True

            with open(target_file, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        else:
            raise NameError("File couldn\'t be retrieved")

    def download_xeno_canto_page(self, species_name, page=1):
        params = {"query": species_name, "page": page}

        response = requests.get(url=self.xeno_canto_url, params=params)

        return response.json()

    def download_species_labels(self, species_name):
        # download first page to get total number of pages and number of recordings
        first_page = self.download_xeno_canto_page(species_name)

        if int(first_page["numSpecies"]) != 1:
            raise NameError(
                "Multiple species found for {}".format(species_name))

        number_of_pages = int(first_page["numPages"])
        labels = first_page["recordings"]

        # download remaining pages
        progress_bar = tqdm.tqdm(
            total=number_of_pages, desc="Download label files for {}...".format(species_name), position=0)
        progress_bar.update(1)

        for page in range(2, number_of_pages+1):
            current_page = self.download_xeno_canto_page(
                species_name, page)

            labels.extend(current_page["recordings"])

            progress_bar.update(1)

        # store all labels as json file
        label_file_path = self.label_file_path(species_name)

        with open(label_file_path, "w") as label_file:
            json.dump(labels, label_file, indent=2, separators=(',', ':'))

        return labels, first_page["numRecordings"]

    def retrieve_urls_from_labels(self, species_name, labels):
        url_list = set()
        for item in labels:
            file = item["file"]
            url_list.add("https:{}".format(file))

        url_file_path = self.url_file_path(species_name)

        # store urls in txt file for later access
        with open(url_file_path, "w+") as url_file:
            for item in url_list:
                url_file.write("{}\n".format(item))

        return list(url_list)

    def download_species_data(self, species_name):
        label_file_path = self.label_file_path(species_name)
        if not os.path.exists(label_file_path):
            try:
                labels, _ = self.download_species_labels(species_name)
            except Exception as e:
                print(e)
                return
        else:
            with open(label_file_path) as label_file:
                labels = json.load(label_file)

        url_file_path = self.url_file_path(species_name)
        if not os.path.exists(url_file_path):
            urls = self.retrieve_urls_from_labels(species_name, labels)
        else:
            with open(url_file_path) as url_file:
                urls = url_file.read().splitlines()

        if not os.path.exists(self.audio_dir_path(species_name)):
            os.mkdir(self.audio_dir_path(species_name))

        progress_bar = tqdm.tqdm(
            total=len(urls), desc="Download audio files for {}...".format(species_name), position=0)

        for url in urls:
            file_id = url.split("/")[3]
            target_file = os.path.join(
                self.audio_dir_path(species_name), file_id + ".mp3")
            if not os.path.exists(target_file):
                try:
                    self.download_file(url, target_file)
                except Exception:
                    progress_bar.write(
                        "Could not retrieve file with id {}".format(file_id))
            progress_bar.update(1)

    def download_species_list_data(self, species_list):
        for species_name in species_list:
            self.download_species_data(species_name)

    def download_species_csv_file_data(self, csv_file, column_name="Scientific_name"):
        df = pd.read_csv(csv_file)
        species_list = df[column_name]
        self.download_species_list_data(species_list)
