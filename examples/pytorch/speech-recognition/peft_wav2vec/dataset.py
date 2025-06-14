import numpy as np
import pandas as pd
from glob import glob
import os
# Parameters
audio_path = os.path.join(os.getcwd(), "libri10h/audio/")
csv_path = os.path.join(os.getcwd(), "libri10h/splits")


def remove_processed_files(csv_path):
  processed_files = glob(f"{csv_path}*/*_processed.tsv")
  len(processed_files)
  for f in processed_files:
    os.remove(f)
  print("Processed files removed successfully")



def prepare_data(audio_path, csv_path):
    split_list = ["librispeech"] #Change according to the number of splits
    for split in split_list:
        csv_file_list = glob(f"{csv_path}/{split}/*.tsv")
        for csv_file in csv_file_list:
            split_file = os.path.basename(csv_file).split(".")[0]
            print(split_file)
            df = pd.read_csv(csv_file, sep="\t")
            df["path"] = audio_path + df['audio']
            df = df.dropna(subset=["path"])
            df = df.drop(columns=['audio'])
            df = df.rename(columns={'path':'wav'})
            df = df.rename(columns={'sentence':'wrd'})
            df = df[["wav", "wrd"]]
            df.to_csv(f"{os.getcwd()}/{split_file}.csv", index=False)
            print(f"{split_file}_processed: ", len(df))

if __name__ == '__main__':
    remove_processed_files(csv_path)
    prepare_data(audio_path, csv_path)

