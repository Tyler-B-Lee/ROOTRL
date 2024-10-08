# simple script to change the zip files in the zoo
# directory to have the ending .pt

import os
import config

def main():
    zoo_dir = config.MODELDIR
    for zoo_dir_item in os.listdir(zoo_dir):
        # check folders only
        if not os.path.isdir(os.path.join(zoo_dir, zoo_dir_item)):
            continue
        print(f'{zoo_dir_item=}')
        for model_name in os.listdir(os.path.join(zoo_dir, zoo_dir_item)):
            print(f'{model_name=}')
            if model_name.endswith(".zip"):
                os.rename(os.path.join(zoo_dir, zoo_dir_item, model_name), os.path.join(zoo_dir, zoo_dir_item, model_name[:-4] + ".pt"))

if __name__ == "__main__":
    main()