import os
import re
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image as pil

# Usage example:
#   python covidx/ct/seq_preprocessing.py --size 224 224 data/raw data/seqct
#
if __name__ == '__main__':
    # Instantiate the command line arguments parser
    parser = argparse.ArgumentParser(description='COVIDx-CT-2A 3D Slice Images Dataset Processor.')
    parser.add_argument('src', type=str, help='The source directory of the COVIDx-CT-2A dataset.')
    parser.add_argument('dest', type=str, help='The destination directory of the processed SeqCT dataset.')
    parser.add_argument(
        '--size', nargs=2, type=int, default=(224, 224), help='The size of the output images.'
    )
    args = parser.parse_args()

    # TODO: load from the DVC's YAML file the "ct_depth" and "random_seed" parameters
    ct_depth = 16
    random_seed = 42

    # Set the random seed
    np.random.seed(random_seed)

    # Create the destination images directories
    train_path = os.path.join(args.dest, 'train')
    if not os.path.isdir(train_path):
        os.makedirs(train_path)
    valid_path = os.path.join(args.dest, 'valid')
    if not os.path.isdir(valid_path):
        os.makedirs(valid_path)
    test_path = os.path.join(args.dest, 'test')
    if not os.path.isdir(test_path):
        os.makedirs(test_path)

    # Load the labels CSVs
    column_names = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    train_label_filepath = os.path.join(args.src, 'train_COVIDx_CT-2A.txt')
    valid_label_filepath = os.path.join(args.src, 'val_COVIDx_CT-2A.txt')
    test_label_filepath = os.path.join(args.src, 'test_COVIDx_CT-2A.txt')
    dataset_labels = {
        'train': pd.read_csv(train_label_filepath, sep=' ', names=column_names),
        'valid': pd.read_csv(valid_label_filepath, sep=' ', names=column_names),
        'test': pd.read_csv(test_label_filepath, sep=' ', names=column_names)
    }

    # Initialize the preprocessed labels CSVs
    dest_column_names = ['filename', 'class']
    dest_dataset_labels = {
        'train': pd.DataFrame(columns=dest_column_names),
        'valid': pd.DataFrame(columns=dest_column_names),
        'test': pd.DataFrame(columns=dest_column_names)
    }

    # Process rows
    for split in ['train', 'valid', 'test']:
        ct_infos = []
        prev_ct_id = None
        df = dataset_labels[split]
        tk = tqdm(total=len(df), miniters=100)
        idx = 0

        while idx < len(df):
            sample = df.iloc[idx]
            filename = sample['filename']
            target = sample['class']
            box = (sample['xmin'], sample['ymin'], sample['xmax'], sample['ymax'])

            # Some incomprehensible regexes to match all the slices filenames and find the CT id
            result = re.search('^(.+)[_-]([0-9]+).png$', filename)
            if result is None:
                result = re.search('^(.+)[_-]IM([0-9]+).png$', filename)
            assert result is not None, 'Regex mismatch - {}'.format(filename)
            ct_id = result.group(1)

            if prev_ct_id is None:
                prev_ct_id = ct_id
                ct_infos.clear()
            if prev_ct_id == ct_id:
                filepath = os.path.join(args.src, '2A_images', filename)
                ct_infos.append((filepath, box, target))
                idx += 1
                tk.update()
                continue

            ct_id = prev_ct_id
            prev_ct_id = None
            num_ct_images = len(ct_infos)
            if num_ct_images < ct_depth:
                continue

            # Preprocess each slice
            ct_slices = []
            for img_filepath, img_box, img_target in ct_infos:
                with pil.open(img_filepath) as img:
                    img = img.convert(mode='L').crop(img_box).resize(args.size, resample=pil.BICUBIC)
                    ct_slices.append((img, img_target))

            # Use TIFF file format
            dest_filename = '{}.tiff'.format(ct_id)
            dest_filepath = os.path.join(args.dest, split, dest_filename)
            images, targets = zip(*ct_slices)
            assert len(set(targets)) == 1, 'Targets mismatch - {} : {}'.format(ct_id, targets)
            ct_class = targets[0]

            # Sub-sample the CT 3D images along the depth dimension
            step_size = num_ct_images // ct_depth
            sample_indices = np.arange(0, num_ct_images, step=step_size)
            mask = np.random.choice(np.arange(len(sample_indices)), size=ct_depth, replace=False)
            sample_indices = sample_indices[mask]
            sample_indices = list(sorted(sample_indices))
            filtered_images = []
            for i in sample_indices:
                filtered_images.append(images[i])
            filtered_images[0].save(dest_filepath, append_images=filtered_images[1:], save_all=True)

            # Append a data row to the preprocessed output CSV
            dest_dataset_labels[split] = dest_dataset_labels[split].append({
                'filename': dest_filename,
                'class': target
            }, ignore_index=True)

        tk.close()
        dest_labels_filepath = os.path.join(args.dest, '{}.csv'.format(split))
        dest_dataset_labels[split].to_csv(dest_labels_filepath, index=False)
