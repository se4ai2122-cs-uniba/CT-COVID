import os
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image as pil

# Usage example:
#   python covidx/ct/preprocessing.py --size 224 224 data/raw data/ct
#
if __name__ == '__main__':
    # Instantiate the command line arguments parser
    parser = argparse.ArgumentParser(description='COVIDx-CT-2A Image Dataset Processor.')
    parser.add_argument('src', type=str, help='The source directory of the COVIDx-CT-2A dataset.')
    parser.add_argument('dest', type=str, help='The destination directory of the processed CT dataset.')
    parser.add_argument(
        '--size', nargs=2, type=int, default=(224, 224), help='The size of the output images.'
    )
    args = parser.parse_args()

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
        df = dataset_labels[split]
        for _, sample in tqdm(df.iterrows(), total=len(df), miniters=100):
            filename = sample['filename']
            target = sample['class']
            box = (sample['xmin'], sample['ymin'], sample['xmax'], sample['ymax'])

            # Build the image filepath
            dest_filename = '{}.png'.format(os.path.splitext(filename)[0])
            src_filepath = os.path.join(args.src, '2A_images', filename)
            with pil.open(src_filepath) as img:
                # Preprocess the image
                img = img.convert(mode='L').crop(box).resize(args.size, resample=pil.BICUBIC)

            # Save the PNG image
            dest_filepath = os.path.join(args.dest, split, dest_filename)
            img.save(dest_filepath)

            # Append a data row to the preprocessed output CSV
            dest_dataset_labels[split] = dest_dataset_labels[split].append({
                'filename': dest_filename,
                'class': target
            }, ignore_index=True)

        dest_labels_filepath = os.path.join(args.dest, '{}.csv'.format(split))
        dest_dataset_labels[split].to_csv(dest_labels_filepath, index=False)
