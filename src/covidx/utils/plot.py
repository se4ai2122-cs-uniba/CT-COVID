import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torchvision.utils

MISS_IMAGES_PATH = 'misses'


def save_history(history, filepath):
    fig, axs = plt.subplots(2, tight_layout=True)
    axs[0].set_title('loss')
    axs[0].plot(history['train']['loss'], label='train')
    axs[0].plot(history['validation']['loss'], label='validation')
    axs[1].set_title('accuracy')
    axs[1].plot(history['train']['accuracy'], label='train')
    axs[1].plot(history['validation']['accuracy'], label='validation')
    axs[0].legend()
    axs[1].legend()
    fig.savefig(filepath, dpi=192)


def plot_cxr2_errors(dataset, errors, model_name):
    label_names = ['negative', 'positive']

    miss_images_path = os.path.join('cxr2-' + MISS_IMAGES_PATH, model_name)
    if not os.path.isdir(miss_images_path):
        os.mkdir(miss_images_path)

    for (idx, label) in errors:
        tensor, _ = dataset.__getitem__(idx)
        img = tensor.numpy().squeeze()
        plt.figure(figsize=(4, 4))
        plt.title(label_names[label])
        path = os.path.join(miss_images_path, 'predict_error{}_{}.png'.format(idx, label_names[label]))
        plt.imsave(path, img, cmap='gray', format='png')
        plt.close()


def save_binary_attention_map(filepath_or_stream, img, att1, att2):
    # Move on CPU
    img = img.cpu()
    att1 = att1.cpu()
    att2 = att2.cpu()

    # Upsample attention maps
    att1 = torch.nn.functional.interpolate(att1, scale_factor=(16, 16), mode='bilinear', align_corners=True)
    att2 = torch.nn.functional.interpolate(att2, scale_factor=(32, 32), mode='bilinear', align_corners=True)

    # Combine attention maps
    att = torch.sqrt(att1 * att2)
    att = (att - att.min()) / (att.max() - att.min())
    att = 255.0 * att

    # Convert to Numpy arrays
    img = 255.0 * img
    att = att.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    img = img.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

    # Apply OTZU Method
    _, att = cv2.threshold(att, 0, 255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply red colormap
    tmp = np.zeros((att.shape[0], att.shape[1], 3), dtype=np.uint8)
    tmp[:, :, 0] = att
    att = tmp

    # Combine heatmap and image
    img = np.repeat(img, repeats=3, axis=2)
    img_att = cv2.addWeighted(img, 0.7, att, 0.3, 0.0)

    # Plot the image and heatmaps
    img = torch.tensor(img).permute(2, 0, 1) / 255.0
    img_att = torch.tensor(img_att).permute(2, 0, 1) / 255.0
    img_grid = torch.stack([img, img_att])
    torchvision.utils.save_image(img_grid, filepath_or_stream, padding=4, nrow=2, pad_value=0, format='png')


def save_attention_map(filepath, img, att1, att2):
    # Move on CPU
    img = img.cpu()
    att1 = att1.cpu()
    att2 = att2.cpu()

    # Un-normalize attention maps
    att1 = (att1 - att1.min()) / (att1.max() - att1.min()) * 255
    att2 = (att2 - att2.min()) / (att2.max() - att2.min()) * 255
    img = img * 255.0

    # Upsample attention maps
    att1 = torch.nn.functional.interpolate(att1, scale_factor=(16, 16), mode='bilinear', align_corners=True)
    att2 = torch.nn.functional.interpolate(att2, scale_factor=(32, 32), mode='bilinear', align_corners=True)

    # Convert to Numpy arrays
    att1 = att1.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    att2 = att2.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    img = img.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

    # Apply colormap
    att1 = cv2.applyColorMap(att1, cv2.COLORMAP_JET)
    att2 = cv2.applyColorMap(att2, cv2.COLORMAP_JET)

    # Combine heatmaps
    img = np.repeat(img, repeats=3, axis=2)
    img_att1 = cv2.addWeighted(img, 0.7, att1, 0.3, 0.0)
    img_att2 = cv2.addWeighted(img, 0.7, att2, 0.3, 0.0)
    img_att1 = cv2.cvtColor(img_att1, cv2.COLOR_BGR2RGB)
    img_att2 = cv2.cvtColor(img_att2, cv2.COLOR_BGR2RGB)

    # Plot the image and heatmaps
    img = torch.tensor(img).permute(2, 0, 1) / 255.0
    img_att1 = torch.tensor(img_att1).permute(2, 0, 1) / 255.0
    img_att2 = torch.tensor(img_att2).permute(2, 0, 1) / 255.0
    img_grid = torch.stack([img, img_att1, img_att2])
    torchvision.utils.save_image(img_grid, filepath, padding=4, nrow=3, pad_value=0)


def save_attention_map_sequence(filepath, img, att1, att2, seqs):
    # Move on CPU
    img = img.cpu()
    att1 = att1.cpu()
    att2 = att2.cpu()
    seqs = seqs.cpu()

    # Un-normalize attention maps
    att1 = torch.stack([(a - a.min()) / (a.max() - a.min()) for a in att1[0]]).unsqueeze(0)
    att2 = torch.stack([(a - a.min()) / (a.max() - a.min()) for a in att2[0]]).unsqueeze(0)
    seqs = (seqs - seqs.min()) / (seqs.max() - seqs.min())
    img = img * 255.0

    # Upsample attention maps
    att1 = torch.nn.functional.interpolate(att1, scale_factor=(16, 16), mode='bilinear', align_corners=True)
    att2 = torch.nn.functional.interpolate(att2, scale_factor=(32, 32), mode='bilinear', align_corners=True)

    # Weight attention maps according to sequence's attentions
    att1 = 255.0 * torch.sqrt(att1 * seqs.unsqueeze(2).unsqueeze(3))
    att2 = 255.0 * torch.sqrt(att2 * seqs.unsqueeze(2).unsqueeze(3))

    # Conver to Numpy arrays
    att1 = att1.squeeze(0).numpy().astype(np.uint8)
    att2 = att2.squeeze(0).numpy().astype(np.uint8)
    img = img.squeeze(0).numpy().astype(np.uint8)
    img = np.expand_dims(img, axis=3)
    img = np.repeat(img, repeats=3, axis=3)

    # Apply colormap
    att1 = np.array([cv2.applyColorMap(np.expand_dims(a, axis=2), cv2.COLORMAP_JET) for a in att1])
    att2 = np.array([cv2.applyColorMap(np.expand_dims(a, axis=2), cv2.COLORMAP_JET) for a in att2])

    # Combine heatmaps
    img_att1 = np.array([cv2.addWeighted(i, 0.7, a, 0.3, 0.0) for i, a in zip(img, att1)])
    img_att2 = np.array([cv2.addWeighted(i, 0.7, a, 0.3, 0.0) for i, a in zip(img, att2)])

    # Convert from BGR to RGB
    img_att1 = np.array([cv2.cvtColor(a, cv2.COLOR_BGR2RGB) for a in img_att1])
    img_att2 = np.array([cv2.cvtColor(a, cv2.COLOR_BGR2RGB) for a in img_att2])

    # Convert again to tensors
    img = torch.tensor(img).permute(0, 3, 1, 2) / 255.0
    img_att1 = torch.tensor(img_att1).permute(0, 3, 1, 2) / 255.0
    img_att2 = torch.tensor(img_att2).permute(0, 3, 1, 2) / 255.0
    n, c, h, w = img.shape

    # Plot the image and heatmaps in a grid
    img_grid = torch.cat([img, img_att1, img_att2], dim=0)
    img_grid = img_grid.reshape(3, n, c, h, w).transpose(1, 0).reshape(n * 3, c, h, w)
    torchvision.utils.save_image(img_grid, filepath, padding=4, nrow=3, pad_value=0)
