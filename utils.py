import matplotlib.pyplot as plt
import torchvision
import io

import PIL.Image
from torchvision.transforms import ToTensor
import wandb


def get_plot(input_times, output_times, gt_times, split=None, gt_dict_dialog=None):
    plt.close()
    plt.plot(input_times, output_times, 'r-', label='prediction')
    if split is None:
        plt.scatter(gt_times[0], gt_times[1], label='gt')
    else:
        plt.scatter(gt_times[0][split['train']], gt_times[1][split['train']], c='g', label='gt train')
        plt.scatter(gt_times[0][split['val']], gt_times[1][split['val']], c='orange', label='gt validation')

    if gt_dict_dialog is not None:
        plt.scatter(gt_dict_dialog[0], gt_dict_dialog[1], c='b', label='pseudo-gt dialogs')
    plt.legend()
    wandb.log({'Mapping': plt})
    # buf = io.BytesIO()
    # plt.savefig(buf, format='jpeg')
    # buf.seek(0)
    # image = PIL.Image.open(buf)
    # image = ToTensor()(image)
    # writer.add_image('Mapping', image, epoch)
    # buf.close()



def plot_diff(input, prediction, output,
              titles=None):
    if titles is None:
        titles = ['Original image', 'Predicted features',
                  'Reverse mapping from \n reverse mapping',
                  'Difference between reverse \n mapping and prediction']
    plt.close()
    diff = prediction - output
    _min = min(input.min(), prediction.min(), output.min(),
               diff.min())
    _max = max(input.max(), prediction.max(), output.max(),
               diff.max())
    X = -1 #512
    Y = -1 #300
    images = []
    plt.figure(figsize=(30, 30))
    f, axarr = plt.subplots(2, 2)
    images.append(axarr[0, 1].imshow(input[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[0], ax=axarr[0, 1])
    axarr[0, 1].set_title(titles[0])
    images.append(axarr[1, 0].imshow(prediction[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[1], ax=axarr[1, 0])
    axarr[1, 0].set_title(titles[1])
    images.append(axarr[1, 1].imshow(output[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[2], ax=axarr[1, 1])
    axarr[1, 1].set_title(titles[2])
    images.append(axarr[0, 0].imshow(diff[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[3], ax=axarr[0, 0])
    axarr[0, 0].set_title(titles[3])
    wandb.log({'Warping Visualization': plt})
    # buf = io.BytesIO()
    # plt.savefig(buf, format='jpeg')
    # buf.seek(0)
    # image = PIL.Image.open(buf)
    # image = ToTensor()(image)
    # writer.add_image('Training Visualization', image, epoch)
    # buf.close()


def visualize_input(input, output):
    plt.close()
    _min = min(input.min(), output.min())
    _max = max(input.max(), output.max())
    X, Y = -1, -1 #512, 300
    images = []
    plt.figure(figsize=(30, 30))
    f, axarr = plt.subplots(1, 2)
    images.append(axarr[0].imshow(input[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[0], ax=axarr[0])
    axarr[0].set_title('Input features')
    images.append(axarr[1].imshow(output[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[1], ax=axarr[1])
    axarr[1].set_title('Output features')
    wandb.log({'Input/Output Visualization': plt})
    # buf = io.BytesIO()
    # plt.savefig(buf, format='jpeg')
    # buf.seek(0)
    # image = PIL.Image.open(buf)
    # image = ToTensor()(image)
    # writer.add_image('Input Visualization', image, 0)
    # buf.close()
