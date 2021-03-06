import matplotlib.pyplot as plt
import torchvision
import io

import PIL.Image
from torchvision.transforms import ToTensor
import wandb


def get_plot(input_times, output_times, gt_times):
    plt.close()
    plt.plot(input_times, output_times, 'r-')
    plt.plot(input_times, gt_times, 'g-')
    wandb.log({'Mapping': plt})
    # buf = io.BytesIO()
    # plt.savefig(buf, format='jpeg')
    # buf.seek(0)
    # image = PIL.Image.open(buf)
    # image = ToTensor()(image)
    # writer.add_image('Mapping', image, epoch)
    # buf.close()



def plot_diff(text_feats, pred_output_ft, gt_output_feats):
    plt.close()
    diff = pred_output_ft - gt_output_feats
    _min = min(text_feats.min(), pred_output_ft.min(), gt_output_feats.min(),
               diff.min())
    _max = max(text_feats.max(), pred_output_ft.max(), gt_output_feats.max(),
               diff.max())
    X = 512
    Y = 300
    images = []
    plt.figure(figsize=(30, 30))
    f, axarr = plt.subplots(2, 2)
    images.append(axarr[0, 1].imshow(text_feats[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[0], ax=axarr[0, 1])
    axarr[0, 1].set_title('Original image')
    images.append(axarr[1, 0].imshow(pred_output_ft[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[1], ax=axarr[1, 0])
    axarr[1, 0].set_title('Predicted features')
    images.append(axarr[1, 1].imshow(gt_output_feats[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[2], ax=axarr[1, 1])
    axarr[1, 1].set_title('Reverse mapping from \n reverse mapping')
    images.append(axarr[0, 0].imshow(diff[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[3], ax=axarr[0, 0])
    axarr[0, 0].set_title('Difference between reverse \n mapping and prediction')
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
    X, Y = 512, 300
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