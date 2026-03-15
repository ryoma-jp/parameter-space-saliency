import yaml
import urllib
import torch
import torch.backends.cudnn as cudnn
import torchvision
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings

from utils import show_heatmap_on_image, test_and_find_incorrectly_classified, transform_raw_image
from parameter_saliency.saliency_model_backprop import SaliencyModel, find_testset_saliency
from model_adapter.factory import build_model_adapter
from task_adapter.classification import ClassificationTaskAdapter
from target.spec import TargetSpec, TargetType

parser = argparse.ArgumentParser(description='Parameter-Space and Input-Space Saliency')

# ----- Model -----
parser.add_argument('--model', default='resnet50', type=str,
                    help='torchvision model name (used when --model_source torchvision)')
parser.add_argument('--model_source', default='torchvision',
                    choices=['torchvision', 'custom_module'],
                    help='source of the model')
parser.add_argument('--model_class_path', default=None, type=str,
                    help='fully-qualified class path for custom_module, e.g. mypkg.models.MyNet')
parser.add_argument('--model_weights_path', default=None, type=str,
                    help='path to weights checkpoint for custom_module')

# ----- Dataset -----
parser.add_argument('--data_to_use', default='ImageNet', type=str,
                    help='which dataset to use (currently only ImageNet)')
parser.add_argument('--imagenet_val_path', default='<insert-ImageNet-val-path-here>',
                    type=str, help='ImageNet validation set path')

# ----- Target -----
parser.add_argument('--target_type', default='true_label',
                    choices=['true_label', 'predicted_top1', 'specified_class'],
                    help='what to use as the saliency target')
parser.add_argument('--target_class_id', default=None, type=int,
                    help='class id when --target_type specified_class')

# ----- Label map -----
parser.add_argument('--label_map_path', default=None, type=str,
                    help='path to YAML label map {int: str}; '
                         'if omitted, ImageNet labels are downloaded for torchvision models')

# ----- Output -----
parser.add_argument('--figure_folder_name', default='input_space_saliency', type=str,
                    help='subdirectory under output_root for input-space figures')
parser.add_argument('--output_root', default='figures', type=str,
                    help='root directory to save output figures')

# ----- Saliency options -----
parser.add_argument('--signed', action='store_true', help='Use signed saliency')
parser.add_argument('--logit', action='store_true',
                    help='[deprecated, not implemented] Use logits rather than cross-entropy')
parser.add_argument('--logit_difference', action='store_true',
                    help='[deprecated, not implemented] Use logit difference as loss')

# ----- Input-space saliency (boosting) -----
parser.add_argument('--boost_factor', default=100.0, type=float,
                    help='boost factor for salient filters')
parser.add_argument('--k_salient', default=10, type=int,
                    help='number of top salient filters to boost')
parser.add_argument('--compare_random', action='store_true',
                    help='boost k random filters for comparison')

# ----- SmoothGrad-like options -----
parser.add_argument('--noise_iters', default=1, type=int,
                    help='number of noise iterations to average')
parser.add_argument('--noise_percent', default=0, type=float,
                    help='std of the noise')

# ----- Reference image -----
parser.add_argument('--image_path',
                    default='raw_images/great_white_shark_mispred_as_killer_whale.jpeg',
                    type=str, help='path to a raw image file')
parser.add_argument('--image_target_label', default=None, type=int,
                    help='ground-truth class index (0-based) for the raw image')
parser.add_argument('--reference_id', default=None, type=int,
                    help='index of image in the validation set')

def _cache_key(args) -> str:
    """Derive a filesystem-safe cache key from model arguments."""
    if args.model_source == 'torchvision':
        return args.model
    return args.model_class_path.split('.')[-1]


def _load_label_map(args) -> dict:
    """Load or download a label map {int -> str}."""
    if args.label_map_path:
        with open(args.label_map_path) as f:
            return yaml.load(f, Loader=yaml.Loader)
    if args.model_source == 'torchvision':
        label_url = urllib.request.urlopen(
            'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a'
            '/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        )
        raw = ''.join(f.decode('utf-8') for f in label_url)
        return yaml.load(raw, Loader=yaml.Loader)
    return {}


def _build_model_spec(args) -> dict:
    """Build a model spec dict from CLI args."""
    if args.model_source == 'torchvision':
        return {'source': 'torchvision', 'name': args.model, 'pretrained': True}
    spec = {'source': 'custom_module', 'class_path': args.model_class_path}
    if args.model_weights_path:
        spec['weights_path'] = args.model_weights_path
    return spec


def save_gradients(grads_to_save, args, reference_image, inv_transform_test):
    grads_to_save, _ = grads_to_save.max(dim=1)
    grads_to_save = grads_to_save.detach().cpu().numpy().reshape((224, 224))
    grads_to_save = np.abs(grads_to_save)
    # grads_to_save[grads_to_save < 0] = 0.0

    #Percentile thresholding
    grads_to_save[grads_to_save > np.percentile(grads_to_save, 99)] = np.percentile(grads_to_save, 99)
    grads_to_save[grads_to_save < np.percentile(grads_to_save, 90)] = np.percentile(grads_to_save, 90)

    plt.figure()
    plt.imshow(grads_to_save)

    save_path = os.path.join(args.output_root, args.figure_folder_name)
    os.makedirs(save_path, exist_ok=True)
    ck = _cache_key(args)
    save_name = (str(args.reference_id) if args.reference_id is not None
                 else args.image_path.split('/')[-1].split('.')[0])
    save_name += '_' + ck
    plt.axis('off')

    grads_to_save = (grads_to_save - np.min(grads_to_save)) / (np.max(grads_to_save) - np.min(grads_to_save))

    reference_image_to_compare = inv_transform_test(reference_image[0].cpu()).permute(1, 2, 0)
    gradients_heatmap = np.ones_like(grads_to_save) - grads_to_save
    gradients_heatmap = cv2.GaussianBlur(gradients_heatmap, (3, 3), 0)

    heatmap_superimposed = show_heatmap_on_image(
        reference_image_to_compare.detach().cpu().numpy(), gradients_heatmap
    )
    plt.imshow(heatmap_superimposed)
    plt.axis('off')
    out_path = os.path.join(save_path, f'input_saliency_heatmap_{save_name}.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Input space saliency saved to {out_path}\n')
    return

def compute_input_space_saliency(
    reference_inputs, reference_targets, net, args,
    task_adapter, target_spec,
    testset_mean_stat=None, testset_std_stat=None,
    inv_transform_test=None, readable_labels=None,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Log prediction summary via TaskAdapter
    task_adapter.summarize_prediction(
        net,
        reference_inputs.to(device),
        reference_targets.to(device),
        readable_labels or {},
    )

    filter_saliency_model = SaliencyModel(
        net, task_adapter,
        device=device, mode='std',
        aggregation='filter_wise', signed=args.signed,
    )
    reference_inputs  = reference_inputs.to(device)
    reference_targets = reference_targets.to(device)

    grad_samples = []
    for _ in range(args.noise_iters):
        perturbed_inputs = reference_inputs.detach().clone()
        perturbed_inputs = (
            (1 - args.noise_percent) * perturbed_inputs
            + args.noise_percent * torch.randn_like(perturbed_inputs)
        )
        perturbed_inputs.requires_grad_()

        filter_saliency = filter_saliency_model(
            perturbed_inputs, reference_targets, target_spec,
            testset_mean_abs_grad=testset_mean_stat,
            testset_std_abs_grad=testset_std_stat,
        ).to(device)

        if args.compare_random:
            sorted_filters = torch.randperm(filter_saliency.size(0)).cpu().numpy()
        else:
            sorted_filters = torch.argsort(filter_saliency, descending=True).cpu().numpy()

        filter_saliency_boosted = filter_saliency.detach().clone()
        filter_saliency_boosted[sorted_filters[:args.k_salient]] *= args.boost_factor

        matching_criterion = torch.nn.CosineSimilarity()
        matching_loss = matching_criterion(
            filter_saliency[None, :], filter_saliency_boosted[None, :]
        )
        matching_loss.backward()

        grad_samples.append(perturbed_inputs.grad.detach().cpu())

    grads_to_save = torch.stack(grad_samples).mean(0)
    return grads_to_save, filter_saliency


def sort_filters_layer_wise(filter_profile, layer_to_filter_id, filter_std = None):
    layer_sorted_profile = []
    means = []
    stds = []
    for layer in layer_to_filter_id:
        layer_inds = layer_to_filter_id[layer]
        layer_sorted_profile.append(np.sort(filter_profile[layer_inds])[::-1])
        means.append(np.ones_like(filter_profile[layer_inds])*np.mean(filter_profile[layer_inds]))
        if filter_std is not None:
            stds.append(filter_std[layer_inds][np.argsort(filter_profile[layer_inds])[::-1]])
    layer_sorted_profile = np.concatenate(layer_sorted_profile)
    sal_means = np.concatenate(means)
    if filter_std is not None:
        sal_stds = np.concatenate(stds)
        return layer_sorted_profile, sal_means, sal_stds
    else:
        return layer_sorted_profile, sal_means

if __name__ == '__main__':

    torch.manual_seed(40)
    np.random.seed(40)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args   = parser.parse_args()

    if args.logit or args.logit_difference:
        raise NotImplementedError('--logit and --logit_difference are not yet implemented.')

    os.makedirs(args.output_root, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    print('==> Building model..')
    adapter = build_model_adapter(_build_model_spec(args))
    net     = adapter.build_model()

    # Saliency units: derived BEFORE DataParallel to keep layer names clean
    layer_to_filter_id = adapter.iter_saliency_units(net)
    total_filters = sum(len(v) for v in layer_to_filter_id.values())
    print(f'Total filters: {total_filters}')
    print(f'Total layers:  {len(layer_to_filter_id)}')

    transform_test     = adapter.get_preprocess()
    inv_transform_test = adapter.get_inv_preprocess()

    net = net.to(device)
    net.eval()
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark    = False
        cudnn.deterministic = True

    # ------------------------------------------------------------------ #
    # Task & target                                                        #
    # ------------------------------------------------------------------ #
    task_adapter = ClassificationTaskAdapter()
    target_spec  = TargetSpec.from_args(args.target_type, args.target_class_id)

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    print('==> Preparing data..')
    testset = None
    if args.data_to_use == 'ImageNet':
        images_path = args.imagenet_val_path
    else:
        raise NotImplementedError(f'data_to_use={args.data_to_use!r} is not supported.')

    if images_path != '<insert-ImageNet-val-path-here>':
        testset = torchvision.datasets.ImageFolder(images_path, transform=transform_test)
    else:
        print(
            '\n  ImageNet validation set path is not specified.\n'
            '  The code will only work with --image_path and --image_target_label.\n'
            '  --reference_id requires the validation set path.\n'
        )

    # ------------------------------------------------------------------ #
    # Label map                                                            #
    # ------------------------------------------------------------------ #
    readable_labels = _load_label_map(args)

    # ------------------------------------------------------------------ #
    # Cache paths                                                          #
    # ------------------------------------------------------------------ #
    ck = _cache_key(args)
    model_helpers_root_path = os.path.join('helper_objects', ck)
    os.makedirs(model_helpers_root_path, exist_ok=True)

    # Include target_type in stats filename so different targets use separate caches
    target_suffix = '' if args.target_type == 'true_label' else f'_{args.target_type}'

    # ------------------------------------------------------------------ #
    # Inference cache                                                      #
    # ------------------------------------------------------------------ #
    inference_file = os.path.join(
        model_helpers_root_path,
        f'ImageNet_val_inference_results_{ck}.pth',
    )
    if os.path.isfile(inference_file):
        inf_results           = torch.load(inference_file)
        incorrect_id          = inf_results['incorrect_id']
        incorrect_predictions = inf_results['incorrect_predictions']
        correct_id            = inf_results['correct_id']
    elif testset is not None:
        warnings.warn('Computing inference; check cache filenames if unintended.')
        incorrect_id, incorrect_predictions, correct_id = \
            test_and_find_incorrectly_classified(net, testset)
        torch.save(
            {'incorrect_id': incorrect_id,
             'incorrect_predictions': incorrect_predictions,
             'correct_id': correct_id},
            inference_file,
        )

    # ------------------------------------------------------------------ #
    # Testset saliency statistics cache                                    #
    # ------------------------------------------------------------------ #
    filter_stats_file = os.path.join(
        model_helpers_root_path,
        f'ImageNet_val_saliency_stat_{ck}{target_suffix}_filter_wise.pth',
    )
    if os.path.isfile(filter_stats_file):
        filter_stats                 = torch.load(filter_stats_file)
        filter_testset_mean_abs_grad = filter_stats['mean']
        filter_testset_std_abs_grad  = filter_stats['std']
    elif testset is not None:
        warnings.warn('Computing testset stats; check cache filenames if unintended.')
        filter_testset_mean_abs_grad, filter_testset_std_abs_grad = find_testset_saliency(
            net, testset, 'filter_wise', task_adapter, target_spec, signed=args.signed,
        )
        torch.save(
            {'mean': filter_testset_mean_abs_grad, 'std': filter_testset_std_abs_grad},
            filter_stats_file,
        )
    else:
        filter_testset_mean_abs_grad = None
        filter_testset_std_abs_grad  = None

    # ------------------------------------------------------------------ #
    # Reference image                                                      #
    # ------------------------------------------------------------------ #
    if args.reference_id is None:
        print(f'\n  Using image {args.image_path} with target label {args.image_target_label}\n')
        reference_image  = transform_raw_image(
            args.image_path, preprocess=transform_test
        ).unsqueeze(0)
        reference_target = torch.tensor(int(args.image_target_label)).unsqueeze(0)
    else:
        print(f'\n  Using {args.reference_id}-th image from the validation set.\n')
        reference_image, reference_target = testset.__getitem__(args.reference_id)
        reference_target = torch.tensor(reference_target).unsqueeze(0)
        reference_image.unsqueeze_(0)

    # ------------------------------------------------------------------ #
    # Compute saliency                                                     #
    # ------------------------------------------------------------------ #
    grads_to_save, filter_saliency = compute_input_space_saliency(
        reference_image, reference_target, net, args,
        task_adapter, target_spec,
        filter_testset_mean_abs_grad, filter_testset_std_abs_grad,
        inv_transform_test, readable_labels,
    )

    layer_sorted_profile, _ = sort_filters_layer_wise(
        filter_saliency.detach().cpu().numpy(), layer_to_filter_id,
    )

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    save_gradients(grads_to_save, args, reference_image, inv_transform_test)

    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pal = sns.color_palette('colorblind')
    ax.plot(layer_sorted_profile, label='Sorted filter saliency', c=pal.as_hex()[0])
    ax.legend()
    ax.get_legend().get_frame().set_alpha(0.0)
    ax.set_xlabel('Filter ID')
    ax.set_ylabel('Saliency')

    save_name = (str(args.reference_id) if args.reference_id is not None
                 else args.image_path.split('/')[-1].split('.')[0])
    save_name += '_' + ck
    out_path = os.path.join(args.output_root, f'filter_saliency_{save_name}.png')
    fig.savefig(out_path)
    print(f'Filter saliency saved to {out_path}')
#Run this: python3 parameter_and_input_saliency.py --model resnet50 --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg --image_target_label 2
