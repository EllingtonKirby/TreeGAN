import pathlib
import numpy as np
import torch
from scipy.linalg import sqrtm
from evaluation.pointnet import PointNetCls
from data.NuscenesObjectsDataLoader import NuscenesObjectsDataLoader

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

"""Calculate Frechet Pointcloud Distance referened by Frechet Inception Distance."
    [ref] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
    github code  : (https://github.com/bioinf-jku/TTUR)
    paper        : (https://arxiv.org/abs/1706.08500)

"""

def get_activations(dataloader, model, dims=1808, device=None, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- pointcloud       : pytorch Tensor of pointclouds.
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : If set to device, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    pred_arr = []
    for i, batch in tqdm(enumerate(dataloader)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, len(dataloader)),
                  end='', flush=True)

        pointcloud_batch = batch[0]
        
        if device is not None:
            pointcloud_batch = pointcloud_batch.to(device)

        pointcloud_batch = pointcloud_batch.transpose(1,2)

        _, _, actv = model(pointcloud_batch)

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.shape[2] != 1 or pred.shape[3] != 1:
        #    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr.append(actv.cpu().data.numpy().reshape(pointcloud_batch.shape[0], -1))

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(dataloader, model, dims=1808, device=None, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- pointcloud       : pytorch Tensor of pointclouds.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : If set to device, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(dataloader, model, dims, device, verbose)
    act = np.array(act).squeeze(1)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['m'][:], f['s'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s

def save_statistics(real_pointclouds, path, model, batch_size, dims, cuda):
    m, s = calculate_activation_statistics(real_pointclouds, model, batch_size,
                                         dims, cuda)
    np.savez(path, m = m, s = s)
    print('save done !!!')

def calculate_fpd(gen_dl, real_dl, dims=1808, device=None):
    """Calculates the FPD of two pointclouds"""

    PointNet_path = './evaluation/cls_model_39.pth'
    statistic_save_path = './evaluation/pre_statistics_real.npz'
    statistic_save_path_gen = './evaluation/pre_statistics_xs_4_1a_cross_pointnet_impcgf_4ch_gen_recreate.npz'
    model = PointNetCls(k=16)
    model.load_state_dict(torch.load(PointNet_path))
    
    if device is not None:
        model.to(device)

    m1, s1 = calculate_activation_statistics(gen_dl, model, dims, device)
    np.savez(statistic_save_path_gen, m = m1, s = s1)
    if real_dl is not None:
        m2, s2 = calculate_activation_statistics(real_dl, model, dims, device)
        np.savez(statistic_save_path, m = m2, s = s2)
    else: # Load saved statistics of real pointclouds.
        f = np.load(statistic_save_path)
        m2, s2 = f['m'][:], f['s'][:]
        f.close()
        
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value
    
def get_nuscenes_FPD():
    model_to_test = 'xs_4_1a_cross_pointnet_impcgf_4ch_gen_recreate'
    print(f'Evaluating: {model_to_test}')
    root = '/home/ekirby/scania/ekirby/datasets/nuscenes_generated_bikes/' + model_to_test
    dl_generated = NuscenesObjectsDataLoader(root=root, real_or_generated='generated', split='train', num_points=1024)
    dl_generated = torch.utils.data.DataLoader(dl_generated, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    dl_real =      NuscenesObjectsDataLoader(root=root, real_or_generated='real', split='train', num_points=1024)
    dl_real =      torch.utils.data.DataLoader(dl_real, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    device = torch.device('cuda:0')
    fid_value = calculate_fpd(dl_generated, dl_real, device=device)
    
    print('Frechet Pointcloud Distance <<< {:.10f} >>>'.format(fid_value))
    with open(f'{model_to_test}_fpd.txt', 'w') as f:
        print('Frechet Pointcloud Distance <<< {:.10f} >>>'.format(fid_value), file=f)