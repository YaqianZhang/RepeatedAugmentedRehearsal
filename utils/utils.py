import torch
import numpy as np

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
# from https://github.com/drimpossible/GDumb/blob/74a5e814afd89b19476cd0ea4287d09a7df3c7a8/src/utils.py#L102:5
def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5,index= "None"):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    if(index == "None"):
        index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def cutmix_data_two_data(x_a,y_a,x_b,y_b, alpha=1.0, ):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x_a.size()[0]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x_a.size(), lam)
    x_a[:, :, bbx1:bbx2, bby1:bby2] = x_b[:, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_a.size()[-1] * x_a.size()[-2]))
    return x_a, y_a, y_b, lam

def maybe_cuda(what, use_cuda=True, **kw):
    """
    Moves `what` to CUDA and returns it, if `use_cuda` and it's available.
        Args:
            what (object): any object to move to eventually gpu
            use_cuda (bool): if we want to use gpu or cpu.
        Returns
            object: the same object but eventually moved to gpu.
    """

    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count == 0:
            return 0
        return float(self.sum) / self.count


def mini_batch_deep_features(model, total_x, num):
    """
        Compute deep features with mini-batches.
            Args:
                model (object): neural network.
                total_x (tensor): data tensor.
                num (int): number of data.
            Returns
                deep_features (tensor): deep feature representation of data tensor.
    """
    is_train = False
    if model.training:
        is_train = True
        model.eval()
    if hasattr(model, "features"):
        model_has_feature_extractor = True
    else:
        model_has_feature_extractor = False
        # delete the last fully connected layer
        modules = list(model.children())[:-1]
        # make feature extractor
        model_features = torch.nn.Sequential(*modules)

    with torch.no_grad():
        bs = 64
        num_itr = num // bs + int(num % bs > 0)
        sid = 0
        deep_features_list = []
        for i in range(num_itr):
            eid = sid + bs if i != num_itr - 1 else num
            batch_x = total_x[sid: eid]

            if model_has_feature_extractor:
                batch_deep_features_ = model.features(batch_x)
            else:
                batch_deep_features_ = torch.squeeze(model_features(batch_x))

            deep_features_list.append(batch_deep_features_.reshape((batch_x.size(0), -1)))
            sid = eid
        if num_itr == 1:
            deep_features_ = deep_features_list[0]
        else:
            deep_features_ = torch.cat(deep_features_list, 0)
    if is_train:
        model.train()
    return deep_features_


def euclidean_distance(u, v):
    euclidean_distance_ = (u - v).pow(2).sum(1)
    return euclidean_distance_


def ohe_label(label_tensor, dim, device="cpu"):
    # Returns one-hot-encoding of input label tensor
    n_labels = label_tensor.size(0)
    zero_tensor = torch.zeros((n_labels, dim), device=device, dtype=torch.long)
    return zero_tensor.scatter_(1, label_tensor.reshape((n_labels, 1)), 1)


def nonzero_indices(bool_mask_tensor):
    # Returns tensor which contains indices of nonzero elements in bool_mask_tensor
    return bool_mask_tensor.nonzero(as_tuple=True)[0]


class EarlyStopping():
    def __init__(self, min_delta, patience, cumulative_delta):
        self.min_delta = min_delta
        self.patience = patience
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score = None

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False

    def reset(self):
        self.counter = 0
        self.best_score = None
