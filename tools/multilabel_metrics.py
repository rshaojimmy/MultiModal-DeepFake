import math
from urllib.request import urlretrieve
import torch
import numpy as np


def get_multi_label(label, image):
    multi_label = torch.zeros([len(label), 4], dtype=torch.long).to(image.device) 
    # origin cls = [0, 0, 0, 0]
    real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
    multi_label[real_label_pos,:] = torch.tensor([0, 0, 0, 0]).to(image.device) 
    # face_swap cls = [1, 0, 0, 0]
    pos = np.where(np.array(label) == 'face_swap')[0].tolist() 
    multi_label[pos,:] = torch.tensor([1, 0, 0, 0]).to(image.device) 
    # face_attribute cls = [0, 1, 0, 0]
    pos = np.where(np.array(label) == 'face_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 1, 0, 0]).to(image.device) 
    # text_swap cls = [0, 0, 1, 0]
    pos = np.where(np.array(label) == 'text_swap')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 0, 1, 0]).to(image.device) 
    # text_attribute cls = [0, 0, 0, 1]
    pos = np.where(np.array(label) == 'text_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 0, 0, 1]).to(image.device) 
    #  face_swap&text_swap cls = [1, 0, 1, 0]
    pos = np.where(np.array(label) == 'face_swap&text_swap')[0].tolist()
    multi_label[pos,:] = torch.tensor([1, 0, 1, 0]).to(image.device) 
    #  face_swap&text_attribute cls = [1, 0, 0, 1]
    pos = np.where(np.array(label) == 'face_swap&text_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([1, 0, 0, 1]).to(image.device) 
    #  face_attribute&text_swap cls = [0, 1, 1, 0]
    pos = np.where(np.array(label) == 'face_attribute&text_swap')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 1, 1, 0]).to(image.device) 
    #  face_attribute&text_attribute cls = [0, 1, 0, 1]
    pos = np.where(np.array(label) == 'face_attribute&text_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 1, 0, 1]).to(image.device) 

    return multi_label, real_label_pos


def get_multi_label_TS(label, image):
    TS_pos = []
    
    multi_label = torch.zeros([len(label), 4], dtype=torch.long).to(image.device) 
    # origin cls = [0, 0, 0, 0]
    real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
    multi_label[real_label_pos,:] = torch.tensor([0, 0, 0, 0]).to(image.device) 
    # face_swap cls = [1, 0, 0, 0]
    pos = np.where(np.array(label) == 'face_swap')[0].tolist() 
    multi_label[pos,:] = torch.tensor([1, 0, 0, 0]).to(image.device) 
    # face_attribute cls = [0, 1, 0, 0]
    pos = np.where(np.array(label) == 'face_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 1, 0, 0]).to(image.device) 
    # text_swap cls = [0, 0, 1, 0]
    pos = np.where(np.array(label) == 'text_swap')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 0, 1, 0]).to(image.device) 
    TS_pos.extend(pos)
    # text_attribute cls = [0, 0, 0, 1]
    pos = np.where(np.array(label) == 'text_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 0, 0, 1]).to(image.device) 
    #  face_swap&text_swap cls = [1, 0, 1, 0]
    pos = np.where(np.array(label) == 'face_swap&text_swap')[0].tolist()
    multi_label[pos,:] = torch.tensor([1, 0, 1, 0]).to(image.device) 
    TS_pos.extend(pos)
    #  face_swap&text_attribute cls = [1, 0, 0, 1]
    pos = np.where(np.array(label) == 'face_swap&text_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([1, 0, 0, 1]).to(image.device) 
    #  face_attribute&text_swap cls = [0, 1, 1, 0]
    pos = np.where(np.array(label) == 'face_attribute&text_swap')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 1, 1, 0]).to(image.device) 
    TS_pos.extend(pos)
    #  face_attribute&text_attribute cls = [0, 1, 0, 1]
    pos = np.where(np.array(label) == 'face_attribute&text_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 1, 0, 1]).to(image.device) 

    return multi_label, real_label_pos, TS_pos

# def get_multi_label(label, image):
#     multi_label = torch.zeros([len(label), 3], dtype=torch.long).to(image.device) 
#     # origin cls = [0, 0, 0]
#     real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
#     multi_label[real_label_pos,:] = torch.tensor([0, 0, 0]).to(image.device) 
#     # face_swap cls = [1, 0, 0]
#     pos = np.where(np.array(label) == 'face_swap')[0].tolist() 
#     multi_label[pos,:] = torch.tensor([1, 0, 0]).to(image.device) 
#     # face_attribute cls = [0, 1, 0]
#     pos = np.where(np.array(label) == 'face_attribute')[0].tolist()
#     multi_label[pos,:] = torch.tensor([0, 1, 0]).to(image.device) 
#     # text_attribute cls = [0, 0, 1]
#     pos = np.where(np.array(label) == 'text_attribute')[0].tolist()
#     multi_label[pos,:] = torch.tensor([0, 0, 1]).to(image.device) 
#     #  face_swap&text_attribute cls = [1, 0, 1]
#     pos = np.where(np.array(label) == 'face_swap&text_attribute')[0].tolist()
#     multi_label[pos,:] = torch.tensor([1, 0, 1]).to(image.device) 
#     #  face_attribute&text_attribute cls = [0, 1, 1]
#     pos = np.where(np.array(label) == 'face_attribute&text_attribute')[0].tolist()
#     multi_label[pos,:] = torch.tensor([0, 1, 1]).to(image.device) 

#     return multi_label, real_label_pos



class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1


