import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import torchvision.models as models

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', alpha=1.0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, 
                                  ignore_index=self.ignore_index, reduction=self.reduction)
        
        # Additional custom term
        custom_term = torch.mean(torch.pow(torch.abs(input - target), self.alpha))

        return ce_loss + custom_term

# Example usage
# Assuming input and target are your model's predictions and ground truth labels respectively
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randint(5, (3,), dtype=torch.int64)

# criterion = CustomCrossEntropyLoss(alpha=0.5)  # You can specify your alpha value here
# loss = criterion(input, target)
# print(loss)


class LabelCalibratedCrossEntropyLoss(nn.Module):
    def __init__(self, label_distrib=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', tau=1.0):
        super(LabelCalibratedCrossEntropyLoss, self).__init__()
        self.label_distrib = label_distrib
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.tau = tau

    def forward(self, logit, y_logit):
        cal_logit = torch.exp(
            logit
            - (
                self.tau
                * torch.pow(self.label_distrib, -1 / 4)
                .expand((logit.shape[0], -1))
            )
        )
        y_logit = torch.gather(cal_logit, dim=-1, index=y_logit.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        return loss.sum() / logit.shape[0]

class CostSensitiveCrossEntropyLoss(nn.Module):
    def __init__(self, cost_matrix):
        super().__init__()
        self.is_train = False

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        if self.is_train:            
            gather_log_probs = log_probs.gather(1, targets.unsqueeze(1))
            gather_costs = self.cost_matrix[targets, self.predicted]            
            # Cost-sensitive loss computation
            loss = -torch.sum(gather_log_probs * gather_costs) / batch_size
            # print(f"\ngathered cost: {gather_costs}, loss: {loss}")
            self.is_train = False
        else:
            log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = -log_probs.mean()        
        # print(f"loss: {loss}")
        return loss
    def set_train(self):
        self.is_train = True
    def set_cost_matrix(self, cost_matrix, predicted):
        self.is_train = True
        self.cost_matrix = cost_matrix
        self.predicted = predicted

class CostSensitiveCrossEntropyLossN_old(nn.Module):
    def __init__(self, cost_matrix):
        super().__init__()
        self.is_train = False
        self.cost_matrix = cost_matrix
        self.beta = 2.0
        self.beta1 = 1.0
        self.beta2 = 3.0

    def forward(self, outputs, targets):
        if torch.isnan(outputs).any():
            print(f"\nNan found in outputs: {outputs}")
            raise RuntimeError  
        batch_size = outputs.size(0)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        _, predicted = torch.max(log_probs, 1)
        for x,y in zip(targets, predicted):
            self.cost_matrix[x][y] = self.cost_matrix[x][y] + 1 
        # cost_matrix = torch.zeros(self.cost_matrix.shape, device=self.cost_matrix.device)
        # for i in range(cost_matrix.shape[0]):
        #     cost_matrix[i] = (self.cost_matrix[i] * self.beta) / max(1,torch.sum(self.cost_matrix[i]))
        cost_matrix = copy.deepcopy(self.cost_matrix)
        # print(f"Cost_mtrix 1: {cost_matrix}")
        cost_matrix = torch.pow(cost_matrix, 1/4) # making higher value high close to 1.
        # print(f"Cost_mtrix 2: {cost_matrix}")
        if torch.isnan(cost_matrix).any():
            print(f"\nNan occured in cost_matrix: {cost_matrix}")
            raise RuntimeError 
        cost_matrix = cost_matrix * (1 - torch.eye(cost_matrix.shape[0], device=cost_matrix.device)) 
        cost_matrix = torch.clip(cost_matrix, min=self.beta1, max=self.beta2)
        # cost_matrix = 1 - cost_matrix 
        # cost_matrix = 1 + cost_matrix
        # Logits calibration using cost matrix
        # cost_for_logits = torch.stack([cost_matrix[y] for y in targets])
        # calibrated_log_probs = log_probs - (log_probs * cost_for_logits)

        gather_log_probs = log_probs.gather(1, targets.unsqueeze(1))    
        gather_costs = cost_matrix[targets, predicted]               
        # Cost-sensitive loss computation
        # After computing log probabilities
        # print("gather_log_probs grad_fn:", gather_log_probs.grad_fn)

        # After modifying with gather_costs
        temp_loss = gather_log_probs * gather_costs
        # print("temp_loss grad_fn after subtraction:", temp_loss.grad_fn)

        # Final loss
        loss = -temp_loss.mean()
        # print("Final loss grad_fn:", loss.grad_fn)
        # loss = -torch.sum(gather_log_probs - (gather_log_probs * gather_costs)) / batch_size
        # loss = -torch.sum(gather_log_probs - gather_costs) / batch_size
        if torch.isnan(loss).any():
            print(f"\nNan occured in loss: {loss}")
            raise RuntimeError  
        # print(f"\nPredicted: {predicted}, \nTargets: {targets},\nLoss: {loss}, \nLog Probs: {log_probs},\nCosts: {gather_costs} \nGather logs probs: {gather_log_probs}")  
        return loss
    def add_global_cost(self, global_cost_matrix):
        self.cost_matrix = self.cost_matrix + global_cost_matrix

class CostSensitiveCrossEntropyLossN(nn.Module):
    def __init__(self, cost_matrix):
        super().__init__()
        self.is_train = False
        self.cost_matrix = cost_matrix
        self.beta = 2.0
        self.beta1 = 1.0
        self.beta2 = 2.0

    def forward(self, outputs, targets):
        if torch.isnan(outputs).any():
            print(f"\nNan found in outputs: {outputs}")
            raise RuntimeError  
        batch_size = outputs.size(0)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        _, predicted = torch.max(log_probs, 1)
        for x,y in zip(targets, predicted):
            self.cost_matrix[x][y] = self.cost_matrix[x][y] + 1 
        cost_matrix = copy.deepcopy(self.cost_matrix)
        # print(f"Cost_mtrix 1: {cost_matrix}")
        cost_matrix = torch.pow(cost_matrix, 1/4) # making higher value high close to 1.
        # print(f"Cost_mtrix 2: {cost_matrix}")
        if torch.isnan(cost_matrix).any():
            print(f"\nNan occured in cost_matrix: {cost_matrix}")
            raise RuntimeError 
        cost_matrix = cost_matrix * (1 - torch.eye(cost_matrix.shape[0], device=cost_matrix.device)) 
        cost_matrix = cost_matrix * (self.beta2/torch.max(cost_matrix).item())
        cost_matrix = torch.clip(cost_matrix, min=self.beta1, max=self.beta2)

        gather_log_probs = log_probs.gather(1, targets.unsqueeze(1))    
        gather_costs = cost_matrix[targets, predicted]              

        # After modifying with gather_costs
        temp_loss = gather_log_probs * gather_costs
        # print("temp_loss grad_fn after subtraction:", temp_loss.grad_fn)

        # Final loss
        loss = -temp_loss.mean()
        if torch.isnan(loss).any():
            print(f"\nNan occured in loss: {loss}")
            raise RuntimeError  
        return loss
    def add_global_cost(self, global_cost_matrix):
        self.cost_matrix = self.cost_matrix + global_cost_matrix

class CostSensitiveCrossEntropyLossN_old(nn.Module):
    def __init__(self, cost_matrix):
        super().__init__()
        self.is_train = False
        self.cost_matrix = cost_matrix
        self.beta = 3
        self.beta1 = 1e-7
        self.beta2 = 1.0

    def forward(self, outputs, targets):
        if torch.isnan(outputs).any():
            print(f"\nNan found in outputs: {outputs}")
            raise RuntimeError  
        batch_size = outputs.size(0)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        _, predicted = torch.max(log_probs, 1)
        for x,y in zip(targets, predicted):
            self.cost_matrix[x][y] = self.cost_matrix[x][y] + 1 
        cost_matrix = torch.zeros(self.cost_matrix.shape, device=self.cost_matrix.device)
        for i in range(cost_matrix.shape[0]):
            cost_matrix[i] = (self.cost_matrix[i] * self.beta) / max(1,torch.sum(self.cost_matrix[i]))
        # cost_matrix = (1/self.cost_matrix)*self.beta   
        # cost_matrix = cost_matrix * (1 - torch.eye(cost_matrix.shape[0], device=cost_matrix.device))  
        # cost_matrix = 1 - cost_matrix 
        # cost_matrix = 1 + cost_matrix
        # cost_matrix = torch.clip(cost_matrix, min=self.beta1, max=self.beta2)
        if torch.isnan(cost_matrix).any():
            print(f"\nNan occured in cost_matrix: {cost_matrix}")
            raise RuntimeError    
        gather_log_probs = log_probs.gather(1, targets.unsqueeze(1))
        gather_costs = cost_matrix[targets, predicted]                  
        # Cost-sensitive loss computation
        # After computing log probabilities
        # print("gather_log_probs grad_fn:", gather_log_probs.grad_fn)

        # After modifying with gather_costs
        temp_loss = gather_log_probs * gather_costs
        # print("temp_loss grad_fn after subtraction:", temp_loss.grad_fn)

        # Final loss
        loss = -temp_loss.mean()
        # print("Final loss grad_fn:", loss.grad_fn)
        # loss = -torch.sum(gather_log_probs - (gather_log_probs * gather_costs)) / batch_size
        # loss = -torch.sum(gather_log_probs - gather_costs) / batch_size
        if torch.isnan(loss).any():
            print(f"\nNan occured in loss: {loss}")
            raise RuntimeError  
        # print(f"\ngathered cost: {gather_costs}, loss: {loss}")
        # print(f"\nPredicted: {predicted}, \nTargets: {targets},\nLoss: {loss}, \nLog Probs: {log_probs},\nCosts: {gather_costs} \nGather logs probs: {gather_log_probs}")  
        return loss

def cross_entropy_loss(outputs, targets):
    """
    Manually computed cross-entropy loss for educational purposes.
    
    Args:
    - outputs (torch.Tensor): The raw logits output by the neural network.
                              Shape: [batch_size, num_classes]
    - targets (torch.Tensor): The ground truth labels.
                              Shape: [batch_size]
                              Each label is an integer in [0, num_classes-1].
    
    Returns:
    - loss (torch.Tensor): The mean cross-entropy loss.
    """
    # Step 1: Compute log softmax
    # log_softmax(x_i) = log(exp(x_i) / sum_j(exp(x_j)))
    log_probs = F.log_softmax(outputs, dim=1)

    # Step 2: Gather the log probabilities of the correct classes
    # targets.unsqueeze(1) changes shape from [batch_size] to [batch_size, 1]
    # gather(dim=1, index=targets.unsqueeze(1)) picks out the log_probs for each target class
    log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Step 3: Compute the negative log likelihood loss
    # nll_loss = -1 * (sum of log_probs for the correct classes) / batch_size
    loss = -log_probs.mean()

    return loss


class AnchorLoss(nn.Module):
    def __init__(self, cls_num, feature_num, ablation=0):
        """
        :param cls_num: class number
        :param feature_num: feature dimens
        """
        super().__init__()
        self.cls_num = cls_num
        self.feature_num = feature_num

        # initiate anchors
        if cls_num > feature_num:
            self.anchor = nn.Parameter(
                F.normalize(torch.randn(cls_num, feature_num)), requires_grad=True
            )
        elif ablation == 1:
            self.anchor = nn.Parameter(
                F.normalize(torch.randn(cls_num, feature_num)), requires_grad=True
            )
        elif ablation == 2:
            self.anchor = nn.Parameter(
                F.normalize(torch.randn(cls_num, feature_num)), requires_grad=True
            )
            self.anchor.data = torch.load("utils/converged_anchors_data.pt")
        else:
            I = torch.eye(feature_num, feature_num)
            index = torch.LongTensor(random.sample(range(feature_num), cls_num))
            init = torch.index_select(I, 0, index)
            # for i in range(cls_num):
            #     if i % 2 == 0:
            #         init[i] = -init[i]
            self.anchor = nn.Parameter(init, requires_grad=True)

    def forward(self, feature, _target, Lambda = 0.1):
        """
        :param feature: input
        :param _target: label/targets
        :return: anchor loss 
        """
        # broadcast feature anchors for all inputs
        centre = self.anchor.cuda().index_select(dim=0, index=_target.long())
        # compute the number of samples in each class
        counter = torch.histc(_target, bins=self.cls_num, min=0, max=self.cls_num-1)
        count = counter[_target.long()]
        centre_dis = feature - centre				# compute distance between input and anchors
        pow_ = torch.pow(centre_dis, 2)				# squre
        sum_1 = torch.sum(pow_, dim=1)				# sum all distance
        dis_ = torch.div(sum_1, count.float())		# mean by class
        sum_2 = torch.sum(dis_)/self.cls_num						# mean loss
        res = Lambda*sum_2   							# time hyperparameter lambda 
        return res

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))
        return loss
    
# class CenterLoss(torch.nn.Module):
#     def __init__(self, num_classes, feat_dim, alpha=0.5):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.alpha = alpha
#         self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))

#     def forward(self, x, labels):
#         centers = self.centers
#         dists = torch.cdist(x, centers, p=2)
#         loss = torch.mean(dists[torch.arange(x.size(0)), labels])
#         return loss


    
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        self.device = device

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.centers.size(0)) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.centers.size(0), batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=0.5, alpha=0.3)

        classes = torch.arange(self.centers.size(0)).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.centers.size(0))
        mask = labels.eq(classes.expand(batch_size, self.centers.size(0)))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

# ArcFace Layer
class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, out_features, device, s=40.0, m=0.40):
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features).to(device))
        nn.init.xavier_uniform_(self.weight)
        #added
        self.eps = 1e-7
        
        # Pre-compute these values as tensors
        self.cos_m = torch.cos(torch.tensor(self.m))
        self.sin_m = torch.sin(torch.tensor(self.m))
        self.threshold = torch.cos(torch.tensor(torch.pi - self.m))
        self.mm = self.sin_m * self.m

    def forward(self, input, label):
        #1
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        # one_hot = torch.zeros_like(cosine, device=input.device)
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # target_logit = theta + one_hot * self.m
        # output = self.s * torch.cos(target_logit)
        # return output
        #2
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine, device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output
        #3
        # # L2 normalize input and weight
        # input_norm = F.normalize(input)
        # weight_norm = F.normalize(self.weight)
        
        # # Compute cosine similarity
        # cosine = F.linear(input_norm, weight_norm)
        
        # # Clip cosine values to prevent acos from returning NaN
        # cosine = torch.clamp(cosine, -1 + self.eps, 1 - self.eps)
        
        # # Compute sine
        # sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # # Compute phi
        # phi = cosine * self.cos_m - sine * self.sin_m
        
        # # Apply conditional operations
        # phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        # # Create one-hot encoding of labels
        # one_hot = torch.zeros_like(cosine, device=input.device)
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # # Compute output
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # output *= self.s
        
        # return output
    
# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device, s=30.0, m=0.50, gamma=2):
        super(CombinedLoss, self).__init__()
        self.arcface = ArcFaceLayer(feat_dim, num_classes, device, s=s, m=m)
        self.focal = FocalLoss(gamma=gamma)

    def forward(self, features, labels):
        logits = self.arcface(features, labels)
        return self.focal(logits, labels)
    
def generate_orthogonal_vectors(num_vectors, vector_dim):
    # Create a random matrix
    random_matrix = torch.randn(num_vectors, vector_dim)
    
    # Perform QR decomposition
    q, r = torch.linalg.qr(random_matrix, mode='reduced')
    
    # Ensure the diagonal of r is positive for uniqueness
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    
    # If we have more dimensions than vectors, we need to pad
    if vector_dim > num_vectors:
        padding = torch.randn(num_vectors, vector_dim - num_vectors)
        q = torch.cat([q, padding], dim=1)
    
    # Normalize the vectors
    q = F.normalize(q, p=2, dim=1)
    
    return q

class CrossEntropyPlusCosineLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, device, initial_target_vectors=None, lambda_feature=0.3, lambda_diversity=0.01, lamda_similarity=0.1):
        super(CrossEntropyPlusCosineLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.mse_loss = nn.MSELoss()
        
        if initial_target_vectors is None:
            initial_target_vectors = generate_orthogonal_vectors(num_classes, feature_dim)
        else:
            initial_target_vectors = torch.tensor(initial_target_vectors, dtype=torch.float32)
        
        self.target_vectors = nn.Parameter(initial_target_vectors.to(device))
        self.lambda_feature = lambda_feature
        self.lambda_diversity = lambda_diversity
        self.lambda_similarity = lamda_similarity

    def forward(self, logits, features, labels):
        # Classification loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Feature matching loss
        target_features = self.target_vectors[labels]
        cosine_loss = self.cosine_loss(features, target_features, torch.ones(features.size(0)).to(features.device))
        mse_loss = self.mse_loss(features, target_features)
        # Diversity loss to maximize distance between target vectors
        normalized_targets = F.normalize(self.target_vectors, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_targets, normalized_targets.t())
        
        # We want to minimize similarity (maximize distance) between different class vectors
        diversity_loss = torch.mean(torch.triu(similarity_matrix, diagonal=1))
        
        # Combined loss
        total_loss = ce_loss + self.lambda_feature * cosine_loss - self.lambda_diversity * diversity_loss + mse_loss * self.lambda_similarity
        # total_loss = ce_loss + self.lambda_feature * cosine_loss + mse_loss * self.lambda_similarity
        
        return total_loss #, ce_loss, cosine_loss, diversity_loss
    
# Contrastive loss
class Contrastive_loss(nn.Module):
    def __init__(self, margin=1.0):
        super(Contrastive_loss, self).__init__()
        self.margin = margin
    def forward(self, features, targets, class_feature_vectors):
        print("Computing contrastive loss")
        batch_size = targets.size(0)
        loss = torch.tensor(0.0, device=targets.device)
        for i in range(batch_size):
            positive_class_id = targets[i].item()
            # positive_feature_vector = class_feature_vectors[f'class_{positive_class_id}']
            
            # positive_distance = F.mse_loss(features[i], positive_feature_vector, reduction='sum')
            # loss += positive_distance
            
            for j in range(len(class_feature_vectors)):
                if j != positive_class_id:
                    negative_feature_vector = class_feature_vectors[f'class_{j}']
                    negative_distance = F.mse_loss(features[i], negative_feature_vector, reduction='sum')
                    loss += F.relu(self.margin - negative_distance)
        
        loss /= batch_size
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features, targets, class_feature_vectors):
        # print("Computing contrastive loss")
        batch_size = targets.size(0)
        num_classes = len(class_feature_vectors)
        
        # Stack all class feature vectors into a tensor
        all_class_vectors = torch.stack([class_feature_vectors[f'class_{i}'] for i in range(num_classes)])

        # Get the positive class vectors for each sample in the batch
        positive_class_vectors = all_class_vectors[targets]
        
        # Compute the positive distances
        positive_distances = torch.sum((features - positive_class_vectors) ** 2, dim=1)

        # Compute the negative distances and apply the margin
        negative_distances = torch.sum((features.unsqueeze(1) - all_class_vectors.unsqueeze(0)) ** 2, dim=2)

        # Ensure that the positive class distances are not included in the negative distances
        mask = torch.eye(num_classes, device=features.device)[targets].bool()
        negative_distances[mask] = float('inf')
        
        # Apply the margin and use ReLU
        negative_distances = F.relu(self.margin - negative_distances)
        
        # Sum the losses
        loss = positive_distances.sum() + negative_distances.sum(dim=1).sum()
        
        # Normalize the loss by batch size
        loss /= batch_size
        
        return loss
    
# Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.feature_extractor = None
        # try:
        #     vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device)
        #     self.feature_extractor = nn.Sequential(*list(vgg.children())).eval()
        #     for param in self.feature_extractor.parameters():
        #         param.requires_grad = False
        # except:
        #     print("Warning: VGG16 model not available. Using MSE loss instead.")
        #     self.feature_extractor = None

    def forward(self, input, target):
        if self.feature_extractor is None:
            return F.mse_loss(input, target)
        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)
        return F.mse_loss(input_features, target_features)
    
class VAELoss(nn.Module):
    def __init__(self, device):
        super(VAELoss, self).__init__()
        self.perceptual_loss = PerceptualLoss(device)
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.beta = 0.1
        self.perceptual_weight = 0.1

    def forward(self,x, recon_x, n_x, mu, logvar):
        recon_loss = self.mse_loss(recon_x, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        perceptual = self.perceptual_loss(recon_x, n_x)
        print(f"\nRecon Loss: {recon_loss}, Kld Loss: {kld_loss}, Perceptual Loss: {perceptual}")
        # print(f"\nX: {x[0]}, \nRecon X: {recon_x[0]}, \nN_X: {n_x}, \nmu: {mu},  Logvar: {logvar}")
        return recon_loss + self.beta * kld_loss + self.perceptual_weight * perceptual




def get_loss_fun(loss):
    if loss == "CE":
        return torch.nn.CrossEntropyLoss
    if loss == "MSE":
        return torch.nn.MSELoss
    if loss == "CL":
        return LabelCalibratedCrossEntropyLoss
    if loss == "CS":
        return CostSensitiveCrossEntropyLoss
    if loss == "CSN":
        return CostSensitiveCrossEntropyLossN
    if loss == "TripLet":
        return TripletLoss
    if loss == "Center":
        return CenterLoss
    if loss == "Combined":
        return CombinedLoss
    if loss == "CCL":
        return CrossEntropyPlusCosineLoss
    if loss == "Contrastive":
        return ContrastiveLoss
    if loss == "PLoss":
        return PerceptualLoss
    if loss == "VLoss":
        return VAELoss

