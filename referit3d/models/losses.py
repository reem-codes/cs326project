import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class MultiheadLoss(nn.Module):
    def __init__(self, margin, 
        weights={'ce': 1.0, 'triplet': 1.0, 'contrastive': 1.0}):
        super(MultiheadLoss, self).__init__()
        self.triplet = TripletLoss(margin)
        self.contrastive = ContrastiveLoss(margin)
        self.weights = weights
        
    def forward(self, logits, target, embeddings1, embeddings2):
		"""
		Given the logits, the target and the embedding of the anchor and positive, 
		compute the loss
		"""
        weighted_loss = torch.zeros(1, dtype=torch.float).to(logits.device)
        if self.weights['ce'] > 0.0:
        	# standard cross entropy, uses the logits and target
            ce = F.cross_entropy(logits, target) * self.weights['ce']
            weighted_loss += ce

        if self.weights['triplet'] > 0.0:
        	# triplet loss, takes the anchor and positive embeddings, in addition to the target position
            t = self.triplet(embeddings1, embeddings2, target) * self.weights['triplet']
            weighted_loss += t
            
        if self.weights['contrastive'] > 0.0:
        	# contrastive loss, takes the anchor and positive embeddings, in addition to the target position
            c = self.contrastive(embeddings1, embeddings2, target) * self.weights['contrastive']
            weighted_loss += c
        return weighted_loss

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def triplet_loss(self, anchor, positive, negative, reduction=True):
    	# since there's only one anchor/positive and 51 negatives
    	# they were first repeated
        anchor = anchor.expand(negative.size(0), -1)
        positive = positive.expand(negative.size(0), -1)

		# calculate triplet loss
        distance_positive = (anchor - positive).pow(2).sum(1).clamp(min=1e-12).sqrt()
        distance_negative = (anchor - negative).pow(2).sum(1).clamp(min=1e-12).sqrt()
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if reduction else losses.sum()

    
    def forward(self, embedding1, embedding2, target, reduction=True):
        total = 0.
        # the embeddings are first normalized
        # the performance increased slightly when they were normalized first
        embedding1 = F.normalize(embedding1, p=2, dim=-1)
        embedding2 = F.normalize(embedding2, p=2, dim=-1)
        
        for b_i in range(embedding1.shape[0]): # for each scene in the batch
        	# get all objects in the scene that are not the target
            nid = list(range(embedding1.size(1)))
            nid.remove(target[b_i])
            a = embedding1[b_i, target[b_i]] # object embedding of anchor
            p = embedding2[b_i, target[b_i]] # positive
            n1 = embedding1[b_i, nid] # negative1
            n2 = embedding2[b_i, nid] # negative2
            
            # make the anchor close to the positive and far from the negative
            # these negatives are the other object fused with the same utterance as the anchor
            # the positive is the same anchor object with a different utterance
            total += self.triplet_loss(a, p, n1, reduction)
            
            # total += self.triplet_loss(a, p, n2, reduction)
            # total += self.triplet_loss(p, a, n1, reduction)
            
            # make the positive close to the anchor and far from the negative
            # these negatives are the other objects fused with the same utterance as the positive 
            total += self.triplet_loss(p, a, n2, reduction)
        return total / (embedding1.shape[0] * 2) # normalize
    
    
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def contrastive_loss(self, output1, output2, target, reduction=True):
    	# only one anchor/positive against 51 negatives
    	# so expand to make sizes compatible
        output1 = output1.expand(output2.size(0), -1)
        
        # contrastive loss
        distances = (output2 - output1).pow(2).sum(1).clamp(min=1e-12).sqrt()  # squared distances
        losses = 0.5 * (target * distances +
                        (1 + -1 * target) * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean() if reduction else losses.sum()

    def forward(self, embedding1, embedding2, target, reduction=True):
        total = 0.
        # the embeddings are first normalized
        # the performance increased slightly when they were normalized first
        embedding1 = F.normalize(embedding1, p=2, dim=-1)
        embedding2 = F.normalize(embedding2, p=2, dim=-1)

        for b_i in range(embedding1.shape[0]): # for each scene in the batch
            # get all objects in the scene that are not the target
            nid = list(range(embedding1.size(1)))
            nid.remove(target[b_i])
            a = embedding1[b_i, target[b_i]] # object embedding of anchor
            p = embedding2[b_i, target[b_i]] # positive
            n1 = embedding1[b_i, nid] # negative1
            n2 = embedding2[b_i, nid] # negative2
            total += self.contrastive_loss(a, p, 1, reduction) # same class
            total += self.contrastive_loss(a, n1, 0, reduction) # different class
            total += self.contrastive_loss(p, n2, 0, reduction) # different class
        return total / (embedding1.shape[0] * 3)
