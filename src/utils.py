import torch
import torch.nn.functional as F

def binary_accuracy(logits, labels, threshold=0.5, soft=False):
    # Convert logits to probabilities
    probs = torch.sigmoid(logits)
    
    # Apply threshold to get predictions (0 or 1)
    preds = (probs >= threshold).float()
    
    # Compare predictions with the true labels
    if soft:
        correct = torch.abs(preds - labels) < 0.5
        accuracy = correct.float().mean()
    else:
        correct = (preds == labels).float()
        accuracy = correct.sum() / len(labels)
    
    return accuracy.item() * 100


import torch.distributed as dist

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def min_max_normalize(features):
    # Find the min and max across the second and third dimensions (document and feature dimensions)
    min_vals = features.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]  # Shape: (batch_size, 1, 1)
    max_vals = features.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]  # Shape: (batch_size, 1, 1)
    
    normalized_features = (features - min_vals) / (max_vals - min_vals + 1e-8)  # Add small value to avoid division by zero
    return normalized_features

def z_score_normalize(features):
    # Compute the mean and standard deviation across the document and feature dimensions
    mean_vals = features.mean(dim=(1, 2), keepdim=True)  # Shape: (batch_size, 1, 1)
    std_vals = features.std(dim=(1, 2), keepdim=True)    # Shape: (batch_size, 1, 1)
    
    standardized_features = (features - mean_vals) / (std_vals + 1e-8)  # Add small value to avoid division by zero
    return standardized_features

def one_hot_binary_batch(batch):
    return torch.stack([1 - batch, batch], dim=1).float()

def distance_prob(prob1, prob2, distance_type='kl'):
    
    if distance_type == 'kl':
        # Ensure non-zero values to avoid log(0) in KL divergence
        prob1 = prob1 + 1e-8
        prob2 = prob2 + 1e-8
        kl_div = F.kl_div(prob1.log(), prob2, reduction='batchmean')  # KL Divergence
        return kl_div
    
    elif distance_type == 'js':
        # Jensen-Shannon Divergence (symmetrized version of KL)
        prob1 = prob1 + 1e-8
        prob2 = prob2 + 1e-8
        m = 0.5 * (prob1 + prob2)
        js_div = 0.5 * (F.kl_div(prob1.log(), m, reduction='batchmean') + F.kl_div(prob2.log(), m, reduction='batchmean'))
        return js_div
    
    elif distance_type == 'l2':
        # L2 Distance (Euclidean distance)
        l2_dist = torch.norm(prob1 - prob2, p=2)
        return l2_dist
    
    elif distance_type == 'l1':
        # L1 Distance (Manhattan distance)
        l1_dist = torch.norm(prob1 - prob2, p=1)
        return l1_dist

from PIL import Image

def merge_images(image_paths, output_path, direction="horizontal"):

    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(i.size for i in images))

    if direction == "horizontal":
        total_width = sum(widths)
        max_height = max(heights)
        new_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for image in images:
            new_image.paste(image, (x_offset, 0))
            x_offset += image.size[0]

    elif direction == "vertical":
        max_width = max(widths)
        total_height = sum(heights)
        new_image = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for image in images:
            new_image.paste(image, (0, y_offset))
            y_offset += image.size[1]

    else:
        raise ValueError("Invalid direction. Choose 'horizontal' or 'vertical'")

    new_image.save(output_path)

def sample_without_replacement_with_prob(delta, pos):
    weights = torch.ones_like(pos)
    remaining_idx = []

    #print('pos', pos)

    if delta*len(pos)<1:
        idx = torch.multinomial(pos, len(pos), replacement=False)
        return pos[idx]
    
    delta_sample_idx = torch.multinomial(pos,int(delta*len(pos)), replacement=False)
    #print ('retaining', pos[delta_sample_idx])
    weights[delta_sample_idx] = 0
    #remaining_idx = len(pos) - int(delta*len(pos))

    for i,j in enumerate(pos):
        if j not in pos[delta_sample_idx]:
            remaining_idx.append(i)

    #print ('perturbing', remaining_idx)

    d_pos = pos.clone()

    for i in remaining_idx:
        # Normalize weights to ensure they sum to 1
        normalized_weights = weights / weights.sum()

        # Sample one index based on normalized weights
        if weights.sum() == 0:
            return d_pos
        
        sampled_index = torch.multinomial(normalized_weights, 1).item()

        #print (i, sampled_index, normalized_weights)

        d_pos[sampled_index] = pos[i]

        # Set the weight of the sampled index to 0 for the next iteration
        weights[sampled_index] = 0

    return d_pos


def sample_swap(pos, click=None, fn='swap_rand'):
    
    def swap(pos, idx, swap_idx=-1):    
        temp = pos[idx].clone()
        pos[idx] = pos[swap_idx]
        pos[swap_idx] = temp
        return pos
    
    if fn == 'swap_rand':
        return swap(pos, idx=0)
    
    if click==None:
        return pos
    
    if click.sum()>0:
        indices = (click == 1).nonzero()
        idx = indices[0].item()
        if fn=='swap_first_click_bot':
            swap_idx = -1
        elif fn=='swap_first_click_top':
            swap_idx = 0
        elif fn=='swap_first_click_rand':
            swap_idx = torch.multinomial(pos, 1)
        return swap(pos, idx, swap_idx)
        
    return pos