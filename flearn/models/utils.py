import torch

def kmeans_pytorch(X, labels, classes, num_iters=10):
    """
    Performs KMeans clustering using PyTorch.
    
    Args:
    X (torch.Tensor): The features to be clustered. Shape: (num_samples, feature_dim)
    num_clusters (int): The number of clusters.
    num_iters (int): The number of iterations for the KMeans algorithm.
    
    Returns:
    centroids (torch.Tensor): The final cluster centroids.
    labels (torch.Tensor): The labels of the clusters for each sample.
    """
    # num_samples, feature_dim = X.shape
    # Flatten the features and centroids
    # print(f"Features shape: {X.shape}")
    N, *S = X.shape
    # print(S)
    SS = torch.prod(torch.tensor(S))
    # print(SS)
    X = X.view(N, SS)
    label_map, reverse_map = {}, {}
    labels = labels.cpu().tolist()
    classes = classes.cpu().tolist()
    for idx, k in enumerate(classes):
        label_map[k] = idx
        reverse_map[idx] = k
    # print(labels, label_map)
    # Update labels according to the mapping
    labels = [label_map[label] for label in labels]
    labels = torch.tensor(labels, device=X.device)
    classes = [label_map[label] for label in classes]
    for i in range(num_iters):
        # Compute distances between points and centroids
        # print(f"old labels: {labels}")
        if i !=0:
            # print(len(X), len(centroids), len(classes))
            # print(f"Features shape: {X.shape}")
            # print(f"Centroids shape: {centroids.shape}")

            distances = torch.cdist(X, centroids)

            # Assign each point to the closest centroid
            labels = torch.argmin(distances, dim=1)
            # print(f"new labels: {labels}")

        # Compute new centroids as the mean of assigned points
        new_centroids = torch.stack([X[labels == k].mean(dim=0) for k in classes])
        
        # Check for convergence (if centroids do not change)
        if i !=0:
            if torch.all(centroids == new_centroids):
                # print(f"\nIterations took to reach final centroid is: {i}")
                break
        
        centroids = new_centroids
    # print("Finished!")
    centroids_dict = {}
    for k, centroid in zip(classes, centroids):
        centroids_dict[reverse_map[k]] = centroid.view(S)
    # print(label_map,reverse_map, centroids_dict.keys())
    return centroids_dict

def cluster_features_pytorch(features, labels, classes):
    """
    Clusters the features based on the number of classes in the dataset.
    
    Args:
    features (torch.Tensor): The features from the last layer of the model. Shape: (num_samples, feature_dim)
    labels (torch.Tensor): The actual class labels of the features. Shape: (num_samples,)
    num_classes (int): The number of classes in the dataset.
    
    Returns:
    cluster_labels (torch.Tensor): The labels of the clusters. Shape: (num_samples,)
    """
    # Normalize features
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    
    # Perform KMeans clustering
    centroids = kmeans_pytorch(features, labels, classes)
    
    return centroids

# Example usage:
if __name__ == "__main__":
    # Example features from the last layer of a model
    features = torch.rand(100, 512)  # 100 samples, 512-dimensional features
    labels = torch.randint(0, 10, (100,))  # 100 samples, 10 classes
    classes = torch.unique(labels)
    
    cluster_dict = cluster_features_pytorch(features, labels, classes)
    print(cluster_dict)
