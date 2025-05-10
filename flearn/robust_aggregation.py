class RobustAggregation:
    def __init__(self, method='coordinate_median', eps=0.1, sigma=0.5, expansion=20.0):
        """Initialize robust aggregation
        
        Args:
            method (str): Aggregation method ('coordinate_median', 'krum', or 'trimmed_mean')
            eps (float): Attack scale threshold
            sigma (float): Standard deviation for noise
            expansion (float): Expansion factor for updates
        """
        self.method = method
        self.eps = eps
        self.sigma = sigma
        self.expansion = expansion

    def aggregate(self, updates, weights=None):
        """Aggregate updates using robust aggregation"""
        if not updates:
            return None
            
        # Convert updates to tensors and move to CPU for aggregation
        updates = [torch.tensor(update, dtype=torch.float32) for update in updates]
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32)
            
        # Move to CPU for aggregation
        updates = [update.cpu() for update in updates]
        if weights is not None:
            weights = weights.cpu()
            
        # Perform aggregation
        if self.method == 'coordinate_median':
            return self._coordinate_median(updates, weights)
        elif self.method == 'krum':
            return self._krum(updates, weights)
        elif self.method == 'trimmed_mean':
            return self._trimmed_mean(updates, weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

    def _coordinate_median(self, updates, weights=None):
        """Coordinate-wise median aggregation"""
        if not updates:
            return None
            
        # Stack updates into a tensor
        stacked_updates = torch.stack(updates)
        
        # Compute median along the first dimension
        median_update = torch.median(stacked_updates, dim=0)[0]
        
        return median_update

    def _krum(self, updates, weights=None):
        """Krum aggregation"""
        if not updates:
            return None
            
        # Stack updates into a tensor
        stacked_updates = torch.stack(updates)
        
        # Compute pairwise distances
        n_updates = len(updates)
        distances = torch.zeros((n_updates, n_updates))
        for i in range(n_updates):
            for j in range(n_updates):
                if i != j:
                    distances[i, j] = torch.norm(stacked_updates[i] - stacked_updates[j])
        
        # Select update with minimum sum of distances to closest neighbors
        k = max(1, n_updates - 2)  # Number of closest neighbors to consider
        closest_distances = torch.topk(distances, k=k, dim=1, largest=False)[0]
        sum_distances = torch.sum(closest_distances, dim=1)
        selected_idx = torch.argmin(sum_distances)
        
        return updates[selected_idx]

    def _trimmed_mean(self, updates, weights=None):
        """Trimmed mean aggregation"""
        if not updates:
            return None
            
        # Stack updates into a tensor
        stacked_updates = torch.stack(updates)
        
        # Sort updates along each coordinate
        sorted_updates, _ = torch.sort(stacked_updates, dim=0)
        
        # Remove top and bottom 10% of updates
        n_updates = len(updates)
        trim_size = int(0.1 * n_updates)
        trimmed_updates = sorted_updates[trim_size:-trim_size]
        
        # Compute mean of remaining updates
        mean_update = torch.mean(trimmed_updates, dim=0)
        
        return mean_update 