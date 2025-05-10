import torch
import numpy as np
import logging
from typing import List, Dict, Tuple

# Set up logging for attack
logging.basicConfig(level=logging.WARNING, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FedAttack")

class Attack:
    def __init__(self, attack_type='model_poisoning', scale=1.0, fraction=0.3, 
                 target_layer=None, malicious_behavior='converge', lie_z=1, **kwargs):
        """
        Initialize the attack with parameters.
        
        Args:
            attack_type: Type of the attack to perform
            scale: Scale factor for the attack
            fraction: Fraction of clients that are malicious
            target_layer: Specific layer to target in attack
            malicious_behavior: Whether malicious clients should help converge or diverge (for backward compatibility)
            lie_z: Z score for LIE attack (for backward compatibility)
            **kwargs: Additional arguments for backward compatibility
        """
        self.attack_type = attack_type
        self.scale = scale
        self.fraction = fraction
        self.target_layer = target_layer
        self.malicious_behavior = malicious_behavior  # Store for backward compatibility
        self.lie_z = lie_z  # Store for backward compatibility
        
        # Set up logging
        self.logger = logging.getLogger("Attack")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.WARNING)
        
        self.logger.debug(f"Initialized {attack_type} attack with scale {scale}, fraction {fraction}")
        
    def apply_attack(self, updates, global_model=None):
        """
        Apply the attack based on the attack type.
        
        Args:
            updates: List of model updates from clients
            global_model: Current global model state
            
        Returns:
            List of potentially modified updates
        """
        if not updates:
            return updates
            
        self.logger.info(f"Applying {self.attack_type} attack with scale {self.scale}")
        
        # Calculate number of malicious clients
        num_malicious = max(1, int(self.fraction * len(updates)))
        self.logger.info(f"Creating {num_malicious} malicious updates out of {len(updates)} total updates")
        
        # Analyze benign updates to get statistics
        benign_stats = self._analyze_benign_updates(updates)
        
        # Create malicious updates
        malicious_updates = self._create_malicious_updates(updates, benign_stats, num_malicious)
        
        # Mix malicious and benign updates
        mixed_updates = updates[:-num_malicious] + malicious_updates
        self.logger.info(f"Final mixed update set: {len(mixed_updates)} updates ({len(updates)-num_malicious} benign, {num_malicious} malicious)")
        
        return mixed_updates
    
    def _analyze_benign_updates(self, updates: List[Dict[str, torch.Tensor]]) -> Dict:
        """Analyze statistical properties of benign updates."""
        if not updates:
            return {}
            
        # Calculate average update
        avg_update = self._average_updates(updates)
        
        # Calculate norms for each update
        norms = []
        for update in updates:
            norm = 0.0
            for key in update.keys():
                norm += torch.norm(update[key]).item()
            norms.append(norm)
            
        return {
            'avg_update': avg_update,
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'max_norm': np.max(norms),
            'min_norm': np.min(norms)
        }
    
    def _create_malicious_updates(self, updates: List[Dict[str, torch.Tensor]], 
                                benign_stats: Dict, num_malicious: int) -> List[Dict[str, torch.Tensor]]:
        """Create malicious updates based on attack type."""
        if self.attack_type == 'model_poisoning':
            return self._model_poisoning_attack(updates, benign_stats, num_malicious)
        else:
            self.logger.warning(f"Unknown attack type: {self.attack_type}")
            return []
    
    def _model_poisoning_attack(self, updates: List[Dict[str, torch.Tensor]], 
                              benign_stats: Dict, num_malicious: int) -> List[Dict[str, torch.Tensor]]:
        """Implement a balanced model poisoning attack."""
        malicious_updates = []
        avg_update = benign_stats['avg_update']
        
        # Calculate target norm that's slightly higher than benign updates
        target_norm = benign_stats['mean_norm'] * (1.0 + self.scale)
        
        for i in range(num_malicious):
            malicious_update = {}
            
            # For each parameter in the model
            for key in avg_update.keys():
                # Get the average update direction
                avg_direction = avg_update[key]
                
                # Calculate attack direction (opposite to average)
                attack_direction = -avg_direction * self.scale
                
                # Add small noise to avoid detection
                noise = torch.randn_like(avg_direction) * 0.01 * torch.norm(avg_direction)
                malicious_update[key] = attack_direction + noise
            
            # Scale to target norm
            current_norm = sum(torch.norm(v).item() for v in malicious_update.values())
            if current_norm > 0:
                scale_factor = target_norm / current_norm
                malicious_update = {k: v * scale_factor for k, v in malicious_update.items()}
            
            malicious_updates.append(malicious_update)
        
        return malicious_updates
    
    def _average_updates(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Calculate the average update from a list of updates."""
        if not updates:
            return {}
            
        avg_update = {}
        for key in updates[0].keys():
            tensors = [update[key] for update in updates]
            avg_update[key] = torch.mean(torch.stack(tensors), dim=0)
            
        return avg_update