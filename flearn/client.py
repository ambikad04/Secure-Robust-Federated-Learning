import torch
import torch.nn as nn
from flearn.utils.torch_utils import select_optimizer

class Client:
    def __init__(self, id, train_data, test_data, model, args):
        self.id = id
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.args = args
        
        # Set device
        if args.gpu >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)  # Set the default CUDA device first
            self.device = torch.device(f'cuda:{args.gpu}')
            print(f"Client {id} using GPU: {torch.cuda.get_device_name(self.device)}")
            # Enable cuDNN benchmarking for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
            print(f"Client {id} using CPU")
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = select_optimizer(self.args.optimizer, self.model, self.args.learning_rate, self.args.weight_decay)
        self.loss_func = nn.CrossEntropyLoss().to(self.device)

    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        self.model.train()
        for epoch in range(num_epochs):
            for batch_id, batch in enumerate(self.train_data):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss_func(output, y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Clear CUDA cache periodically
                if batch_id % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def test(self, batch_size=10):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_id, batch in enumerate(self.test_data):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                output = self.model(x)
                test_loss += self.loss_func(output, y).item()
                
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()
                total += y.size(0)
                
                # Clear CUDA cache periodically
                if batch_id % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return correct / total if total > 0 else 0, test_loss / len(self.test_data) 