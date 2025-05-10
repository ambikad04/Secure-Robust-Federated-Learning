import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
ARGS = {
    "mnist": (1, 256, 10),
    "emnist": (1, 256, 62),
    "fmnist": (1, 256, 10),
    "cifar": (3, 400, 10),
    "cifar10": (3, 400, 10),
    "cifar100": (3, 400, 100),
    "tinyimagenet": (3, 400, 100),
}

AVG_POOL_K = {18: 3, 32:4, 64:6} 

func = (lambda x: x.detach().clone())
class elu(nn.Module):
    def __init__(self) -> None:
        super(elu, self).__init__()

    def forward(self, x):
        return torch.where(x >= 0, x, 0.2 * (torch.exp(x) - 1))

class linear(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super(linear, self).__init__()
        self.w = nn.Parameter(
            torch.randn(out_c, in_c) * torch.sqrt(torch.tensor(2 / in_c))
        )
        self.b = nn.Parameter(torch.randn(out_c))

    def forward(self, x):
        return F.linear(x, self.w, self.b)

class MLP_MNIST(nn.Module):
    def __init__(self) -> None:
        super(MLP_MNIST, self).__init__()
        self.fc1 = linear(28 * 28, 80)
        self.fc2 = linear(80, 60)
        self.fc3 = linear(60, 10)
        self.flatten = nn.Flatten()
        self.activation = elu()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x
    
    def set_params(self, model_params=None):
        if model_params is not None:
            with torch.no_grad():
                for param, value in zip(self.parameters(), model_params):
                    # print(type(value))
                    if isinstance(value, np.ndarray):
                        param.copy_(torch.from_numpy(value))
                    elif isinstance(value, torch.Tensor):
                        param.copy_(value)
                    else:
                        print("Variable is neither a numpy.ndarray nor a torch.Tensor")
                        # print("check loaded model:  ->" , model_params)
                        self.load_state_dict(model_params)
                        break

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]
        
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]
        
    def get_representation_features(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)
        return func(x)
    
    def get_feature_dim(self):
        return 60

    def classifier(self, x):
        return self.activation(self.fc3(x))


class LeNet5(nn.Module):
    def __init__(self, dataset) -> None:
        super(LeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ARGS[dataset][0], 6, 5),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(ARGS[dataset][1], 120),
            nn.PReLU(),
            nn.Linear(120, 84),
            nn.PReLU(),
            nn.Linear(84, ARGS[dataset][2]),
        )

    def forward(self, x):
        return self.net(x)
    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]
        
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]
        
    def get_representation_features(self, x):
       # Debugging print statements
        # print("Initial input requires_grad:", x.requires_grad)
        for i, layer in enumerate(self.net[:10]):
            x = layer(x)
            # print(f"Layer {i}, requires_grad={x.requires_grad}")
        return x
    def get_representation_params_t(self):
        return self.net[:10].parameters()
    def get_representation_params(self):
        return list(self.net[:10].parameters())
    
    def classifier(self, x):
        return self.net[10:](x)
    
    def get_classifier_params(self):
        return list(self.net[10:].parameters())
    
    def get_classifier_params_t(self):
        return self.net[10:].parameters()
    
    def get_classifier_named_params_t(self):
        return self.net[10:].named_parameters()
    
    def get_feature_dim(self):
        return 84
    
class MLP_CIFAR10(nn.Module):
    def __init__(self) -> None:
        super(MLP_CIFAR10, self).__init__()
        self.fc1 = linear(32 * 32 * 3, 80)
        self.fc2 = linear(80, 60)
        self.fc3 = linear(60, 10)
        self.flatten = nn.Flatten()
        self.activation = elu()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x
    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]
        
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]
        
    def get_representation_features(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)
        return func(x)
    
    def classifier(self, x):
        return self.activation(self.fc3(x))

    def get_feature_dim(self):
        return 60
        
class MLP_CIFAR100(nn.Module):
    def __init__(self) -> None:
        super(MLP_CIFAR100, self).__init__()
        self.fc1 = linear(32 * 32 * 3, 512)
        self.fc2 = linear(512, 256)
        self.fc3 = linear(256, 100)
        self.flatten = nn.Flatten()
        self.activation = elu()
        self.classifier = None

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x

    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]
        
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]
        
    def get_representation_features(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)
        return func(x)
    
    def classifier(self, x):
        return self.activation(self.fc3(x))
    
    def get_feature_dim(self):
        return 256
    
class MNIST_SOLVER(nn.Module):
    def __init__(self) -> None:
        super(MNIST_SOLVER, self).__init__()
        self.fc1 = linear(28*28, 128)
        self.fc2 = linear(128, 128)
        self.fc3 = linear(128, 10)
        self.flatten = nn.Flatten()
        self.activation = elu()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)
        
        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x
    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]
        
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]
        
    def get_representation_features(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)
        return func(x)
    
    def classifier(self, x):
        return self.activation(self.fc3(x))
    
    def get_feature_dim(self):
        return 128
        
class TorchResNet(nn.Module):
    def __init__(self, num_classes=None):
        super(TorchResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Load pre-trained ResNet-18
        # Modify the last fully connected layer
        if num_classes != None:
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]
        
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]
        
    def get_representation_features(self, x):
        x= self.resnet.conv1(x)
        x= self.resnet.bn1(x)
        x= self.resnet.relu(x)
        x= self.resnet.maxpool(x)
        x= self.resnet.layer1(x)
        x= self.resnet.layer2(x)
        x= self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return func(x)
    
    def classifier(self, x):
        return self.resnet.fc(x)
    
    def get_feature_dim(self):
        return 512
        
class TorchResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(TorchResNet20, self).__init__()
        self.resnet18 = models.resnet18()
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet18(x)
    
    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]
        
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

def torch_resnet18(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) 
    return model
# TResNet18 = torch_resnet18(10)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

Resnet8_avg_pool = {"mnist": 3, "cifar10": 4, "cifar": 4, "cifar100": 4, "tinyimagenet": 6}
class ResNet8(nn.Module):
    def __init__(self, block, num_blocks, dataset, num_classes=10, input_channels=3):
        super(ResNet8, self).__init__()
        self.dataset = dataset
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        
        # Calculate the size of the feature maps after the last layer
        # For CIFAR-10: 32x32 -> 16x16 after layer2 (stride=2)
        # Then 16x16 -> 4x4 after avg_pool
        self.feature_size = 128 * block.expansion * 4 * 4  # 128 channels * 4x4 spatial size
        
        # Final fully connected layer
        self.linear = nn.Linear(self.feature_size, num_classes)
        
        # Feature extraction layers
        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.layer1,
            self.layer2
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = F.avg_pool2d(out, Resnet8_avg_pool[self.dataset])
        out = out.view(out.size(0), -1)  # Flatten
        out = self.linear(out)
        return out

    def get_feature_dim(self):
        return self.feature_size

Resnet18_avg_pool = {"mnist":3, "cifar10":4, "cifar":4, "cifar100":4, "tinyimagenet":6}
# Define ResNet18 model for CIFAR-10
class ResNet18(nn.Module):
    def __init__(self, block, num_blocks, dataset, input_channels=3, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.dataset=dataset
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.PReLU(),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        # print(f'\nshape1: {out.shape}')
        out = F.avg_pool2d(out, Resnet18_avg_pool[self.dataset])
        out = out.view(out.size(0), -1)
        # print(f'\nshape2: {out.shape}')
        out = self.linear(out)
        return out
    
    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]
        
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]
        
    def get_representation_features(self, x):
        out = self.features(x)        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def get_representation_params_t(self):
        return self.features.parameters()
    
    def get_classifier_params_t(self):
        return self.linear.parameters()
    
    def classifier(self, x):
        return self.linear(x)
    
    def get_feature_dim(self):
        return 512
    

class EnhancedClassificationNet(nn.Module):
    def __init__(self, input_size, input_channels, conv_channel1, conv_channel2, conv_channel3, fc1_features, fc2_features, use_batch_norm, use_dropout, dropout_rate):
        super(EnhancedClassificationNet, self).__init__()
        # First conv block with stride=2 for faster downsampling
        self.conv1 = nn.Conv2d(input_channels, conv_channel1, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channel1) if use_batch_norm else nn.Identity()
        
        # Second conv block with stride=2
        self.conv2 = nn.Conv2d(conv_channel1, conv_channel2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channel2) if use_batch_norm else nn.Identity()
        
        # Calculate the size of flattened features
        self.flatten_size = conv_channel2 * (input_size // 4) * (input_size // 4)
        
        # FC layers with reduced sizes for faster training
        self.fc1 = nn.Linear(self.flatten_size, fc1_features // 4)
        self.fc2 = nn.Linear(fc1_features // 4, fc2_features)
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        
        # Use inplace operations for memory efficiency
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First conv block with batch norm and relu
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second conv block with batch norm and relu
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_feature_dim(self):
        return self.fc1.out_features
        
    def get_params(self):
        with torch.no_grad():
            return [param.clone().cpu().detach().numpy() for param in self.parameters()]
            
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]
            
    def get_representation_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return x
        
    def set_params(self, model_params=None):
        if model_params is not None:
            with torch.no_grad():
                for param, value in zip(self.parameters(), model_params):
                    if isinstance(value, np.ndarray):
                        param.copy_(torch.from_numpy(value))
                    elif isinstance(value, torch.Tensor):
                        param.copy_(value)
                    else:
                        self.load_state_dict(model_params)
                        break
                        
    def get_representation_params_t(self):
        return [p for p in self.parameters() if p.requires_grad and p not in self.fc2.parameters()]
        
    def get_representation_params(self):
        return [p.detach().cpu().numpy() for p in self.get_representation_params_t()]
        
    def get_classifier_params_t(self):
        return self.fc2.parameters()
        
    def get_classifier_params(self):
        return [p.detach().cpu().numpy() for p in self.fc2.parameters()]
        
    def get_classifier_named_params_t(self):
        return self.fc2.named_parameters()
        
    def classifier(self, x):
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PBClassificationNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PBClassificationNet, self).__init__()
        self.classifier = nn.Sequential(
        nn.Linear(input_dim, 256),
        F.relu(),
        nn.BatchNorm1d(256),  # Batch normalization
        nn.Linear(256, 128),
        nn.Dropout(0.5),  # Regularization to prevent overfitting
        nn.Linear(128, 2),
        )
        self.parallel_classifier = nn.ModuleList([self.classifier for _ in range(num_classes)])
    
    def forward(self, x):
        x = self.parallel_classifier(x)
        return x
    
    
    def get_params_t(self):
        with torch.no_grad():
            return [param.clone().detach() for param in self.parameters()]

# Define the Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        # Define a linear layer
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # Apply the linear transformation
        return self.fc(x)

def MnistResNet8():
    print("mnist")
    return ResNet8(BasicBlock, [1, 1], input_channels=1, num_classes=10)

N_channels = {"mnist": [1,18], "cifar": [3,32], "cifar100": [3,32], "tinyimagenet": [3,64]}
class AutoencoderSimple(nn.Module):
    def __init__(self, dataset):
        super(AutoencoderSimple, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(N_channels[dataset][0], N_channels[dataset][1], kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv4 = nn.Conv2d( N_channels[dataset][1],  N_channels[dataset][0], kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        return x

    def decode(self, x):
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = torch.sigmoid(self.dec_conv4(x))  # Sigmoid to ensure output is between 0 and 1
        return x

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
    
def weights_init(m):
    # print("Setting initial weight..")
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Autoencoder(nn.Module):
    def __init__(self, dataset):
        super(Autoencoder, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(N_channels[dataset][0], N_channels[dataset][1], kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(N_channels[dataset][1], 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64,N_channels[dataset][1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv4 = nn.Conv2d(N_channels[dataset][1], N_channels[dataset][0], kernel_size=3, stride=1, padding=1)

        # Initialize weights
        self.apply(weights_init)

    def encode(self, x):
        x = F.leaky_relu(self.enc_conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.enc_conv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.enc_conv3(x), negative_slope=0.01)
        x = F.leaky_relu(self.enc_conv4(x), negative_slope=0.01)
        return x

    def decode(self, x):
        x = F.leaky_relu(self.dec_conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.dec_conv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.dec_conv3(x), negative_slope=0.01)
        x = torch.sigmoid(self.dec_conv4(x))
        return x

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    
class AutoencoderO(nn.Module):
    def __init__(self, dataset):
        super(AutoencoderO, self).__init__()
        in_channels = N_channels[dataset][0]
        
        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, 32),
            self._conv_block(32, 64, stride=2),
            self._conv_block(64, 128, stride=2),
            self._conv_block(128, 256, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            self._conv_transpose_block(256, 128),
            self._conv_transpose_block(128, 64),
            self._conv_transpose_block(64, 32),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def _conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _conv_transpose_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class AutoencoderR(nn.Module):
    def __init__(self, dataset):
        super(Autoencoder, self).__init__()
        in_channels = N_channels[dataset][0]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VAE(nn.Module):
    def __init__(self, dataset, latent_dim=128):
        super(VAE, self).__init__()
        input_channels = N_channels[dataset][0]
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute the flattened size
        with torch.no_grad():
            sample = torch.randn(1, input_channels, 32, 32)  # Assuming 32x32 input
            flat_size = self.encoder(sample).shape[1]
        
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, flat_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 128, 4, 4)  # Reshape to match decoder input
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



MODEL_DICT = {"mnist": MLP_MNIST, 
              "cifar": MLP_CIFAR10, 
              "cifar10": MLP_CIFAR10,
              "cifar100": MLP_CIFAR100, 
              "nmnist": MLP_MNIST, 
              "tinyimagenet": MLP_CIFAR100,
              }
CLASSES = {"mnist":10, "nmnist":10,"cifar10":10, "cifar":10, "cifar100":100, "tinyimagenet":200, "emnist_digits":26, "emnist_balanced":47}

def get_model(dataset, device='cuda'):
    """Get model for the dataset"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    if dataset == 'cifar':
        model = ResNet8(BasicBlock, [1, 1], dataset='cifar', num_classes=10, input_channels=3)
    elif dataset == 'cifar10':
        model = ResNet8(BasicBlock, [1, 1], dataset='cifar10', num_classes=10, input_channels=3)
    elif dataset == 'cifar100':
        model = ResNet8(BasicBlock, [1, 1], dataset='cifar100', num_classes=100, input_channels=3)
    elif dataset == 'mnist':
        model = MLP_MNIST()
    elif dataset == 'tinyimagenet':
        model = ResNet8(BasicBlock, [1, 1], dataset='tinyimagenet', num_classes=200, input_channels=3)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    model = model.to(device)
    return model

CHANNELS = {"mnist": 1, 
            "cifar": 3, 
            "cifar10": 3,
            "cifar100": 3, 
            "tinyimagenet":3 
            }

MLP = {"mnist": MLP_MNIST(), 
        "cifar": MLP_CIFAR10(), 
        "cifar10": MLP_CIFAR10(),
        "cifar100": MLP_CIFAR100(), 
}

def get_model_by_name(dataset, model, device='cuda'):
    """Get model by name"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    if model == 'resnet8':
        model = ResNet8(BasicBlock, [1, 1], dataset=dataset, 
                       num_classes=CLASSES[dataset], 
                       input_channels=CHANNELS[dataset])
    elif model == 'mlp':
        model = MLP_MNIST()
    elif model == 'lenet5':
        model = LeNet5(dataset=dataset)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    model = model.to(device)
    return model

