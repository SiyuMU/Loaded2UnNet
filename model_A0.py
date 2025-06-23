"""
Load2UnNet: Graph Neural Network for Cardiac Unloading

Predicts unloaded cardiac geometry from end-diastolic meshes using 
Graph Attention Networks with cycle consistency loss.

Author: Siyu Mu
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp  # Add mixed precision training support
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import os
import glob
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import matplotlib
matplotlib.use('Agg')
import argparse
import pyvista as pv
from scipy import ndimage

# ========== CUDA/GPU Configuration ==========
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✓ CUDA optimizations enabled")
else:
    device = torch.device('cpu')
    print("Using CPU")

# ========== Path Configurations ==========
MODEL_NAME = "model_A0"  # Single model name variable

PATH_CONFIG = {
    'DATA_DIR': "dataset",
    'MODEL_NAME': MODEL_NAME,
    'OUTPUT_DIR': f"{MODEL_NAME}/predicted_unloaded",
    'MODEL_PATH': f"{MODEL_NAME}/models/{MODEL_NAME}.pt",
    'VAL_BEST_MODEL_PATH': f"{MODEL_NAME}/models/model_val_best.pt",
    'STATS_FILE': f"{MODEL_NAME}/test_statistics.txt",
    'CACHE_FILE': f"{MODEL_NAME}/dataset_cache.pt"
}

# ========== Training Configuration ==========
TRAINING_CONFIG = {
    'USE_MIXED_PRECISION': False,   # Mixed precision training (FP16) - Saves 50% memory, 1.5-2x speedup
    'ACCUMULATION_STEPS': 1,       # Gradient accumulation steps - Effective batch size = 1 * 10 = 10
    'USE_GRADIENT_CLIPPING': False, # Gradient clipping - Prevents gradient explosion
    'USE_ADAMW': True,             # AdamW optimizer - Generally better than Adam
    'NUM_WORKERS': 8,              # Data loader worker processes
    'PREFETCH_FACTOR': 4,          # Data prefetch factor
    'ENABLE_CUDA_OPTIMIZATIONS': True,  # CUDA optimizations - CUDNN benchmark etc.
    'WEAK_SUPERVISION_RATIO': 0.0  # Percentage of training data to use as weak supervision (0-1)
}

# ========== Model Configuration ==========
MODEL_CONFIG = {
    'HIDDEN_DIM': 128,            # Hidden dimension size
    'OUTPUT_DIM': 3,              # Output dimension (3D coordinates)
    'INCLUDE_GLOBAL_IN_NODES': False,  # Whether to include global parameters in node features
    'NUM_HEADS': 4,               # Number of attention heads
    'DROPOUT': 0.1                # Dropout rate
}

print("=" * 50)
print("Load2UnNet - Cardiac Unloading")
print("=" * 50)
for key, value in TRAINING_CONFIG.items():
    print(f"{key:25}: {value}")
print("=" * 50)

# ========== VTK Mesh I/O ==========
def load_vtu(path):
    """Load VTU mesh file."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def save_vtu(mesh, path):
    """Save mesh to VTU file."""
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(path)
    writer.SetInputData(mesh)
    writer.Write()

def extract_parameters_from_filename(filename):
    """Extract simulation parameters from filename.
    
    Format: ED1_4_50_60_60.vtu
    Returns: pressure, stiffness, endo_helix, epi_helix
    """
    try:
        basename = os.path.basename(filename)
        if 'ED' in basename:
            parts = basename.split('_')
            if len(parts) >= 5:
                pressure = float(parts[1])
                stiffness = float(parts[2])
                endo_helix = float(parts[3])
                epi_helix = float(parts[4].split('.')[0])
                return pressure, stiffness, endo_helix, epi_helix
        print(f"Warning: Could not extract parameters from {filename}, using default values")
        return 4.0, 50.0, 60.0, 60.0
    except:
        print(f"Warning: Could not extract parameters from {filename}, using default values")
        return 4.0, 50.0, 60.0, 60.0

def normalize_parameters(pressure, stiffness, endo_helix, epi_helix):
    """Normalize parameters to [0,1] range."""
    pressure_norm = (pressure - 4.0) / (14.0 - 4.0)
    stiffness_norm = (stiffness - 50.0) / (300.0 - 50.0)
    endo_helix_norm = (endo_helix - 60.0) / (70.0 - 60.0)
    epi_helix_norm = (epi_helix - 60.0) / (70.0 - 60.0)
    return pressure_norm, stiffness_norm, endo_helix_norm, epi_helix_norm

def mesh_to_graph(mesh):
    """Convert VTK mesh to PyTorch Geometric graph"""
    # Extract points and cells
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    
    # Create edges from cells (assuming tetrahedral mesh)
    cells = []
    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TETRA:
            # Add edges for each pair of points in the tetrahedron
            ids = [cell.GetPointId(j) for j in range(4)]
            for j in range(4):
                for k in range(j+1, 4):
                    cells.append([ids[j], ids[k]])
                    cells.append([ids[k], ids[j]])  # Bidirectional edges
    
    # Remove duplicate edges
    edges = np.unique(cells, axis=0)
    
    # Create PyTorch tensors
    x = torch.tensor(points, dtype=torch.float)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)

def normalize_mesh(mesh, center=None, scale=None):
    """Normalize mesh to have zero mean and unit variance"""
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    
    if center is None:
        center = np.mean(points, axis=0)
    points = points - center
    
    if scale is None:
        scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale
    
    # Update mesh points
    mesh_points = vtk.vtkPoints()
    mesh_points.SetData(numpy_to_vtk(points))
    mesh.SetPoints(mesh_points)
    
    return mesh, center, scale

def normalize_pressure(pressure):
    """Normalize pressure values to range [0, 1]"""
    min_pressure = 6.0  # Minimum pressure in dataset
    max_pressure = 15.0  # Maximum pressure in dataset
    return (pressure - min_pressure) / (max_pressure - min_pressure)

class ScaledMSELoss(nn.Module):
    def __init__(self, scale=1e6):
        super(ScaledMSELoss, self).__init__()
        self.scale = scale
        
    def forward(self, pred, target):
        # Calculate MSE and scale it
        mse = torch.mean((pred - target) ** 2)
        return mse * self.scale

class MeshDeformationModel(nn.Module):
    """Graph neural network for cardiac mesh deformation with cycle consistency."""
    
    def __init__(self, hidden_dim=64, output_dim=3, include_global_in_nodes=True):
        super(MeshDeformationModel, self).__init__()
        
        self.include_global_in_nodes = include_global_in_nodes
        
        input_dim = 7 if include_global_in_nodes else 3  # 3 coordinates + 4 global params or just 3 coordinates
        
        # Shared encoder components
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.1)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.1)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.1)
        
        self.global_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
        )
        
        self.skip_connection = nn.Linear(input_dim, hidden_dim * 4)
        
        # Attention fusion layers
        self.attention_fusion = nn.MultiheadAttention(hidden_dim * 4, num_heads=4, dropout=0.1)
        self.fusion_norm = nn.LayerNorm(hidden_dim * 4)
        self.fusion_proj = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        
        # Forward decoder (ED -> unloaded)
        self.forward_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Reverse decoder (unloaded -> ED)
        self.reverse_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def encode(self, data, global_params):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if global_params.dim() == 1:
            global_params = global_params.unsqueeze(0)
        pressure, stiffness, endo_helix, epi_helix = global_params[:, 0], global_params[:, 1], global_params[:, 2], global_params[:, 3]
        pressure_norm, stiffness_norm, endo_helix_norm, epi_helix_norm = normalize_parameters(
            pressure, stiffness, endo_helix, epi_helix
        )
        norm_params = torch.stack([pressure_norm, stiffness_norm, endo_helix_norm, epi_helix_norm], dim=1)

        global_features = self.global_encoder(norm_params)
        global_features = global_features[batch]

        if self.include_global_in_nodes:
            x_input = torch.cat([x, norm_params[batch]], dim=1)
        else:
            x_input = x

        x_skip = self.skip_connection(x_input)
        x1 = torch.relu(self.conv1(x_input, edge_index)) + x_skip
        x2 = torch.relu(self.conv2(x1, edge_index)) + x1
        x3 = self.conv3(x2, edge_index) + x2

        global_shape = global_mean_pool(x3, batch)  # [batch_size, hidden_dim*4]
        global_shape = global_shape[batch]          # [num_nodes, hidden_dim*4]

        # Reshape features for attention
        x3_attn = x3.unsqueeze(0)  # [1, num_nodes, hidden_dim*4]
        global_features_attn = global_features.unsqueeze(0)  # [1, num_nodes, hidden_dim*4]
        global_shape_attn = global_shape.unsqueeze(0)  # [1, num_nodes, hidden_dim*4]

        # Attention fusion
        fused_features1, _ = self.attention_fusion(x3_attn, global_features_attn, global_features_attn)
        fused_features1 = self.fusion_norm(fused_features1 + x3_attn)
        
        fused_features2, _ = self.attention_fusion(fused_features1, global_shape_attn, global_shape_attn)
        fused_features = self.fusion_norm(fused_features2 + fused_features1)
        
        return self.fusion_proj(fused_features.squeeze(0))  # [num_nodes, hidden_dim*4]
        
    def forward(self, data, global_params):
        # Encode the input mesh
        encoded_features = self.encode(data, global_params)
        
        # Forward transformation (ED -> unloaded)
        forward_deformation = self.forward_decoder(encoded_features)
        
        # 在GPU上直接计算变形后的点，不创建VTK网格
        deformed_points = data.x + forward_deformation
        # 使用变形后的点直接创建新的PyG数据
        intermediate_data = Data(x=deformed_points, edge_index=data.edge_index)
        
        # 确保intermediate_data有正确的batch属性
        if hasattr(data, 'batch'):
            intermediate_data.batch = data.batch
        else:
            intermediate_data.batch = torch.zeros(intermediate_data.x.size(0), dtype=torch.long, device=intermediate_data.x.device)
        
        # Encode the intermediate mesh
        intermediate_features = self.encode(intermediate_data, global_params)
        
        # Reverse transformation (unloaded -> ED)
        reverse_deformation = self.reverse_decoder(intermediate_features)
        
        return forward_deformation, reverse_deformation

def create_mesh_from_deformation(data, deformation):
    """Create a VTK mesh from PyTorch Geometric data and deformation"""
    # Get original points
    points = data.x.cpu().numpy()
    
    # Apply deformation
    deformed_points = points + deformation.cpu().numpy()
    
    # Create VTK mesh
    mesh = vtk.vtkUnstructuredGrid()
    
    # Create points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(deformed_points))
    mesh.SetPoints(vtk_points)
    
    # Create cells (assuming tetrahedral mesh)
    for i in range(data.edge_index.size(1)):
        cell = vtk.vtkTetra()
        cell.GetPointIds().SetId(0, data.edge_index[0, i])
        cell.GetPointIds().SetId(1, data.edge_index[1, i])
        cell.GetPointIds().SetId(2, data.edge_index[0, i])  # Reuse points for tetrahedron
        cell.GetPointIds().SetId(3, data.edge_index[1, i])  # Reuse points for tetrahedron
        mesh.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
    
    return mesh

def calculate_dice_coefficient(mesh1, mesh2):
    """Calculate DICE coefficient between two meshes based on point distances"""
    # Get points from both meshes
    points1 = vtk_to_numpy(mesh1.GetPoints().GetData())
    points2 = vtk_to_numpy(mesh2.GetPoints().GetData())
    
    # Calculate distances between corresponding points
    distances = np.linalg.norm(points1 - points2, axis=1)
    
    # Count points that are close enough (within threshold)
    threshold = 0.01  # Distance threshold in mm
    intersection = np.sum(distances < threshold)
    
    # Calculate DICE coefficient
    total_points = len(points1)
    dice = (2 * intersection) / (total_points + total_points)
    
    return dice

def calculate_geometric_similarity(mesh1, mesh2):
    """Calculate geometric similarity metrics between two meshes"""
    points1 = vtk_to_numpy(mesh1.GetPoints().GetData())
    points2 = vtk_to_numpy(mesh2.GetPoints().GetData())
    
    # Hausdorff distance
    hd1 = directed_hausdorff(points1, points2)[0]
    hd2 = directed_hausdorff(points2, points1)[0]
    hausdorff = max(hd1, hd2)
    
    # Mean distance
    tree = cKDTree(points2)
    distances, _ = tree.query(points1)
    mean_dist = np.mean(distances)
    
    # Standard deviation of distances
    std_dist = np.std(distances)
    
    # DICE coefficient
    dice = calculate_dice_coefficient(mesh1, mesh2)
    
    return {
        'hausdorff': hausdorff,
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'dice': dice
    }

def plot_metrics(train_losses, val_losses, output_dir):
    """Plot and save loss curves"""
    plt.figure(figsize=(10, 5))
    
    # Plot loss curves with log scale
    plt.plot(range(10, len(train_losses)), train_losses[10:], label='Training Loss')
    plt.plot(range(10, len(val_losses)), val_losses[10:], label='Validation Loss')
    plt.yscale('log')  # Use log scale for y-axis
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    
    # Save loss curves to txt file
    with open(os.path.join(output_dir, 'loss_curves.txt'), 'w') as f:
        f.write("Epoch\tTrain Loss\tValidation Loss\n")
        f.write("-" * 60 + "\n")
        for epoch in range(10, len(train_losses)):
            f.write(f"{epoch}\t{train_losses[epoch]:.6f}\t{val_losses[epoch]:.6f}\n")

def prepare_dataset(data_dir, include_global_in_nodes=True):
    """Prepare dataset from the directory structure"""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    train_dataset = []
    val_dataset = []
    test_dataset = []
    
    # First pass: compute global normalization parameters
    all_points = []
    for split_dir in [train_dir, val_dir, test_dir]:
        ed_dir = os.path.join(split_dir, 'ED')
        ed_files = sorted(glob.glob(os.path.join(ed_dir, '*.vtu')))
        for ed_file in ed_files:
            ed_mesh = load_vtu(ed_file)
            all_points.append(vtk_to_numpy(ed_mesh.GetPoints().GetData()))
    
    if not all_points:
        raise ValueError("No VTU files found in the dataset directories")
    
    all_points = np.vstack(all_points)
    global_center = np.mean(all_points, axis=0)
    global_scale = np.max(np.linalg.norm(all_points - global_center, axis=1))
    
    # Process each split
    for split_dir, dataset in [(train_dir, train_dataset), (val_dir, val_dataset), (test_dir, test_dataset)]:
        ed_dir = os.path.join(split_dir, 'ED')
        unloaded_dir = os.path.join(split_dir, 'unloaded')
        
        ed_files = sorted(glob.glob(os.path.join(ed_dir, '*.vtu')))
        for ed_file in ed_files:
            # Find corresponding unloaded file
            unloaded_file = os.path.join(unloaded_dir, os.path.basename(ed_file).replace('ED', 'unloaded'))
            if not os.path.exists(unloaded_file):
                print(f"Warning: No matching unloaded file for {ed_file}")
                continue
            
            # Load meshes
            ed_mesh = load_vtu(ed_file)
            unloaded_mesh = load_vtu(unloaded_file)
            
            # Normalize meshes
            ed_mesh, _, _ = normalize_mesh(ed_mesh, global_center, global_scale)
            unloaded_mesh_norm, _, _ = normalize_mesh(unloaded_mesh, global_center, global_scale)
            
            # Extract parameters
            pressure, stiffness, endo_helix, epi_helix = extract_parameters_from_filename(ed_file)
            
            # Convert to graph
            ed_graph = mesh_to_graph(ed_mesh)
            
            # Extract points
            ed_points = vtk_to_numpy(ed_mesh.GetPoints().GetData())
            unloaded_points = vtk_to_numpy(unloaded_mesh_norm.GetPoints().GetData())
            
            # Calculate deformation field
            deformation = unloaded_points - ed_points
            deformation_tensor = torch.tensor(deformation, dtype=torch.float)
            
            # Create parameter tensors
            params_tensor = torch.tensor([pressure, stiffness, endo_helix, epi_helix], dtype=torch.float)
           
            # Add to dataset
            if split_dir == test_dir:
                dataset.append((ed_graph, deformation_tensor, params_tensor, ed_file, unloaded_file))
            else:
                dataset.append((ed_graph, deformation_tensor, params_tensor))
    
    # Apply weak supervision: randomly select percentage of data for weak supervision
    if train_dataset:
        np.random.seed(42)  # Ensure reproducibility
        num_samples = len(train_dataset)
        num_weak = int(num_samples * TRAINING_CONFIG['WEAK_SUPERVISION_RATIO'])
        weak_indices = set(range(num_weak))
        
        print(f"\nWeak Supervision Settings:")
        print(f"Total training samples: {num_samples}")
        print(f"Strong supervision samples: {num_samples - num_weak}")
        print(f"Weak supervision samples: {num_weak} ({TRAINING_CONFIG['WEAK_SUPERVISION_RATIO']*100:.0f}%)")
        
        # Mark selected samples as weak supervision, but keep GT labels for cycle loss
        new_train_dataset = []
        for i, (ed_graph, deformation_tensor, params_tensor) in enumerate(train_dataset):
            if i in weak_indices:
                # Weak supervision sample: add flag but keep GT for cycle loss
                new_train_dataset.append((ed_graph, deformation_tensor, params_tensor, True))  # True indicates weak supervision
            else:
                # Strong supervision sample: add flag
                new_train_dataset.append((ed_graph, deformation_tensor, params_tensor, False))  # False indicates strong supervision
        
        train_dataset = new_train_dataset
    
    return train_dataset, val_dataset, test_dataset, global_center, global_scale

def train_model(model, train_loader, val_loader, epochs=1000, lr=0.001, output_dir=None, patience=50, val_best_model_path=None):
    """Main training function for the model"""
    
    # Get optimization configuration
    USE_MIXED_PRECISION = TRAINING_CONFIG['USE_MIXED_PRECISION']
    ACCUMULATION_STEPS = TRAINING_CONFIG['ACCUMULATION_STEPS']
    USE_GRADIENT_CLIPPING = TRAINING_CONFIG['USE_GRADIENT_CLIPPING']
    USE_ADAMW = TRAINING_CONFIG['USE_ADAMW']
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer and scheduler
    if USE_ADAMW:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Loss function and mixed precision
    loss_fn = ScaledMSELoss(scale=1e6)
    scaler = torch.cuda.amp.GradScaler() if (torch.cuda.is_available() and USE_MIXED_PRECISION) else None
    
    # Print configuration information
    print(f"Training Configuration:")
    print(f"  Mixed Precision: {USE_MIXED_PRECISION}")
    print(f"  Gradient Accumulation: {ACCUMULATION_STEPS} steps")
    print(f"  Gradient Clipping: {USE_GRADIENT_CLIPPING}")
    print(f"  Optimizer: {'AdamW' if USE_ADAMW else 'Adam'}")
    print(f"  Effective Batch Size: {ACCUMULATION_STEPS}")
    
    # Training state variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    no_improvement_count = 0
    cycle_weight = 0.2  # Cycle consistency weight
    
    # Early stopping variables
    min_val_loss = float('inf')
    min_val_loss_epoch = 0
    val_loss_history = []
    window_size = 5  # Window size for inflection point detection
    patience = 50    # Early stopping patience

    for epoch in range(epochs):
        # Add this line to trigger output refresh, ensuring epoch information is displayed
        print(f"Starting epoch {epoch+1}...", flush=True)
        
        # ========== Training Phase ==========
        model.train()
        train_forward_loss = 0
        train_cycle_loss = 0
        train_total_loss = 0
        
        optimizer.zero_grad()
        accumulated_loss = 0
        
        for i, batch in enumerate(train_loader):
            # Unpack batch data, handle weak supervision samples
            if len(batch) == 4:
                data, target, global_params, is_weak_supervision = batch
                is_weak_supervision = is_weak_supervision[0]  # Get scalar value
            elif len(batch) == 3:
                data, target, global_params = batch
                is_weak_supervision = False  # Validation set samples are all strongly supervised
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
            
            # Move data to device
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            global_params = global_params.to(device, non_blocking=True)
            
            # Forward pass (with mixed precision support)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    forward_deformation, reverse_deformation = model(data, global_params)
            
                    # Calculate loss
                    if not is_weak_supervision:  # Strong supervision samples
                        if target.dim() == 3 and target.size(0) == 1:
                            target = target.squeeze(0)
                        forward_loss = loss_fn(forward_deformation, target)
                    else:  # Weak supervision samples
                        forward_loss = torch.tensor(0.0, device=device)
                    
                    # Cycle consistency loss (calculated for all samples)
                    cycle_loss = loss_fn(reverse_deformation, -forward_deformation)
                    
                    # Total loss
                    if not is_weak_supervision:
                        total_loss = (forward_loss + cycle_weight * cycle_loss) / ACCUMULATION_STEPS
                    else:
                        total_loss = cycle_weight * cycle_loss / ACCUMULATION_STEPS
                
                # Backward pass
                scaler.scale(total_loss).backward()
                accumulated_loss += total_loss.item()
                
                # Gradient accumulation and update
                if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    if USE_GRADIENT_CLIPPING:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    train_total_loss += accumulated_loss * ACCUMULATION_STEPS
                    accumulated_loss = 0
            else:
                # Standard precision training
                forward_deformation, reverse_deformation = model(data, global_params)
                
                if target.dim() == 3 and target.size(0) == 1:
                    target = target.squeeze(0)
                forward_loss = loss_fn(forward_deformation, target)
                
                original_points = data.x
                cycle_loss = loss_fn(reverse_deformation, -forward_deformation)
                total_loss = (forward_loss + cycle_weight * cycle_loss) / ACCUMULATION_STEPS
                
                total_loss.backward()
                accumulated_loss += total_loss.item()
                
                if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    if USE_GRADIENT_CLIPPING:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    train_total_loss += accumulated_loss * ACCUMULATION_STEPS
                    accumulated_loss = 0
            
            train_forward_loss += forward_loss.item()
            train_cycle_loss += cycle_loss.item()
        
        # Calculate average losses
        num_batches = len(train_loader)
        if num_batches > 0:
            train_forward_loss /= num_batches
            train_cycle_loss /= num_batches
            train_total_loss /= num_batches
        train_losses.append(train_total_loss)

        # ========== Validation Phase ==========
        model.eval()
        val_forward_loss = 0
        val_cycle_loss = 0
        val_total_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for data, target, global_params in val_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                global_params = global_params.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        forward_deformation, reverse_deformation = model(data, global_params)
                
                    if target.dim() == 3 and target.size(0) == 1:
                        target = target.squeeze(0)
                    forward_loss = loss_fn(forward_deformation, target)
                
                    original_points = data.x
                    cycle_loss = loss_fn(reverse_deformation, -forward_deformation)
                    total_loss = forward_loss + cycle_weight * cycle_loss
                else:
                    forward_deformation, reverse_deformation = model(data, global_params)
                
                    if target.dim() == 3 and target.size(0) == 1:
                        target = target.squeeze(0)
                    forward_loss = loss_fn(forward_deformation, target)
                    
                    original_points = data.x
                    cycle_loss = loss_fn(reverse_deformation, -forward_deformation)
                total_loss = forward_loss + cycle_weight * cycle_loss
                
                val_forward_loss += forward_loss.item()
                val_cycle_loss += cycle_loss.item()
                val_total_loss += total_loss.item()
                val_batches += 1
                
        if val_batches > 0:
            val_forward_loss /= val_batches
            val_cycle_loss /= val_batches
            val_total_loss /= val_batches
        val_losses.append(val_total_loss)
        
        # ========== Training Monitoring ==========
        # Print information every 10 epochs or first 10 epochs
        if epoch % 10 == 0 or epoch < 10:
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train - Forward: {train_forward_loss:.6f}, Cycle: {train_cycle_loss:.6f}, Total: {train_total_loss:.6f}')
            print(f'Val   - Forward: {val_forward_loss:.6f}, Cycle: {val_cycle_loss:.6f}, Total: {val_total_loss:.6f}')

            # GPU memory monitoring
            if torch.cuda.is_available():
                print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB')
                torch.cuda.reset_peak_memory_stats()

        # Update plots every 50 epochs
        if (epoch + 1) % 50 == 0 and output_dir:
            plot_metrics(train_losses, val_losses, output_dir)

        # Learning rate scheduling
        scheduler.step(val_total_loss)

        # Save best validation model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            if val_best_model_path is not None:
                torch.save(model.state_dict(), val_best_model_path)
            print(f"Saved new val best model at epoch {epoch+1} with val loss {val_total_loss:.6f}")
            min_val_loss = val_total_loss
            min_val_loss_epoch = epoch

        # Update validation loss history
        val_loss_history.append(val_total_loss)
        
        # Detect inflection point
        if len(val_loss_history) >= window_size * 2:  # Ensure enough history data
            recent_avg = np.mean(val_loss_history[-window_size:])
            prev_avg = np.mean(val_loss_history[-2*window_size:-window_size])
            
            if not np.isnan(recent_avg) and not np.isnan(prev_avg):
                if recent_avg > prev_avg * 1.05:  # 5% threshold
                    print(f"Detected inflection point at epoch {epoch+1}")
                    print(f"Previous average: {prev_avg:.6f}, Recent average: {recent_avg:.6f}")
                    break

        # Early stopping check
        if epoch - min_val_loss_epoch >= patience:
            print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
            break

    # Load best validation model
    if val_best_model_path is not None and os.path.exists(val_best_model_path):
        model.load_state_dict(torch.load(val_best_model_path))
        print(f"Loaded best validation model from epoch {min_val_loss_epoch+1}")
    
    return model

def evaluate_model(model, test_dataset, output_dir, global_center, global_scale):
    """Evaluate model on test dataset and compare with ground truth"""
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    results = []
    metrics = {
        'hausdorff': [],
        'mean_distance': [],
        'std_distance': [],
        'dice': []
    }
    
    with torch.no_grad():
        for ed_graph, deformation, params, ed_file, unloaded_file in test_loader:
            ed_mesh = load_vtu(ed_file[0])
            gt_unloaded_mesh = load_vtu(unloaded_file[0])

            ed_mesh_norm, _, _ = normalize_mesh(ed_mesh, global_center, global_scale)
            ed_graph_norm = mesh_to_graph(ed_mesh_norm)
            ed_graph_norm = ed_graph_norm.to(device)
            params = params.to(device)
            params = params.view(-1)
 
            num_nodes = ed_graph_norm.x.shape[0]
            ed_graph_norm.batch = torch.zeros(num_nodes, dtype=torch.long, device=ed_graph_norm.x.device)
            
            # Predict deformation in normalized space
            forward_deformation, _ = model(ed_graph_norm, params)
            
            # Create predicted mesh in normalized space
            predicted_mesh_norm = create_mesh_from_deformation(ed_graph_norm, forward_deformation)
            
            # Normalize ground truth mesh using same parameters
            gt_unloaded_mesh_norm, _, _ = normalize_mesh(gt_unloaded_mesh, global_center, global_scale)
            
            # Calculate geometric similarity in normalized space
            similarity = calculate_geometric_similarity(predicted_mesh_norm, gt_unloaded_mesh_norm)
            
            metrics['hausdorff'].append(similarity['hausdorff']* global_scale)
            metrics['mean_distance'].append(similarity['mean_distance']* global_scale)
            metrics['std_distance'].append(similarity['std_distance']* global_scale)
            metrics['dice'].append(similarity['dice'])
            
            # Denormalize predicted mesh for saving
            points = vtk_to_numpy(predicted_mesh_norm.GetPoints().GetData())
            points = points * global_scale + global_center
            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(numpy_to_vtk(points))
            predicted_mesh = vtk.vtkUnstructuredGrid()
            predicted_mesh.DeepCopy(ed_mesh)
            predicted_mesh.SetPoints(vtk_points)
            
            results.append({
                'case': os.path.basename(ed_file[0]),
                'similarity': similarity,
                'predicted_mesh': predicted_mesh
            })

            output_file = os.path.join(output_dir, f"predicted_{os.path.basename(ed_file[0])}")
            save_vtu(predicted_mesh, output_file)
            
            print(f"Processed {os.path.basename(ed_file[0])}")
            print(f"DICE score: {similarity['dice']:.6f}")
            print(f"Average distance: {similarity['mean_distance']* global_scale:.6f}")
            print(f"HD: {similarity['hausdorff']* global_scale:.6f}")
            print(f"scaling: {global_scale}")
            print("shift: ", global_center)
    
    stats = {}
    for metric, values in metrics.items():
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return results, stats

def main():
    """Main function for training and evaluation"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Only evaluate on test set using saved model')
    args = parser.parse_args()

    # ========== System Configuration ==========
    if torch.cuda.is_available() and TRAINING_CONFIG['ENABLE_CUDA_OPTIMIZATIONS']:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ CUDA optimizations enabled")

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = not TRAINING_CONFIG['ENABLE_CUDA_OPTIMIZATIONS']
    
    # Create output directories
    os.makedirs(PATH_CONFIG['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(os.path.dirname(PATH_CONFIG['MODEL_PATH']), exist_ok=True)
    
    # ========== Dataset Preparation ==========
    if os.path.exists(PATH_CONFIG['CACHE_FILE']):
        print("Loading cached dataset...")
        cache = torch.load(PATH_CONFIG['CACHE_FILE'], weights_only=False)
        train_dataset = cache['train_dataset']
        val_dataset = cache['val_dataset']
        test_dataset = cache['test_dataset']
        global_center = cache['global_center']
        global_scale = cache['global_scale']
    else:
        print("Preparing dataset and saving cache...")
        train_dataset, val_dataset, test_dataset, global_center, global_scale = prepare_dataset(
            PATH_CONFIG['DATA_DIR'], 
            include_global_in_nodes=MODEL_CONFIG['INCLUDE_GLOBAL_IN_NODES']
        )
        torch.save({
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'global_center': global_center,
            'global_scale': global_scale
        }, PATH_CONFIG['CACHE_FILE'])
    
    # ========== Data Loaders ==========
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=TRAINING_CONFIG['NUM_WORKERS'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=TRAINING_CONFIG['PREFETCH_FACTOR'],
        generator=torch.Generator().manual_seed(seed)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=TRAINING_CONFIG['NUM_WORKERS'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=TRAINING_CONFIG['PREFETCH_FACTOR']
    )
    
    # ========== Model Initialization ==========
    model = MeshDeformationModel(
        hidden_dim=MODEL_CONFIG['HIDDEN_DIM'],
        include_global_in_nodes=MODEL_CONFIG['INCLUDE_GLOBAL_IN_NODES']
    )

    # ========== Training or Evaluation ==========
    if args.eval:
        print("Evaluation mode: loading model and running on test set...")
        if not os.path.exists(PATH_CONFIG['VAL_BEST_MODEL_PATH']):
            raise FileNotFoundError(f"Best validation model file {PATH_CONFIG['VAL_BEST_MODEL_PATH']} not found!")
        model.load_state_dict(torch.load(PATH_CONFIG['VAL_BEST_MODEL_PATH'], map_location='cpu'))
        print("Best validation model loaded.")
    else:
        print("Training model...")
        model = train_model(
            model, train_loader, val_loader, epochs=1000, lr=0.001, 
            output_dir=PATH_CONFIG['OUTPUT_DIR'],
            val_best_model_path=PATH_CONFIG['VAL_BEST_MODEL_PATH']
        )
        print("Model training finished.")

    # ========== Test Evaluation ==========
    print("Evaluating on test set...")
    test_results, test_stats = evaluate_model(
        model, test_dataset, PATH_CONFIG['OUTPUT_DIR'], 
        global_center, global_scale
    )
    print("Evaluation finished. Saving statistics...")
    
    # Save test statistics
    with open(PATH_CONFIG['STATS_FILE'], 'w') as f:
        f.write("Test Statistics\n")
        f.write("==============\n\n")
        f.write("Metric\t\tMean\t\tStd\n")
        f.write("-" * 50 + "\n")
        f.write(f"Hausdorff\t{test_stats['hausdorff']['mean']:.6f}\t{test_stats['hausdorff']['std']:.6f}\n")
        f.write(f"Mean Distance\t{test_stats['mean_distance']['mean']:.6f}\t{test_stats['mean_distance']['std']:.6f}\n")
        f.write(f"Std Distance\t{test_stats['std_distance']['mean']:.6f}\t{test_stats['std_distance']['std']:.6f}\n")
        f.write(f"DICE\t\t{test_stats['dice']['mean']:.6f}\t{test_stats['dice']['std']:.6f}\n")
    
    print(f"Test statistics saved to {PATH_CONFIG['STATS_FILE']}")
    print("Test Statistics Summary:")
    print(f"Hausdorff: {test_stats['hausdorff']['mean']:.6f} ± {test_stats['hausdorff']['std']:.6f}")
    print(f"Mean Distance: {test_stats['mean_distance']['mean']:.6f} ± {test_stats['mean_distance']['std']:.6f}")
    print(f"Std Distance: {test_stats['std_distance']['mean']:.6f} ± {test_stats['std_distance']['std']:.6f}")
    print(f"DICE: {test_stats['dice']['mean']:.6f} ± {test_stats['dice']['std']:.6f}")

if __name__ == "__main__":
    main() 


