import torch
import yaml
from utils.model_utils import build_model, get_model, load_model_checkpoint
from data.utils import get_device

# --- Set paths here ---
'''checkpoint_path = "./pretrained/skelite_2d/check/model_000400.pt"
config_path = "./pretrained/skelite_2d/config_drive.yaml"
output_path = "skelite_scripted.pt"'''

device = "cpu"

#crackseg9k fine tuned
checkpoint_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\fine_tuned_outputs\crackseg9k\model_results\skelite_epoch050.pth"
config_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\ft_cfg_cs9k.yml"
output_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\fine_tuned_outputs\crackseg9k\exported_last\skelite_ft_scripted.pt"

#concrete3k fine tuned
'''checkpoint_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\fine_tuned_outputs\concrete3k\model_results\skelite_epoch050.pth"
config_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\ft_cfg_concrete3k.yml"
output_path = r"D:\camerer_ml\adaptive_pca_3dreproject\Skelite\fine_tuned_outputs\concrete3k\exported_last\skelite_ft_scripted.pt"'''

# --- Load YAML config ---
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# --- Build model as in your code ---
model_module = get_model(config["net_type"])
model = build_model(model_module, config, device)
model = model.to(device)
model.eval()

# --- Load checkpoint ---
model = load_model_checkpoint(model, checkpoint_path, device)

# --- Export to TorchScript ---
print("PARAMETERS:")
for name, p in model.named_parameters():
    print(name, p.shape if p is not None else p)
print("BUFFERS:")
for name, b in model.named_buffers():
    print(name, b.shape if b is not None else b)
scripted_model = torch.jit.script(model)
scripted_model.save(output_path)
print(f"TorchScript model saved as: {output_path}")