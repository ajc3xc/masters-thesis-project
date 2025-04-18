import torch
from WaveMixSR import WaveMixSR_V2

def export_onnx(weights, onnx_out):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaveMixSR_V2(
        sr=2, blocks=2, mult=1,
        final_dim=144, ff_channel=144, dropout=0.3
    ).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    dummy = torch.randn(1,1,256,256, device=device)
    torch.onnx.export(
        model, dummy, onnx_out,
        opset_version=15,
        input_names=['input'], output_names=['output'],
        dynamic_axes={
          'input':  {0:'batch',2:'h',3:'w'},
          'output': {0:'batch',2:'h',3:'w'}
        }
    )
    print(f'ONNX model saved to {onnx_out}')

if __name__=='__main__':
    export_onnx('models/bsd100_2x_y_df2k_33.2.pth', 'models/sr_block_2x.onnx')