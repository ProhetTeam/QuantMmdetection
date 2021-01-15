import torch
from thirdparty.mtransformer import build_mtransformer
from thirdparty.mtransformer.APOT import APOTQuantConv2d

try:
    from pytorch_memlab import LineProfiler, MemReporter
except ImportError:
    raise ImportError("Please Install pytorch_memlab: pip install pytorch_memlab")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

use_quant = True
if  use_quant: 
    conv = APOTQuantConv2d(32, 32, 3, 3, bit = 4).to(device)
else:
    conv = torch.nn.Conv2d(32, 32, 3, 3).to(device)
input = torch.ones(()).new_empty(
                (1, 32, 640, 640),
                dtype=next(conv.parameters()).dtype,
                device=next(conv.parameters()).device)
# pass in a model to automatically infer the tensor names
test = conv(input).mean()
reporter = MemReporter(conv)
out = conv(input).mean()
print('========= before backward =========')
reporter.report(verbose=True)
out.backward()
print('========= after backward =========')
print(reporter.report(verbose=True))