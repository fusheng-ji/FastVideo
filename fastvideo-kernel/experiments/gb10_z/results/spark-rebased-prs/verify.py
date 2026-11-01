import hashlib,importlib.metadata as m,json,os,subprocess
from pathlib import Path
import torch,triton,fastvideo,fastvideo_kernel
import fastvideo_kernel.triton_kernels.attn_qat_train as qat
repo=Path.cwd().resolve();kernel=(repo/'fastvideo-kernel').resolve()
assert Path(fastvideo.__file__).resolve().is_relative_to(repo)
assert Path(fastvideo_kernel.__file__).resolve().is_relative_to(kernel)
assert Path(qat.__file__).resolve().is_relative_to(kernel)
direct=json.loads(m.distribution('fastvideo-kernel').read_text('direct_url.json'))
assert direct['dir_info']['editable'] and direct['url']=='file://'+str(kernel)
print(json.dumps({'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'kernel_editable':True,'kernel_source_sha256':hashlib.sha256(Path(qat.__file__).read_bytes()).hexdigest(),'torch':torch.__version__,'triton':triton.__version__,'flash_attn':m.version('flash-attn'),'cuda':torch.version.cuda,'capability':torch.cuda.get_device_capability()}))
