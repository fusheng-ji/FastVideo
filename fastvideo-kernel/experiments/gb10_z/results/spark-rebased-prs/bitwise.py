import hashlib,json,os,sys
import torch
from fastvideo_kernel.triton_kernels import attn_qat_train as kernel
n=int(sys.argv[1]); output=sys.argv[2]
def digest(t):
 t=t.detach().contiguous().cpu();return {'sha256':hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest(),'dtype':str(t.dtype),'shape':list(t.shape),'finite':bool(torch.isfinite(t).all())}
torch.manual_seed(11)
q,k,v=[torch.randn((1,3,n,128),device='cuda',dtype=torch.bfloat16) for _ in range(3)];do=torch.randn_like(q)
record={'tokens':n,'seed':11,'heads':3,'inputs':{name:digest(t) for name,t in zip(('q','k','v','do'),(q,k,v,do))},'qat_source_sha256':hashlib.sha256(open(kernel.__file__,'rb').read()).hexdigest()}
try:
 q,k,v=[x.requires_grad_(True) for x in (q,k,v)]
 out=kernel.attention(q,k,v,False,128**-0.5,True,False,False,True,False,True,True,False,False,False)
 ste,m=out.grad_fn.saved_tensors[3:5]
 dq,dk,dv=torch.autograd.grad(out,(q,k,v),do)
 record['outputs']={name:digest(t) for name,t in zip(('O','M','dQ','dK','STE','dV'),(out,m,dq,dk,ste,dv))};record['status']='completed'
except BaseException as e:
 record.update(status='failed',error_type=type(e).__name__,error=str(e));raise
finally:
 with open(output,'w') as f:json.dump(record,f,indent=2)
