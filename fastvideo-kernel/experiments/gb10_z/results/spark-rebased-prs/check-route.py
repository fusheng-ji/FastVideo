import os,json,torch
from fastvideo_kernel.triton_kernels import attn_qat_train as k
q=torch.zeros((1,1,128,128),dtype=torch.bfloat16,device='cuda')
os.environ['FASTVIDEO_ATTN_QAT_SM121_FWD_DV']='0'
legacy=k._use_sm121_forward_consistent_dv(q,q,q,False,True,True,False,False,True,False)
os.environ['FASTVIDEO_ATTN_QAT_SM121_FWD_DV']='1'
corrected=k._use_sm121_forward_consistent_dv(q,q,q,False,True,True,False,False,True,False)
print(json.dumps({'override_zero_forward_consistent':legacy,'override_one_forward_consistent':corrected}))
assert not legacy and corrected
