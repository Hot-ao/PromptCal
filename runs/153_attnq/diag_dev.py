q = models["brecq"]; qm = q.model.to(device)
print("[dev] device 변수:", device, flush=True)
for n, p in list(qm.named_parameters()) + list(qm.named_buffers()):
    if p.device.type == 'cpu': print("[dev] CPU 텐서(등록됨):", n, flush=True)
for n, m in qm.named_modules():
    for k, v in vars(m).items():
        if torch.is_tensor(v) and v.device.type == 'cpu': print("[dev] CPU 텐서(미등록 속성):", n, k, tuple(v.shape), flush=True)
print("[dev] calib 장치:", calib[0].device, flush=True)
