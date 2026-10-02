# PTQ_POST_BUILD 훅: BRECQ 모델의 블록별 출력 상대오차(FP 대비, calib 앞 32장). 붕괴 run과 정상 run을 나란히 비교용.
q = models["brecq"]; qm = q.model.to(device); fm = fp.model.to(device)   # 빌드 직후 일부 파라미터가 CPU에 남아 있음(평가 경로는 val()이 옮김)
def hook_all(m, store):
    hs = []
    for i, b in enumerate(m.model):
        hs.append(b.register_forward_hook(lambda mod, inp, out, i=i: store.setdefault(i, []).append(out)))
    head = m.model[-1]
    for br in ('cv2', 'cv3', 'cv4'):
        for l, sub in enumerate(getattr(head, br)):
            hs.append(sub.register_forward_hook(lambda mod, inp, out, k=f'{br}.{l}': store.setdefault(k, []).append(out)))
    return hs
def flat(o):
    if torch.is_tensor(o): return [o]
    if isinstance(o, (list, tuple)): return [t for x in o for t in flat(x)]
    return []
err = {}
with torch.no_grad():
    for x in calib[:32]:
        x = x.to(device)
        sq, sf = {}, {}
        hq = hook_all(qm, sq); hf = hook_all(fm, sf)
        qm(x); fm(x)
        for h in hq + hf: h.remove()
        for k in sf:
            if k not in sq: continue
            a = flat(sq[k][0]); b = flat(sf[k][0])
            if not a or len(a) != len(b) or any(u.shape != v.shape for u, v in zip(a, b)): continue
            num = sum(((u - v).float().pow(2).sum()) for u, v in zip(a, b)); den = sum((v.float().pow(2).sum()) for v in b)
            err.setdefault(k, []).append((num / den.clamp(min=1e-12)).sqrt().item())
for k, v in err.items():
    print(f"[diag4] {k}: 상대오차 {sum(v)/len(v):.4f}", flush=True)
