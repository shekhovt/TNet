# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
#%%
import os, relimport
os.chdir(relimport.proj_dir())
from . import interactive
import importlib
from . import exp_plotting
importlib.reload(exp_plotting)
from .exp_plotting import *
# globals().update(vars(exp_plotting))
#%%
ll = []

# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=[0,0,0,0])' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 3 + 2 = 6")]
# # perhaps recompute with no extra layers after the last reduction
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 6 + 2 = 9")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 9 + 2 = 12")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(3,3,3,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 12 + 2 = 15")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(5,5,5,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 18 + 2 = 21")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(7,7,7,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 24 + 2 = 27")]
# plain models scaling with double inner width
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DW=True,DB=False,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 6 + 2 = 9")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DW=True,DB=False,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 9 + 2 = 12")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DW=True,DB=False,layers=(3,3,3,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 12 + 2 = 15")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DW=True,DB=False,layers=(5,5,5,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 18 + 2 = 21")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DW=True,DB=False,layers=(7,7,7,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 24 + 2 = 27")]
# tower networks
# T2, T4, T8
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 3 + 2 = 9")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 6 + 2 = 15")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 9 + 2 = 21")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(3,3,3,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 12 + 2 = 27")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(5,5,5,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 18 + 2 = 38")]
#         
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=4,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower4 x 3 + 2 = 15")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=4,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower4 x 6 + 2 = 27")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=4,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower4 x 9 + 2 = 38")]

ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=8,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower8 x 3 + 2 = 27")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=8,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower8 x 6 + 2 = 51")]

# refernce fp32
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 6 + 2 = 15")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32' --compile", "1 + Tower2 x 6 + 2 = 15 fp32")]
# BiNeal

# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet(m=1)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32' --compile", "BiNeal(m=1) fp32")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet(m=1)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25' --fp16 --compile", "BiNeal(m=1) fp16")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet(m=1)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32-FP1' --compile", "BiNeal(m=1) fp32")]
# no difference...
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet(m=1.5)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32' --compile", "BiNeal(m=1.5) fp32")]
#

# rr = plot(ll);
rr = plot(ll, title="", experiment="../tex/fix/scaling/scaling", loc=2)

#%%
# final validation accuracy vs depth: Normal, Tower-2, Tower-4, Tower-8
import re
import io
import contextlib
from matplotlib import patheffects as pe

arch_data = {}
for r, spec in zip(rr, ll):
    cmd = spec[0] if isinstance(spec, tuple) else str(spec)
    label = spec[1] if isinstance(spec, tuple) else str(spec)
    m = re.search(r'(Normal|Tower(\d+))\s+x\s+(\d+).*?=\s*(\d+)', label)
    if not m:
        continue
    arch = 'Normal' if m.group(1) == 'Normal' else f'Tower-{m.group(2)}'
    if arch == 'Normal' and 'DW=True' in cmd:
        arch = 'Plain-DW'  # plain net with doubled inner width (same label as plain, distinguished by cmd)
    n_blocks = int(m.group(3))
    depth = int(m.group(4))
    acc = r.hist['val_A1'].max() * 100
    tr_acc = r.hist['A1'].max() * 100
    # Build network to count parameters
    n_params = None
    try:
        m_net = re.search(r"--net\s+'([^']+)'", cmd)
        m_A = re.search(r'\s-A\s+(\d+)', cmd)
        m_W = re.search(r'\s-W\s+(\d+)', cmd)
        if m_net:
            _A = m_A.group(1) if m_A else '2'
            _W = m_W.group(1) if m_W else '2'
            _cmd = f"--net '{m_net.group(1)}' --method ST -A {_A} -W {_W}"
            _o = o_from_str(_cmd)
            setup_o(_o)
            _o.num_classes = 100
            with contextlib.redirect_stdout(io.StringIO()):
                _net = create_net(_o)
            n_params = sum(p.numel() for p in _net.parameters())
            del _net
    except Exception as e:
        print(f'Warning: could not count params for {label}: {e}')
    arch_data.setdefault(arch, []).append((depth, acc, tr_acc, n_blocks, n_params))

# Print summary table organized by family
print(f"\n{'Family':<12} {'Name':<18} {'Depth':>8} {'Val Acc (%)':>12} {'Params':>10}")
print('-' * 64)
for arch in ['Normal', 'Plain-DW', 'Tower-2', 'Tower-4', 'Tower-8']:
    if arch not in arch_data:
        continue
    for depth, acc, tr_acc, nb, n_params in sorted(arch_data[arch]):
        if arch.startswith('Tower'):
            name = f'Tower{arch.split("-")[1]} x{nb}'
        elif arch == 'Plain-DW':
            name = f'Plain-DW x{nb}'
        else:
            name = f'Plain x{nb}'
        params_str = f'{n_params/1e6:.2f}M' if n_params is not None else 'N/A'
        print(f'{arch:<12} {name:<18} {depth:>8} {acc:>12.2f} {params_str:>10}')

figsize = (8, 5)
plt.figure(figsize=figsize)
import matplotlib.patheffects as pe
texts = []
lines = []
for arch in ['Normal', 'Plain-DW', 'Tower-2', 'Tower-4', 'Tower-8']:
    if arch not in arch_data:
        continue
    pts = sorted(arch_data[arch])
    x, y, y_tr, nb, n_p = zip(*pts)
    line, = plt.plot(x, y, marker='o', label=arch, linewidth=4, markersize=9,
                     path_effects=[pe.Stroke(linewidth=5, foreground='white'), pe.Normal()])
    plt.plot(x, y_tr, marker='o', linestyle='--', color=line.get_color(), label=None,markersize=4)
    if arch.startswith('Tower'):
        order = arch.split('-')[1]  # '2', '4', '8'
    for idx, (xi, yi, ni) in enumerate(zip(x, y, nb)):
        if arch.startswith('Tower'):
            txt = f'Tower{order} x{ni}' if idx == 0 else f'x{ni}'
        elif arch == 'Plain-DW':
            txt = f'Plain-DW x{ni}' if idx == 0 else f'x{ni}'
        else:
            txt = f'Plain x{ni}' if idx == 0 else f'x{ni}'
        t = plt.text(xi, yi, txt, fontsize=13, color=line.get_color(),
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                               edgecolor='none', alpha=0.8),
                     path_effects=[
                         pe.withSimplePatchShadow(offset=(1, -1), shadow_rgbFace='black', alpha=0.0),
                         pe.Normal(),
                     ])
        texts.append(t)

# avoid_x, avoid_y = [], []
# for line in lines:
#     xd = np.array(line.get_xdata(), dtype=float)
#     yd = np.array(line.get_ydata(), dtype=float)
#     for i in range(len(xd) - 1):
#         t = np.linspace(0, 1, 30)
#         avoid_x.extend(xd[i] + t * (xd[i + 1] - xd[i]))
#         avoid_y.extend(yd[i] + t * (yd[i + 1] - yd[i]))

# adjust_text(texts, x=avoid_x, y=avoid_y, expand=(1.1, 1.1))
adjust_text(texts, add_objects=lines, expand=(1.3, 1.3))

plt.xlabel('Total Network Depth')
plt.ylabel('Final Validation Accuracy (%)')
# plt.title('Final validation accuracy vs depth')
plt.grid(True, alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.legend()
plt.tight_layout()
path = '../tex/fig/scaling/scaling-nice.pdf'
force_path(path)
savefig(path)
plt.show()

# Accuracy vs parameters
plt.figure(figsize=figsize)
texts2 = []
lines2 = []
for arch in ['Normal', 'Plain-DW', 'Tower-2', 'Tower-4', 'Tower-8']:
    if arch not in arch_data:
        continue
    pts = sorted(arch_data[arch])
    pts_valid = [(d, a, ta, nb, np_) for d, a, ta, nb, np_ in pts if np_ is not None]
    print(f'{arch}: {len(pts)} points, {len(pts_valid)} with valid params')
    if not pts_valid:
        continue
    try:
        _, y, y_tr, nb, n_params_list = zip(*pts_valid)
        x_p = [p / 1e6 for p in n_params_list]
        if arch.startswith('Tower'):
            order = arch.split('-')[1]
        line2, = plt.plot(x_p, y, marker='o', label=arch, linewidth=4, markersize=9,
                          path_effects=[pe.Stroke(linewidth=5, foreground='white'), pe.Normal()])
        lines2.append(line2)
        plt.plot(x_p, y_tr, marker='o', linestyle='--', color=line2.get_color(), label=None, markersize=4)
        for idx, (xi, yi, ni) in enumerate(zip(x_p, y, nb)):
            if arch.startswith('Tower'):
                txt = f'Tower{order} x{ni}' if idx == 0 else f'x{ni}'
            elif arch == 'Plain-DW':
                txt = f'Plain-DW x{ni}' if idx == 0 else f'x{ni}'
            else:
                txt = f'Plain x{ni}' if idx == 0 else f'x{ni}'
            t = plt.text(xi, yi, txt, fontsize=13, color=line2.get_color(),
                         bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                   edgecolor='none', alpha=0.8),
                         path_effects=[
                             pe.withSimplePatchShadow(offset=(1, -1), shadow_rgbFace='black', alpha=0.0),
                             pe.Normal(),
                         ])
            texts2.append(t)
    except Exception as e:
        print(f'Error plotting {arch}: {e}')
        import traceback; traceback.print_exc()
adjust_text(texts2, add_objects=lines2, expand=(1.3, 1.3))
plt.xlabel('Parameters (M)')
plt.ylabel('Final Validation Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
path2 = '../tex/fig/scaling/scaling-params.pdf'
force_path(path2)
savefig(path2)
plt.show()
# %%
#%%
ll = []

# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=[0,0,0,0])' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 3 + 2 = 6")]
# # perhaps recompute with no extra layers after the last reduction
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "Normal x 6 + 3 = 9")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 9 + 2 = 12")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(3,3,3,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "Normal x 12 + 3 = 15")]
#ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(5,5,5,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 18 + 2 = 21")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(7,7,7,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 24 + 2 = 27")]
# tower networks
# T2, T3, T5
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 3 + 2 = 9")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "Tower2 x 6 + 3 = 15")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 9 + 2 = 21")]
#         
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=4,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower4 x 3 + 2 = 15")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=4,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "Tower4 x 6 + 3 = 27")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=8,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower8 x 3 + 2 = 27")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=8,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "Tower8 x 6 + 3 = 51")]

# refernce fp32
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 6 + 2 = 15")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32' --compile", "1 + Tower2 x 6 + 2 = 15 fp32")]
# BiNeal

# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet(m=1)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32' --compile", "BiNeal(m=1) fp32")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet(m=1)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25' --fp16 --compile", "BiNeal(m=1) fp16")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet(m=1)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32-FP1' --compile", "BiNeal(m=1) fp32")]
# no difference...
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet(m=1.5)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32' --compile", "BiNeal(m=1.5) fp32")]
#

ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8"]

rr = plot(ll, loc=4);
# plot(ll, title="", experiment="Quant-GCPR25/exp/scaling", loc=2)

# %%
# Tower vs Fusion
ll = []
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Fusion8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Fusion8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
# >>
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16", "sparse forward d, d//2")]
# >>
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'rec'"]
# >>
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'fw'"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'fw-alpha'"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'fw-alpha1'"] # 1/(i-j) init weight
rr = plot(ll, loc=4, title="");
val_summary(rr)
# %%
gr_hist(rr[-3])


# %%
# classifier variants
ll = []
# >>
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16","c3")]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 --v 'c5'"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 --v 'c5-relu'"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 --v 'c5-relu2'"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 --v 'c5-relu1'"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 --v 'c5-max'"]
rr = plot(ll, loc=4, title="");
val_summary(rr)
#


# %% res 512
ll= []
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
rr = plot(ll, loc=4, title='gr');

# %%
gr_hist(rr[-1])


# %% SE architecture
ll = []
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16","c3")]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'rec'"]
rr = plot(ll, loc=4, title='');
# %%

# %% SE architecture
ll = []
ll += ["--batch_size 256 --data 'imagenet-10(set=LA)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
ll += ["--batch_size 256 --data 'imagenet-10(set=LA)' --MD --net 'QResNet18(gate=Fusion8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
ll += ["--batch_size 256 --data 'imagenet-10(set=LA)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
rr = plot(ll, loc=4, title='');


# %% SE architecture
ll = []
ll += ["--batch_size 256 --data 'imagenet-100(set=LA)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
ll += ["--batch_size 256 --data 'imagenet-100(set=LA)' --MD --net 'QResNet18(gate=Fusion8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
ll += ["--batch_size 256 --data 'imagenet-100(set=LA)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
rr = plot(ll, loc=4, title='');

# %%

#
ll = []
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 6")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 9")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(3,3,3,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Normal x 12")]
# tower networks
# T2, T3, T5
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 3")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 6")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 9")]
#         
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=4,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower4 x 3")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=4,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower4 x 6")]
#
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=8,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower8 x 3")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=8,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower8 x 6")]

# rr = plot(ll);
plot(ll, title="", experiment="Quant-GCPR25/exp/scaling-slide", loc="center left", acc=False, bbox_to_anchor = (1,0.5));
# %%


#
ll = []
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "Plain-8")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "Plain-12")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(3,3,3,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "Plain-14")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(5,5,5,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "Plain-21")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(7,7,7,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "Plain-27")]
# tower networks
# T2, T3, T5
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 3")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 6")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower2 x 9")]
#         
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=4,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower4 x 3")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=4,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower4 x 6")]
#
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=8,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower8 x 3")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=8,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile", "1 + Tower8 x 6")]

# rr = plot(ll);
plot(ll, title="", experiment="Quant-GCPR25/exp/scaling-plain", loc=0, acc=False);
# %%
