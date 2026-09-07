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
import relimport

import os

from adjustText import adjust_text

from ..utilities.drawing import *
from ..train import *

def load_hist(l):
    res = dotdict()
    o = o_from_str(l)
    r = find_root(o)
    print(r)
    rf = r + 'out/hist.pkl'
    res.o = o
    try:
        res.hist = load_object(rf)
    except FileNotFoundError:
        l = l + ' --pathversion 4'
        o = o_from_str(l)
        r = find_root(o)
        print(r)
        rf = r + 'out/hist.pkl'
        res.hist = load_object(rf)
    
    return res,o


def load_test_results(l, test_variant = 'final-test.pkl'):
    o = o_from_str(l)
    r = find_root(o)
    rf = r + test_variant
    try:
        test_r = load_object(rf)
    except FileNotFoundError:
        l = l + ' --pathversion 4'
        o = o_from_str(l)
        r = find_root(o)
        # print(r)
        rf = r + test_variant
        test_r = load_object(rf)
    return test_r

def get_xy(res, what, method):
    if 'epoch_' in what:
        if '_A1' in what:
            y = res.hist.A1
        else: #"_L1"
            y = res.hist.L1
        x = res.hist.epoch
    else: # 'val_'
        if what == 'val_L1':
            if method in ['Mean', 'ReLU', 'Clamp']: # method is real
                y = res.hist.v_real_L
            else:
                y = res.hist.v_det_L # method is det
        else:
            y = res.hist[what]
        x = res.hist.val_epoch
    return x,y

def sort_dicts_by_commonality(dict_list):
    
    from collections import Counter
    # Step 1: Count key occurrences
    key_counter = Counter()
    for d in dict_list:
        # print(d.keys())
        key_counter.update(d.keys())
    # Step 2: Sort the keys in each dict by frequency (descending)
    print(key_counter)
    def sort_dict_by_key_frequency(d, freq_map):
        sorted_keys = sorted(d.keys(), key=lambda k: (-freq_map[k], k))
        # print(sorted_keys)
        return {k: d[k] for k in sorted_keys}
    # Step 3: Apply to each dict
    sorted_dicts = [sort_dict_by_key_frequency(d, key_counter) for d in dict_list]
    return sorted_dicts


#: Root directory the experiment scripts write their figures into.  A figure destination is a
#: *name* (``"scaling"``, ``"width"``), not a path, so no script names a directory outside the
#: repository.  Set $TNET_FIG_DIR to send the figures somewhere else -- e.g. straight into a
#: paper's figure directory -- without editing any of them.
FIG_ROOT = os.environ.get('TNET_FIG_DIR', os.path.join('res', 'figures'))


def fig_path(experiment, name):
    """Where a figure called *name* belonging to *experiment* is written."""
    return os.path.join(FIG_ROOT, experiment, name + '.pdf')


def plot(lln, title=None, experiment=None, loc=0, Abottom=40, acc=True, bbox_to_anchor = None):
    unimportant = {'args_str', 'data_seed', 'test', 'unlock', 'seed', 'compile', 'cudagraph', 'pathversion'} #,'variant'}
    important = {'method' ,'lr'}
    rr = []
    oo = []
    od_default = dict(o_from_str('')) # default
    for k in important:
        od_default.pop(k, None) # remove important from default

    ll = []
    nn = []
    for i,x in enumerate(lln):
        if isinstance(x, tuple):
            ll.append(x[0])
            nn.append(x[1])
        else:
            ll.append(x)
            nn.append(None)
            

    # if isinstance(lln[0], tuple):
    #     ll = [l[0] for l in lln]
    #     nn = [l[1] for l in lln]
    # else:
    #     ll = lln
    #     nn =[None for k in lln]

    for (i,l) in enumerate(ll):
        res, o = load_hist(l)
        rr += [res]
        od = dict(o)
        for k in unimportant:
            if k in od.keys():
                del od[k]
        oo += [set(od.items())]

    common = set.intersection(*oo)
    print(f'common options: ', common)

    # compute common non-default
    common_nondef = common - set(dict(od_default).items())
    print(f'common non-default: ', common_nondef)

    uniqs = []
    for i in range(len(rr)):
                res = rr[i]
                o = oo[i]
                # remove common and default
                # unique = o - common - set(dict(od_default).items())
                # remove common
                unique = o - common
                unique = dict(unique)
                # remove unimportant
                unique = {k: v for k, v in unique.items() if k not in unimportant}
                uniqs.append(unique)

    uniqs = sort_dicts_by_commonality(uniqs)

    # for what in ['epoch_L1', 'epoch_A1', 'v_det_A', 'tr_det_A']:
    for what in ['epoch_L1', 'epoch_A1', 'tr_det_A', 'val_A1']:
    # for what in ['epoch_L1', 'epoch_A1', 'tr_det_A', 'v_det_A']:
    # for what in ['L1', 'A1']:
        # plt.figure(figsize=(16,10))
        if experiment is None:
            plt.figure(figsize=(10,7))
        else:
            plt.figure(figsize=(8,5))
        # for p_what in ['epoch_', 'val_']:
        a = plt.gca()
        for i in range(len(rr)):
            res = rr[i]
            o = oo[i]
            unique = uniqs[i]
            o = rr[i].o
            x,y = get_xy(res, what, o.method)
            if 'epoch_' in what:
                st = '-'
            else:
                st = '-'
            if 'epoch_' in what:
                markevery = np.arange(9,len(x),10)
                marker_indices = np.arange(9, len(x), 10)  # every 10 points
            else:
                markevery = 1
                marker_indices = np.arange(0, len(x), 1)  # every 1 point
                
            if '_L1' in what:
                y_units = 1
            else:
                y_units = 100

            label = ' '
            if nn[i] is not None:
                label = nn[i]
            else:
                for (k,v) in unique.items():
                    # print(k, end=' ')
                    if k=='winit' and v != '':
                        label += f'{k} '
                    else:
                        if isinstance(v,bool):
                            if v is True:
                                label += f'{k} '
                        elif v != '' and v is not None:
                            if type(v) == str:
                                if 'QResNet18' in v:
                                    n, dv = parse_expression(v)
                                    for k1,v1 in dv.items():
                                        if type(v1) == str and v1 not in ['True', 'False','None']:
                                            label += f'{v1} ' # abbreviate keys
                                        else:
                                            label += f'{k1}={v1} ' # abbreviate keys
                                else:
                                    label += f'{v} '
                            else:
                                label += f'{k}={v} '
                
            if len(rr) > 3 and i < len(rr) - 3 and experiment is None:
                lw = 0.5
                ms = 2
            else:
                lw = 2
                ms = 2
            # if st == '-':
            # a.plot(x,y*100,st, color = cc[i], marker= markers[i], label=label,linewidth=lw, markevery=markevery, markersize=ms)
            a.plot(x[0:2],y[0:2]*y_units,st, color = cc[i], marker= markers[i], label=label,linewidth=lw, markevery=100, markersize=ms*4)
            a.plot(x, y[0:len(x)]*y_units, st, color = cc[i], label=None,linewidth=lw)
            z_marker = np.random.randint(1, 5, size=len(marker_indices))
            for z in np.unique(z_marker):
                mask = z_marker == z
                # plt.scatter(x[mask], y[mask], zorder=z)
                a.scatter(x[marker_indices[mask]], y[marker_indices[mask]]*y_units, marker= markers[i], s=ms*16, color=cc[i], zorder=z, label=None)  # suppress duplicate legend

            if acc:
                a.text(x[-1] + 3, y[-1]*y_units, f"{y[-1]*y_units:3.2f}" if '_L1' in what else f"{y[-1]*y_units:3.1f}", 
                ha='left', va='bottom',  # anchor bottom-left of text to (x+0.1, y+0.1)
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgray'), 
                zorder=50,         # ensures it's drawn on top
                clip_on=False      # prevents it from being clipped at axes boundaries
                )


            # a.annotate(
            #     "Label",
            #     xy=(x[-1], y[-1]*y_units), xytext=(x[-1]+50, y[-1]*y_units+0.5),
            #     textcoords='data',
            #     ha='left', va='bottom',
            #     bbox=dict(boxstyle="round", fc="w", ec="k"),
            #     arrowprops=dict(arrowstyle="->", lw=0.5),
            #     zorder=1000,
            #     clip_on=False
            # )


        if not '_L1' in what:
            a.set_ylim(bottom=Abottom)
            if 'val_' in what:
                plt.ylabel('Validation Accuracy')
            else:
                plt.ylabel('Training Accuracy')
            a.legend(loc=loc, bbox_to_anchor = bbox_to_anchor)
        else:
            if 'val_' in what:
                plt.ylabel('Validation Loss')
            else:
                plt.ylabel('Training Loss')
            a.set_yscale('log')
            # a.set_xscale('log')
            # plt.ylim(0.3, 4)

        # plt.xlim(0,210)
        # plt.ylim(0.5,0.85)
        a.grid()
        if title is None:
            title = "".join(f"{k}={v}, " for k,v in dict(common_nondef).items())
            if len(title)>2:
                title = title[:-2]
        if title != "":
            if experiment is None:
                plt.title(title + f" ({what})")
            else:
                plt.title(title)
        plt.xlabel('epochs')
        plt.draw()
        if experiment is not None:
            path = fig_path(experiment, what)
            force_path(path)
            savefig(path)
        plt.show()
    return rr
    # %%

def gr_hist(rec, class_name = 'QConv2d', param_name = 'weight' ):
    gr = rec.hist.gr
    plt.figure(figsize=(14,14))
    L  = 0
    texts = []

    layer_i = 0
    for (id, d) in gr[0].items():
        if d.class_name != class_name:
            continue
        if d.param_name == param_name or d.param_name in param_name:
            layer_i += 1
            gg = []
            n = len(gr)
            # K = d.K
            n_in = np.prod(np.array(d.shape[1:]))
            for e in range(n):
                g = gr[e][id].grad_sq**0.5 #* (K**2-1)**0.5
                gg.append(g)
            c = d.module_name.count(".residual")
            if c == 0:
                L += 1
            plt.plot(gg, linewidth=1, color = cc[layer_i], marker = markers[layer_i], label = str(L-c))
            texts.append(plt.text(n-1, gg[-1], str(L-c),fontsize=12,ha='left', va='center'))
            # texts.append(plt.text(0, gg[0], str(L-c),fontsize=12, ha='right', va='center'))
            print(str(L-c) + ' | ' + d.module_name + ' | ' + d.param_name + ' ' + str(d.shape))
    plt.yscale('log')
    # plt.xscale('log')
    plt.grid(axis='x')
    plt.legend()
    # adjust_text(texts)
    # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', shrinkA=0, shrinkB=0, lw=1), 
    #             # only_move={'text': 'xy'},  # move horizontally only
    #             # force_text=0.5, # repel between labels
    #             # expand_points=(0, 0),  # no extra space around points
    #             # force_points=0.0, # pull label arrow to exact point
    #             # expand_text=(0, 0),    # no extra space around text
    #             );



def val_summary(rr):
    for (i,r) in enumerate(rr):
        Wb = int(np.log2(r.o.W))
        Ab = int(np.log2(r.o.A))
        dilation = 'Dilation' in r.o.args_str
        acc = f"{(r.hist['val_A1'].max())*100:3.1f}"
        running = len(r.hist['val_A1']) < 21
        # print(f'Dilation={dilation} Fb={Ab}, Wb={Wb}, Acc={acc}{"+" if running else ""}')
        print(f'{r.o.net}  {r.o.method} & {Ab} & {Wb} & {acc}{"+" if running else ""}')
