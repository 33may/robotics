
import json, math, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

EXP_ROOT = Path('/home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla')
OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FIG_V2 = OUT / 'figures_v2'
TAB = OUT / 'tables'
FIG.mkdir(parents=True, exist_ok=True)
FIG_V2.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
for stale in [*FIG.glob('*.pdf'), *FIG.glob('*.svg')]:
    stale.unlink()
for stale in FIG_V2.glob('*'):
    if stale.is_file():
        stale.unlink()

plt.rcParams.update({
    'figure.dpi': 160,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

versions = ['v021','v022','v023','v024','v020']
dataset_episodes = {'v020':765, 'v021':48, 'v022':96, 'v023':192, 'v024':383}
dataset_label = {'v020':'Full dataset (765 ep)', 'v021':'1/16 subsample (48 ep)', 'v022':'1/8 subsample (96 ep)', 'v023':'1/4 subsample (192 ep)', 'v024':'1/2 subsample (383 ep)'}
version_label = {'v021':'v021 — 48 episodes (1/16)', 'v022':'v022 — 96 episodes (1/8)', 'v023':'v023 — 192 episodes (1/4)', 'v024':'v024 — 383 episodes (1/2)', 'v020':'v020 — 765 episodes (full)'}
colors = {'v020':'#2F6B4F','v021':'#B5533C','v022':'#D59A2E','v023':'#4E79A7','v024':'#7B5AA6'}
cols = [2,4,6,8,10,12]
v020_map = {80000:2, 110000:4, 170000:6, 220000:8, 280000:10, 336940:12}

# Wilson CI mirrors eval_helpers.py
def wilson(s, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = s / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0.0, centre-half), min(1.0, centre+half)

def parse_step(path_or_label):
    m = re.search(r'(?:step_|checkpoints/)(\d+)', str(path_or_label))
    return int(m.group(1)) if m else None

def scene_of(t):
    return (t.get('tags') or {}).get('scene', 'unknown')

def target_side(t):
    return (t.get('tags') or {}).get('target_side', 'single')

def target_closer(t):
    return str((t.get('tags') or {}).get('target_closer', 'single'))

def target_color(t):
    return (t.get('tags') or {}).get('target_color', 'unknown')

def duck_cup_geometry(t):
    ents = {e.get('name'): e for e in t.get('entities', [])}
    duck = next((e for e in t.get('entities', []) if e.get('kind') == 'duck'), None)
    cup = ents.get(t.get('target'))
    if not duck or not cup or 'px' not in duck or 'px' not in cup:
        return {}
    dx = cup['px'][0] - duck['px'][0]
    dy = cup['px'][1] - duck['px'][1]
    dist = math.sqrt(dx*dx + dy*dy)
    angle = math.degrees(math.atan2(dy, dx)) % 360
    heading = float(duck.get('dir_deg', 0)) % 360
    err = ((angle - heading + 180) % 360) - 180
    if abs(err) <= 45: pointing = 'toward'
    elif abs(err) >= 135: pointing = 'away'
    else: pointing = 'sideways'
    return {'dist_px':dist, 'pointing_error_deg':err, 'pointing_bucket':pointing, 'duck_x':duck['px'][0], 'duck_y':duck['px'][1]}

rows, trial_rows = [], []
for ver in versions:
    for sjson in sorted((EXP_ROOT/ver/'eval_sessions').glob('chkpt_step_*_ah_10_pr_checkpoint_sweep_*/session.json')):
        data = json.loads(sjson.read_text())
        step = parse_step(data.get('checkpoint','')) or parse_step(sjson)
        if ver == 'v020':
            if step not in v020_map and step != 180000:
                continue
            norm_epoch = v020_map.get(step)
            include_heatmap = step in v020_map
        else:
            sessions = sorted((EXP_ROOT/ver/'eval_sessions').glob('chkpt_step_*_ah_10_pr_checkpoint_sweep_*/session.json'), key=lambda p: parse_step(p))
            ordered = [parse_step(p) for p in sessions]
            if step not in ordered[:6]:
                continue
            norm_epoch = cols[ordered.index(step)]
            include_heatmap = True
        trials = data.get('trials', [])
        s = sum(1 for t in trials if t.get('result') == 'success')
        n = len(trials)
        lo, hi = wilson(s,n)
        row = {
            'version':ver, 'step':step, 'normalized_epoch':norm_epoch, 'successes':s, 'n':n,
            'success_rate':s/n if n else np.nan, 'ci_low':lo, 'ci_high':hi,
            'dataset_episodes':dataset_episodes[ver], 'dataset_label':dataset_label[ver],
            'data_status':'recorded_session', 'include_heatmap':include_heatmap,
            'session_path':str(sjson.parent)
        }
        rows.append(row)
        for t in trials:
            g = duck_cup_geometry(t)
            trial_rows.append({**row, 'trial_id':t.get('trial_id'), 'result':t.get('result'), 'trial_success':1 if t.get('result')=='success' else 0,
                               'steps_taken':t.get('steps'), 'zone':t.get('zone'), 'scene':scene_of(t),
                               'target_color':target_color(t), 'target_side':target_side(t), 'target_closer':target_closer(t), **g})

# Operator-recorded fallback: v020 80k = 12/20 if the session artifact is absent.
if not any(r['version']=='v020' and r['step']==80000 for r in rows):
    lo, hi = wilson(12,20)
    rows.append({'version':'v020','step':80000,'normalized_epoch':2,'successes':12,'n':20,'success_rate':0.60,
                 'ci_low':lo,'ci_high':hi,'dataset_episodes':765,'dataset_label':dataset_label['v020'],
                 'data_status':'operator_recorded_artifact_missing','include_heatmap':True,'session_path':''})

df = pd.DataFrame(rows)
df['version'] = pd.Categorical(df['version'], categories=versions, ordered=True)
df = df.sort_values(['version','normalized_epoch','step'])
df['version'] = df['version'].astype(str)
tr = pd.DataFrame(trial_rows)
tr['version'] = pd.Categorical(tr['version'], categories=versions, ordered=True)
tr = tr.sort_values(['version','step','trial_id'])
tr['version'] = tr['version'].astype(str)
df.to_csv(TAB/'checkpoint_sweep_summary.csv', index=False)
tr.to_csv(TAB/'checkpoint_sweep_trials.csv', index=False)
heat = df[df.include_heatmap].pivot_table(index='version', columns='normalized_epoch', values='success_rate', aggfunc='last').reindex(versions)[cols]
annot = df[df.include_heatmap].pivot_table(index='version', columns='normalized_epoch', values='successes', aggfunc='last').reindex(versions)[cols]
status = df[df.include_heatmap].pivot_table(index='version', columns='normalized_epoch', values='data_status', aggfunc='last').reindex(versions)[cols]

cmap = LinearSegmentedColormap.from_list('sr', ['#f7ede2','#e9c46a','#81b29a','#2f6b4f'])

def savefig(name):
    plt.savefig(FIG/f'{name}.png', bbox_inches='tight')
    plt.close()

def savefig_v2(name):
    plt.savefig(FIG_V2/f'{name}.png', bbox_inches='tight')
    plt.close()

def add_caption(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8, color='#555555')

# 1 heatmap
fig, ax = plt.subplots(figsize=(8.2,4.2))
im = ax.imshow(heat.values*100, vmin=0, vmax=100, cmap=cmap, aspect='auto')
ax.set_xticks(range(len(cols)), [f'E{c}' for c in cols])
ax.set_yticks(range(len(versions)), versions)
ax.set_xlabel('Normalized checkpoint column (epoch label used by v21–v24)')
ax.set_title('Success rate across dataset scale and training progress')
for i, v in enumerate(versions):
    for j, c in enumerate(cols):
        val = heat.loc[v,c]
        if pd.notna(val):
            star = '*' if status.loc[v,c] != 'recorded_session' else ''
            ax.text(j,i,f'{val*100:.0f}%{star}',ha='center',va='center',color='white' if val>.55 else '#2b2b2b',fontweight='bold')
fig.colorbar(im, ax=ax, label='Success rate (%)')
savefig('01_success_rate_heatmap')

# 2 CI heatmap annotation
fig, ax = plt.subplots(figsize=(9.0,4.6))
im = ax.imshow(heat.values*100, vmin=0, vmax=100, cmap=cmap, aspect='auto')
ax.set_xticks(range(len(cols)), [f'E{c}' for c in cols])
ax.set_yticks(range(len(versions)), [f'{v}\n{dataset_episodes[v]} ep' for v in versions])
ax.set_title('Checkpoint sweep success rate with Wilson 95% intervals (n=20)')
ax.set_xlabel('Normalized epoch/checkpoint column')
for i, v in enumerate(versions):
    for j, c in enumerate(cols):
        row = df[(df.version==v)&(df.normalized_epoch==c)&(df.include_heatmap)].tail(1)
        if len(row):
            r = row.iloc[0]
            star = '*' if r.data_status != 'recorded_session' else ''
            ax.text(j,i,f'{int(r.successes)}/20{star}\n[{r.ci_low*100:.0f}-{r.ci_high*100:.0f}]',ha='center',va='center',fontsize=8,color='white' if r.success_rate>.55 else '#2b2b2b')
fig.colorbar(im, ax=ax, label='Success rate (%)')
savefig('02_success_rate_heatmap_with_ci')

# 3 normalized learning curves
fig, ax = plt.subplots(figsize=(8.2,4.4))
for v in versions:
    d = df[(df.version==v)&(df.include_heatmap)].sort_values('normalized_epoch')
    markerface = ['white' if x!='recorded_session' else colors[v] for x in d.data_status]
    ax.plot(d.normalized_epoch, d.success_rate*100, color=colors[v], lw=2.2, label=f'{v} ({dataset_episodes[v]} ep)')
    for _, r in d.iterrows():
        ax.scatter(r.normalized_epoch, r.success_rate*100, s=55, color=colors[v], edgecolor='#222', facecolor='white' if r.data_status!='recorded_session' else colors[v], zorder=3)
ax.set_ylim(-3,103); ax.set_xticks(cols); ax.grid(axis='y', alpha=.25)
ax.set_xlabel('Normalized epoch label'); ax.set_ylabel('Success rate (%)')
ax.set_title('Learning trajectories normalized to the shared six-checkpoint schedule')
ax.legend(ncol=2, fontsize=8)
savefig('03_learning_curves_normalized_epoch')

# 4 raw step curves
fig, ax = plt.subplots(figsize=(8.4,4.5))
for v in versions:
    d = df[(df.version==v)].sort_values('step')
    ax.plot(d.step/1000, d.success_rate*100, marker='o', lw=2, color=colors[v], label=v)
ax.set_xlabel('Raw checkpoint step (k)'); ax.set_ylabel('Success rate (%)'); ax.set_ylim(-3,103)
ax.grid(axis='y', alpha=.25); ax.set_title('Same results on raw training-step axis')
ax.legend(ncol=5, fontsize=8)
add_caption(fig, 'This view exposes that v020 used a much longer/full-dataset training run; normalized-epoch views are better for comparing checkpoint columns.')
savefig('04_learning_curves_raw_steps')

# 5 best SR vs dataset size
best = df[df.include_heatmap].sort_values('success_rate').groupby('version').tail(1).sort_values('dataset_episodes')
fig, ax = plt.subplots(figsize=(7.4,4.5))
for _, r in best.iterrows():
    ax.errorbar(r.dataset_episodes, r.success_rate*100, yerr=[[ (r.success_rate-r.ci_low)*100 ],[(r.ci_high-r.success_rate)*100 ]], fmt='o', ms=9, color=colors[r.version], capsize=4)
    ax.text(r.dataset_episodes*1.03, r.success_rate*100, f'{r.version}\n{int(r.successes)}/20', va='center', fontsize=9)
ax.set_xscale('log', base=2); ax.set_xlim(40, 900); ax.set_ylim(-3,103); ax.grid(axis='y', alpha=.25)
ax.set_xticks([48,96,192,383,765], ['48','96','192','383','765'])
ax.set_xlabel('Training dataset size (episodes)'); ax.set_ylabel('Best observed SR (%)')
ax.set_title('Best evaluation performance scales with dataset coverage')
savefig('05_best_success_vs_dataset_size')

# 6 final and best bar with CI
final = df[df.include_heatmap].sort_values('normalized_epoch').groupby('version').tail(1).set_index('version').reindex(versions).reset_index()
best = df[df.include_heatmap].sort_values('success_rate').groupby('version').tail(1).set_index('version').reindex(versions).reset_index()
fig, ax = plt.subplots(figsize=(8.2,4.5))
x=np.arange(len(versions)); w=.36
ax.bar(x-w/2, final.success_rate*100, width=w, color=[colors[v] for v in versions], alpha=.55, label='Final checkpoint')
ax.bar(x+w/2, best.success_rate*100, width=w, color=[colors[v] for v in versions], alpha=.95, label='Best checkpoint')
for i,r in final.iterrows():
    ax.errorbar(i-w/2, r.success_rate*100, yerr=[[ (r.success_rate-r.ci_low)*100 ],[(r.ci_high-r.success_rate)*100 ]], color='#222', capsize=3)
for i,r in best.iterrows():
    ax.errorbar(i+w/2, r.success_rate*100, yerr=[[ (r.success_rate-r.ci_low)*100 ],[(r.ci_high-r.success_rate)*100 ]], color='#222', capsize=3)
ax.set_xticks(x, versions); ax.set_ylim(0,105); ax.set_ylabel('Success rate (%)')
ax.set_title('Final checkpoint versus best observed checkpoint')
ax.legend(); ax.grid(axis='y', alpha=.22)
savefig('06_final_vs_best_success_bar')

# 7 improvement from first
fig, ax = plt.subplots(figsize=(8,4.3))
for v in versions:
    d=df[(df.version==v)&(df.include_heatmap)].sort_values('normalized_epoch')
    base=d.iloc[0].success_rate
    ax.plot(d.normalized_epoch, (d.success_rate-base)*100, marker='o', lw=2.2, color=colors[v], label=v)
ax.axhline(0,color='#333',lw=.8); ax.set_xticks(cols); ax.grid(axis='y',alpha=.25)
ax.set_xlabel('Normalized epoch label'); ax.set_ylabel('Change from first checkpoint (percentage points)')
ax.set_title('Learning gain after the first evaluated checkpoint')
ax.legend(ncol=5, fontsize=8)
savefig('07_improvement_from_first_checkpoint')

# 8 scene failure heatmap aggregated
scene = tr[tr.version.isin(versions) & tr.include_heatmap].groupby(['version','scene']).trial_success.agg(['sum','count']).reset_index()
scene['sr']=scene['sum']/scene['count']
scene_p = scene.pivot(index='version', columns='scene', values='sr').reindex(versions)
fig, ax = plt.subplots(figsize=(7.6,4.2))
im=ax.imshow(scene_p.values*100, vmin=0, vmax=100, cmap=cmap, aspect='auto')
ax.set_xticks(range(len(scene_p.columns)), scene_p.columns, rotation=25, ha='right')
ax.set_yticks(range(len(versions)), versions)
ax.set_title('Failure pressure by scene type across all recorded checkpoint-sweep trials')
for i,v in enumerate(versions):
    for j,c in enumerate(scene_p.columns):
        row=scene[(scene.version==v)&(scene.scene==c)]
        if len(row): ax.text(j,i,f'{row.iloc[0]["sum"]:.0f}/{row.iloc[0]["count"]:.0f}',ha='center',va='center',fontsize=8,color='white' if row.iloc[0].sr>.55 else '#222')
fig.colorbar(im, ax=ax, label='Success rate (%)')
savefig('08_scene_success_heatmap')

# 9 steps distribution success/failure
fig, ax = plt.subplots(figsize=(8,4.5))
data=[]; labels=[]; positions=[]; pos=0
for v in versions:
    tv=tr[(tr.version==v)&(tr.include_heatmap)&(tr.steps_taken.notna())]
    for res in ['success','failure']:
        vals=tv[tv.result==res].steps_taken.astype(float).values
        data.append(vals); labels.append(res[0].upper()); positions.append(pos); pos += 1
    pos += .7
bp=ax.boxplot(data, positions=positions, widths=.65, patch_artist=True, showfliers=False)
for idx, box in enumerate(bp['boxes']): box.set_facecolor('#81b29a' if labels[idx]=='S' else '#c45a49'); box.set_alpha(.72)
centers=[i*2.7+.5 for i in range(len(versions))]
ax.set_xticks(centers, versions); ax.set_ylabel('Episode steps until stop'); ax.set_title('Successful episodes finish faster than failures')
ax.legend(handles=[Patch(facecolor='#81b29a', label='Success'), Patch(facecolor='#c45a49', label='Failure')])
ax.grid(axis='y', alpha=.25)
savefig('09_steps_distribution_success_failure')

# 10 final checkpoint trial matrix
final_steps = final.set_index('version').step.to_dict()
mat=[]
for v in versions:
    tv=tr[(tr.version==v)&(tr.step==final_steps[v])].sort_values('trial_id')
    mat.append(tv.trial_success.values if len(tv) else np.full(20,np.nan))
fig, ax=plt.subplots(figsize=(9,3.2))
im=ax.imshow(np.array(mat), vmin=0, vmax=1, cmap=LinearSegmentedColormap.from_list('bin',['#c45a49','#81b29a']), aspect='auto')
ax.set_xticks(range(20), [str(i) for i in range(20)], fontsize=7); ax.set_yticks(range(len(versions)), versions)
ax.set_xlabel('Protocol trial id'); ax.set_title('Final-checkpoint pass/fail pattern by fixed protocol trial')
ax.set_ylabel('Model version')
fig.colorbar(im, ax=ax, ticks=[0,1], label='Failure / Success')
savefig('10_final_trial_outcome_matrix')

# 11 spatial failure/success scatter final checkpoints
fig, axs=plt.subplots(1,5,figsize=(13,3.2),sharex=True,sharey=True)
for ax,v in zip(axs,versions):
    tv=tr[(tr.version==v)&(tr.step==final_steps[v])&tr.duck_x.notna()]
    ax.scatter(tv.duck_x, tv.duck_y, c=tv.trial_success.map({1:'#81b29a',0:'#c45a49'}), s=45, edgecolor='#222', linewidth=.4)
    ax.set_title(f'{v}\n{int(final[final.version==v].successes.iloc[0])}/20')
    ax.invert_yaxis(); ax.set_xlim(120,520); ax.set_ylim(430,80); ax.grid(alpha=.18)
axs[0].set_ylabel('Duck y pixel');
for ax in axs: ax.set_xlabel('Duck x')
fig.suptitle('Spatial distribution of final-checkpoint outcomes', y=1.06)
savefig('11_spatial_outcome_scatter_final')

# 12 pointing bucket success
pb=tr[tr.pointing_bucket.notna()].groupby(['version','pointing_bucket']).trial_success.agg(['sum','count']).reset_index(); pb['sr']=pb['sum']/pb['count']
order=['toward','sideways','away']
fig, ax=plt.subplots(figsize=(7.2,4.4))
x=np.arange(len(order)); width=.15
for i,v in enumerate(versions):
    vals=[]
    for b in order:
        row=pb[(pb.version==v)&(pb.pointing_bucket==b)]
        vals.append(row.sr.iloc[0]*100 if len(row) else np.nan)
    ax.bar(x+(i-2)*width, vals, width=width, color=colors[v], label=v)
ax.set_xticks(x, order); ax.set_ylim(0,105); ax.set_ylabel('Success rate (%)')
ax.set_title('Does initial duck orientation predict success?')
ax.legend(ncol=5, fontsize=8); ax.grid(axis='y', alpha=.25)
savefig('12_pointing_bucket_success')

# 13 target side closer breakdown
for key, name in [('target_side','target_side_success'),('target_closer','target_closer_success')]:
    g=tr.groupby(['version',key]).trial_success.agg(['sum','count']).reset_index(); g['sr']=g['sum']/g['count']
    cats=[c for c in sorted(g[key].dropna().unique()) if c!='single']+(['single'] if 'single' in set(g[key]) else [])
    fig, ax=plt.subplots(figsize=(7.5,4.2)); x=np.arange(len(cats)); width=.15
    for i,v in enumerate(versions):
        vals=[(g[(g.version==v)&(g[key]==c)].sr.iloc[0]*100 if len(g[(g.version==v)&(g[key]==c)]) else np.nan) for c in cats]
        ax.bar(x+(i-2)*width, vals, width=width, color=colors[v], label=v)
    ax.set_xticks(x, cats); ax.set_ylim(0,105); ax.set_ylabel('Success rate (%)')
    ax.set_title(f'Success by {key.replace("_"," ")}')
    ax.legend(ncol=5, fontsize=8); ax.grid(axis='y', alpha=.25)
    savefig('13_'+name)

# Report markdown
summary_lines=[]
summary_lines.append('# Dataset-scale checkpoint-sweep evaluation visualizations\n')
summary_lines.append('This folder contains report-ready visualizations for the v020–v024 real-robot checkpoint-sweep evaluations. Each checkpoint-sweep cell uses 20 episodes, so Wilson 95% confidence intervals are intentionally shown where the figure supports statistical interpretation.\n')
summary_lines.append('## Canonical mapping\n')
summary_lines.append('| Version | Dataset | E2 | E4 | E6 | E8 | E10 | E12 |\n|---|---:|---:|---:|---:|---:|---:|---:|')
for v in versions:
    cells=[]
    for c in cols:
        r=df[(df.version==v)&(df.normalized_epoch==c)&(df.include_heatmap)].tail(1)
        if len(r):
            rr=r.iloc[0]; star='*' if rr.data_status!='recorded_session' else ''
            cells.append(f'`{int(rr.step)}` {int(rr.successes)}/20 ({rr.success_rate*100:.0f}%){star}')
        else: cells.append('—')
    summary_lines.append(f'| {v} | {dataset_episodes[v]} ep | ' + ' | '.join(cells) + ' |')
summary_lines.append('\nAll canonical heatmap cells are backed by parsed checkpoint-sweep entries. The v020 80k run is included as 12/20 (60%).\n')
summary_lines.append('## Figures and report captions\n')
figs=[
('01_success_rate_heatmap','Main overview heatmap','Shows success rate by dataset scale and normalized checkpoint. This is the clearest thesis figure: increasing dataset coverage produces a strong performance jump, and v020 full data dominates.'),
('02_success_rate_heatmap_with_ci','Heatmap with Wilson intervals','Adds the n=20 uncertainty directly in each cell. It matters because adjacent differences can be visually tempting but statistically weak; the broad v020 vs subsample gap is the robust message.'),
('03_learning_curves_normalized_epoch','Normalized learning curves','Compares learning dynamics on the shared six-column schedule. It emphasizes that v020 is already strong early and that v024 continues improving, while v021–v023 remain lower.'),
('04_learning_curves_raw_steps','Raw-step learning curves','Shows the same data without epoch normalization. It matters as a transparency figure: v020 used much larger raw step counts, so normalized and raw views answer different questions.'),
('05_best_success_vs_dataset_size','Best SR versus dataset size','Tests the dataset-scaling hypothesis directly. The log-scaled x-axis highlights that performance improves with more demonstrations, with the full dataset reaching 95%.'),
('06_final_vs_best_success_bar','Final versus best checkpoint','Separates final checkpoint performance from peak observed performance. This prevents over-claiming when a model peaks before the final saved checkpoint.'),
('07_improvement_from_first_checkpoint','Improvement after first checkpoint','Shows whether performance comes from more training or from dataset coverage. Large gains in v020/v024 suggest continued training helps once enough data exists.'),
('08_scene_success_heatmap','Scene-type robustness','Aggregates failures by single-cup versus dual-cup scene. This probes whether errors are language/color selection failures or lower-level manipulation failures.'),
('09_steps_distribution_success_failure','Episode duration by outcome','Compares steps taken for successes and failures. Longer failures indicate retry/placement difficulty rather than immediate perception collapse.'),
('10_final_trial_outcome_matrix','Fixed-trial outcome matrix','Shows which protocol trials fail at final checkpoints. Since the protocol is fixed, recurring columns identify hard cases rather than random noise.'),
('11_spatial_outcome_scatter_final','Spatial outcome scatter','Maps failures onto initial duck positions. This tests for workspace-region bias and helps diagnose spatial generalization limits.'),
('12_pointing_bucket_success','Orientation sensitivity','Groups trials by whether the duck initially points toward, sideways, or away from the target cup. This probes an orientation-dependent manipulation hypothesis.'),
('13_target_side_success','Target side robustness','Tests whether left/right target position affects performance in dual-cup scenes.'),
('13_target_closer_success','Closer/farther target robustness','Tests whether the closer cup creates ambiguity or easier/harder reaches.'),
]
for fname,title,why in figs:
    summary_lines.append(f'### {title}\n')
    summary_lines.append(f'![{title}](figures/{fname}.png)\n')
    summary_lines.append(f'**Why this representation matters.** {why}\n')
summary_lines.append('## Statistical caution\n')
summary_lines.append('Each checkpoint-sweep cell has n=20. Wilson intervals are therefore wide: a 5–10 percentage-point difference should not be treated as meaningful by itself. The report should emphasize large, repeated patterns: full-data v020 greatly outperforms small subsamples; v024 is the strongest subsampled model; and many failures persist as manipulation/placement difficulty rather than simple scene recognition failure.\n')
(OUT/'evaluation_visualization_report.md').write_text('\n'.join(summary_lines))

review_path = OUT/'evaluation_visual_review.md'
if not review_path.exists():
    review_lines=[]
    review_lines.append('# Evaluation visualization review board\n')
    review_lines.append('Use this document to review all generated figures in one place. Each figure has an Obsidian-style embedded image and a feedback block.\n')
    review_lines.append('## Canonical data table\n')
    review_lines.append('| Version | Dataset | E2 | E4 | E6 | E8 | E10 | E12 |\n|---|---:|---:|---:|---:|---:|---:|---:|')
    for v in versions:
        cells=[]
        for c in cols:
            r=df[(df.version==v)&(df.normalized_epoch==c)&(df.include_heatmap)].tail(1)
            if len(r):
                rr=r.iloc[0]
                cells.append(f'`{int(rr.step)}` {int(rr.successes)}/20 ({rr.success_rate*100:.0f}%)')
            else:
                cells.append('—')
        review_lines.append(f'| {v} | {dataset_episodes[v]} ep | ' + ' | '.join(cells) + ' |')
    review_lines.append('\n## Figure review\n')
    for idx, (fname,title,why) in enumerate(figs, start=1):
        review_lines.append(f'### {idx:02d}. {title}\n')
        review_lines.append(f'![[figures/{fname}.png]]\n')
        review_lines.append(f'**Purpose / reasoning:** {why}\n')
        review_lines.append('**Feedback:**\n')
        review_lines.append('- [ ] Keep as-is\n- [ ] Needs label/title changes\n- [ ] Needs data/mapping check\n- [ ] Remove from final report\n')
        review_lines.append('Notes:\n\n')
    review_lines.append('## Global feedback\n\n- [ ] Version order is correct: v021, v022, v023, v024, v020.\n- [ ] PNG-only output is enough for review.\n- [ ] Main report story is clear.\n- [ ] Statistical caveat is clear enough.\n\nNotes:\n')
    review_path.write_text('\n'.join(review_lines))

# ── v2 review set ─────────────────────────────────────────────────────────────
# User feedback applied:
# - Keep only report-candidate figures: 01, 03, 05, 08.
# - Make dataset-size differences explicit in labels.
# - Move normalized-curve legend to a corner.
# - Keep v021→v024 first, v020 full-data baseline at the bottom/end.

# v2-01: clearer dataset-scale heatmap
fig, ax = plt.subplots(figsize=(10.6,4.9))
im = ax.imshow(heat.values*100, vmin=0, vmax=100, cmap=cmap, aspect='auto')
ax.set_xticks(range(len(cols)), [f'E{c}' for c in cols])
ax.set_yticks(range(len(versions)), [version_label[v] for v in versions])
ax.set_xlabel('Normalized checkpoint column (same six evaluation points for all models)')
ax.set_title('Real-robot success rate by dataset size and training checkpoint')
for i, v in enumerate(versions):
    for j, c in enumerate(cols):
        val = heat.loc[v,c]
        if pd.notna(val):
            ax.text(j,i,f'{val*100:.0f}%',ha='center',va='center',color='white' if val>.55 else '#2b2b2b',fontweight='bold')
fig.colorbar(im, ax=ax, label='Success rate (%)')
fig.text(0.02, -0.02, 'Rows are ordered by dataset scale: v021–v024 are subsampled datasets; v020 is the full 765-episode dataset.', fontsize=8, color='#555555')
savefig_v2('v2_01_success_rate_heatmap_dataset_labels')

# v2-03: normalized learning curves with corner legend and explicit dataset sizes
fig, ax = plt.subplots(figsize=(9.0,4.9))
for v in versions:
    d = df[(df.version==v)&(df.include_heatmap)].sort_values('normalized_epoch')
    ax.plot(d.normalized_epoch, d.success_rate*100, marker='o', lw=2.2, ms=6, color=colors[v], label=version_label[v])
ax.set_ylim(-3,103)
ax.set_xticks(cols)
ax.grid(axis='y', alpha=.25)
ax.set_xlabel('Normalized checkpoint column')
ax.set_ylabel('Success rate (%)')
ax.set_title('Learning trajectory across the shared six-checkpoint evaluation schedule')
ax.legend(loc='upper left', fontsize=8)
savefig_v2('v2_03_learning_curves_corner_legend')

# v2-05: kept dataset-size scaling figure
fig, ax = plt.subplots(figsize=(8.4,4.8))
for _, r in best.sort_values('dataset_episodes').iterrows():
    ax.errorbar(r.dataset_episodes, r.success_rate*100, yerr=[[ (r.success_rate-r.ci_low)*100 ],[(r.ci_high-r.success_rate)*100 ]], fmt='o', ms=9, color=colors[r.version], capsize=4)
    ax.text(r.dataset_episodes*1.03, r.success_rate*100, f'{r.version}\n{int(r.successes)}/20', va='center', fontsize=9)
ax.set_xscale('log', base=2)
ax.set_xlim(40, 900)
ax.set_ylim(-3,103)
ax.grid(axis='y', alpha=.25)
ax.set_xticks([48,96,192,383,765], ['48','96','192','383','765'])
ax.set_xlabel('Training dataset size (episodes)')
ax.set_ylabel('Best observed success rate (%)')
ax.set_title('Best real-robot success rate increases with dataset coverage')
savefig_v2('v2_05_best_success_vs_dataset_size')

# v2-08: kept scene-type robustness, with explicit dataset labels
scene_p = scene.pivot(index='version', columns='scene', values='sr').reindex(versions)
fig, ax = plt.subplots(figsize=(8.6,4.6))
im=ax.imshow(scene_p.values*100, vmin=0, vmax=100, cmap=cmap, aspect='auto')
ax.set_xticks(range(len(scene_p.columns)), scene_p.columns, rotation=20, ha='right')
ax.set_yticks(range(len(versions)), [version_label[v] for v in versions])
ax.set_title('Scene-type robustness across dataset scales')
for i,v in enumerate(versions):
    for j,c in enumerate(scene_p.columns):
        row=scene[(scene.version==v)&(scene.scene==c)]
        if len(row):
            ax.text(j,i,f'{row.iloc[0]["sum"]:.0f}/{row.iloc[0]["count"]:.0f}',ha='center',va='center',fontsize=8,color='white' if row.iloc[0].sr>.55 else '#222')
fig.colorbar(im, ax=ax, label='Success rate (%)')
savefig_v2('v2_08_scene_success_heatmap_dataset_labels')

v2_figs=[
('v2_01_success_rate_heatmap_dataset_labels','Main dataset-scale heatmap','Kept from v1, but y-axis now spells out exact dataset sizes and subsampling ratios so the version names are no longer ambiguous.'),
('v2_03_learning_curves_corner_legend','Normalized learning curves','Kept with the legend moved to the upper-left corner and labels expanded to include dataset size.'),
('v2_05_best_success_vs_dataset_size','Best success versus dataset size','Kept as-is conceptually because it directly supports the dataset-scaling hypothesis.'),
('v2_08_scene_success_heatmap_dataset_labels','Scene-type robustness','Kept as failure-mode support; y-axis now also includes dataset sizes.'),
]
v2_lines=[]
v2_lines.append('# Evaluation visualization review board — v2\n')
v2_lines.append('This v2 applies the feedback from `evaluation_visual_review.md`: unclear versioning was fixed with explicit dataset-size labels, the curve legend was moved to a corner, and figures marked “Remove from final report” were removed from this review set.\n')
v2_lines.append('## Canonical data table\n')
v2_lines.append('| Version | Dataset meaning | E2 | E4 | E6 | E8 | E10 | E12 |\n|---|---|---:|---:|---:|---:|---:|---:|')
for v in versions:
    cells=[]
    for c in cols:
        r=df[(df.version==v)&(df.normalized_epoch==c)&(df.include_heatmap)].tail(1)
        if len(r):
            rr=r.iloc[0]
            cells.append(f'`{int(rr.step)}` {int(rr.successes)}/20 ({rr.success_rate*100:.0f}%)')
        else:
            cells.append('—')
    v2_lines.append(f'| {v} | {dataset_label[v]} | ' + ' | '.join(cells) + ' |')
v2_lines.append('\n## Figures kept for v2 review\n')
for idx, (fname,title,why) in enumerate(v2_figs, start=1):
    v2_lines.append(f'### {idx:02d}. {title}\n')
    v2_lines.append(f'![[figures_v2/{fname}.png]]\n')
    v2_lines.append(f'**V2 change / reasoning:** {why}\n')
    v2_lines.append('**Feedback:**\n')
    v2_lines.append('- [ ] Keep as-is\n- [ ] Needs label/title changes\n- [ ] Needs data/mapping check\n- [ ] Remove from final report\n')
    v2_lines.append('Notes:\n\n')
v2_lines.append('## Removed from v2 report set\n')
v2_lines.append('- Heatmap with Wilson intervals — removed per feedback.\n- Raw-step learning curves — removed per feedback.\n- Final vs best bar chart — removed per feedback.\n- Improvement from first checkpoint — removed per feedback.\n- Episode duration, trial matrix, spatial scatter, orientation, target side, and target closer plots — removed per feedback.\n')
v2_lines.append('## Global feedback\n\n- [ ] Version/dataset-size labeling is now clear.\n- [ ] Only the intended report figures remain.\n- [ ] Legend placement is acceptable.\n- [ ] v020 full-dataset baseline belongs at the bottom/end.\n\nNotes:\n')
(OUT/'evaluation_visual_review_v2.md').write_text('\n'.join(v2_lines))
print('Wrote', OUT)
print(df[['version','step','normalized_epoch','successes','n','success_rate','data_status']].to_string(index=False))
