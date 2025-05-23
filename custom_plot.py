import concurrent.futures
import functools
import json
import re
import warnings

import elements
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
import tqdm
import elements 


COLORS = [
    '#0022ff', '#33aa00', '#ff0011', '#ddaa00', '#cc44dd', '#0088aa',
    '#001177', '#117700', '#990022', '#885500', '#553366', '#006666',
    '#7777cc', '#999999', '#990099', '#888800', '#ff00aa', '#444444',
]


def load_config(config_path):
    """Load and parse a config.yaml file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.YAML(typ='safe').load(f)
        return config
    except Exception as e:
        elements.print(f"Error loading config {config_path}: {e}", color='red')
        return None

def get_run_metadata(run_dir):
    """Extract metadata from a run directory."""
    config_path = run_dir / 'config.yaml'
    config = load_config(config_path)
    if not config:
        return None
    
    # Extract method from directory structure
    method_parts = str(run_dir).split('/')
    for part in method_parts:
        if part in ['default', 'latent_reward_disagreement', 'optimized_replay_buffer']:
            method = part
            break
    else:
        method = "unknown"
    
    # Extract game name from config
    if 'task' in config:
        # For atari100k_battle_zone format, extract just the game part
        if 'atari100k_' in config['task']:
            game = config['task'].split('atari100k_')[1]
        else:
            game = config['task']
    
    # Extract seed and other relevant parameters
    metadata = {
        'method': method,
        'game': game,
        'seed': config.get('seed', 'unknown'),
        'config': config
    }
    
    # Add method-specific grouping parameters
    if 'latent_reward_disagreement' in method:
        intrinsic = config.get('agent', {}).get('intrinsic', {})
        metadata['intrinsic_reward_lambda'] = intrinsic.get('intrinsic_reward_lambda', 0)
        metadata['learn_strategy'] = intrinsic.get('learn_strategy', 'unknown')
    
    return metadata


def load_run(filename, xkeys, ykeys, ythres=None):
  try:
    try:
      df = pd.read_json(filename, lines=True)
    except ValueError:
      print('Falling back to robust JSONL reader.')
      records = []
      for line in filename.read_text().split('\n')[:-1]:
        try:
          records.append(json.loads(line))
        except json.decoder.JSONDecodeError:
          print(f'Skipping invalid JSONL line: {line}')
      df = pd.DataFrame(records)
    assert len(df), 'no timesteps in run'
    xkey = [k for k in xkeys if k in df]
    ykey = [k for k in ykeys if k in df]
    assert xkey, (filename, df.columns, xkeys)
    assert ykey, (filename, df.columns, ykeys)
    xs = df[xkey[0]].to_list()
    ys = df[ykey[0]].to_list()
    assert isinstance(xs, list), type(xs)
    assert isinstance(ys, list), type(ys)
    if ythres:
      ys = [1 if y > ythres else 0 for y in ys]
    return xs, ys
  except Exception as e:
    elements.print(f'Exception loading {filename}: {e}', color='red')
    return None


def load_runs(args):
  indirs = [elements.Path(x) for x in args.indirs]
  assert len(set(x.name for x in indirs)) == len(indirs), indirs
  records, filenames = [], []
  methods = re.compile(args.methods)
  tasks = re.compile(args.tasks)
  for indir in indirs:
    found = list(indir.glob(args.pattern))
    assert found, (indir, args.pattern)
    for filename in found:
      if args.newstyle:
        _, task, method, seed = filename.parent.name.split('-')
      else:
        task, method, seed = str(filename).split('/')[-4: -1]
      if not (methods.search(method) and tasks.search(task)):
        continue
      seed = f'{indir.name}_{seed}' if len(args.indirs) > 1 else seed
      method = f'{indir.name}_{method}' if args.indir_prefix else method
      records.append(dict(task=task, method=method, seed=seed))
      filenames.append(filename)
  print(f'Loading {len(records)} runs...')
  load = functools.partial(
      load_run, xkeys=args.xkeys, ykeys=args.ykeys, ythres=args.ythres)
  if args.workers:
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
      runs = list(tqdm.tqdm(pool.map(load, filenames), total=len(filenames)))
  else:
    runs = list(tqdm.tqdm((load(x) for x in filenames), total=len(filenames)))
  assert len(runs) > 0
  records, runs = zip(*[(x, y) for x, y in zip(records, runs) if y])
  for record, (xs, ys) in zip(records, runs):
    record.update(xs=xs, ys=ys)
  return pd.DataFrame(records)


def bin_runs(df, args):
  print('Binning runs...')
  if args.xlim:
    df['xlim'] = args.xlim
  else:
    xlim = df.groupby('task')['xs'].agg(lambda xs: max(max(x) for x in xs))
    df = pd.merge(df, xlim.rename('xlim'), on='task', how='left')
  if args.binsize:
    df['xlim'] = df['xlim'].max()
    df['binsize'] = args.binsize
  else:
    assert args.bins <= 1000, args.bins
    df['binsize'] = df['xlim'].apply(lambda x: x / args.bins)
  def binning(row):
    bins = np.arange(0, row['xlim'] + 0.99 * row['binsize'], row['binsize'])
    sums = np.histogram(row['xs'], bins=bins, weights=row['ys'])[0]
    nums = np.histogram(row['xs'], bins=bins)[0]
    xs = bins[1:]
    ys = np.divide(sums, nums, out=np.full(len(xs), np.nan), where=(nums != 0))
    return xs, ys
  df['xs'], df['ys'] = zip(*df.apply(binning, axis=1))
  df = df.drop(columns=['xlim', 'binsize'])
  assert len(df['xs'].apply(len).unique()) == 1
  return df


def comp_stat(name, df, fn, baseline=None):
  df = df.copy()
  if not df['xs'].apply(lambda xs: np.array_equal(xs, df['xs'][0])).all():
    assert len(df['xs'].apply(len).unique()) == 1
    domain = np.linspace(0, 1, len(df['xs'][0]))
    df['xs'] = df['xs'].apply(lambda _: domain)

  df = df.groupby(['task', 'method'])[['xs', 'ys']].agg(np.stack).reset_index()
  df['xs'] = df['xs'].apply(lambda xs: nanmean(xs, axis=0))
  df['ys'] = df['ys'].apply(lambda ys: nanmean(ys, axis=0))
  if baseline is not None:
    def normalize(row):
      task = row['task']
      if task in baseline:
        lo, hi = baseline[task]
      else:
        # If task not in baseline, use min/max of this task's data
        ys_flat = row['ys'].flatten()
        valid_ys = ys_flat[np.isfinite(ys_flat)]
        if len(valid_ys) > 0:
          lo, hi = np.min(valid_ys), np.max(valid_ys)
        else:
          lo, hi = 0, 1
      
      # Avoid division by zero
      if hi == lo:
        return row['ys'] - lo
      return (row['ys'] - lo) / (hi - lo)
    df['ys'] = df.apply(normalize, axis=1)
  df = df.groupby('method')[['xs', 'ys']].agg(np.stack).reset_index()

  df['xs'] = df['xs'].apply(lambda xs: nanmean(xs, axis=0))
  df['ys'] = df['ys'].apply(fn)
  df['name'] = name
  return df


def comp_count(name, df):
  df = df.copy()
  if not df['xs'].apply(lambda xs: np.array_equal(xs, df['xs'][0])).all():
    assert len(df['xs'].apply(len).unique()) == 1
    domain = np.linspace(0, 1, len(df['xs'][0]))
    df['xs'] = df['xs'].apply(lambda _: domain)
  df = df.groupby(['method'])[['xs', 'ys']].agg(np.stack).reset_index()
  df['xs'] = df['xs'].apply(lambda xs: nanmean(xs, axis=0))
  df['ys'] = df['ys'].apply(lambda ys: np.isfinite(ys).sum(0))
  df['name'] = name
  return df


def comp_stats(df, args):
    print('Computing stats...')

    # Derive baselines from default_* runs
    default_df = df[df['method'].str.startswith('default')]
    self_baseline = {}
    
    # Only compute baselines if we have default runs
    if not default_df.empty:
        for task in df['task'].unique():
            task_df = default_df[default_df['task'] == task]
            if not task_df.empty:
                all_ys = np.concatenate(task_df['ys'].tolist())
                valid_ys = all_ys[np.isfinite(all_ys)]
                if len(valid_ys) > 0:
                    self_baseline[task] = (np.min(valid_ys), np.max(valid_ys))
                else:
                    # Use 0-1 range as fallback
                    self_baseline[task] = (0, 1)
            else:
                # Use 0-1 range as fallback
                self_baseline[task] = (0, 1)

    stats = []
    choices = list(args.stats)
    choices = [x for x in choices if x != 'none']
    if not choices:
        return None

    if 'auto' in choices:
        choices.remove('auto')
        choices += ['mean', 'median']

    ax0 = lambda fn: functools.partial(fn, axis=0)
    for stat in choices:
        if stat == 'runs':
            x = comp_count('Runs', df)
        elif stat == 'mean':
            x = comp_stat('Mean', df, ax0(np.mean))
        elif stat == 'median':
            x = comp_stat('Median', df, ax0(np.median))
        elif stat == 'self_mean':
            # Only compute self_mean if we have baselines
            if self_baseline:
                x = comp_stat('Self Mean', df, ax0(nanmean), self_baseline)
            else:
                print("Skipping 'self_mean' stat - no baseline data available")
                continue
        elif stat == 'self_median':
            # Only compute self_median if we have baselines
            if self_baseline:
                x = comp_stat('Self Median', df, ax0(nanmedian), self_baseline)
            else:
                print("Skipping 'self_median' stat - no baseline data available")
                continue
        else:
            print(f"Stat '{stat}' is not supported without baselines.yaml.")
            continue
        stats.append(x)

    return pd.concat(stats) if stats else None



def plot_runs(df, stats, args, title=None):
    print('Plotting...')
    tasks = natsort(df.task.unique())
    snames = [] if stats is None else stats.name.unique()
    methods = natsort(df.method.unique())
    total = len(tasks) + len(snames)
    cols = args.cols or (4 + (total > 24) + (total > 35) + (total > 48))
    
    bigger_size = (6, 5)  
    fig, axes = plots(total, cols, bigger_size)

    # For game-specific plots, show individual seeds with different colors
    for task, ax in zip(tasks, axes[:len(tasks)]):
        style(ax, xticks=args.xticks, yticks=args.yticks)
        title_text = title or task.replace('_', ' ').replace(':', ' ').title()
        ax.set_title(title_text, fontsize=14)  # Larger title font
        args.xlim and ax.set_xlim(0, 1.03 * args.xlim)
        args.ylim and ax.set_ylim(0, 1.03 * args.ylim)
        task_df = df[df['task'] == task]
        
        # Plot each method and seed combination separately
        color_idx = 0 
        for method in methods:
            method_df = task_df[task_df['method'] == method]
            
            if method_df.empty:
                continue
                
            # Plot each seed with its own color
            for seed, seed_data in method_df.groupby('seed'):
                xs = seed_data['xs'].iloc[0]
                ys = seed_data['ys'].iloc[0]
                label = f"{method} (seed {seed})"
                curve(ax, xs, ys, None, None, label, color_idx, linestyle='-')
                color_idx += 1 

    # For stat plots (Mean, Self Mean), add variance shading
    if stats is not None:
        try:
            # First, prepare data for variance calculation
            mean_data = {}
            for task in tasks:
                task_df = df[df['task'] == task]
                for method in methods:
                    method_df = task_df[task_df['method'] == method]
                    if method_df.empty:
                        continue
                    
                    # Stack all seed data for this method
                    xs = method_df['xs'].iloc[0]  # Assume all xs are aligned
                    all_ys = np.stack(method_df['ys'].values)
                    
                    # Calculate mean and std
                    mean_ys = nanmean(all_ys, axis=0)
                    std_ys = nanstd(all_ys, axis=0)
                    
                    # Store for later use
                    if (task, method) not in mean_data:
                        mean_data[(task, method)] = (xs, mean_ys, mean_ys - std_ys, mean_ys + std_ys)
            
            # Now plot the stat plots with variance
            grouped_stats = stats.groupby(['name', 'method'])
            for sname, ax in zip(snames, axes[len(tasks):]):
                style(ax, xticks=args.xticks, yticks=args.yticks, darker=True)
                ax.set_title(sname, fontsize=14)  # Larger title font
                args.xlim and ax.set_xlim(0, 1.03 * args.xlim)
                
                for i, method in enumerate(methods):
                    try:
                        sub = grouped_stats.get_group((sname, method))
                        xs = sub['xs'].iloc[0]
                        ys = sub['ys'].iloc[0]
                        
                        # For Mean plot, add variance shading
                        if sname == 'Mean':
                            # Aggregate variance across tasks
                            all_lo = []
                            all_hi = []
                            
                            for task in tasks:
                                if (task, method) in mean_data:
                                    _, _, lo, hi = mean_data[(task, method)]
                                    all_lo.append(lo)
                                    all_hi.append(hi)
                            
                            if all_lo and all_hi:
                                lo = nanmean(np.stack(all_lo), axis=0)
                                hi = nanmean(np.stack(all_hi), axis=0)
                                curve(ax, xs, ys, lo, hi, method, i)
                            else:
                                curve(ax, xs, ys, None, None, method, i)
                        else:
                            # For other stats, no variance shading
                            curve(ax, xs, ys, None, None, method, i)
                    except (KeyError, ValueError) as e:
                        print(f"Error plotting stats for method '{method}' on '{sname}': {e}")
                        continue
        except Exception as e:
            print(f"Error plotting stats: {e}")

    # Increase legend font size and spacing
    legend(fig, adjust=True, ncol=args.legendcols or min(4, cols, len(axes)), fontsize=12)

    # Create output directory
    outdir = elements.Path(args.outdir) / title if title else elements.Path(args.outdir) / elements.Path(args.indirs[0]).stem
    outdir.mkdir(parents=True, exist_ok=True)
    filename = outdir / 'curves.png'
    fig.savefig(filename, dpi=200)  # Higher DPI for better quality
    print('Saved', filename)


def plots(amount, cols=4, size=(3, 3), **kwargs):
  rows = int(np.ceil(amount / cols))
  cols = min(cols, amount)
  kwargs['figsize'] = kwargs.get('figsize', (size[0] * cols, size[1] * rows))
  fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False, **kwargs)
  for ax in axes.flatten()[amount:]:
    ax.axis('off')
  ax = axes.flatten()[:amount]
  return fig, ax


def style(ax, xticks=4, yticks=4, grid=(1, 1), logx=False, darker=False):
  ax.tick_params(axis='x', which='major', length=2, labelsize=10, pad=3)
  ax.tick_params(axis='y', which='major', length=2, labelsize=10, pad=2)
  ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(xticks))
  ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(yticks))
  ax.xaxis.set_major_formatter(lambda x, pos: natfmt(x))
  ax.yaxis.set_major_formatter(lambda x, pos: natfmt(x))
  if grid:
    color = '#cccccc' if darker else '#eeeeee'
    ax.grid(which='both', color=color)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(grid[0]))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(grid[1]))
    ax.tick_params(which='minor', length=0)
  if logx:
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(plt.LogLocator(10, numticks=3))
    ax.xaxis.set_minor_locator(plt.LogLocator(10, subs='all', numticks=100))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
  if darker:
    ax.set_facecolor((0.95, 0.95, 0.95))


def curve(
    ax, xs, ys, lo=None, hi=None, label=None, order=None, color=None,
    scatter=True, linestyle='-', **kwargs):
  color = color or (None if order is None else COLORS[order])
  order = order or 0
  kwargs['color'] = color
  mask = np.isfinite(ys)
  ax.plot(xs[mask], ys[mask], label=label, zorder=200 - order, linestyle=linestyle, **kwargs)
  if scatter:
    ax.scatter(xs, ys, s=5, label=None, zorder=3000 - order, **kwargs)  # Remove label from scatter to avoid duplicate legend entries
  if lo is not None:
    ax.fill_between(
        xs[mask], lo[mask], hi[mask],
        zorder=100 - order, lw=0, **{**kwargs, 'alpha': 0.2})


def legend(fig, names=None, reverse=False, adjust=False, fontsize=10, **kwargs):
  options = dict(
      fontsize=fontsize, numpoints=1, labelspacing=0, columnspacing=1.2,
      handlelength=1.5, handletextpad=0.5, ncol=4, loc='lower center')
  options.update(kwargs)
  entries = {}
  for ax in fig.axes:
    for handle, label in zip(*ax.get_legend_handles_labels()):
      entries[label] = handle
  if names:
    entries = {name: entries[label] for label, name in names.items()}
  if reverse:
    entries = dict(list(reversed(list(entries.items()))))
  leg = fig.legend(entries.values(), entries.keys(), **options)
  leg.get_frame().set_edgecolor('white')
  leg.set_zorder(2000)
  [line.set_linewidth(2) for line in leg.legend_handles]
  if adjust:
    extent = leg.get_window_extent(fig.canvas.get_renderer())
    extent = extent.transformed(fig.transFigure.inverted())
    yloc, xloc = options['loc'].split()
    y0 = dict(lower=extent.y1, center=0, upper=0)[yloc]
    y1 = dict(lower=1, center=1, upper=extent.y0)[yloc]
    x0 = dict(left=extent.x1, center=0, right=0)[xloc]
    x1 = dict(left=1, center=1, right=extent.x0)[xloc]
    fig.tight_layout(rect=[x0, y0, x1, y1], h_pad=1, w_pad=1)
  return leg


def silent(fn):
  def wrapped(*args, **kwargs):
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      return fn(*args, **kwargs)
  return wrapped
nanmean = silent(np.nanmean)
nanmedian = silent(np.nanmedian)
nanstd = silent(np.nanstd)
nanmax = silent(np.nanmax)
nanmin = silent(np.nanmin)


def natsort(sequence):
  pattern = re.compile(r'([0-9]+)')
  return sorted(sequence, key=lambda x: [
      (int(y) if y.isdigit() else y) for y in pattern.split(x)])


def natfmt(x):

  if abs(x) < 1e3:
    x, suffix = x, ''
  elif 1e3 <= abs(x) < 1e6:
    x, suffix = x / 1e3, 'K'
  elif 1e6 <= abs(x) < 1e9:
    x, suffix = x / 1e6, 'M'
  elif 1e9 <= abs(x):
    x, suffix = x / 1e9, 'B'
  if abs(x) <= 1:
    return f'{x:.3f}{suffix}'
  elif 1 <= abs(x) < 10:
    return f'{x:.1f}{suffix}'
  elif 10 <= abs(x):
    return f'{x:.0f}{suffix}'


def print_summary(df):
  methods = natsort(df.method.unique())
  tasks = natsort(df.task.unique())
  seeds = natsort(df.seed.unique())
  print('-' * 79)
  print(f'Methods ({len(methods)}):', ', '.join(methods))
  print('-' * 79)
  print(f'Tasks ({len(tasks)}):', ', '.join(tasks))
  print('-' * 79)
  print(f'Seeds ({len(seeds)}):', ', '.join(seeds))
  print('-' * 79)


def find_and_group_runs(logdir, pattern="**/scores.jsonl"):
    """Find all runs and group them by method, game, and relevant parameters."""
    logdir = elements.Path(logdir)
    all_runs = list(logdir.glob(pattern))
    
    grouped_runs = {}
    for run_file in all_runs:
        run_dir = run_file.parent
        metadata = get_run_metadata(run_dir)
        if not metadata:
            continue
        
        # Create grouping key based on method
        if 'latent_reward_disagreement' in metadata['method']:
            # Include lambda in the key to keep different lambda values separate
            group_key = (
                metadata['method'],
                metadata['game'],
                metadata['intrinsic_reward_lambda'],
                metadata['learn_strategy']
            )
            # Create a method name that includes the lambda value
            method_name = f"{metadata['method']}_lambda{metadata['intrinsic_reward_lambda']}"
        else:  # default or optimized_replay_buffer
            group_key = (
                metadata['method'],
                metadata['game']
            )
            method_name = metadata['method']
        
        if group_key not in grouped_runs:
            grouped_runs[group_key] = []
        
        grouped_runs[group_key].append({
            'file': run_file,
            'metadata': metadata,
            'method_name': method_name
        })
    
    return grouped_runs


def main(args):
    # Find and group runs
    grouped_runs = find_and_group_runs(args.logdir, args.pattern)
    
    # Create a dictionary to store all runs by game
    games_data = {}
    
    # Process each group and organize by game
    for group_key, runs in grouped_runs.items():
        method = runs[0]['method_name']  # Use the method name with lambda if applicable
        game = group_key[1]
        
        print(f"Processing group: {game}_{method} with {len(runs)} runs")
        
        # Load data from each run
        records = []
        filenames = []
        for run in runs:
            records.append({
                'task': game,
                'method': method,  # Method name already includes lambda if needed
                'seed': run['metadata']['seed']
            })
            filenames.append(run['file'])
        
        # Load run data
        load = functools.partial(
            load_run, xkeys=args.xkeys, ykeys=args.ykeys, ythres=args.ythres)
        
        if args.workers:
            with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
                loaded_runs = list(tqdm.tqdm(pool.map(load, filenames), total=len(filenames)))
        else:
            loaded_runs = list(tqdm.tqdm((load(x) for x in filenames), total=len(filenames)))
        
        # Filter out failed loads
        valid_data = [(rec, run) for rec, run in zip(records, loaded_runs) if run]
        if not valid_data:
            print(f"No valid data for group {game}_{method}")
            continue
            
        records, loaded_runs = zip(*valid_data)
        
        # Update records with data
        for record, (xs, ys) in zip(records, loaded_runs):
            record.update(xs=xs, ys=ys)
        
        # Add to the game's data collection
        if game not in games_data:
            games_data[game] = []
        
        games_data[game].extend(records)
    
    # Now process each game separately
    for game, records in games_data.items():
        if not records:
            continue
            
        print(f"Creating plots for game: {game}")
        
        # Create DataFrame for this game
        df = pd.DataFrame(records)
        
        # Bin runs
        df = bin_runs(df, args)
        
        # Compute stats
        stats = comp_stats(df, args)
        
        # Plot
        plot_runs(df, stats, args, title=game)


if __name__ == '__main__':
    main(elements.Flags(
        logdir='SCRATCH/scur2603logdir/',
        pattern='**/scores.jsonl',
        indirs=[''],
        outdir='plots',
        methods='.*',
        tasks='.*',
        newstyle=True,
        indir_prefix=False,
        workers=16,
        xkeys=['xs', 'step'],
        ykeys=['ys', 'episode/score'],
        ythres=0.0,
        xlim=0,
        ylim=0,
        binsize=0,
        bins=30,
        cols=0,
        legendcols=0,
        size=[3, 3],
        xticks=4,
        yticks=10,
        stats=['mean'],
        agg=True,
        todf='',
    ).parse())
