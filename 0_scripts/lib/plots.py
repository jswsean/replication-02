import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.markers as mmarkers

# Official party colors.
USA_COL = {'dem': '#3333FF', 'rep': '#E91D0E'}
CA_COL = {'lib': '#D71920', 'con': '#1A4782', 'ndp': '#F28000', 'bloc': '#33B2CC', 'ref': '#3CB371'}
UK_COL = {'lab': '#D50000', 'con': '#0087DC', 'lib': '#FDBB3A'}

# Party markers.
USA_MK = {'dem': 'o', 'rep': 's'}
CA_MK = {'lib': 's', 'con': '^', 'ndp': 'o', 'bloc': 'D', 'ref': 'x'}
UK_MK = {'lab': 'o', 'con': '^', 'lib': 's'}

# The following lines are not needed per se, but this reproduces the exact font and formatting used in the paper:
mpl.rcParams["font.family"] = "Verdana"
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['font.size'] = 14

def mscatter(x, y, ax=None, m=None, **kw):

    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)

    return sc

def plot_2a(Z, labels, cols, mkers, savepath='figure2a.pdf', xlim=(-20, 20)):

    plt.figure(figsize=(22,15))
    mscatter(Z.dim1, Z.dim2, alpha=0.6, color=cols, m=mkers, s = 400)
    for idx, (label, x, y, c) in enumerate(zip(labels, Z.dim1, Z.dim2, cols)):
        # Prints only one out of three labels for clarity:
        if idx%3==0:
            plt.annotate(
                label,
                xy=(x, y), xytext=(-40, 40), fontsize=12,
                textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc=c, alpha=0.2),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.xlim(xlim)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,y1,y2 + 1))
    # Adding legend and axis labels:
    legend_elements = [Line2D([], [], marker=USA_MK['dem'],
                              color=USA_COL['dem'], alpha=1, label='Democrats',
                              linestyle='None', markersize=15),
                        Line2D([], [], marker=USA_MK['rep'],
                              color=USA_COL['rep'], label='Republicans',
                              linestyle='None', markersize=15)]
    plt.legend(handles=legend_elements,loc='lower left',numpoints = 1,prop={'size': 20} )
    # Save resulting plot in current directory.
    plt.savefig(savepath, dpi=600, bbox_inches='tight')
    plt.close()

def plot_timeseries(Z, fullnames, cols, dimension=1, savepath='figure2b.pdf', legend='upper left'):
    # Reshaping the dataset for time-series plot
    reshaped = Z
    newvars = reshaped.label.str.split(n=1,expand=True)
    reshaped['year'] = newvars[1].astype(float)
    reshaped['party'] = fullnames
    reshaped['color'] = cols
    fig, ax = plt.subplots(figsize=(22,15), sharex='all')
    for idx, (key, grp) in enumerate(reshaped.groupby('party')):
        if dimension==1:
            if idx==0:
                grp.plot(ax=ax, kind='line', x='year', y='dim1', linewidth=5, ls='dashed', c=grp.color.values[0], label=key)
            else:
                grp.plot(ax=ax, kind='line', x='year', y='dim1', linewidth=5, ls='solid', c=grp.color.values[0], label=key)
        else:
            if idx==0:
                grp.plot(ax=ax, kind='line', x='year', y='dim2', linewidth=5, ls='dashed', c=grp.color.values[0], label=key)
            else:
                grp.plot(ax=ax, kind='line', x='year', y='dim2', linewidth=5, ls='solid', c=grp.color.values[0], label=key)
    # Adding legend and axis labels
    plt.legend(loc=legend,prop={'size': 20})
    plt.xlabel("Year")
    if dimension==1:
        plt.ylabel("Ideological Placement (First Principal Component)")
    else:
        plt.ylabel("South-North Axis (Second Principal Component)")
    # Save resulting plot in current directory.
    plt.savefig(savepath, dpi=600, bbox_inches='tight')
    plt.close()


def plot_3a(Z, labels, cols, mkers, savepath='figure3a.pdf', xlim=(-20, 20)):
    plt.figure(figsize=(22,15))
    mscatter(Z.dim1, Z.dim2, alpha=0.6, color=cols, m=mkers, s = 400)
    uk_examples = ['Labour 1987', 'Cons 1987', 'Labour 2001', 'Cons 2010', 'LibDems 1997']
    for idx, (label, x, y, c) in enumerate(zip(labels, Z.dim1, Z.dim2, cols)):
        # Prints only a few labels for clarity:
        if label in uk_examples:
            plt.annotate(
                label,
                xy=(x, y), xytext=(-40, 40), fontsize=12,
                textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc=c, alpha=0.2),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.xlim(xlim)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,y1,y2 + 1))
    # Adding legend and axis labels:
    legend_elements = [Line2D([], [], marker=UK_MK['lab'],
                              color=UK_COL['lab'], alpha=0.6, label='Labour',
                              linestyle='None', markersize=15),
                        Line2D([], [], marker=UK_MK['lib'],
                              color=UK_COL['lib'], label='Liberal-Democrat',
                              linestyle='None', markersize=15),
                       Line2D([], [], marker=UK_MK['con'],
                              color=UK_COL['con'], label='Conservative',
                              linestyle='None', markersize=15)]
    plt.legend(handles=legend_elements,loc='upper left',numpoints = 1, prop={'size': 20} )
    # Save resulting plot in current directory.
    plt.savefig(savepath, dpi=600, bbox_inches='tight')
    plt.close()

def plot_3b(Z, labels, cols, mkers, savepath='figure3b.pdf', xlim=(-20, 20)):
    plt.figure(figsize=(22,15))
    mscatter(Z.dim1, Z.dim2, alpha=0.6, color=cols, m=mkers, s = 400)
    can_examples = ['Cons 2015', 'Liberal 2015', 'NDP 1984', 'Bloc 1993', 'RefAll 1993']
    for idx, (label, x, y, c) in enumerate(zip(labels, Z.dim1, Z.dim2, cols)):
        # Prints only a few labels for clarity:
        if label in can_examples:
            plt.annotate(
                label,
                xy=(x, y), xytext=(-40, 40), fontsize=12,
                textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc=c, alpha=0.2),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.xlim(xlim)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,y1,y2 + 1))
    # Adding legend and axis labels:
    legend_elements = [Line2D([], [], marker=CA_MK['ndp'],
                              color=CA_COL['ndp'], alpha=0.6, label='New Democratic',
                              linestyle='None', markersize=15),
                        Line2D([], [], marker=CA_MK['lib'],
                              color=CA_COL['lib'], label='Liberal',
                              linestyle='None', markersize=15),
                       Line2D([], [], marker=CA_MK['con'],
                              color=CA_COL['con'], label='Conservative',
                              linestyle='None', markersize=15),
                       Line2D([], [], marker=CA_MK['bloc'],
                              color=CA_COL['bloc'], alpha=0.6, label='Bloc Quebecois',
                              linestyle='None', markersize=15),
                       Line2D([], [], marker=CA_MK['ref'],
                              color=CA_COL['ref'], alpha=0.6, label='Reform/Canadian Alliance',
                              linestyle='None', markersize=15)]
    plt.legend(handles=legend_elements,loc='upper left',numpoints = 1 , prop={'size': 20})
    # Save resulting plot in current directory.
    plt.savefig(savepath, dpi=600, bbox_inches='tight')
    plt.close()

def plot_4a(df, savepath='figure4a.pdf'):
    plt.figure(figsize=(22,15))
    plt.xlim((1870,2018))
    plt.plot(df.year, df.democrat, color=USA_COL['dem'], ls='dashed', linewidth=5)
    plt.plot(df.year, df.republican, color=USA_COL['rep'], linewidth=5)
    legend_elements = [Line2D([], [], color=USA_COL['dem'], label='Democrats',
                                  linestyle='dashed', linewidth=5),
                            Line2D([], [], color=USA_COL['rep'], label='Republicans',
                                  linestyle='solid', linewidth=5)]
    plt.xlabel("Year")
    plt.ylabel("WordFish Estimate")
    plt.legend(handles=legend_elements, loc='upper left', prop={'size': 40})
    plt.savefig(savepath, dpi=600, bbox_inches='tight')
    plt.close()

def plot_4b(df, savepath='figure4b.pdf'):
    plt.figure(figsize=(22,15))
    plt.xlim((2006,2016))
    plt.plot(df.year, df.democrat, color=USA_COL['dem'], linestyle='--', linewidth=5)
    plt.plot(df.year, df.republican, color=USA_COL['rep'], linewidth=5)
    plt.xlabel("Year")
    plt.ylabel("WordFish Estimate")
    legend_elements = [Line2D([], [], color=USA_COL['dem'], label='Democrats',
                                  linestyle='--', linewidth=5),
                            Line2D([], [], color=USA_COL['rep'], label='Republicans',
                                  linestyle='solid', linewidth=5)]
    plt.legend(handles=legend_elements, loc='lower right', prop={'size': 40})
    plt.savefig(savepath, dpi=600, bbox_inches='tight')
    plt.close()

def plot_5(Z, dems, reps, savepath='figure5.pdf'):
    cols = [USA_COL['dem']]*len(dems) + [USA_COL['rep']]*len(reps)
    mkers = [USA_MK['dem']]*len(dems) + [USA_MK['rep']]*len(reps)
    legend_elements = [Line2D([], [], marker=USA_MK['dem'],
                              color=USA_COL['dem'], alpha=1, label='Democrats',
                              linestyle='None', markersize=15),
                        Line2D([], [], marker=USA_MK['rep'],
                              color=USA_COL['rep'], label='Republicans',
                              linestyle='None', markersize=15)]
    plt.figure(figsize=(21, 15))
    mscatter(Z.dim1, Z.dim2, alpha=1, color=cols, m=mkers, s = 400)
    plt.xlim((-21,21))
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(handles=legend_elements, loc='upper left', numpoints = 1 , prop={'size': 30})
    plt.savefig(savepath, dpi=600, bbox_inches='tight')
    plt.close()

def plot_A2(Z, cols, mkers, savepath='figureA2.pdf'):
    uk_examples = ['Labour 1987', 'Cons 1987', 'Labour 2001', 'Cons 2010', 'LibDems 1997']
    plt.figure(figsize=(22,15))
    mscatter(Z.dim1, Z.dim2, alpha=0.6, color=cols, m=mkers, s = 400)
    for idx, (label, x, y, c) in enumerate(zip(Z.label, Z.dim1, Z.dim2, cols)):
        if label in uk_examples:
            plt.annotate(
                label,
                xy=(x, y), xytext=(-40, 40), fontsize=12,
                textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc=c, alpha=0.2),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.xlabel("Economic Left-Right")
    plt.ylabel("Social Left-Right")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,y1,y2 + 2))
    legend_elements = [Line2D([], [], marker=UK_MK['lab'],
                              color=UK_COL['lab'], alpha=0.6, label='Labour',
                              linestyle='None', markersize=15),
                        Line2D([], [], marker=UK_MK['lib'],
                              color=UK_COL['lib'], label='Liberal-Democrat',
                              linestyle='None', markersize=15),
                       Line2D([], [], marker=UK_MK['con'],
                              color=UK_COL['con'], label='Conservative',
                              linestyle='None', markersize=15)]
    plt.legend(handles=legend_elements,loc='upper left',numpoints = 1, prop={'size': 20} )
    plt.savefig(savepath, dpi=600, bbox_inches='tight')
    plt.close()
