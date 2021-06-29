def states_display(state_seq, title=None, figsize=(10,10), annot=True, fmt="0.1f", linewidths=.5, square=True, cbar=False, cmap="Reds", ax=None):
    state_array = np.array(state_seq).T

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(state_array, annot=annot, fmt=fmt, linewidths=linewidths, square=square, cbar=cbar, cmap=cmap, ax=ax)

    if title is not None:
        ax.set_title(title)

    if ax is None:
        plt.show()
    else:
        return ax

def policy_display(policy):
    labels = np.vectorize(lambda x: env.actions_dic[x])(policy)
    states_display(policy.T, title="Policy", cbar=False, annot=labels, fmt='s')
