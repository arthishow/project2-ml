import numpy as np
import matplotlib.pyplot as plt


def plot_number_of_ratings(data):
    num_items_per_user = np.array((data != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((data != 0).sum(axis=1).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_movies_per_user)
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_users_per_movie)
    ax2.set_xlabel("Items")
    ax2.set_ylabel("Number of ratings (sorted)")
    ax2.grid()

    plt.tight_layout()
    plt.savefig("plots/num_ratings")
    plt.show()
