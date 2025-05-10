import matplotlib.pyplot as plt
import os

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
def plot_figure(X,Y,label_x, label_y, title, path=f'./figures/'):
    if not os.path.exists(path):
        os.makedirs(path)
    # Create a plot
    plt.plot(X, Y)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    path = f'./figures/'
    # Save the plot as a PDF file
    plt.savefig(path+'plot.pdf')

    # Show the plot (optional)
    plt.show()
# plot_figure(x,y, 'X-axis', 'Y-axis', 'Sample Plot')

def plot_loass_accuracy_in_pdf(X,Y, loss,label_x, label_y, title, path=f'./figures/'):
    if not os.path.exists(path):
        os.makedirs(path)
    # Create a plot
    plt.plot(X, Y, label='Accuracy')
    plt.plot(X, loss, label='Loss')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    # Add a legend
    plt.legend()
    path = f'./figures/'
    # Save the plot as a PDF file
    plt.savefig(path+'plot.pdf')

    # Show the plot (optional)
    plt.show()

def plot_data_dict_in_pdf(data: dict, title="Title", path='./figures/', show=False):
    import os
    import matplotlib.pyplot as plt

    if not os.path.exists(path):
        os.makedirs(path)

    # Create a plot
    X = data["x"]
    file_name = data['name']
    if data["dual_axis"]:
        fig, ax1 = plt.subplots()
        Y_1, Y_2 = data["y"]
        Legend_1, Legend_2 = data["legends"]
        label_x, (label_y1, label_y2) = data["labels"]

        ax1.set_xlabel(label_x)
        ax1.set_ylabel(label_y1, color='b')
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # List of colors
        for i, (Y, legend) in enumerate(zip(Y_1, Legend_1)):
            ax1.plot(X, Y, label=legend, color=colors[i % len(colors)])  # Use modulo to loop through colors
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.set_xlabel(label_x)
        ax2.set_ylabel(label_y2, color='r')
        for i, (Y, legend) in enumerate(zip(Y_2, Legend_2)):
            ax2.plot(X, Y, label=legend, color=colors[(i + len(Y_1)) % len(colors)])  # Start from a new color
        ax2.legend(loc='lower right')
    max_acc = data["max_acc_g"]
    plt.title(title+f":max({max_acc})")

    # Save the plot as a PDF file
    plt.savefig(f'{path}{file_name}.pdf')

    # Show the plot (optional)
    if show:
        plt.show()

# Example usage:
# plot_data_dict_in_pdf(loaded_data)
















# def plot_data_dict_in_pdf(data: dict, title, path=f'./figures/', ):
#     if not os.path.exists(path):
#         os.makedirs(path)
#     # Create a plot
#     X = data["x"] 
#     Y_list = data["y"]
#     Legen_labels = data["legends"]
#     label_x, label_y = data["label"]
#     file_name = data['name']
#     for Y, legend in zip(Y_list, Legen_labels):
#         plt.plot(X, Y, label=legend)
#         plt.xlabel(label_x)
#         plt.ylabel(label_y)
#         plt.title(title)
#         # Add a legend
#     plt.legend()
#     path = f'./figures/'
#     # Save the plot as a PDF file
#     plt.savefig(path+f'{file_name}.pdf')

#     # Show the plot (optional)
#     plt.show()

