import imageio
import matplotlib.pyplot as plt


class Gifs:
    # Build GIF 3D
    @staticmethod
    def save_gif_3d_with_range(min_range, max_range):
        with imageio.get_writer('mygif3d.gif', mode='I') as writer:
            for i in range(min_range, max_range):
                if i < 10:
                    i = "0" + str(i)
                image = imageio.imread(str(i) + "3d" + ".png")
                writer.append_data(image)

    # Build GIF 3D
    @staticmethod
    def save_gif_3d():
        with imageio.get_writer('mygif3d.gif', mode='I') as writer:
            for i in range(1, 22):
                if i < 10:
                    i = "0" + str(i)
                image = imageio.imread(str(i) + "3d" + ".png")
                writer.append_data(image)

    # Build GIF 2D
    @staticmethod
    def save_gif_2d(patient_str):
        with imageio.get_writer(patient_str + '.gif', mode='I') as writer:
            for i in range(1, 21):
                if i < 10:
                    i = "0" + str(i)
                image = imageio.imread(str(i) + ".png")
                writer.append_data(image)

    # Build GIF 2D for graph images
    @staticmethod
    def save_gif_2d_graph():
        with imageio.get_writer('mygraph.gif', mode='I') as writer:
            for i in range(1, 21):
                if i < 10:
                    i = "0" + str(i)
                image = imageio.imread(str(i) + "graph.png")
                writer.append_data(image)

    # Build GIF 2D combined graph and normal
    @staticmethod
    def save_gif_2d_combined():
        with imageio.get_writer('comb.gif', mode='I') as writer:
            for i in range(1, 21):
                if i < 10:
                    i = "0" + str(i)
                fig = plt.figure(figsize=(9, 9))
                ax = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                plt.gray()

                image = imageio.imread(str(i) + ".png")
                image2 = imageio.imread(str(i) + "graph.png")
                ax.imshow(image)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax2.imshow(image2)
                ax2.axes.get_xaxis().set_visible(False)
                ax2.axes.get_yaxis().set_visible(False)
                fig.savefig(str(i) + 'comb.png')
                figure = imageio.imread(str(i) + "comb.png")
                writer.append_data(figure)
