from os import listdir
#from segmentation.simple_segmentation import segment_patient
from segmentation.dot_segmentation import segment_patient
from segmentation.dot_segmentation import save_gif_2d
import sys
import pickle as pickle5
from random import randint
from segmentation.clustering import data_to_cluster
import seaborn as sns
import matplotlib.pyplot as plt
import os



def get_pallete(n):
  return sns.color_palette('viridis', n).as_hex()


def plot_graphs(all_cluster_info, predicted_clusters, colors, fig_list_cluster, path_to_file,path_to_output_folder):
    patient_str = path_to_file.split('.')[0].split('/')[-1]
    
    for frame in range(0, len(all_cluster_info)):
        cluster_info = all_cluster_info[frame]
        print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        print(len(cluster_info[2]))

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)
        plt.gray()

        ax.imshow(cluster_info[0])
        
        snake = cluster_info[1]
        
        xn2 = snake[:, 1]
        yn2 = snake[:, 0]

        for i in range(len(xn2)):
            
            if i % 5 == 0:
                    
                print(len(predicted_clusters))
                print(i)
                ax.plot(cluster_info[2][int(i/5)].x, cluster_info[2][int(i/5)].y, 'o', color=colors[predicted_clusters[int(i/5)]])
                ax.plot(xn2[i], yn2[i], 'o', color=colors[predicted_clusters[int(i/5)]])
        fig.savefig(cluster_info[3])
    save_gif_2d(path_to_output_folder + '/CLUSTER_' + patient_str, fig_list_cluster)
    for file in fig_list_cluster:
        os.remove(file)




if __name__ == '__main__':
    path_folder_1 = '/Users/antonioneto/Antonio/tese/Dados/ACDC_segmented/OUTPUT_ACDC/'
    path_folder_2 ='/Users/antonioneto/Antonio/tese/Dados/GotaResults'
    if path_folder_1:
        counter = 0
        data_dict={}
        colors = []
        for i in range(100):
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        print(len(colors))
        for file in listdir(path_folder_1):
            if file.endswith('nii.gz'):
                print("here")
                data_by_frame, all_cluster_info, fig_list_cluster = segment_patient(path_folder_1 + file.replace('.nii.gz','_0000.nii.gz'), path_folder_1 + file, path_folder_2, colors)
                predicted_clusters = data_to_cluster(data_by_frame, 100,6,400,15)
                cluster_colors = get_pallete(6)
                plot_graphs(all_cluster_info, predicted_clusters, cluster_colors, fig_list_cluster, sys.argv[1] + file, sys.argv[3])

                """
                try:
                    data_dict[counter].append(data_by_frame)
                except KeyError:
                    data_dict[counter] = data_by_frame
                counter+=1
                """
                pickle_file_path = sys.argv[4] + file.split(".")[0] + ".pkl"
                print(pickle_file_path + " IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
                pickle5.dump( data_by_frame, open( pickle_file_path, "wb" ) )

