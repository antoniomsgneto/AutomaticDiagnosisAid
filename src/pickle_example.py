# Save a dictionary into a pickle file.
import pickle5 
import sys
import matplotlib.pyplot as plt

def organize_by_point(slice_distances):
    aux_dict={}
    for item in slice_distances:
        try:
            aux_dict[item[0]].append(item[2])
        except KeyError:
            aux_dict[item[0]] = [item[2]]
    return aux_dict

def get_info_from_point(slice_distance_bPoint):
    values_dict={}
    
    for key in slice_distance_bPoint.keys():
        values_dict[key] = {}
        min_val = min(list(slice_distance_bPoint[key]))
        max_val = max(list(slice_distance_bPoint[key]))
        values_dict[key]['min'] = min_val
        values_dict[key]['max'] = max_val
        values_dict[key]['frame_min'] = list(slice_distance_bPoint[key]).index(min_val)+1
        values_dict[key]['frame_max'] = list(slice_distance_bPoint[key]).index(max_val)+1
    return values_dict


def plot_points(bla):
    graph = plt.figure(figsize=(10, 10))
    ax = graph.add_subplot(111)
    plt.gray()
    x_axis = list(range(0, len(bla)))
    ax.plot(x_axis, list(lista_pontos), '-', color='#ff7f0e')
    #graph.savefig('/home/joao/Desktop/code_tese/AutomaticDiagnosisAid/bla.png')

if __name__ == '__main__':
    if sys.argv[1]:
        data_stored = pickle5.load( open( sys.argv[1], "rb" ) )
        print(data_stored)
        print("\nNUMBER OF SLICES \n")
        print(len(data_stored))
        data_aux = data_stored
        aux = {}
        min_max_aux = {}
        for key in list(data_aux):
            if len(data_aux[key]) == 0:
                del data_aux[key]
            else:
                print("\n\n\n---------------\n\n\n")
                lel = organize_by_point(data_aux[key])
                min_max_aux[key] = get_info_from_point(lel)
                try:
                    aux[key].append(lel)
                except KeyError:
                    aux[key] = [lel]
             

        #plot_(aux[7][0][0])
        print("\n\n\n\nNumber of slices in new data:\n\n")
        print(len(data_aux))
        
        print(min_max_aux[7])
        


