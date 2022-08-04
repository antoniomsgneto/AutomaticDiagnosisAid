# Save a dictionary into a pickle file.
import pickle5 
import sys

def organize_by_point(slice_distances):
    aux_dict={}
    for item in slice_distances:
        try:
            aux_dict[item[0]].append(item[2])
        except KeyError:
            aux_dict[item[0]] = [item[2]]
    return aux_dict


if __name__ == '__main__':
    if sys.argv[1]:
        data_stored = pickle5.load( open( sys.argv[1], "rb" ) )
        
        print("\nNUMBER OF SLICES \n")
        print(len(data_stored))
        data_aux = data_stored
        print(data_aux[2])
        aux = {}
        for key in list(data_aux):
            if len(data_aux[key]) == 0:
                del data_aux[key]
            else:
                print("\n\n\n---------------\n\n\n")
                lel = organize_by_point(data_aux[key])
                try:
                    aux[key].append(lel)
                except KeyError:
                    aux[key] = [lel]
                print(aux)
        print("\n\n\n\nNumber of slices in new data:\n\n")
        print(len(data_aux))
        


