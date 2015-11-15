import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import csv

def plot_graph(x_axis, y_axis, x_axis_name, y_axis_name, z_axis = None):
    if z_axis:
        plt.plot(x_axis,y_axis, 'bs', z_axis, y_axis, 'r^')
    else:
        plt.plot(x_axis,y_axis)
    plt.show()
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    out_file = x_axis_name + "_"+ y_axis_name + ".jpeg"
    plt.savefig(out_file)
    return out_file


def main_func(filename):
  with open(filename, 'rb') as csvfile:
     x_axis_name = "number_of_images"
     y_axis_name = filename.split(".csv")[0]
     lines = csv.reader(csvfile, delimiter=',')
     z_axis = []
     y_axis = []
     x_axis = []
     prev_images = 0
     current_images = -1
     for row in lines:
         if len(row) == 3:
            z_axis.append(row[2])
         
         x_axis.append(row[0])
         y_axis.append(row[1])
      
     if len(z_axis) > 1:
        return plot_graph(x_axis, y_axis, "acuracy", y_axis_name, z_axis = z_axis)  
     else:
        return plot_graph(x_axis, y_axis, x_axis_name, y_axis_name, z_axis = None)   

if __name__ == '__main__':
     print main_func(sys.argv[1])         
