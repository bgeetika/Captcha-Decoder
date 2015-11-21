import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import csv



def plot_graph(x_axis, y_axis, x_axis_name, y_axis_name, z_axis = None):
    if z_axis:
        plt.plot(x_axis,y_axis, 'bs', z_axis, y_axis, 'r^')
    else:
        plt.plot(x_axis,y_axis)
    plt.show()
    if "/" in x_axis_name:
        x_axis_name = x_axis_name.rsplit("/",1)[1]
    if "/" in y_axis_name:
        y_axis_name = y_axis_name.rsplit("/",1)[1]
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    out_file = x_axis_name + "_"+ y_axis_name + ".jpeg"
    return plt
    #plt.savefig(out_file)
    return out_file

'''
def main_chart(direc,prefix):
    list_files = get_list_of_files(direc, prefix)
    print list_files
    list_of_axis = plot_chart(direc, list_files)
    charts(list_of_axis)
'''
def get_list_of_files(direc, prefix):
    files = os.listdir(direc)
    list_files = []
    for filename in files:
        if prefix in filename:
           list_files.append(direc+filename)
    return list_files

def charts(list_of_files):
   m = len(list_of_files)/2 + 1
   n = 2
   f, axarr = plt.subplots(m, 2)
   i = 0
   for x in range(0,m):
       for y in range(0,n):  
           if i == len(list_of_files):
              plt.savefig("out.jpeg")
              return 
           prefix, xaxis, yaxis, zaxis = list_of_files[i]
           if zaxis == []:
              axarr[x,y].plot(xaxis,yaxis)
              
           else:
              axarr[x,y].plot(xaxis,yaxis, 'bs', zaxis, yaxis, 'r^')   
           axarr[x,y].set_title(prefix)
   plt.savefig("out.jpeg")


def listof_axis(direc,list_files):
 list_sub = []
 for filename in list_files:
   with open(filename, 'rb') as csvfile:
     prefix = filename
     x_axis_name = "number_of_images"
     y_axis_name = filename.split(".csv")[0]
     lines = csv.reader(csvfile, delimiter=',')
     z_axis = []
     y_axis = []
     x_axis = []
     prev_images = 0
     current_images = 0
     for row in lines:
         if prev_images > int(row[0]):
	    current_images = prev_images+ int(row[0])
         else:
            current_images = int(row[0])     
         prev_images = current_images 
         if len(row) == 3:
            z_axis.append(row[2])
         
         x_axis.append(current_images)
         y_axis.append(row[1])
     list_sub.append((prefix, x_axis, y_axis, z_axis))
 return list_sub
 
def main_func(filename):
  with open(filename, 'rb') as csvfile:
     x_axis_name = "number_of_images"
     y_axis_name = filename.split(".csv")[0]
     lines = csv.reader(csvfile, delimiter=',')
     z_axis = []
     y_axis = []
     x_axis = []
     prev_images = 0
     current_images = 0
     for row in lines:
         if prev_images > int(row[0]):
	    current_images = prev_images+ int(row[0])
         else:
            current_images = int(row[0])     
         prev_images = current_images 
         if len(row) == 3:
            z_axis.append(row[2])
         
         x_axis.append(current_images)
         y_axis.append(row[1])
      
     if len(z_axis) > 1:
        return plot_graph(x_axis, y_axis, "acuracy", y_axis_name, z_axis = z_axis)  
     else:
        return plot_graph(x_axis, y_axis, x_axis_name, y_axis_name, z_axis = None)   

if __name__ == '__main__':
     #print main_func(sys.argv[1])  
     main_chart(sys.argv[1], sys.argv[2])       
