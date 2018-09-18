"""
Created in 2018 by Adrian Lapico for completion of a Bachelors of Information Technology Honours.
(Github Adroso, Adrian Lapico)

To quickly sort data. Named oystersizer9000 because this helpped assist with processing 9000 images

"""
import csv
from os import listdir
import matplotlib.pyplot as plt
import cv2


PATH_TO_IMAGES = "F:\\Honours Image Library\\Oysters\\archive\\3x3\\"
CSV = "3x3csv.csv"
CSV_NEW = "keptOysters.csv"

input_csv = open(CSV, 'r')
output_csv = open(CSV_NEW, 'w')

i = 0
for line in input_csv:
    if i !=0:
        line = line.split(',')
        image_to_process = PATH_TO_IMAGES+line[0]+".JPG"
        raw_image = cv2.imread(image_to_process)
        plt.imshow(raw_image)
        plt.show()
        user_input = input("Keep?")
        if user_input == "1":
            output_csv.write(line[0]+ "," +line[1]+","+line[2])
        else:
            pass

        if user_input == "9":
            break
    i+=1

input_csv.close()
output_csv.close()