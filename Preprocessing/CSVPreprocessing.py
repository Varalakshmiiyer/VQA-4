import os
import csv
import re
from skimage import io
import sys,argparse
new_names = {}
path = ("C:/Users/hits/vqa assignment/VQAMed2018Train/VQAMed2018Train-images/output/").replace('\\','/')
for f in os.listdir(path):
	if f=='Thumbs.db':
		continue		
	new_names[f] = set()
	for item in os.listdir(os.path.join(path,f)):
		if re.sub(".jpg",'',item):
			temp = re.sub(".jpg",'',item)
		else:
			temp = re.sub(".jpeg",'',item)
		new_names.get(f).add(temp)

question_path = "C:/Users/hits/vqa assignment/VQAMed2018Valid/VQAMed2018Valid-QA.csv"
fp = open(question_path, "r")
data = fp.readlines()

out_csvfile = open("C:/Users/hits/Downloads/cv-tricks.com-master/cv-tricks.com-master/Tensorflow-tutorials/tutorial-2-image-classifier/csv_output/outputvalidation.csv", "w")
filewriter = csv.writer(out_csvfile, delimiter=',', lineterminator='\n')
total_rows = 0
#1	rjv03401	what does mri show?	lesion at tail of pancreas
for item in data:
    if(len(item) == 0):
        continue
    row = item.split("\t")
    image_id = row[1].strip()
    question = row[2]
    answer = row[3]
    label = ''
    # find label based on the which part of the new names conatins the image_id
    for key, temp_image_ids in new_names.items():
        if image_id in temp_image_ids:
            label = key

    if (len(label) == 0):
        print('could not find lable for the image:' + image_id)
        continue
    #remove items from the list
    #new_names.get(label).remove(image_id)

    #image_id label question_answer
    filewriter.writerow([image_id, label, (question+ ' ' + answer).strip()])
    total_rows = total_rows + 1

print('processed total rows: '+ str(total_rows))
#print('file could not processed:')
#print(new_names)
