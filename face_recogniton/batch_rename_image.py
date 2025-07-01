import os
import glob

current_files = glob.glob('./*.jpg')
print(current_files)

# quit()
index = 1
for i, filename in enumerate(current_files):
    os.rename(filename, './' + "Barry" + str(index) + '.jpg')
    index += 1