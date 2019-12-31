import glob
import re
import numpy as np
from os import rename
for filename in glob.glob('T*.mat'):
	#print filename.split('.')[0]
	a='S'+str(int(np.asarray(re.findall(r'\d+',filename)))+21)+'.mat'
	rename(filename, a)
