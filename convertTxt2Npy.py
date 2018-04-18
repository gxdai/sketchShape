import numpy as np

def convertTxt2Npy(inputListFile, featureFile, labelFile):
    with open(inputListFile, 'r') as fid:
        lines = fid.readlines()
    fileNum = len(lines)

    feaArray = np.zeros(fileNum, 4096)
    labelArray = np.zeros(fileNum, 1)

    for i, line in enumerate(lines):
        splitLine = line.split(' ')
        feaArray[i] = np.loadtxt(splitLine[0])
        labelArray[i] = int(splitLine[1]) 



    np.save(featureFile, feaArray)
    np.save(labelFile, labelArray)


    


