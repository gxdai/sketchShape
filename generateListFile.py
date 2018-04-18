import os
import argparse
import numpy as np

def generateListFile(inputDir, outputList):
    fid = open(outputList, 'w')
    for rootdir, subdir, files in os.walk(inputDir):
        for f in files:
            fileExt = f.split('.')[-1]
            if fileExt == 'df' or fileExt == 'sdf':
                filepath = os.path.join(rootdir, f)
                fid.write(filepath+'\n')
    fid.close()

def generateTxtListFile(inputDir, outputTrainList, outputTestList):

    def getSubList(fid, subPath, label):
        fileSet = sorted(os.listdir(subPath))
        for f in fileSet:
            split_f = f.split('.')
            if split_f[-1] == 'txt':
                fid.write(os.path.join(subPath, f) + ' ' + str(label) + '\n')

        return fid

    fid_train = open(outputTrainList, 'w')
    fid_test = open(outputTestList, 'w')
    classesAll = sorted(os.listdir(inputDir))
    # remove .DS_store
    classesAll = [groupName for groupName in classesAll if 'DS_Store' not in groupName]

    for label, groupName in enumerate(classesAll):
        fid_train = getSubList(fid_train, os.path.join(inputDir, groupName, 'train'), label)
        fid_test = getSubList(fid_test, os.path.join(inputDir, groupName, 'test'), label)

    fid_train.close()
    fid_test.close()

def generateShapeTxtListFile(inputDir, outputList):

    def getSubList(fid, subPath, label):
        fileSet = sorted(os.listdir(subPath))
        for f in fileSet:
            split_f = f.split('.')
            if split_f[-1] == 'txt':
                fid.write(os.path.join(subPath, f) + ' ' + str(label) + '\n')

        return fid

    fid = open(outputList, 'w')
    classesAll = sorted(os.listdir(inputDir))
    print(type(os.listdir(inputDir)))
    # classesAll = os.listdir(inputDir).sort()
    # remove .DS_store
    classesAll = [groupName for groupName in classesAll if 'DS_Store' not in groupName]

    print(classesAll)

    for label, groupName in enumerate(classesAll):
        fid = getSubList(fid, os.path.join(inputDir, groupName), label)

    fid.close()

def generateListFileByClass(inputDir, outputList):
    fid = open(outputList, 'w')
    classSet = os.listdir(inputDir)
    for i, cls in enumerate(classSet):
        classPath = os.path.join(inputDir, cls)
        for rootdir, subdir, files in os.walk(classPath):
            for tmp_file in files:
                splitFileName = tmp_file.split('.')
                fileExt = splitFileName[-1]
                if fileExt == 'PNG' or fileExt == 'JPG' or \
                        fileExt == 'png' or fileExt == 'jpg':
                    fullpath = os.path.join(rootdir, tmp_file)
                    fid.write("{} {}\n".format(fullpath, i))
    fid.close()



def convertTxt2Npy(inputListFile, featureFile, labelFile):
    with open(inputListFile, 'r') as fid:
        lines = fid.readlines()
    fileNum = len(lines)

    feaArray = np.zeros((fileNum, 4096))
    labelArray = np.zeros((fileNum, 1))

    for i, line in enumerate(lines):
        splitLine = line.split(' ')
        feaArray[i] = np.loadtxt(splitLine[0])
        labelArray[i] = int(splitLine[1])

    np.save(featureFile, feaArray)
    np.save(labelFile, labelArray)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers')
    parser.add_argument('--inputDir', type=str)
    parser.add_argument('--outputList', type=str)
    parser.add_argument('--outputTrainList', type=str)
    parser.add_argument('--outputTestList', type=str)
    parser.add_argument('--inputListFile', type=str)
    parser.add_argument('--featureFile', type=str)
    parser.add_argument('--labelFile', type=str)

    args = parser.parse_args()

    # convertTxt2Npy(inputListFile=args.inputListFile, featureFile=args.featureFile, labelFile=args.labelFile)
    generateTxtListFile(inputDir=args.inputDir, outputTrainList=args.outputTrainList, outputTestList=args.outputTestList)
    # generateShapeTxtListFile(inputDir=args.inputDir, outputList=args.outputList)
