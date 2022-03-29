import pickle
import errno
import os


def LoadFromPickleFile(localFile):
    try:
            f = open(localFile,'rb')
    except OSError:
            print('Error! Reading local pickle file：%s' %(localFile))
            return -1
    pickle_file = pickle.load(f)
    f.close()
    #print('Success! Reading local pickle file：%s' %(localFile))
    return pickle_file


def SaveToPickleFile(myValue, localFileName):
    path = os.path.dirname(localFileName)
    try:
        os.makedirs(path)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        elif path == "":
            pass
        else: raise
    try:
        f = open(localFileName,'wb') #Overwrite the original file
    except OSError:
        print('Error! Writing context to local pickle file：%s' %(localFileName))
        return

    pickle.dump(myValue, f)
    f.close()