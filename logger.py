import pdb

def changeLoggerTrain(input_to_logg):
    with open("logg.txt", 'r') as f:
        tmpList = []
        old_logg = []
        for line in  f:
            old_logg.append(line)

    with open("logg.txt",'w') as f:
        old_logg = input_to_logg + old_logg
        old_logg.append("\n")
        for el in old_logg:
            f.write(el)

#    pdb.set_trace()
#    tmpList = []
    return

def changeLoggerNet(input_to_logg):
    with open("logg.txt", 'r+') as f:
        new_logg = []
        old_logg = []
        for line in f:
            old_logg

    return
