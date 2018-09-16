import os
limit = 5
net=0
for i in range(1,5000000):
    try:
        os.system("python3 train.py")
    except ValueError:
        pass
    os.system("find . -size -1c -delete")
