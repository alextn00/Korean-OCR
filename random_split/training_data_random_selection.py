import os
import random
from shutil import copyfile

def main():
    print("select source directory")
    ldir = os.listdir()
    ldir = [l for l in ldir if os.path.isdir(l)]
    print(ldir)
    source = input()

    print("select target directory")
    target = input()

    ldir2 = os.listdir(source)
    print(f"There are {len(ldir2)} directory in source directory")
    print("how many items do you want to select randomly?")
    nums = input()
    val = 0

    try : 
        val = int(nums)
    except ValueError : 
        print("Not Number !")
        return
    print(f"You want to select {nums} in each the directories !")

    for dir in ldir2 : 
        os.makedirs(f"./{target}/{dir}/")

    for dir in ldir2 :
        index = 0
        ldir3 = os.listdir(f"./{source}/{dir}/")
        selected = random.sample(ldir3, val)
        print(f"{len(selected)} selected in {dir} !!")
        for file in selected :
            copyfile(f"./{source}/{dir}/{file}", f"./{target}/{dir}/{dir}_{index}.png")
            index += 1

if __name__ == '__main__':
    main()