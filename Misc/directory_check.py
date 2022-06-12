import os


def main():
    cur_dir = os.getcwd()
    cur_dir = cur_dir + "\section"
    
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
    
    print(cur_dir)
    file_hand = open(cur_dir + "\dummy.txt", "w")
    



if __name__ == "__main__":
    main()
    
    