import time
import os

def main():
    current_dir_contents = os.listdir(os.getcwd())
    print(current_dir_contents)
    for i in range(0, 100):
        print(i)
        print('sleep for 10 seconds...')
        time.sleep(10)

if __name__ == '__main__':
    main()
