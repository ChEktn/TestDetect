
from PIL import Image, ImageTk
from random import randint
import csv

FILENAME = "train_t.csv"
FILENAME_VAL = 'test_t.csv'


if __name__ == '__main__':
    for i in range(500):
        color_rand = (randint(110, 255), randint(110, 255), randint(110, 255))
        img = Image.new(mode='RGB', size=(500, 800), color=color_rand)
        menu_i = randint(1, 7)
        path_menu = f'burger_menus/burger_menu_{menu_i}.png'
        img1 = Image.open(path_menu)
        size_menu = (randint(20, 31), randint(20, 31))
        img1 = img1.resize(size_menu)
        pos_menu = (randint(10, 490), randint(10, 250))
        img.paste(img1, pos_menu, mask=img1)

        for j in range(20):
            ex_i = randint(0, 26)
            path_ex = f'example_menu/ex{ex_i}.png'
            img_ex = Image.open(path_ex)
            size_ex= (randint(27, 45), randint(27, 45))
            img_ex = img_ex.resize(size_ex)
            pos_ex = (randint(10, 490), randint(10, 790))
            img.paste(img_ex, pos_ex, mask=img_ex)

        img.save(f't_image/image_{i}.jpg')

        if i == 350:
            FILENAME = FILENAME_VAL
        with open(FILENAME, "a", newline="") as file:
            writer = csv.writer(file)
            list_masks = [f'image_{i}.jpg', pos_menu[0], pos_menu[1], pos_menu[0]+size_menu[0], pos_menu[1]+size_menu[1], 'burger-menu']
            writer.writerow(list_masks)

    #img_mask = Image.new(mode='1', size=(500, 800), color=0)
    #img_mask_menu = Image.new(mode='1', size=size_menu, color=1)
    #img_mask.paste(img_mask_menu, pos_menu)
    #img_mask.save(f'masks/mask_{i}.jpg')