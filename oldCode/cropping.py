from PIL import Image
import glob


def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


photos = glob.glob("./bone_age/training/*.png")
for photo in photos:
    img = Image.open(photo)
    img = make_square(img)
    # img.show()
    width = 512
    height = 512
    img = img.resize((width, height), Image.ANTIALIAS)  # best down-sizing filter
    # img.show()
    img.save(photo.replace("bone_age", "normalized"), "PNG")
