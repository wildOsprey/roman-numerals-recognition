from PIL import Image
import os

def crop(image, rows,columns, subimgheight,subimgwidth):
    for i in range(rows):
        for j in range(columns):
            box = (j*subimgwidth, i*subimgheight, (j+1)*subimgwidth, (i+1)*subimgheight)
            yield image.crop(box)


if __name__=='__main__':
    infile='C:\\Users\\BIONIC Admin\\Downloads\\Telegram Desktop\\IMG_1761.PNG'
    rows=15
    columns=24

    im = Image.open(infile)
    imgwidth, imgheight = im.size
    subimgwidth=imgwidth//columns
    subimgheight=imgheight//rows

    for k,piece in enumerate(crop(im,rows,columns,subimgheight, subimgwidth)):
        img=Image.new('RGB', (subimgwidth,subimgheight), 255)
        img.paste(piece)
        path=os.path.join('F:\\MachineLearning\\Int20h\\TestTask\\dataset\\Kostia',"IMG-%s.png" % k)
        img.save(path)
        