import random
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt

class GneratedCode:
    # 生成大写26个英文字母
    def getCode(self):
        a=random.randint(65,90)
        b=random.randint(97,122)
        c=[a,b]
        return chr(random.choice(c))
    # chr可以将0~255的整数变为字符
    # 1.添加小写(done!)
    # 2.写汉字

    #背景
    def bg_color(self):
        return  (random.randint(0,120),
                 random.randint(0,120),
                 random.randint(0,120))
    #前景
    def fd_color(self):
        return  (random.randint(150,255),
                 random.randint(150,255),
                 random.randint(150,255))
    # 生成图片
    def gen_pic(self):
        #画板
        w,h=240,60
        # 将w,h作为参数，或者直接使用算法只需传入w的值，这么做的化draw.text的参数值会变


        # 生成一个新的画板
        img=Image.new(size=(w,h),mode='RGB',color=(255,255,255))
        draw=ImageDraw.Draw(img)

        # windows字体文件 C://Windows/font
        font=ImageFont.truetype(font=r'/src/Dengl.ttf',size=30)
        for n in range(10):
            for y in range(h):
                for x in range(w):
                    draw.point((x,y),fill=self.bg_color())
            for i in range(4):
                draw.text((60*i+20,15),text=self.getCode(),fill=self.fd_color(),font=font)

        # img.show()
            img.save(r'C:\Users\Administrator\Desktop\Project：777\CODE\python\src\test{}.jpg'.format(n))

g=GneratedCode()
pic=g.gen_pic()
# plt.imshow(pic)
# plt.show()






















