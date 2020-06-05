import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


class GneratedCode:
    # 生成大写26个英文字母
    def getCode(self):
        a = random.randint(65, 90)
        b = random.randint(97, 122)
        c = [a, b]
        return chr(random.choice(c))

    # chr可以将0~255的整数变为字符
    # 1.添加小写(done!)

    # 背景
    def bg_color(self):
        return (random.randint(0, 120),
                random.randint(0, 120),
                random.randint(0, 120))

    # 前景
    def fd_color(self):
        return (random.randint(150, 255),
                random.randint(150, 255),
                random.randint(150, 255))

    # 生成图片
    def gen_pic(self):
        # 画板
        w, h = 60, 60
        # 将w,h作为参数，或者直接使用算法只需传入w的值，这么做的化draw.text的参数值会变

        # 生成一个新的画板
        img = Image.new(size=(w, h), mode='RGB', color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # windows字体文件 C://Windows/font
        font = ImageFont.truetype(font=r'/src/Dengl.ttf', size=30)
        for n in range(500):
            for y in range(h):
                for x in range(w):
                    draw.point((x, y), fill=self.bg_color())
            filename =""
            # 生成字符的长度
            for i in range(1):
                code = self.getCode()
                filename=filename+code
                draw.text((60 * i + 20, 15), text=code, fill=self.fd_color(), font=font)
            img.save(r'C:\Users\Administrator\Desktop\verifyCode\{}{}.jpg'.format(filename,n))
            # img.save(r'C:\Users\Administrator\Desktop\verifyCode\test{}.jpg'.format(n))


g = GneratedCode()
pic = g.gen_pic()
