'''关键字参数'''
# https://docs.pythontab.com/python/python3.5/controlflow.html#tut-keywordargs
# keyword=value
def parrot(voltage,state='a stiff',action='voom',type='Norwegian Blue'):
    print("--this parrot wouldn't ",action,end='')   #end=' ' 表示不换行输出
    print(" if you put ",voltage,"vlots through it.")
    print("--Lovely plimage,the ",type)
    print("--It's ",state,"!")

# parrot(1000)                                          # 1 positional argument
# parrot(voltage=1000)                                  # 1 keyword argument
# parrot(voltage=1000000, action='VOOOOOM')             # 2 keyword arguments
# parrot(action='VOOOOOM', voltage=1000000)             # 2 keyword arguments
# parrot('a million', 'bereft of life', 'jump')         # 3 positional arguments
# parrot('a thousand', state='pushing up the daisies')  # 1 positional, 1 keyword
"""以下调用是无效的"""
# parrot(voltage=5.0, 'dead')  # 不给keyword默认positional，但是参数列表中唯一的positional是voltage
# parrot(110, voltage=220)     # 同上，TypeError: function() got multiple values for keyword argument
# parrot(actor='John Cleese')  # 参数列表中无该keyword

