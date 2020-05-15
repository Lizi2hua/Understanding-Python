# https://cloud.tencent.com/developer/section/1370514
# 下面的代码是一个 Python 程序，它接受一个整数列表并产生总和或最大值：
import argparse

# 创建一个解析器
parser = argparse.ArgumentParser(description="process some integers.")
#添加参数，通过调用add_argument来完成关于参数的信息填充，告诉ArgumentParser如何解析命令
parser.add_argument("integers", metavar="N", type=int, nargs="+",
                    help="an integer for the accumulator")
parser.add_argument("--sum", dest="accumulate", action="store_const",
                    const=sum, default=max, help="sum the integers(default:find the max")
#上面的信息在parse_args()调用时被存储和使用
args = parser.parse_args()
print(args.accumulate(args.integers))
