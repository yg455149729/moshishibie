def opetxt(path):
    f=open(path)
    return f.readlines()
def expression(lines):
    dic={}#构建相应的字典
    for line in lines:
        line=line-line[0]-line[-1]
        line.split(':')
        dic=dic+{line[0]:line[1]}
    for i in dic:
        if(dic.key)


