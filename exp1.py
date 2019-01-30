def opentxt(path):
    f=open(path)
    symbol=[' ', '=', '*', '/', '+', '#', '<', '<=', '>', '>=', ':=', '(', ')', ',', ';', '-', '\n', '.' ,':']
    result=f.readlines()
    TMP=result
    for j in symbol:
        temp=[]
        for i in TMP:
            temp=temp+i.split(j)
        TMP=temp
    result=temp
    return result
def main():
    path='C:/Users/45514/Desktop/sample.txt'
    flag=['begin', 'call', 'const', 'do', 'end', 'if', 'odd', 'procedure', 'read', 'then' , 'var',  'while', 'write']
    dic=opentxt(path)
    dic=[i.lower() for i in dic]
    dic1=list(set(dic))
    dic1.remove('')
    for i in flag:
        if i in dic1:
            dic1.remove(i)
    for i in dic1:
        if i.isdigit()==True:
            dic1.remove(i)
    for i in dic1:
        print(i,':',dic.count(i))
main()